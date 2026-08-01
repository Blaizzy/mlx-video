"""GatedDeltaNet (Escha linear_attn) — unit tests.

Coverage:
    - shape smoke                     (synthetic weights, sanity dims)
    - numeric finiteness              (zeros / ones input → no NaN)
    - state-cache correctness         (autoregressive == parallel; critical)
    - multi-layer chaining            (2 GDN layers, no blow-up)
    - real-weight shape smoke         (skipped if checkpoint absent)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import mlx.core as mx


MODEL_DIR = Path.home() / "models" / "Qwen3.6-35B-A3B-Escha-W2"

# Escha-W2 layer dims (from config.json).
CFG = dict(
    hidden_size=2048,
    num_v_heads=32,
    num_k_heads=16,
    head_k_dim=128,
    head_v_dim=128,
    conv_kernel_size=4,
    rms_norm_eps=1e-6,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _synthetic_weights(seed: int = 0, dtype_int8=mx.int8):
    """Build a small dict of synthetic weights matching Escha's linear_attn shapes.

    Uses random int8 codes with fp16 per-row scales so `Int8Linear` dequantizes
    to a valid bf16 weight (values ~small so activations stay in-range).
    """
    mx.random.seed(seed)
    hidden = CFG["hidden_size"]
    nk, dk, nv, dv = (
        CFG["num_k_heads"], CFG["head_k_dim"],
        CFG["num_v_heads"], CFG["head_v_dim"],
    )
    key_dim = nk * dk
    value_dim = nv * dv
    conv_dim = key_dim * 2 + value_dim
    K = CFG["conv_kernel_size"]

    def _int8(out_f, in_f):
        # Uniform in [-8, 8] → dequant weights ~0.05 (with scale=0.01) — small.
        w = (mx.random.uniform(-8, 8, (out_f, in_f)) + 0.5).astype(mx.int8)
        s = (mx.random.uniform(0.005, 0.02, (out_f,))).astype(mx.float16)
        return w, s

    qkv_w, qkv_s = _int8(key_dim * 2 + value_dim, hidden)
    z_w, z_s = _int8(value_dim, hidden)
    out_w, out_s = _int8(hidden, value_dim)

    return {
        "in_proj_qkv.weight_int8": qkv_w,
        "in_proj_qkv.weight_scale": qkv_s,
        "in_proj_z.weight_int8": z_w,
        "in_proj_z.weight_scale": z_s,
        "in_proj_a.weight": (mx.random.normal((nv, hidden)) * 0.02).astype(mx.float16),
        "in_proj_b.weight": (mx.random.normal((nv, hidden)) * 0.02).astype(mx.float16),
        # conv1d weight in ESCHA layout: (C, 1, K).
        "conv1d.weight": (mx.random.normal((conv_dim, 1, K)) * 0.1).astype(mx.float16),
        # A_log: log(A) with A in [0.1, 16]  → A_log in ~[-2.3, 2.77].
        "A_log": mx.log(mx.random.uniform(0.5, 8.0, (nv,))).astype(mx.float16),
        "dt_bias": (mx.random.normal((nv,)) * 0.1 - 3.0).astype(mx.float16),
        "norm.weight": mx.ones((dv,)).astype(mx.float16),
        "out_proj.weight_int8": out_w,
        "out_proj.weight_scale": out_s,
    }


def _make_layer():
    from mlx_video.models.qwen3_5_moe_escha.gated_deltanet import GatedDeltaNet
    return GatedDeltaNet(_synthetic_weights(), **CFG)


# --------------------------------------------------------------------------- #
# Shape smoke
# --------------------------------------------------------------------------- #


def test_forward_shape_synthetic():
    layer = _make_layer()
    x = mx.random.normal((1, 32, CFG["hidden_size"])).astype(mx.bfloat16)
    y = layer(x)
    assert y.shape == (1, 32, CFG["hidden_size"])
    assert y.dtype == mx.bfloat16


def test_forward_zeros_finite():
    layer = _make_layer()
    x = mx.zeros((1, 8, CFG["hidden_size"]), dtype=mx.bfloat16)
    y = layer(x)
    assert y.shape == (1, 8, CFG["hidden_size"])
    assert bool(mx.all(mx.isfinite(y.astype(mx.float32))).item())


def test_forward_ones_finite():
    layer = _make_layer()
    x = mx.ones((1, 8, CFG["hidden_size"]), dtype=mx.bfloat16)
    y = layer(x)
    assert bool(mx.all(mx.isfinite(y.astype(mx.float32))).item())


# --------------------------------------------------------------------------- #
# The critical test: cache correctness (autoregressive vs. parallel)
# --------------------------------------------------------------------------- #


def test_cache_matches_parallel():
    """A GDN layer must produce the SAME output when fed one big prompt vs.
    fed the same tokens one-by-one through a stateful cache.

    This is the highest-signal sanity check on the recurrence + conv-state
    plumbing — if it fails, either the SSM state update or the conv1d cache
    is wrong.
    """
    from mlx_video.models.qwen3_5_moe_escha.gated_deltanet import GDNCache

    layer = _make_layer()
    mx.random.seed(42)
    S = 6
    x = (mx.random.normal((1, S, CFG["hidden_size"])) * 0.5).astype(mx.bfloat16)

    # Parallel pass.
    y_par = layer(x)                                        # (1, S, H)

    # Sequential pass.
    cache = GDNCache()
    ys = []
    for t in range(S):
        ys.append(layer(x[:, t : t + 1, :], cache=cache))
    y_seq = mx.concatenate(ys, axis=1)                      # (1, S, H)

    # Cast to fp32 for the numerical comparison. bf16's ~3-decimal accuracy
    # means we can't demand tight equality — atol 5e-2 is what the reference
    # Qwen3Next test uses too (see mlx_lm test_qwen3_next).
    diff = np.array((y_par - y_seq).astype(mx.float32))
    par = np.array(y_par.astype(mx.float32))
    max_abs = float(np.max(np.abs(diff)))
    max_rel = float(np.max(np.abs(diff) / (np.abs(par) + 1e-3)))
    assert max_abs < 5e-2, (
        f"cache mismatch: max_abs={max_abs:.4e} max_rel={max_rel:.4e}"
    )


def test_cache_matches_parallel_chunked():
    """Split the same 6-token prompt into a (3, 3) chunk pair — result should
    match the single 6-token forward. Exercises multi-token cache updates,
    not just token-by-token."""
    from mlx_video.models.qwen3_5_moe_escha.gated_deltanet import GDNCache

    layer = _make_layer()
    mx.random.seed(7)
    S = 6
    x = (mx.random.normal((1, S, CFG["hidden_size"])) * 0.5).astype(mx.bfloat16)

    y_par = layer(x)

    cache = GDNCache()
    y_a = layer(x[:, :3, :], cache=cache)
    y_b = layer(x[:, 3:, :], cache=cache)
    y_seq = mx.concatenate([y_a, y_b], axis=1)

    diff = np.array((y_par - y_seq).astype(mx.float32))
    max_abs = float(np.max(np.abs(diff)))
    assert max_abs < 5e-2, f"chunked cache mismatch: max_abs={max_abs:.4e}"


# --------------------------------------------------------------------------- #
# Multi-layer chain
# --------------------------------------------------------------------------- #


def test_two_layer_chain_finite():
    from mlx_video.models.qwen3_5_moe_escha.gated_deltanet import GatedDeltaNet

    l1 = GatedDeltaNet(_synthetic_weights(seed=1), **CFG)
    l2 = GatedDeltaNet(_synthetic_weights(seed=2), **CFG)
    x = (mx.random.normal((1, 16, CFG["hidden_size"])) * 0.3).astype(mx.bfloat16)
    y = l2(l1(x))
    assert y.shape == x.shape
    assert bool(mx.all(mx.isfinite(y.astype(mx.float32))).item())


# --------------------------------------------------------------------------- #
# Real-weight smoke — skipped if the checkpoint isn't on disk.
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    not (MODEL_DIR / "model.safetensors.index.json").exists(),
    reason=f"Escha-W2 checkpoint not present at {MODEL_DIR}",
)
def test_real_weights_layer0_shape_smoke():
    """Load layer-0 linear_attn weights from the actual checkpoint and run a
    forward. Verifies the projection-split matches Escha's export layout."""
    from mlx_video.models.qwen3_5_moe_escha.weight_loader import load_state
    from mlx_video.models.qwen3_5_moe_escha.gated_deltanet import GatedDeltaNet

    state = load_state(MODEL_DIR)
    prefix = "model.language_model.layers.0.linear_attn."
    weights = {
        k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)
    }
    assert weights, "no linear_attn weights found for layer 0"

    layer = GatedDeltaNet(weights, **CFG)
    x = mx.random.normal((1, 4, CFG["hidden_size"])).astype(mx.bfloat16) * 0.1
    y = layer(x)
    assert y.shape == (1, 4, CFG["hidden_size"])
    assert bool(mx.all(mx.isfinite(y.astype(mx.float32))).item())
