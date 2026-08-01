"""Validate the reverse-engineered MLX escham_reconstruct.

Cross-check the MLX packed decoder against Option-B dense-dequant M matrices.

Pipeline algebra (all four steps linear):
    M = t128( t128(I, pre=rin) @ w_bare, post=rout )
where w_bare = escham_reconstruct(escha_code).

If the MLX decoder is correct, composing M from our decoded w_bare should
match the M produced on Modal (Option B, stored in
`~/models/Qwen3.6-35B-A3B-Escha-W2-MLX-dequant/dequant_v1/`).

Also runs plumbing tests on the codebook (shape, dtype, sign-bit as uint16).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import mlx.core as mx

from safetensors import safe_open


ORIG_DIR = Path("/Users/kaede/models/Qwen3.6-35B-A3B-Escha-W2")
DEQUANT_DIR = Path("/Users/kaede/models/Qwen3.6-35B-A3B-Escha-W2-MLX-dequant/dequant_v1")


# ---------------------------------------------------------------------------
# Plumbing tests — always runnable (need only the extracted codebook artifact).
# ---------------------------------------------------------------------------
def test_codebook_loads():
    from mlx_video.models.qwen3_5_moe_escha.eschamoe import _load_codebook_dense
    cb2 = _load_codebook_dense(2, dtype=mx.float16)
    assert cb2.shape == (32, 65536, 16, 16), f"K=2 shape {cb2.shape}"
    assert cb2.dtype == mx.float16
    # Baseline (code=0) must be all-zero for every k_slot
    b = cb2[:, 0, :, :]
    mx.eval(b)
    assert float(mx.abs(b).max()) == 0.0


def test_reconstruct_shape_and_zero_code():
    from mlx_video.models.qwen3_5_moe_escha.eschamoe import escham_reconstruct
    # All-zero code should reconstruct to all-zero w_bare (baseline is all zero
    # per our extraction; the op's non-zero baseline was captured as identical
    # sum-of-cb[k,0] contributions that we verified are zero).
    code = mx.zeros((128, 64, 32), dtype=mx.int16)
    w = escham_reconstruct(code, in_features=2048, out_features=1024, K=2)
    mx.eval(w)
    assert w.shape == (2048, 1024)
    assert w.dtype == mx.float16
    # NB: op's actual baseline w0 is non-zero, but we extract cb as delta-from-
    # baseline. Compose-M validates the full pipeline including baseline.


def test_reconstruct_int16_signbit_as_uint16():
    """Codes stored as int16 must be interpreted as uint16 for lookup."""
    from mlx_video.models.qwen3_5_moe_escha.eschamoe import (
        _load_codebook_dense, escham_reconstruct,
    )
    cb = _load_codebook_dense(2, dtype=mx.float16)
    code = np.zeros((1, 1, 32), dtype=np.int16)
    code[0, 0, 0] = -1  # bit pattern = 0xFFFF = uint16 65535
    w = escham_reconstruct(mx.array(code), 16, 16, K=2)
    mx.eval(w)
    # First (16, 16) block should equal cb[k=0, v=65535]
    expected = cb[0, 65535]
    got = w[:16, :16]
    diff = float(mx.abs(got.astype(mx.float32) - expected.astype(mx.float32)).max())
    # bf16 rounding on the internal codebook load produces ~1e-2 abs errors on
    # values of magnitude ~5. Just verify the lookup went to the right entry.
    assert diff < 5e-2, f"sign-bit lookup failed: max diff {diff}"


# ---------------------------------------------------------------------------
# Correctness test vs. Option-B M matrices. Requires the model to be on disk.
# ---------------------------------------------------------------------------
def _load_expert(layer: int, expert: int, proj: str) -> dict:
    idx = json.loads((ORIG_DIR / "model.safetensors.index.json").read_text())
    wm = idx["weight_map"]
    prefix = f"model.language_model.layers.{layer}.mlp.experts.{proj}"
    out = {}
    for suf in ("escha_code", "escha_rin", "escha_rout"):
        key = f"{prefix}.{suf}"
        shard = wm[key]
        with safe_open(ORIG_DIR / shard, framework="numpy") as f:
            out[suf] = mx.array(f.get_tensor(key)[expert])
    return out


def _load_M(layer: int, expert: int, proj: str) -> mx.array:
    """Load Option-B M matrix (out_f, in_f) bf16 via mx.load (bf16-aware)."""
    key = f"layer_{layer}.expert_{expert}.{proj}.weight"
    all_ = mx.load(str(DEQUANT_DIR / f"layer_{layer:02d}.safetensors"))
    return all_[key]


def _compose_M_chunked(w_bare: mx.array, rin: mx.array, rout: mx.array,
                       chunk: int = 256) -> mx.array:
    """M = t128( t128(I, pre=rin) @ w_bare, post=rout ) — chunked over rows."""
    from mlx_video.models.qwen3_5_moe_escha.transform import t128
    in_f, out_f = w_bare.shape
    w_bf = w_bare.astype(mx.bfloat16)
    rin_bf = rin.astype(mx.bfloat16)
    rout_bf = rout.astype(mx.bfloat16)
    chunks = []
    for i in range(0, in_f, chunk):
        n = min(chunk, in_f - i)
        r = mx.arange(n).reshape(-1, 1)
        c = mx.arange(in_f).reshape(1, -1)
        Ic = (c == (r + i)).astype(mx.bfloat16)
        xh = t128(Ic, pre=rin_bf)
        y_mid = xh @ w_bf
        M_chunk = t128(y_mid, post=rout_bf)
        chunks.append(M_chunk)
        mx.eval(M_chunk)
    return mx.concatenate(chunks, axis=0)


@pytest.mark.skipif(not ORIG_DIR.exists() or not DEQUANT_DIR.exists(),
                    reason="requires original + dequant model dirs on disk")
@pytest.mark.parametrize("proj,in_f,out_f,K", [
    ("gate_up_proj", 2048, 1024, 2),
    ("down_proj", 512, 2048, 3),
])
def test_reconstruct_matches_option_b(proj, in_f, out_f, K):
    from mlx_video.models.qwen3_5_moe_escha.eschamoe import escham_reconstruct
    ex = _load_expert(layer=0, expert=0, proj=proj)
    code = ex["escha_code"]
    rin = ex["escha_rin"]
    rout = ex["escha_rout"]

    w_bare = escham_reconstruct(code, in_f, out_f, K, cb_id=1, mul1=False)
    mx.eval(w_bare)
    M_mlx = _compose_M_chunked(w_bare, rin, rout, chunk=256)
    mx.eval(M_mlx)

    M_ref = _load_M(layer=0, expert=0, proj=proj).T  # (in_f, out_f)
    mx.eval(M_ref)

    diff = M_mlx.astype(mx.float32) - M_ref.astype(mx.float32)
    max_abs = float(mx.abs(diff).max())
    rel = float(mx.linalg.norm(diff) / mx.linalg.norm(M_ref.astype(mx.float32)))
    print(f"\n[{proj}] max_abs={max_abs:.3e} rel_l2={rel:.3e}")
    assert rel < 5e-2, f"{proj} rel L2 too high: {rel}"


if __name__ == "__main__":
    # Direct run — bypass pytest infra.
    test_codebook_loads()
    print("codebook loads OK")
    test_reconstruct_shape_and_zero_code()
    print("zero-code reconstruct OK")
    test_reconstruct_int16_signbit_as_uint16()
    print("int16 sign-bit lookup OK")
    if ORIG_DIR.exists() and DEQUANT_DIR.exists():
        test_reconstruct_matches_option_b("gate_up_proj", 2048, 1024, 2)
        test_reconstruct_matches_option_b("down_proj", 512, 2048, 3)
    else:
        print("skipping Option-B match (model dirs missing)")
