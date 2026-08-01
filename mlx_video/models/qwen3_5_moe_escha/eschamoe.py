"""escham_reconstruct + PackedScaledExpertLinear — the 2-bit AQLM decode.

WORLD-FIRST NON-ESCHA IMPLEMENTATION of the packed 2-bit format.

Reverse-engineered from `torch.ops.escha.escham_reconstruct` via three Modal
A10G probes (`codebooks/modal_op_audit.py`, `codebooks/modal_smart_probe.py`):

1. Op signature: `escham_reconstruct(Tensor packed, int in_f, int out_f, int K,
   bool cbA, bool mul1) -> Tensor(in_f, out_f) fp16`
2. Op is EXACTLY LINEAR in code entries (superposition |diff|=0).
3. Codebook is (bi, bj)-INVARIANT: the same code value at any block position
   produces the same 16x16 delta, only offset by (bi*16, bj*16).
4. Each (K, k_slot, v) triplet maps to a 16x16 dense block with ~5-9 nonzero
   entries. The union-of-nonzero positions per (K, k_slot) is 10-15 (very
   sparse), giving a compact codebook of ~120 MB total for K=2 and K=3.

Decode formula:
    w_bare[bi*16:(bi+1)*16, bj*16:(bj+1)*16]
        = sum over k of place(cb[K, k, code[bi, bj, k]], positions[K, k])

Codebook artifact: `codebooks/layout_v2/compact.pkl` produced by the smart
prober on Modal A10G. Compact format:
    K{2,3}_positions[k] : (n_nz, 2) int8  — (row, col) positions in the block
    K{2,3}_values[k]    : (65536, n_nz) fp16 — cb values per code index
"""

from __future__ import annotations

import pickle
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .transform import t128


_CB_PATH = Path(__file__).parent / "codebooks" / "layout_v2" / "compact.pkl"
# Cache: keyed by (K, dtype) -> dense (k_max, 65536, 16, 16) mx.array codebook
_CB_CACHE: dict[tuple[int, str], mx.array] = {}


# --- BASELINE_V2 WIRE-IN (auto-inserted) ---
# See codebooks/wire_baseline_v2.py. `w0 = op(all_zeros_code, in_f, out_f, K)`
# captured on the reference CUDA runtime; subtracting it isolates the
# codebook sum from Escha's shape-dependent additive bias.
_BASELINE_NPZ = Path(__file__).parent / "codebooks" / "baseline_v2.npz"
_BASELINE_CACHE: dict[tuple[int, int, int], "mx.array"] = {}


def _baseline_w0(in_features: int, out_features: int, K: int, dtype) -> "mx.array":
    key = (int(in_features), int(out_features), int(K))
    if key in _BASELINE_CACHE:
        return _BASELINE_CACHE[key].astype(dtype)
    if not _BASELINE_NPZ.exists():
        raise FileNotFoundError(
            f"baseline_v2.npz not found at {_BASELINE_NPZ}. "
            f"Run codebooks/baseline_probe_colab.ipynb on a T4 and wire via "
            f"codebooks/wire_baseline_v2.py."
        )
    import numpy as _np
    with _np.load(_BASELINE_NPZ, allow_pickle=False) as npz:
        want = f"in{in_features}_out{out_features}_K{K}"
        match = [k for k in npz.files if k.endswith(f"__{want}")]
        if not match:
            raise KeyError(
                f"baseline_v2.npz has no entry for {want}; "
                f"present keys: {list(npz.files)}"
            )
        arr = mx.array(npz[match[0]].astype(_np.float32))
    _BASELINE_CACHE[key] = arr
    return arr.astype(dtype)
# --- /BASELINE_V2 WIRE-IN ---


def _load_codebook_dense(K: int, dtype: mx.Dtype = mx.bfloat16) -> mx.array:
    """Load the codebook for K in {2, 3}, densified to (k_max, 65536, 16, 16).

    Storage cost (dense): K=2 → 32 × 65536 × 256 × 2 B = 1.07 GB fp16.
                         K=3 → 48 × 65536 × 256 × 2 B = 1.61 GB fp16.
    Held on-device (unified memory) for zero-copy lookups. Cached across calls.
    """
    key = (K, str(dtype))
    if key in _CB_CACHE:
        return _CB_CACHE[key]
    if not _CB_PATH.exists():
        raise FileNotFoundError(
            f"Escha packed codebook not found at {_CB_PATH}.\n"
            f"Run: modal run mlx_video/models/qwen3_5_moe_escha/codebooks/modal_smart_probe.py\n"
            f"(requires the escha wheel + a CUDA GPU; ~1024 op calls, ~2 min A10G)."
        )
    import numpy as np
    with open(_CB_PATH, "rb") as f:
        data = pickle.load(f)
    positions = data[f"K{K}_positions"]      # list of (n_nz, 2) int8
    values = data[f"K{K}_values"]            # list of (65536, n_nz) fp16
    k_max = len(positions)
    dense = np.zeros((k_max, 65536, 16, 16), dtype=np.float16)
    for k, (pos, val) in enumerate(zip(positions, values)):
        for i, (r, c) in enumerate(pos):
            dense[k, :, r, c] = val[:, i]
    cb = mx.array(dense).astype(dtype)
    _CB_CACHE[key] = cb
    return cb


def escham_reconstruct(
    code: mx.array,
    in_features: int,
    out_features: int,
    K: int,
    cb_id: int = 1,
    mul1: bool = False,
) -> mx.array:
    """Decode packed 2-bit Escha codes to a dense fp16 weight matrix.

    Args:
        code: int16 of shape (in_features/16, out_features/16, 16*K).
        in_features, out_features: padded dims (in_p, out_p). Escha-W2 has no padding.
        K: residual depth (2 for gate_up, 3 for down).
        cb_id: codebook variant — always 1 for Escha-W2 (cbA); other values would
               swap the codebook and are not implemented (unused in the model).
        mul1: unused; kept for schema parity with the reference op signature.

    Returns:
        fp16 tensor of shape (in_features, out_features).

    Algorithm:
        cb has shape (k_max, 65536, 16, 16). For each (bi, bj, k):
            block[bi*16:(bi+1)*16, bj*16:(bj+1)*16] += cb[k, code[bi,bj,k]]
        Vectorized via a single `mx.take` per k_slot (or one gather over all k).
    """
    if cb_id != 1:
        raise NotImplementedError(f"Only cb_id=1 (Escha-W2 default) supported; got {cb_id}")
    k_max = 16 * K
    if code.shape[-1] != k_max:
        raise ValueError(f"code last dim {code.shape[-1]} != 16*K={k_max}")
    bi_max = in_features // 16
    bj_max = out_features // 16
    if code.shape[0] != bi_max:
        raise ValueError(f"code.shape[0]={code.shape[0]} != in_features/16={bi_max}")
    if code.shape[1] != bj_max:
        raise ValueError(f"code.shape[1]={code.shape[1]} != out_features/16={bj_max}")

    cb = _load_codebook_dense(K, dtype=mx.bfloat16)   # (k_max, 65536, 16, 16) bf16

    # Reinterpret int16 → uint16 index in [0, 65536).
    idx = (code.astype(mx.int32) & 0xFFFF)             # (bi, bj, k_max)
    # Per-k gather: for each k, cb[k] is (65536, 16, 16). We index with idx[:,:,k]
    # to get (bi, bj, 16, 16) blocks. Do this for all k, then sum on k axis.
    # Vectorized via advanced indexing on axis 0 = k, axis 1 = code index.
    # We build shape-(k_max, bi_max, bj_max) index of code-value pairs, then
    # gather (16, 16) blocks -> (k_max, bi_max, bj_max, 16, 16) -> sum over k
    # -> (bi_max, bj_max, 16, 16) -> transpose to (bi, 16, bj, 16) -> reshape.
    idx_k_major = idx.transpose(2, 0, 1)               # (k, bi, bj)
    # cb: (k, 65536, 16, 16). Gather cb[k, idx_k_major[k]] for each k.
    # mx.take_along_axis needs same-rank; simpler: reshape+gather manually.
    # We compute:  blocks[k, bi, bj] = cb[k, idx_k_major[k, bi, bj]]
    # by flattening (k, code) to a single dimension.
    flat_cb = cb.reshape(k_max * 65536, 16, 16)
    flat_idx = (mx.arange(k_max, dtype=mx.int32) * 65536).reshape(k_max, 1, 1) + idx_k_major
    blocks = mx.take(flat_cb, flat_idx, axis=0)        # (k, bi, bj, 16, 16)
    per_block = blocks.sum(axis=0)                     # (bi, bj, 16, 16)
    # Reassemble: (bi, 16, bj, 16) -> (in, out)
    per_block = per_block.transpose(0, 2, 1, 3)        # (bi, 16, bj, 16)
    w = per_block.reshape(in_features, out_features)
    w = w + _baseline_w0(in_features, out_features, K, w.dtype)
    return w.astype(mx.float16)


class PackedScaledExpertLinear(nn.Module):
    """One expert projection: code + rin + rout, all fused into a single matmul.

    Mirrors escha/gptoss_experts.py::PackedScaledExpertLinear semantics exactly.
    Correctness path only — no Metal fusion yet. Decode-once-per-forward.
    """

    def __init__(
        self,
        code: mx.array,        # int16 (in_p/16, out_p/16, 16*K)
        rin: mx.array,         # fp16 (in_p,)
        rout: mx.array,        # fp16 (out_p,)
        *,
        in_f: int,
        out_f: int,
        in_p: int,
        out_p: int,
        K: int,
        cb_id: int = 1,
        mul1: bool = False,
    ) -> None:
        super().__init__()
        self.in_f, self.out_f = int(in_f), int(out_f)
        self.in_p, self.out_p = int(in_p), int(out_p)
        self.K = int(K)
        self.cb_id = int(cb_id)
        self.mul1 = bool(mul1)
        # Buffers (non-trainable). Store as bf16 for internal math; int16 codes
        # stay int16 (or reinterpret through int32 gather).
        self.code = code
        self.rin = rin.astype(mx.bfloat16).reshape(-1)      # (in_p,)
        self.rout = rout.astype(mx.bfloat16).reshape(-1)    # (out_p,)
        # Decoded weight is cached lazily on first forward (pure correctness path).
        # Phase 2 will replace this with a fused Metal kernel that decodes on-the-fly.
        self._w_cache: mx.array | None = None

    def _weight(self) -> mx.array:
        if self._w_cache is None:
            self._w_cache = escham_reconstruct(
                self.code, self.in_p, self.out_p, self.K, self.cb_id, self.mul1
            ).astype(mx.bfloat16)
        return self._w_cache

    def __call__(self, x: mx.array) -> mx.array:
        # x: (..., in_f). in_p == in_f in Escha-W2 (no padding).
        if self.in_p != self.in_f:
            pad = mx.zeros((*x.shape[:-1], self.in_p - self.in_f), dtype=x.dtype)
            xp = mx.concatenate([x, pad], axis=-1)
        else:
            xp = x
        xh = t128(xp, pre=self.rin)                          # (..., in_p) bf16
        y = xh @ self._weight()                              # (..., out_p) bf16
        y = t128(y, post=self.rout)                          # (..., out_p)
        if self.out_p != self.out_f:
            y = y[..., : self.out_f]
        return y


class DequantExpertLinear(nn.Module):
    """Option-B expert projection: uses a pre-composed dense M matrix.

    Bypasses escham_reconstruct + T128 + diag(rin) / diag(rout) entirely:
    since all four are linear, they compose into a single (in_f, out_f) matrix
    M produced offline (Modal identity trick, see codebooks/modal_dequant_all.py).

    Storage:  self.weight  shape (out_f, in_f)  bf16
    Forward:  y = x @ weight.T
    """

    def __init__(self, weight_dequant: mx.array) -> None:
        super().__init__()
        # Store as (out, in) bf16 to match nn.Linear convention.
        self.weight = weight_dequant.astype(mx.bfloat16)

    def __call__(self, x: mx.array) -> mx.array:
        return x.astype(self.weight.dtype) @ self.weight.T
