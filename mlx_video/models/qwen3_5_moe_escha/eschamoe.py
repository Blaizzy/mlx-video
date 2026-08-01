"""escham_reconstruct + PackedScaledExpertLinear — the 2-bit AQLM decode.

This is the single algorithmic hole in the Escha port: `escham_reconstruct`
decodes a 3-D int16 code tensor into a dense fp16 weight matrix by looking
each 16-bit index up in a fixed codebook lattice baked into the reference
Linux .so. See docs/ESCHA_PORT_FEASIBILITY.md §5 for the extraction plan.

Codebook file: mlx_video/models/qwen3_5_moe_escha/codebooks/escha_codebooks_v1.npz
Expected layout after extraction (see codebooks/extract_codebooks.py):
    cb_A_K2 : (65536, 16) fp16   — codebook 1 (default), K=2 slice
    cb_A_K3 : (65536, 16) fp16   — codebook 1, K=3 slice
    (codebooks B, C not needed for Escha-W2 which uses cb_id=1)

If the codebook file is absent, escham_reconstruct raises with a pointer
to the extraction script.
"""

from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .transform import t128


_CB_PATH = Path(__file__).parent / "codebooks" / "escha_codebooks_v1.npz"
_CB_CACHE: dict[int, mx.array] = {}   # keyed by K


def _load_codebook(K: int) -> mx.array:
    """Load codebook for the given K slice. Cached across calls."""
    if K in _CB_CACHE:
        return _CB_CACHE[K]
    if not _CB_PATH.exists():
        raise FileNotFoundError(
            f"Escha codebook not found at {_CB_PATH}.\n"
            f"Run: python -m mlx_video.models.qwen3_5_moe_escha.codebooks.extract_codebooks\n"
            f"on a Linux x86-64 box with the escha wheel installed. See\n"
            f"docs/ESCHA_PORT_FEASIBILITY.md §5b for the one-shot extraction procedure."
        )
    import numpy as np
    data = np.load(_CB_PATH)
    key = f"cb_A_K{K}"
    if key not in data.files:
        raise KeyError(f"Codebook {key} missing from {_CB_PATH} (found: {data.files})")
    cb = mx.array(data[key])   # (65536, 16) fp16
    _CB_CACHE[K] = cb
    return cb


def escham_reconstruct(
    code: mx.array,
    in_features: int,
    out_features: int,
    K: int,
    cb_id: int = 1,
    mul1: bool = False,
) -> mx.array:
    """Decode packed AQLM codes to a dense fp16 weight matrix.

    Args:
        code: int16 (or uint16 view) of shape (in_features/16, out_features/16, 16*K).
        in_features, out_features: padded dims (in_p, out_p). Escha-W2 has no padding.
        K: residual depth (2 for gate_up, 3 for down).
        cb_id: codebook variant — always 1 for Escha-W2 (cbA).
        mul1: multiplicative-1 flag — false for Escha-W2. When true the decoded
              weights are multiplied by an implicit factor of 1 (documented as
              a no-op in the reference; kept for schema parity).

    Returns:
        fp16 tensor of shape (in_features, out_features).

    Correctness model (matches escha_aqlm_gemv semantics):
        For each 16-row × 16-col block indexed by (bi, bj), and for each residual
        layer k ∈ [0, K):
            idx = code[bi, bj, k*16 : (k+1)*16]                       # (16,) uint16
            block[k, row, :] = codebook[cb_id, K, idx[row]]             # (16,)
        Final block = sum over k. Assemble all blocks back into (in, out).
    """
    if cb_id != 1:
        raise NotImplementedError(f"Only cb_id=1 (Escha-W2 default) supported; got {cb_id}")
    if code.shape[-1] != 16 * K:
        raise ValueError(f"code last dim {code.shape[-1]} != 16*K={16*K}")
    if code.shape[0] * 16 != in_features:
        raise ValueError(f"code.shape[0]*16 = {code.shape[0]*16} != in_features {in_features}")
    if code.shape[1] * 16 != out_features:
        raise ValueError(f"code.shape[1]*16 = {code.shape[1]*16} != out_features {out_features}")

    cb = _load_codebook(K)                                              # (65536, 16) fp16
    # int16 → uint16 → int32 for gather. mx.take needs an integer index.
    codes = code.reshape(in_features // 16, out_features // 16, K, 16)  # (bi, bj, k, row)
    # Reinterpret signed int16 as unsigned via masking:
    idx = codes.astype(mx.int32) & 0xFFFF                                # 0..65535
    # Gather: for each (bi, bj, k, row) fetch codebook[idx][:] → (bi, bj, k, row, 16)
    gathered = mx.take(cb, idx, axis=0)
    # Sum over K (residual reconstruction).
    per_block = gathered.sum(axis=2)                                     # (bi, bj, row, 16_col)
    # Reassemble: (bi, bj, row, col_in_block) → (in_features, out_features).
    # bi indexes rows of 16; bj indexes cols of 16.
    per_block = per_block.transpose(0, 2, 1, 3)                          # (bi, row, bj, col)
    w = per_block.reshape(in_features, out_features)
    if mul1:
        pass   # documented as identity in the reference for cb_id=1
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
