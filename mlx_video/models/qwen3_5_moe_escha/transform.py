"""Block-128 Walsh–Hadamard transform (bit-for-bit port of escha/transform.py).

The Escha runtime rotates every activation vector by a normalized 128×128
Hadamard matrix applied block-wise on the last axis, both before and after
the quantized GEMM. This flattens the weight distribution enough that the
codebook lookup stays accurate.

Reference (escha/transform.py::escha_t128):
    x       : (..., IC)  IC % 128 == 0
    pre     : (IC,) or None
    post    : (IC,) or None
    y = (x * pre)                                      # optional
    y = y.reshape(..., IC//128, 128) @ H128            # H128 = kron(H_1..H_1)/sqrt(128)
    y = y.reshape(..., IC)
    y = y * post                                       # optional
"""

from __future__ import annotations

import math
import mlx.core as mx


_HADAMARD_CACHE: dict[tuple[int, str], mx.array] = {}


def hadamard_128(dtype: mx.Dtype = mx.bfloat16) -> mx.array:
    """Normalized 128x128 Walsh-Hadamard matrix, cached per dtype.

    Built by the standard doubling recursion (Sylvester construction) and
    divided by sqrt(128) so the transform is orthonormal (self-inverse-with-
    same-normalization: T128 @ T128 = I).
    """
    key = (128, str(dtype))
    cached = _HADAMARD_CACHE.get(key)
    if cached is not None:
        return cached
    # Build in fp32 for a clean normalization, then cast.
    import numpy as np
    h = np.array([[1.0]], dtype=np.float32)
    while h.shape[0] < 128:
        h = np.block([[h, h], [h, -h]])
    h = h / math.sqrt(128.0)
    out = mx.array(h).astype(dtype)
    _HADAMARD_CACHE[key] = out
    return out


def t128(
    x: mx.array,
    *,
    pre: mx.array | None = None,
    post: mx.array | None = None,
) -> mx.array:
    """Apply y = post * T128(x * pre) on the last axis.

    Either scale may be None. Both are supported for symmetry with the
    reference, though the reference typically uses one per call.
    """
    ic = x.shape[-1]
    if ic % 128 != 0:
        raise ValueError(f"t128: last dim {ic} must be divisible by 128 (got {ic})")
    if pre is not None:
        x = x * pre.astype(x.dtype)
    leading = x.shape[:-1]
    x = x.reshape(*leading, ic // 128, 128)
    H = hadamard_128(x.dtype)
    x = x @ H
    x = x.reshape(*leading, ic)
    if post is not None:
        x = x * post.astype(x.dtype)
    return x
