"""Per-row int8 symmetric quantization — attention / lm_head / embed / router.

Format (verified in ~/models/Qwen3.6-35B-A3B-Escha-W2):
    weight_int8:  int8   (out_features, in_features)   values in [-127, 127]
    weight_scale: fp16   (out_features,)                per-output-row scale

Reconstruction:
    w_bf16[i, :] = weight_int8[i, :].astype(bf16) * weight_scale[i]

This is trivial to port. For a first pass we pre-dequantize at load time
(costs ~1 GB extra memory vs. lazy dequant; buys back a lot of runtime speed
because MLX matmul is much better on bf16 than int8 on unified memory).

For lm_head (248320 × 2048) the pre-dequant cost is ~1 GB — worth it because
lm_head runs every generation step.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def dequant_int8_per_row(w_int8: mx.array, scale: mx.array, dtype: mx.Dtype = mx.bfloat16) -> mx.array:
    """Reconstruct a bf16 weight from per-row int8 + fp16 scale.

    Args:
        w_int8: int8 (out_features, in_features).
        scale : fp16 (out_features,).

    Returns:
        bf16 (out_features, in_features).
    """
    return (w_int8.astype(dtype) * scale.astype(dtype).reshape(-1, 1))


class Int8Linear(nn.Module):
    """Drop-in Linear that stores int8 weights + per-row fp16 scales and
    pre-dequantizes to bf16 at load. No bias (Escha-W2 attention layers are
    bias-free)."""

    def __init__(self, weight_int8: mx.array, weight_scale: mx.array):
        super().__init__()
        self.weight = dequant_int8_per_row(weight_int8, weight_scale)   # (out, in) bf16

    def __call__(self, x: mx.array) -> mx.array:
        return x @ self.weight.T
