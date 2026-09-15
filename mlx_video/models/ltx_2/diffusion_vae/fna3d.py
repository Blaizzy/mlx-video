"""Fused 3D neighborhood attention for the LTX-2.5 diffusion VAE.

The window geometry and tiled-softmax design follow NATTEN's Metal backend
prototype (SHI-Labs/NATTEN#312, MIT). This MLX inference-only implementation
uses ``mx.fast.metal_kernel`` and does not depend on PyTorch or NATTEN.
"""

from __future__ import annotations

import functools
import math

import mlx.core as mx

_MAX_HEAD_DIM = 64


def _validate_inputs(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    kernel_size: tuple[int, int, int],
) -> None:
    if query.ndim != 6:
        raise ValueError(f"Expected BTHWHD tensors, got query shape {query.shape}")
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError("query, key, and value must have identical shapes")
    if len(kernel_size) != 3 or any(size < 1 or size % 2 == 0 for size in kernel_size):
        raise ValueError(
            f"kernel_size must contain three positive odd values, got {kernel_size}"
        )
    if any(size > dim for size, dim in zip(kernel_size, query.shape[1:4])):
        raise ValueError(
            f"kernel_size {kernel_size} exceeds spatial-temporal shape {query.shape[1:4]}"
        )
    if query.shape[-1] > _MAX_HEAD_DIM:
        raise ValueError(f"head dimension {query.shape[-1]} exceeds {_MAX_HEAD_DIM}")
    if query.dtype not in (mx.float32, mx.float16, mx.bfloat16):
        raise ValueError(f"Unsupported dtype {query.dtype}")


def _window_start(index: int, size: int, window: int) -> int:
    """NATTEN's non-causal shifted-window origin."""
    return min(max(index - window // 2, 0), size - window)


def neighborhood_attention_3d_reference(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    kernel_size: tuple[int, int, int],
    *,
    scale: float | None = None,
) -> mx.array:
    """Small eager reference used for numerical tests, not production decode."""
    _validate_inputs(query, key, value, kernel_size)
    batch, frames, height, width, heads, head_dim = query.shape
    scale = head_dim**-0.5 if scale is None else scale
    batches = []
    for batch_index in range(batch):
        frame_rows = []
        for frame in range(frames):
            height_rows = []
            frame_start = _window_start(frame, frames, kernel_size[0])
            for row in range(height):
                width_rows = []
                row_start = _window_start(row, height, kernel_size[1])
                for column in range(width):
                    column_start = _window_start(column, width, kernel_size[2])
                    keys = key[
                        batch_index,
                        frame_start : frame_start + kernel_size[0],
                        row_start : row_start + kernel_size[1],
                        column_start : column_start + kernel_size[2],
                    ].reshape(-1, heads, head_dim)
                    values = value[
                        batch_index,
                        frame_start : frame_start + kernel_size[0],
                        row_start : row_start + kernel_size[1],
                        column_start : column_start + kernel_size[2],
                    ].reshape(-1, heads, head_dim)
                    q = query[batch_index, frame, row, column]
                    scores = mx.sum(
                        keys.astype(mx.float32) * q[None].astype(mx.float32), axis=-1
                    )
                    weights = mx.softmax(scores * scale, axis=0)
                    width_rows.append(
                        mx.sum(weights[..., None] * values.astype(mx.float32), axis=0)
                    )
                height_rows.append(mx.stack(width_rows))
            frame_rows.append(mx.stack(height_rows))
        batches.append(mx.stack(frame_rows))
    return mx.stack(batches).astype(query.dtype)


_METAL_SOURCE = r"""
    uint index = thread_position_in_grid.x;
    uint total = q_shape[0] * q_shape[1] * q_shape[2] * q_shape[3] * q_shape[4];
    if (index >= total) return;

    uint head = index % q_shape[4];
    uint position = index / q_shape[4];
    uint column = position % q_shape[3];
    position /= q_shape[3];
    uint row = position % q_shape[2];
    position /= q_shape[2];
    uint frame = position % q_shape[1];
    uint batch = position / q_shape[1];

    int fs = min(max(int(frame) - KT / 2, 0), int(q_shape[1]) - KT);
    int hs = min(max(int(row) - KH / 2, 0), int(q_shape[2]) - KH);
    int ws = min(max(int(column) - KW / 2, 0), int(q_shape[3]) - KW);
    uint q_base = (((((batch * q_shape[1] + frame) * q_shape[2] + row)
        * q_shape[3] + column) * q_shape[4] + head) * HD);

    float maximum = -INFINITY;
    for (int ft = 0; ft < KT; ++ft) {
        for (int hh = 0; hh < KH; ++hh) {
            for (int ww = 0; ww < KW; ++ww) {
                uint k_base = (((((batch * q_shape[1] + uint(fs + ft)) * q_shape[2]
                    + uint(hs + hh)) * q_shape[3] + uint(ws + ww)) * q_shape[4]
                    + head) * HD);
                float score = 0.0f;
                for (int d = 0; d < HD; ++d) {
                    score += float(q[q_base + d]) * float(k[k_base + d]);
                }
                maximum = max(maximum, score * float(attention_scale[0]));
            }
        }
    }

    float denominator = 0.0f;
    float accumulator[MAX_HD];
    for (int d = 0; d < MAX_HD; ++d) accumulator[d] = 0.0f;
    for (int ft = 0; ft < KT; ++ft) {
        for (int hh = 0; hh < KH; ++hh) {
            for (int ww = 0; ww < KW; ++ww) {
                uint kv_base = (((((batch * q_shape[1] + uint(fs + ft)) * q_shape[2]
                    + uint(hs + hh)) * q_shape[3] + uint(ws + ww)) * q_shape[4]
                    + head) * HD);
                float score = 0.0f;
                for (int d = 0; d < HD; ++d) {
                    score += float(q[q_base + d]) * float(k[kv_base + d]);
                }
                float weight = metal::exp(score * float(attention_scale[0]) - maximum);
                denominator += weight;
                for (int d = 0; d < HD; ++d) {
                    accumulator[d] += weight * float(v[kv_base + d]);
                }
            }
        }
    }
    for (int d = 0; d < HD; ++d) {
        out[q_base + d] = T(accumulator[d] / denominator);
    }
"""


_METAL_SOURCE_HD64 = r"""
    uint group = threadgroup_position_in_grid.x;
    uint lane = thread_index_in_simdgroup;
    uint total = q_shape[0] * q_shape[1] * q_shape[2] * q_shape[3] * q_shape[4];
    if (group >= total) return;

    uint head = group % q_shape[4];
    uint position = group / q_shape[4];
    uint column = position % q_shape[3];
    position /= q_shape[3];
    uint row = position % q_shape[2];
    position /= q_shape[2];
    uint frame = position % q_shape[1];
    uint batch = position / q_shape[1];

    int fs = min(max(int(frame) - KT / 2, 0), int(q_shape[1]) - KT);
    int hs = min(max(int(row) - KH / 2, 0), int(q_shape[2]) - KH);
    int ws = min(max(int(column) - KW / 2, 0), int(q_shape[3]) - KW);
    uint q_base = (((((batch * q_shape[1] + frame) * q_shape[2] + row)
        * q_shape[3] + column) * q_shape[4] + head) * 64);

    float accumulator0 = 0.0f;
    float accumulator1 = 0.0f;
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    for (int ft = 0; ft < KT; ++ft) {
        for (int hh = 0; hh < KH; ++hh) {
            for (int ww = 0; ww < KW; ++ww) {
                uint kv_base = (((((batch * q_shape[1] + uint(fs + ft)) * q_shape[2]
                    + uint(hs + hh)) * q_shape[3] + uint(ws + ww)) * q_shape[4]
                    + head) * 64);
                float partial = float(q[q_base + lane]) * float(k[kv_base + lane]);
                partial += float(q[q_base + lane + 32]) * float(k[kv_base + lane + 32]);
                float score = simd_sum(partial) * float(attention_scale[0]);
                float next_max = max(running_max, score);
                float correction = metal::exp(running_max - next_max);
                float weight = metal::exp(score - next_max);
                running_sum = running_sum * correction + weight;
                accumulator0 = accumulator0 * correction + weight * float(v[kv_base + lane]);
                accumulator1 = accumulator1 * correction + weight * float(v[kv_base + lane + 32]);
                running_max = next_max;
            }
        }
    }
    out[q_base + lane] = T(accumulator0 / running_sum);
    out[q_base + lane + 32] = T(accumulator1 / running_sum);
"""


@functools.cache
def _metal_kernel() -> object:
    return mx.fast.metal_kernel(
        name="ltx_diffvae_na3d_forward",
        input_names=["q", "k", "v", "attention_scale"],
        output_names=["out"],
        source=_METAL_SOURCE,
        ensure_row_contiguous=True,
    )


@functools.cache
def _metal_kernel_hd64() -> object:
    return mx.fast.metal_kernel(
        name="ltx_diffvae_na3d_hd64_forward",
        input_names=["q", "k", "v", "attention_scale"],
        output_names=["out"],
        source=_METAL_SOURCE_HD64,
        ensure_row_contiguous=True,
    )


def neighborhood_attention_3d(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    kernel_size: tuple[int, int, int],
    *,
    scale: float | None = None,
) -> mx.array:
    """Run inference-only fused neighborhood attention on Apple GPU."""
    _validate_inputs(query, key, value, kernel_size)
    head_dim = query.shape[-1]
    scale = head_dim**-0.5 if scale is None else scale
    total_queries = math.prod(query.shape[:-1])
    common = {
        "inputs": [query, key, value, mx.array([scale], dtype=mx.float32)],
        "template": [
            ("T", query.dtype),
            ("KT", kernel_size[0]),
            ("KH", kernel_size[1]),
            ("KW", kernel_size[2]),
        ],
        "output_shapes": [query.shape],
        "output_dtypes": [query.dtype],
    }
    if head_dim == 64:
        output = _metal_kernel_hd64()(
            **common,
            grid=(total_queries * 32, 1, 1),
            threadgroup=(32, 1, 1),
        )[0]
        return output

    output = _metal_kernel()(
        template=[
            *common["template"],
            ("HD", head_dim),
            ("MAX_HD", _MAX_HEAD_DIM),
        ],
        inputs=common["inputs"],
        grid=(total_queries, 1, 1),
        threadgroup=(min(256, total_queries), 1, 1),
        output_shapes=common["output_shapes"],
        output_dtypes=common["output_dtypes"],
    )[0]
    return output
