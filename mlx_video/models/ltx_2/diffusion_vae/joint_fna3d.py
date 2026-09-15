"""Keyframe-aware joint neighborhood attention for LTX-2.5 DiffVAE."""

from __future__ import annotations

import functools

import mlx.core as mx
import numpy as np


def nearest_slot_maps(
    keyframe_times: mx.array, video_length: int, slots: int = 2
) -> tuple[mx.array, mx.array]:
    """Return video-to-keyframe and keyframe-to-video nearest-slot tables."""
    times = np.asarray(keyframe_times.astype(mx.float32)).reshape(-1)
    video = np.arange(video_length, dtype=np.float32)

    def rank(queries: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        distances = np.abs(queries[:, None] - candidates[None])
        take = min(slots, len(candidates))
        selected = np.argsort(distances, axis=1, kind="stable")[:, :take]
        if take < slots:
            selected = np.pad(
                selected,
                ((0, 0), (0, slots - take)),
                constant_values=-1,
            )
        return selected.astype(np.int32)

    return mx.array(rank(video, times)), mx.array(rank(times, video))


_SOURCE = r"""
    uint group = threadgroup_position_in_grid.x;
    uint lane = thread_index_in_simdgroup;
    uint video_groups = q_shape[0] * q_shape[1] * q_shape[2] * q_shape[3] * q_shape[4];
    uint keyframe_groups = kfq_shape[0] * kfq_shape[1] * kfq_shape[2] * kfq_shape[3] * kfq_shape[4];
    if (group >= video_groups + keyframe_groups) return;

    bool is_keyframe = group >= video_groups;
    uint local_group = is_keyframe ? group - video_groups : group;
    uint head_count = q_shape[4];
    uint head = local_group % head_count;
    uint position = local_group / head_count;
    uint column = position % q_shape[3];
    position /= q_shape[3];
    uint row = position % q_shape[2];
    position /= q_shape[2];
    uint temporal = position % (is_keyframe ? kfq_shape[1] : q_shape[1]);
    uint batch = position / (is_keyframe ? kfq_shape[1] : q_shape[1]);

    uint q_base;
    if (is_keyframe) {
        q_base = (((((batch * kfq_shape[1] + temporal) * q_shape[2] + row)
            * q_shape[3] + column) * head_count + head) * 64);
    } else {
        q_base = (((((batch * q_shape[1] + temporal) * q_shape[2] + row)
            * q_shape[3] + column) * head_count + head) * 64);
    }

    float accumulator0 = 0.0f;
    float accumulator1 = 0.0f;
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    #define ACCUMULATE(KEY_PTR, VALUE_PTR, BASE) { \
        float partial = float(KEY_PTR[BASE + lane]) * float(is_keyframe ? kfq[q_base + lane] : q[q_base + lane]); \
        partial += float(KEY_PTR[BASE + lane + 32]) * float(is_keyframe ? kfq[q_base + lane + 32] : q[q_base + lane + 32]); \
        float score = simd_sum(partial) * float(attention_scale[0]); \
        float next_max = max(running_max, score); \
        float correction = metal::exp(running_max - next_max); \
        float weight = metal::exp(score - next_max); \
        running_sum = running_sum * correction + weight; \
        accumulator0 = accumulator0 * correction + weight * float(VALUE_PTR[BASE + lane]); \
        accumulator1 = accumulator1 * correction + weight * float(VALUE_PTR[BASE + lane + 32]); \
        running_max = next_max; \
    }

    int half_t = KT / 2;
    int half_h = KH / 2;
    int half_w = KW / 2;
    if (!is_keyframe) {
        for (int dt = -half_t; dt <= half_t; ++dt) {
            int tf = int(temporal) + dt;
            if (tf < 0 || tf >= int(q_shape[1])) continue;
            for (int dh = -half_h; dh <= half_h; ++dh) {
                int hr = int(row) + dh;
                if (hr < 0 || hr >= int(q_shape[2])) continue;
                for (int dw = -half_w; dw <= half_w; ++dw) {
                    int wc = int(column) + dw;
                    if (wc < 0 || wc >= int(q_shape[3])) continue;
                    uint base = (((((batch * q_shape[1] + uint(tf)) * q_shape[2] + uint(hr))
                        * q_shape[3] + uint(wc)) * head_count + head) * 64);
                    ACCUMULATE(k, v, base);
                }
            }
        }
        for (int slot = 0; slot < 2; ++slot) {
            int plane = video_slots[temporal * 2 + slot];
            if (plane < 0) continue;
            for (int dh = -half_h; dh <= half_h; ++dh) {
                int hr = int(row) + dh;
                if (hr < 0 || hr >= int(q_shape[2])) continue;
                for (int dw = -half_w; dw <= half_w; ++dw) {
                    int wc = int(column) + dw;
                    if (wc < 0 || wc >= int(q_shape[3])) continue;
                    uint base = (((((batch * kfq_shape[1] + uint(plane)) * q_shape[2] + uint(hr))
                        * q_shape[3] + uint(wc)) * head_count + head) * 64);
                    ACCUMULATE(kfk, kfv, base);
                }
            }
        }
        video_out[q_base + lane] = T(accumulator0 / running_sum);
        video_out[q_base + lane + 32] = T(accumulator1 / running_sum);
    } else {
        for (int dh = -half_h; dh <= half_h; ++dh) {
            int hr = int(row) + dh;
            if (hr < 0 || hr >= int(q_shape[2])) continue;
            for (int dw = -half_w; dw <= half_w; ++dw) {
                int wc = int(column) + dw;
                if (wc < 0 || wc >= int(q_shape[3])) continue;
                uint base = (((((batch * kfq_shape[1] + temporal) * q_shape[2] + uint(hr))
                    * q_shape[3] + uint(wc)) * head_count + head) * 64);
                ACCUMULATE(kfk, kfv, base);
            }
        }
        for (int slot = 0; slot < 2; ++slot) {
            int frame = keyframe_slots[temporal * 2 + slot];
            if (frame < 0) continue;
            for (int dh = -half_h; dh <= half_h; ++dh) {
                int hr = int(row) + dh;
                if (hr < 0 || hr >= int(q_shape[2])) continue;
                for (int dw = -half_w; dw <= half_w; ++dw) {
                    int wc = int(column) + dw;
                    if (wc < 0 || wc >= int(q_shape[3])) continue;
                    uint base = (((((batch * q_shape[1] + uint(frame)) * q_shape[2] + uint(hr))
                        * q_shape[3] + uint(wc)) * head_count + head) * 64);
                    ACCUMULATE(k, v, base);
                }
            }
        }
        keyframe_out[q_base + lane] = T(accumulator0 / running_sum);
        keyframe_out[q_base + lane + 32] = T(accumulator1 / running_sum);
    }
"""


@functools.cache
def _kernel() -> object:
    return mx.fast.metal_kernel(
        name="ltx_diffvae_joint_na3d_hd64",
        input_names=[
            "q",
            "k",
            "v",
            "kfq",
            "kfk",
            "kfv",
            "video_slots",
            "keyframe_slots",
            "attention_scale",
        ],
        output_names=["video_out", "keyframe_out"],
        source=_SOURCE,
        ensure_row_contiguous=True,
    )


def joint_neighborhood_attention_3d(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    keyframe_query: mx.array,
    keyframe_key: mx.array,
    keyframe_value: mx.array,
    keyframe_times: mx.array,
    kernel_size: tuple[int, int, int],
    *,
    scale: float | None = None,
) -> tuple[mx.array, mx.array]:
    """Run LTX joint video/keyframe attention with two nearest context slots."""
    if query.ndim != 6 or keyframe_query.ndim != 6:
        raise ValueError("video and keyframe Q/K/V must use BTHWHD layout")
    if query.shape[-1] != 64 or keyframe_query.shape[-1] != 64:
        raise ValueError("joint Metal attention currently requires head_dim=64")
    if (
        query.shape[0] != keyframe_query.shape[0]
        or query.shape[2:] != keyframe_query.shape[2:]
    ):
        raise ValueError(
            "video and keyframe streams must share batch, H/W, heads, and head_dim"
        )
    video_slots, keyframe_slots = nearest_slot_maps(keyframe_times, query.shape[1])
    total_groups = math_prod(query.shape[:-1]) + math_prod(keyframe_query.shape[:-1])
    outputs = _kernel()(
        inputs=[
            query,
            key,
            value,
            keyframe_query,
            keyframe_key,
            keyframe_value,
            video_slots,
            keyframe_slots,
            mx.array([scale or 64**-0.5], dtype=mx.float32),
        ],
        template=[
            ("T", query.dtype),
            ("KT", kernel_size[0]),
            ("KH", kernel_size[1]),
            ("KW", kernel_size[2]),
        ],
        grid=(total_groups * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[query.shape, keyframe_query.shape],
        output_dtypes=[query.dtype, query.dtype],
    )
    return outputs[0], outputs[1]


def math_prod(values: tuple[int, ...]) -> int:
    result = 1
    for value in values:
        result *= value
    return result
