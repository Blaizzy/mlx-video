import mlx.core as mx
import numpy as np

from mlx_video.models.ltx_2.diffusion_vae.joint_fna3d import (
    joint_neighborhood_attention_3d,
    nearest_slot_maps,
)


def test_nearest_slot_maps_tie_break_toward_lower_index():
    video, keyframes = nearest_slot_maps(mx.array([1.0, 3.0]), 5)
    assert mx.array_equal(video[2], mx.array([0, 1], dtype=mx.int32))
    assert mx.array_equal(keyframes[0], mx.array([1, 0], dtype=mx.int32))


def test_joint_metal_attention_produces_both_finite_streams():
    mx.random.seed(31)
    video_shape = (1, 3, 3, 3, 1, 64)
    keyframe_shape = (1, 2, 3, 3, 1, 64)
    tensors = [mx.random.normal(video_shape).astype(mx.bfloat16) for _ in range(3)]
    keyframes = [mx.random.normal(keyframe_shape).astype(mx.bfloat16) for _ in range(3)]
    video, keyframe = joint_neighborhood_attention_3d(
        *tensors,
        *keyframes,
        mx.array([0.5, 2.5]),
        (3, 3, 3),
    )
    mx.eval(video, keyframe)
    assert video.shape == video_shape
    assert keyframe.shape == keyframe_shape
    assert mx.all(mx.isfinite(video))
    assert mx.all(mx.isfinite(keyframe))


def test_joint_attention_uniform_values_remain_uniform():
    video_shape = (1, 3, 3, 3, 1, 64)
    keyframe_shape = (1, 1, 3, 3, 1, 64)
    video = [mx.zeros(video_shape), mx.zeros(video_shape), mx.ones(video_shape)]
    keyframes = [
        mx.zeros(keyframe_shape),
        mx.zeros(keyframe_shape),
        mx.ones(keyframe_shape),
    ]
    video_out, keyframe_out = joint_neighborhood_attention_3d(
        *video,
        *keyframes,
        mx.array([1.0]),
        (3, 3, 3),
    )
    mx.eval(video_out, keyframe_out)
    assert mx.allclose(video_out, mx.ones(video_shape))
    assert mx.allclose(keyframe_out, mx.ones(keyframe_shape))


def test_joint_metal_attention_matches_numpy_reference():
    rng = np.random.default_rng(37)
    shape = (1, 3, 3, 3, 1, 64)
    keyframe_shape = (1, 2, 3, 3, 1, 64)
    video = [rng.normal(size=shape).astype(np.float32) for _ in range(3)]
    keyframes = [rng.normal(size=keyframe_shape).astype(np.float32) for _ in range(3)]
    times = np.array([0.5, 2.5], dtype=np.float32)
    video_slots, keyframe_slots = nearest_slot_maps(mx.array(times), 3)
    video_slots = np.asarray(video_slots)
    keyframe_slots = np.asarray(keyframe_slots)
    scale = 64**-0.5

    def attend(query, keys, values):
        scores = np.asarray(keys) @ query * scale
        weights = np.exp(scores - scores.max())
        return weights @ np.asarray(values) / weights.sum()

    expected_video = np.empty_like(video[0])
    for frame in range(3):
        for row in range(3):
            for column in range(3):
                keys, values = [], []
                for tf in range(max(0, frame - 1), min(3, frame + 2)):
                    for hr in range(max(0, row - 1), min(3, row + 2)):
                        for wc in range(max(0, column - 1), min(3, column + 2)):
                            keys.append(video[1][0, tf, hr, wc, 0])
                            values.append(video[2][0, tf, hr, wc, 0])
                for plane in video_slots[frame]:
                    for hr in range(max(0, row - 1), min(3, row + 2)):
                        for wc in range(max(0, column - 1), min(3, column + 2)):
                            keys.append(keyframes[1][0, plane, hr, wc, 0])
                            values.append(keyframes[2][0, plane, hr, wc, 0])
                expected_video[0, frame, row, column, 0] = attend(
                    video[0][0, frame, row, column, 0], keys, values
                )

    expected_keyframes = np.empty_like(keyframes[0])
    for plane in range(2):
        for row in range(3):
            for column in range(3):
                keys, values = [], []
                for hr in range(max(0, row - 1), min(3, row + 2)):
                    for wc in range(max(0, column - 1), min(3, column + 2)):
                        keys.append(keyframes[1][0, plane, hr, wc, 0])
                        values.append(keyframes[2][0, plane, hr, wc, 0])
                for frame in keyframe_slots[plane]:
                    for hr in range(max(0, row - 1), min(3, row + 2)):
                        for wc in range(max(0, column - 1), min(3, column + 2)):
                            keys.append(video[1][0, frame, hr, wc, 0])
                            values.append(video[2][0, frame, hr, wc, 0])
                expected_keyframes[0, plane, row, column, 0] = attend(
                    keyframes[0][0, plane, row, column, 0], keys, values
                )

    actual_video, actual_keyframes = joint_neighborhood_attention_3d(
        *(mx.array(item) for item in video),
        *(mx.array(item) for item in keyframes),
        mx.array(times),
        (3, 3, 3),
    )
    mx.eval(actual_video, actual_keyframes)
    assert mx.allclose(actual_video, mx.array(expected_video), atol=2e-4, rtol=2e-4)
    assert mx.allclose(
        actual_keyframes, mx.array(expected_keyframes), atol=2e-4, rtol=2e-4
    )
