import mlx.core as mx
import pytest

from mlx_video.models.ltx_2.dfr import (
    create_keyframe_slot_positions,
    denoise_dfr_tokens,
    flatten_video_latents,
    keyframe_token_mask,
    resolve_dfr_canvas,
    unflatten_video_tokens,
)


def test_resolve_dfr_canvas_uses_official_segment_grid():
    assert resolve_dfr_canvas(121) == (121, 24, [24, 48, 72, 96, 120])
    assert resolve_dfr_canvas(97) == (97, 32, [32, 64, 96])
    assert resolve_dfr_canvas(113) == (121, 24, [24, 48, 72, 96, 120])


@pytest.mark.parametrize("frames", [1, 10, 120])
def test_resolve_dfr_canvas_rejects_invalid_frame_counts(frames):
    with pytest.raises(ValueError):
        resolve_dfr_canvas(frames)


def test_keyframe_slot_positions_are_single_frame_and_spatially_aligned():
    positions = create_keyframe_slot_positions([24, 48], height=2, width=3, fps=24.0)
    assert positions.shape == (1, 3, 12, 2)
    assert mx.allclose(positions[0, 0, 0], mx.array([1.0, 25 / 24]), atol=6e-3)
    assert mx.allclose(positions[0, 0, 6], mx.array([2.0, 49 / 24]), atol=6e-3)
    assert mx.array_equal(positions[0, 1, 0], mx.array([0.0, 32.0]))
    assert mx.array_equal(positions[0, 2, 5], mx.array([64.0, 96.0]))


def test_video_token_round_trip_and_keyframe_mask():
    latents = mx.arange(2 * 4 * 3 * 2 * 2).reshape(2, 4, 3, 2, 2)
    tokens = flatten_video_latents(latents)
    restored = unflatten_video_tokens(tokens, frames=3, height=2, width=2)
    assert mx.array_equal(restored, latents)

    mask = keyframe_token_mask(
        batch_size=1, video_tokens=12, slot_tokens=8, tokens_per_frame=4
    )
    assert mx.array_equal(mask[0, :4], mx.ones((4,)))
    assert mx.array_equal(mask[0, 4:12], mx.zeros((8,)))
    assert mx.array_equal(mask[0, 12:], mx.ones((8,)))


def test_dfr_denoiser_appends_slots_and_clean_reference_tokens():
    class ZeroTransformer:
        def __init__(self):
            self.video = None

        def __call__(self, video, audio):
            self.video = video
            return mx.zeros_like(video.latent), mx.zeros_like(audio.latent)

    transformer = ZeroTransformer()
    video = mx.ones((1, 2, 2, 1, 2), dtype=mx.float32)
    slots = mx.full((1, 2, 1, 1, 2), 2.0)
    reference = mx.full((1, 2, 2, 1, 1), 3.0)
    video_positions = mx.zeros((1, 3, 4, 2))
    slot_positions = mx.zeros((1, 3, 2, 2))
    reference_positions = mx.zeros((1, 3, 2, 2))
    audio = mx.ones((1, 1, 2, 2))

    out_video, out_slots, out_audio = denoise_dfr_tokens(
        video,
        slots,
        video_positions,
        slot_positions,
        mx.zeros((1, 1, 2)),
        transformer,
        [1.0, 0.0],
        audio_latents=audio,
        audio_positions=mx.zeros((1, 1, 2, 2)),
        audio_embeddings=mx.zeros((1, 1, 2)),
        reference_latents=reference,
        reference_positions=reference_positions,
    )

    assert mx.array_equal(out_video, video)
    assert mx.array_equal(out_slots, slots)
    assert mx.array_equal(out_audio, audio)
    assert transformer.video.latent.shape == (1, 8, 2)
    assert mx.array_equal(transformer.video.timesteps[0, -2:], mx.zeros((2,)))
    assert mx.array_equal(
        transformer.video.keyframes_mask[0],
        mx.array([1, 1, 0, 0, 1, 1, 0, 0]),
    )
