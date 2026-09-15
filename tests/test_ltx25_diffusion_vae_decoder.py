import mlx.core as mx

from mlx_video.models.ltx_2.diffusion_vae.decoder import (
    DiffusionVideoDecoder,
    patchify_pixels,
    timestep_embedding,
    unpatchify_pixels,
)


def test_pixel_patchify_round_trip_matches_c_r_q_order():
    pixels = mx.arange(1 * 3 * 2 * 8 * 12).reshape(1, 3, 2, 8, 12)
    patches = patchify_pixels(pixels, 4)
    restored = unpatchify_pixels(patches, 4, 3)
    assert patches.shape == (1, 2, 2, 3, 48)
    assert mx.array_equal(restored, pixels)


def test_timestep_embedding_shape_and_finite():
    embedding = timestep_embedding(mx.array([0.0, 1000.0]))
    assert embedding.shape == (2, 256)
    assert mx.all(mx.isfinite(embedding))


def test_checkpoint_sanitizer_splits_fused_qkv():
    weights = {
        "decoder.det_stages.0.0.attn.qkv.weight": mx.arange(12).reshape(6, 2),
        "decoder.det_stages.0.0.attn.qkv.bias": mx.arange(6),
        "decoder.conv_in.weight": mx.ones((4, 2)),
        "per_channel_statistics.mean-of-means": mx.zeros((2,)),
        "encoder.ignored.weight": mx.zeros((1,)),
    }
    sanitized = DiffusionVideoDecoder.sanitize(weights)
    assert set(sanitized) == {
        "det_stages.0.0.attn.to_q.weight",
        "det_stages.0.0.attn.to_k.weight",
        "det_stages.0.0.attn.to_v.weight",
        "det_stages.0.0.attn.to_q.bias",
        "det_stages.0.0.attn.to_k.bias",
        "det_stages.0.0.attn.to_v.bias",
        "conv_in.weight",
        "mean_of_means",
    }
    assert sanitized["det_stages.0.0.attn.to_q.weight"].shape == (2, 2)


def test_spatial_tiling_matches_full_decode_with_receptive_field_halo():
    mx.random.seed(23)
    model = DiffusionVideoDecoder(
        in_channels=8,
        out_channels=1,
        patch_size=2,
        head_dim=64,
        stage_channels=(64, 64, 64, 64, 64),
        stage_depths=(1, 1, 1, 1, 1),
        stage_kernels=((3, 3, 3),) * 5,
        upsamples=(
            ((1, 2, 2), 1),
            ((2, 1, 1), 1),
            ((2, 2, 2), 1),
            ((2, 2, 2), 1),
        ),
    )
    latent = mx.random.normal((1, 8, 1, 3, 3)).astype(mx.float32)
    full = model(latent, seed=29, spatial_tiles=1)
    tiled = model(latent, seed=29, spatial_tiles=2)
    mx.eval(full, tiled)
    assert tiled.shape == full.shape
    assert mx.allclose(tiled, full, atol=2e-4, rtol=2e-4)


def test_keyframe_aware_decoder_runs_both_streams():
    mx.random.seed(41)
    model = DiffusionVideoDecoder(
        in_channels=8,
        out_channels=1,
        patch_size=2,
        head_dim=64,
        stage_channels=(64, 64, 64, 64, 64),
        stage_depths=(1, 1, 1, 1, 1),
        stage_kernels=((3, 3, 3),) * 5,
        upsamples=(
            ((1, 2, 2), 1),
            ((2, 1, 1), 1),
            ((2, 2, 2), 1),
            ((2, 2, 2), 1),
        ),
    )
    latent = mx.random.normal((1, 8, 1, 3, 3)).astype(mx.bfloat16)
    keyframes = mx.random.normal((1, 8, 1, 3, 3)).astype(mx.bfloat16)
    output = model(
        latent,
        seed=43,
        keyframe_latents=keyframes,
        keyframe_positions=[0],
    )
    mx.eval(output)
    assert output.shape == (1, 1, 1, 48, 48)
    assert mx.all(mx.isfinite(output))
    tiled = model(
        latent,
        seed=43,
        spatial_tiles=2,
        keyframe_latents=keyframes,
        keyframe_positions=[0],
    )
    mx.eval(tiled)
    assert mx.allclose(tiled, output, atol=2e-4, rtol=2e-4)

    temporal_latent = mx.random.normal((1, 8, 2, 3, 3)).astype(mx.bfloat16)
    temporal_keyframes = mx.random.normal((1, 8, 1, 3, 3)).astype(mx.bfloat16)
    full_temporal = model(
        temporal_latent,
        seed=47,
        keyframe_latents=temporal_keyframes,
        keyframe_positions=[4],
    )
    tiled_temporal = model(
        temporal_latent,
        seed=47,
        temporal_tiles=2,
        keyframe_latents=temporal_keyframes,
        keyframe_positions=[4],
    )
    mx.eval(full_temporal, tiled_temporal)
    assert full_temporal.shape == (1, 1, 9, 48, 48)
    assert mx.allclose(tiled_temporal, full_temporal, atol=3e-2, rtol=3e-2)
