import mlx.core as mx

from mlx_video.models.ltx_2.diffusion_vae.layers import (
    LinearPixelShuffleUpsample,
    NABlock,
    apply_absolute_rope,
    default_rope_dim_split,
)


def test_default_rope_split_matches_ltx_head_dimension():
    assert default_rope_dim_split(64) == (16, 24, 24)


def test_absolute_rope_preserves_shape_dtype_and_norm():
    mx.random.seed(3)
    value = mx.random.normal((1, 3, 4, 5, 2, 64)).astype(mx.bfloat16)
    rotated = apply_absolute_rope(value, (16, 24, 24))
    mx.eval(rotated)
    assert rotated.shape == value.shape
    assert rotated.dtype == value.dtype
    assert mx.allclose(
        mx.sum(rotated.astype(mx.float32) ** 2, axis=-1),
        mx.sum(value.astype(mx.float32) ** 2, axis=-1),
        atol=0.2,
        rtol=0.02,
    )


def test_na_block_runs_channels_last_with_metal_attention():
    mx.random.seed(5)
    block = NABlock(dim=64, kernel_size=(3, 3, 3), head_dim=64)
    value = mx.random.normal((1, 3, 3, 3, 64)).astype(mx.bfloat16)
    output = block(value)
    mx.eval(output)
    assert output.shape == value.shape
    assert mx.all(mx.isfinite(output))


def test_linear_pixel_shuffle_matches_decoder_geometry():
    value = mx.arange(1 * 2 * 2 * 3 * 8).reshape(1, 2, 2, 3, 8)
    spatial = LinearPixelShuffleUpsample(8, (1, 2, 2), reduction=2)
    spatial_output = spatial(value)
    assert spatial_output.shape == (1, 2, 4, 6, 4)

    temporal = LinearPixelShuffleUpsample(8, (2, 1, 1), reduction=2)
    temporal_output = temporal(value)
    assert temporal_output.shape == (1, 3, 2, 3, 4)
