import mlx.core as mx

from mlx_video.models.ltx_2.video_vae.decoder import (
    sanitize_standalone_vae_weights,
)


def test_sanitize_standalone_vae_weights() -> None:
    conv = mx.zeros((4, 3, 2, 5, 7))
    weights = {
        "decoder.conv_in.conv.weight": conv,
        "decoder.conv_in.conv.bias": mx.zeros((4,)),
        "decoder.up_blocks.1.conv.conv.weight": conv,
        "encoder.conv_in.conv.weight": conv,
        "per_channel_statistics.mean-of-means": mx.zeros((128,)),
        "per_channel_statistics.std-of-means": mx.ones((128,)),
    }

    actual = sanitize_standalone_vae_weights(weights)

    assert set(actual) == {
        "conv_in.conv.conv.weight",
        "conv_in.conv.conv.bias",
        "up_blocks.1.conv.conv.weight",
        "per_channel_statistics.mean",
        "per_channel_statistics.std",
    }
    assert actual["conv_in.conv.conv.weight"].shape == (4, 2, 5, 7, 3)
