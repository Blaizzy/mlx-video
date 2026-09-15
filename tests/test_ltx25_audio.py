import mlx.core as mx

from mlx_video.models.ltx_2.audio_vae.audio_vae import AudioDecoder, AudioEncoder
from mlx_video.models.ltx_2.audio_vae.vocoder import (
    sanitize_split_vocoder_weights,
)
from mlx_video.models.ltx_2.config import (
    AudioDecoderModelConfig,
    AudioEncoderModelConfig,
)


def test_split_audio_encoder_and_decoder_weights_are_sanitized() -> None:
    conv = mx.zeros((4, 3, 2, 5))
    stats = mx.zeros((128,))
    encoder = AudioEncoder(
        AudioEncoderModelConfig(
            norm_type="pixel",
            causality_axis="height",
            mid_block_add_attention=False,
            attn_resolutions=[],
        )
    )
    decoder = AudioDecoder(
        AudioDecoderModelConfig(
            norm_type="pixel",
            causality_axis="height",
            mid_block_add_attention=False,
            attn_resolutions=[],
        )
    )

    encoder_weights = encoder.sanitize(
        {
            "audio_vae.encoder.conv_in.conv.weight": conv,
            "audio_vae.per_channel_statistics.mean-of-means": stats,
        }
    )
    decoder_weights = decoder.sanitize(
        {
            "audio_vae.decoder.conv_in.conv.weight": conv,
            "audio_vae.per_channel_statistics.std-of-means": stats,
        }
    )

    assert encoder_weights["conv_in.conv.weight"].shape == (4, 2, 5, 3)
    assert "per_channel_statistics.mean_of_means" in encoder_weights
    assert decoder_weights["conv_in.conv.weight"].shape == (4, 2, 5, 3)
    assert "per_channel_statistics.std_of_means" in decoder_weights


def test_split_vocoder_weights_are_sanitized() -> None:
    weights = sanitize_split_vocoder_weights(
        {
            "vocoder.vocoder.ups.0.weight": mx.zeros((6, 4, 3)),
            "vocoder.vocoder.conv_pre.weight": mx.zeros((6, 4, 3)),
            "audio_vae.decoder.conv_in.weight": mx.zeros((1,)),
        }
    )

    assert set(weights) == {
        "vocoder.ups.0.weight",
        "vocoder.conv_pre.weight",
    }
    assert weights["vocoder.ups.0.weight"].shape == (4, 3, 6)
    assert weights["vocoder.conv_pre.weight"].shape == (6, 3, 4)
