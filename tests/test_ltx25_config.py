from mlx_video.models.ltx_2.config import LTXModelConfig


def test_legacy_prompt_adaln_enables_explicit_architecture_flags() -> None:
    config = LTXModelConfig(has_prompt_adaln=True)

    assert config.apply_gated_attention is True
    assert config.cross_attention_adaln is True


def test_ltx25_architecture_flags_and_biases_are_independent() -> None:
    config = LTXModelConfig.from_dict(
        {
            "model_version": "2.5.0",
            "has_prompt_adaln": True,
            "apply_gated_attention": True,
            "cross_attention_adaln": True,
            "ff_bias": False,
            "audio_ff_bias": True,
            "use_keyframes_abs_pos_embedding": True,
        }
    )

    video = config.get_video_config()
    audio = config.get_audio_config()
    assert config.model_version == "2.5.0"
    assert config.use_keyframes_abs_pos_embedding is True
    assert video is not None and video.ff_bias is False
    assert audio is not None and audio.ff_bias is True
