__all__ = [
    "LTXModel",
    "LTXModelConfig",
    "AudioDecoder",
    "AudioEncoder",
    "Vocoder",
    "decode_audio",
    "AudioPatchifier",
    "AudioLatentShape",
    "PerChannelStatistics",
    "VideoConditionByLatentIndex",
    "convert_audio_encoder",
    "get_model_path",
    "load_safetensors",
    "load_config",
    "save_weights",
    "WanModel",
    "WanModelConfig",
]


def __getattr__(name):
    if name in {"LTXModel", "LTXModelConfig"}:
        from mlx_video.models.ltx_2 import LTXModel, LTXModelConfig

        return {"LTXModel": LTXModel, "LTXModelConfig": LTXModelConfig}[name]

    if name in {
        "AudioDecoder",
        "AudioEncoder",
        "AudioLatentShape",
        "AudioPatchifier",
        "PerChannelStatistics",
        "Vocoder",
        "decode_audio",
    }:
        from mlx_video.models.ltx_2 import audio_vae

        return getattr(audio_vae, name)

    if name == "VideoConditionByLatentIndex":
        from mlx_video.models.ltx_2.conditioning import VideoConditionByLatentIndex

        return VideoConditionByLatentIndex

    if name in {
        "convert_audio_encoder",
        "get_model_path",
        "load_config",
        "load_safetensors",
        "save_weights",
    }:
        from mlx_video.models.ltx_2 import utils

        return getattr(utils, name)

    if name in {"WanModel", "WanModelConfig"}:
        from mlx_video.models.wan_2 import WanModel, WanModelConfig

        return {"WanModel": WanModel, "WanModelConfig": WanModelConfig}[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
