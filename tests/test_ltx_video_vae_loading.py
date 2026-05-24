import json


def test_video_encoder_falls_back_to_unified_vae_parent(tmp_path, monkeypatch):
    from mlx_video.models.ltx_2.config import VideoEncoderModelConfig
    from mlx_video.models.ltx_2.video_vae.video_vae import VideoEncoder
    import mlx_video.models.ltx_2.video_vae.video_vae as video_vae_module

    vae_dir = tmp_path / "vae"
    encoder_dir = vae_dir / "encoder"
    vae_dir.mkdir(parents=True)
    (vae_dir / "config.json").write_text(
        json.dumps(
            {
                "_class_name": "AutoencoderKLLTX2Video",
                "encoder_spatial_padding_mode": "zeros",
            }
        ),
        encoding="utf-8",
    )
    (vae_dir / "diffusion_pytorch_model.safetensors").write_text("", encoding="utf-8")

    captured = {}

    def fake_init(self, config):
        captured["config"] = config

    def fake_load(path):
        captured["loaded_path"] = path
        return {"vae.encoder.conv_in.conv.weight": "encoder-weight"}

    monkeypatch.setattr(VideoEncoder, "__init__", fake_init)
    monkeypatch.setattr(VideoEncoder, "sanitize", lambda self, weights: weights)
    monkeypatch.setattr(VideoEncoder, "load_weights", lambda self, weights, strict=False: None)
    monkeypatch.setattr(video_vae_module.mx, "load", fake_load)

    VideoEncoder.from_pretrained(encoder_dir)

    assert captured["loaded_path"] == str(vae_dir / "diffusion_pytorch_model.safetensors")
    assert isinstance(captured["config"], VideoEncoderModelConfig)


def test_video_decoder_falls_back_to_unified_vae_parent(tmp_path, monkeypatch):
    from mlx_video.models.ltx_2.video_vae.decoder import LTX2VideoDecoder
    import mlx_video.models.ltx_2.video_vae.decoder as decoder_module

    vae_dir = tmp_path / "vae"
    decoder_dir = vae_dir / "decoder"
    vae_dir.mkdir(parents=True)
    (vae_dir / "config.json").write_text(
        json.dumps(
            {
                "_class_name": "AutoencoderKLLTX2Video",
                "decoder_spatial_padding_mode": "reflect",
                "timestep_conditioning": False,
            }
        ),
        encoding="utf-8",
    )
    (vae_dir / "diffusion_pytorch_model.safetensors").write_text("", encoding="utf-8")

    captured = {}

    def fake_init(self, **kwargs):
        captured["kwargs"] = kwargs

    def fake_load(path):
        captured["loaded_path"] = path
        return {"vae.decoder.conv_in.conv.weight": "decoder-weight"}

    monkeypatch.setattr(LTX2VideoDecoder, "__init__", fake_init)
    monkeypatch.setattr(LTX2VideoDecoder, "_infer_blocks", staticmethod(lambda weights: []))
    monkeypatch.setattr(LTX2VideoDecoder, "sanitize", lambda self, weights: weights)
    monkeypatch.setattr(LTX2VideoDecoder, "load_weights", lambda self, weights, strict=True: None)
    monkeypatch.setattr(decoder_module.mx, "load", fake_load)

    LTX2VideoDecoder.from_pretrained(decoder_dir)

    assert captured["loaded_path"] == str(vae_dir / "diffusion_pytorch_model.safetensors")
    assert captured["kwargs"]["spatial_padding_mode"].value == "reflect"


def test_video_encoder_sanitize_accepts_unprefixed_unified_vae_keys():
    from mlx_video.models.ltx_2.config import VideoEncoderModelConfig
    from mlx_video.models.ltx_2.video_vae.video_vae import VideoEncoder

    encoder = VideoEncoder(VideoEncoderModelConfig())

    sanitized = encoder.sanitize(
        {
            "encoder.conv_in.conv.bias": "conv-bias",
            "encoder.down_blocks.0.resnets.0.conv1.conv.bias": "resnet-bias",
            "encoder.down_blocks.0.downsamplers.0.conv.conv.bias": "downsample-bias",
            "encoder.mid_block.resnets.1.conv2.conv.bias": "mid-bias",
            "decoder.conv_in.conv.bias": "decoder-bias",
        }
    )

    assert sanitized == {
        "conv_in.conv.bias": "conv-bias",
        "down_blocks.0.res_blocks.0.conv1.conv.bias": "resnet-bias",
        "down_blocks.1.conv.conv.bias": "downsample-bias",
        "down_blocks.8.res_blocks.1.conv2.conv.bias": "mid-bias",
    }


def test_video_encoder_config_uses_diffusers_latent_channels():
    from mlx_video.models.ltx_2.config import VideoEncoderModelConfig

    config = VideoEncoderModelConfig.from_dict(
        {
            "_class_name": "AutoencoderKLLTX2Video",
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 128,
            "patch_size": 4,
            "encoder_spatial_padding_mode": "zeros",
        }
    )

    assert config.out_channels == 128


def test_video_decoder_sanitize_accepts_unprefixed_unified_vae_keys():
    from mlx_video.models.ltx_2.video_vae.decoder import LTX2VideoDecoder

    decoder = LTX2VideoDecoder(decoder_blocks=[])

    sanitized = decoder.sanitize(
        {
            "encoder.conv_in.conv.bias": "encoder-bias",
            "decoder.conv_in.conv.bias": "conv-bias",
            "decoder.up_blocks.0.resnets.0.conv1.conv.bias": "resnet-bias",
            "decoder.mid_block.resnets.0.conv1.conv.bias": "mid-bias",
            "decoder.up_blocks.0.upsamplers.0.conv.conv.bias": "upsample-bias",
            "per_channel_statistics.mean-of-means": "mean",
        }
    )

    assert sanitized == {
        "conv_in.conv.conv.bias": "conv-bias",
        "up_blocks.2.res_blocks.0.conv1.conv.conv.bias": "resnet-bias",
        "up_blocks.0.res_blocks.0.conv1.conv.conv.bias": "mid-bias",
        "up_blocks.1.conv.conv.bias": "upsample-bias",
        "per_channel_statistics.mean": "mean",
    }


def test_video_decoder_sanitize_adds_identity_statistics_when_missing():
    import mlx.core as mx

    from mlx_video.models.ltx_2.video_vae.decoder import LTX2VideoDecoder

    decoder = LTX2VideoDecoder(decoder_blocks=[])

    sanitized = decoder.sanitize({"decoder.conv_in.conv.bias": mx.zeros((1,))})

    assert sanitized["per_channel_statistics.mean"].shape == (128,)
    assert sanitized["per_channel_statistics.std"].shape == (128,)
    assert mx.all(sanitized["per_channel_statistics.mean"] == 0).item()
    assert mx.all(sanitized["per_channel_statistics.std"] == 1).item()
