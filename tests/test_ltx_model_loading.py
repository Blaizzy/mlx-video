import json


def test_from_pretrained_ignores_huggingface_config_metadata(tmp_path, monkeypatch):
    from mlx_video.models.ltx_2.config import LTXModelConfig
    from mlx_video.models.ltx_2.ltx_2 import LTXModel
    import mlx_video.models.ltx_2.ltx_2 as ltx_module

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "_class_name": "LTXTransformer3DModel",
                "_diffusers_version": "0.35.0",
                "model_type": "ltx av model",
            }
        ),
        encoding="utf-8",
    )

    captured = {}

    def fake_init(self, config):
        captured["config"] = config

    monkeypatch.setattr(LTXModel, "__init__", fake_init)
    monkeypatch.setattr(LTXModel, "sanitize", lambda self, weights: weights)
    monkeypatch.setattr(LTXModel, "load_weights", lambda self, weights, strict=True: None)
    monkeypatch.setattr(LTXModel, "parameters", lambda self: {})
    monkeypatch.setattr(LTXModel, "eval", lambda self: None)
    monkeypatch.setattr(ltx_module.mx, "eval", lambda *args, **kwargs: None)

    LTXModel.from_pretrained(tmp_path, strict=False)

    assert isinstance(captured["config"], LTXModelConfig)
    assert captured["config"].model_type.value == "ltx av model"
