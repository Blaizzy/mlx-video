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


def test_sanitize_normalizes_unprefixed_diffusers_transformer_keys():
    from mlx_video.models.ltx_2.ltx_2 import LTXModel

    weights = {
        "caption_projection.linear_1.weight": "caption-linear-1",
        "caption_projection.linear_2.bias": "caption-linear-2",
        "transformer_blocks.0.attn1.to_out.0.weight": "attention-output",
        "transformer_blocks.0.ff.net.0.proj.weight": "video-ff-in",
        "transformer_blocks.0.ff.net.2.bias": "video-ff-out",
        "transformer_blocks.0.audio_ff.net.0.proj.weight": "audio-ff-in",
        "transformer_blocks.0.audio_ff.net.2.bias": "audio-ff-out",
        "proj_in.weight": "video-patchify",
        "audio_proj_in.bias": "audio-patchify",
        "time_embed.linear.weight": "video-adaln",
        "audio_time_embed.linear.bias": "audio-adaln",
        "av_cross_attn_video_scale_shift.linear.weight": "video-scale-shift",
        "av_cross_attn_audio_scale_shift.linear.bias": "audio-scale-shift",
        "av_cross_attn_video_a2v_gate.linear.weight": "a2v-gate",
        "av_cross_attn_audio_v2a_gate.linear.bias": "v2a-gate",
        "transformer_blocks.0.attn1.norm_q.weight": "q-norm",
        "transformer_blocks.0.attn1.norm_k.weight": "k-norm",
        "transformer_blocks.0.video_a2v_cross_attn_scale_shift_table": "video-ca-table",
        "transformer_blocks.0.audio_a2v_cross_attn_scale_shift_table": "audio-ca-table",
    }

    assert LTXModel.sanitize(None, weights) == {
        "caption_projection.linear1.weight": "caption-linear-1",
        "caption_projection.linear2.bias": "caption-linear-2",
        "transformer_blocks.0.attn1.to_out.weight": "attention-output",
        "transformer_blocks.0.ff.proj_in.weight": "video-ff-in",
        "transformer_blocks.0.ff.proj_out.bias": "video-ff-out",
        "transformer_blocks.0.audio_ff.proj_in.weight": "audio-ff-in",
        "transformer_blocks.0.audio_ff.proj_out.bias": "audio-ff-out",
        "patchify_proj.weight": "video-patchify",
        "audio_patchify_proj.bias": "audio-patchify",
        "adaln_single.linear.weight": "video-adaln",
        "audio_adaln_single.linear.bias": "audio-adaln",
        "av_ca_video_scale_shift_adaln_single.linear.weight": "video-scale-shift",
        "av_ca_audio_scale_shift_adaln_single.linear.bias": "audio-scale-shift",
        "av_ca_a2v_gate_adaln_single.linear.weight": "a2v-gate",
        "av_ca_v2a_gate_adaln_single.linear.bias": "v2a-gate",
        "transformer_blocks.0.attn1.q_norm.weight": "q-norm",
        "transformer_blocks.0.attn1.k_norm.weight": "k-norm",
        "transformer_blocks.0.scale_shift_table_a2v_ca_video": "video-ca-table",
        "transformer_blocks.0.scale_shift_table_a2v_ca_audio": "audio-ca-table",
    }
