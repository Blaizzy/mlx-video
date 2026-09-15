from types import SimpleNamespace

from mlx_video.models.ltx_2.text_encoder import enhance_prompt_gemma4


def test_gemma4_prompt_enhancement_uses_separate_model_and_image(
    monkeypatch,
) -> None:
    calls = {}
    model = SimpleNamespace(config=SimpleNamespace())
    processor = object()

    def fake_load(repo):
        calls["repo"] = repo
        return model, processor

    def fake_template(actual_processor, config, messages, num_images):
        calls["template"] = (actual_processor, config, messages, num_images)
        return "formatted prompt"

    def fake_generate(*, model, processor, prompt, **kwargs):
        calls["generate"] = (model, processor, prompt, kwargs)
        return SimpleNamespace(text="***Enhanced caption")

    monkeypatch.setattr("mlx_vlm.load", fake_load)
    monkeypatch.setattr("mlx_vlm.generate", fake_generate)
    monkeypatch.setattr("mlx_vlm.prompt_utils.apply_chat_template", fake_template)

    actual = enhance_prompt_gemma4(
        "a cat walking",
        "local/gemma4",
        image="cat.png",
        max_tokens=123,
        seed=7,
        verbose=False,
    )

    assert actual == "Enhanced caption"
    assert calls["repo"] == "local/gemma4"
    assert calls["template"][3] == 1
    assert "REFERENCE IMAGE" in calls["template"][2][0]["content"]
    assert calls["generate"][3]["image"] == "cat.png"
    assert calls["generate"][3]["max_tokens"] == 123
    assert calls["generate"][3]["temperature"] == 0.0
