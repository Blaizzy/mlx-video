from pathlib import Path

import pytest

from mlx_video.webui.runner import (
    GenerationRequest,
    build_command,
    clone_for_retry,
    validate_request,
)


def test_build_ltx_command_includes_selected_options(tmp_path: Path):
    image = tmp_path / "start.png"
    image.write_text("fake", encoding="utf-8")
    request = GenerationRequest(
        engine="ltx",
        prompt="A neon city at night",
        width=768,
        height=512,
        num_frames=97,
        seed=123,
        output_path=tmp_path / "out.mp4",
        ltx_pipeline="dev",
        ltx_model_repo="prince-canuma/LTX-2-dev",
        image_path=image,
        steps=40,
        cfg_scale=3.5,
    )

    command = build_command(request)

    assert Path(command[0]).is_absolute()
    assert command[0].endswith("mlx_video.ltx_2.generate")
    assert command[1] == "--prompt"
    assert "--pipeline" in command
    assert "dev" in command
    assert "--model-repo" in command
    assert "prince-canuma/LTX-2-dev" in command
    assert "--image" in command
    assert str(image) in command
    assert command[-2:] == ["--output-path", str(tmp_path / "out.mp4")]


def test_build_wan_command_includes_model_dir_and_scheduler(tmp_path: Path):
    model_dir = tmp_path / "wan"
    model_dir.mkdir()
    request = GenerationRequest(
        engine="wan",
        prompt="Ocean waves",
        width=640,
        height=480,
        num_frames=81,
        seed=42,
        output_path=tmp_path / "wan.mp4",
        wan_model_dir=model_dir,
        steps=12,
        guide_scale="3.0,4.0",
        shift=12.0,
        scheduler="unipc",
    )

    command = build_command(request)

    assert Path(command[0]).is_absolute()
    assert command[0].endswith("mlx_video.wan_2.generate")
    assert command[1] == "--model-dir"
    assert str(model_dir) in command
    assert "--scheduler" in command
    assert "unipc" in command
    assert command[-2:] == ["--output-path", str(tmp_path / "wan.mp4")]


def test_validate_request_rejects_missing_prompt(tmp_path: Path):
    request = GenerationRequest(
        engine="ltx",
        prompt="",
        width=512,
        height=512,
        num_frames=25,
        seed=1,
        output_path=tmp_path / "out.mp4",
    )

    with pytest.raises(ValueError, match="Prompt is required"):
        validate_request(request)


def test_validate_request_rejects_missing_wan_model_dir(tmp_path: Path):
    request = GenerationRequest(
        engine="wan",
        prompt="A cat",
        width=512,
        height=512,
        num_frames=25,
        seed=1,
        output_path=tmp_path / "out.mp4",
        wan_model_dir=tmp_path / "missing",
    )

    with pytest.raises(ValueError, match="Wan model directory does not exist"):
        validate_request(request)


def test_clone_for_retry_preserves_parameters_with_new_id(tmp_path: Path):
    old = {
        "id": "old",
        "engine": "ltx",
        "prompt": "A cat",
        "params": {"width": 512, "height": 512},
    }

    cloned = clone_for_retry(old, "new", tmp_path)

    assert cloned["id"] == "new"
    assert cloned["engine"] == "ltx"
    assert cloned["prompt"] == "A cat"
    assert cloned["params"] == {"width": 512, "height": 512}
    assert cloned["status"] == "queued"
    assert cloned["output_path"] == str(tmp_path / "videos" / "new.mp4")
