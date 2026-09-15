from pathlib import Path

from mlx_video.models.ltx_2.generate import LTX25_REQUIRED_FILES
from mlx_video.utils import get_model_path


def test_selected_model_files_are_forwarded_to_snapshot_download(
    monkeypatch, tmp_path: Path
) -> None:
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path)

    monkeypatch.setattr("mlx_video.utils.snapshot_download", fake_snapshot_download)

    actual = get_model_path(
        "Lightricks/LTX-2.5", allow_patterns=LTX25_REQUIRED_FILES
    )

    assert actual == tmp_path
    assert calls == [
        {
            "repo_id": "Lightricks/LTX-2.5",
            "allow_patterns": LTX25_REQUIRED_FILES,
        }
    ]


def test_local_model_path_does_not_download(monkeypatch, tmp_path: Path) -> None:
    def fail_snapshot_download(**kwargs):
        raise AssertionError(f"unexpected download: {kwargs}")

    monkeypatch.setattr("mlx_video.utils.snapshot_download", fail_snapshot_download)

    assert get_model_path(str(tmp_path), allow_patterns=["unused"]) == tmp_path
