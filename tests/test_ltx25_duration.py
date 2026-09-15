import mlx.core as mx
import pytest

from mlx_video.models.ltx_2.duration import DurationHead, seconds_to_num_frames


def test_duration_head_shapes_and_requires_tokens() -> None:
    head = DurationHead(
        video_dim=8,
        audio_dim=4,
        hidden_dim=4,
        num_queries=1,
        num_heads=2,
        mlp_hidden=4,
    )

    prediction = head(
        video_tokens=mx.zeros((1, 3, 8)),
        audio_tokens=mx.zeros((1, 2, 4)),
    )

    assert prediction.shape == (1,)
    assert mx.all(mx.isfinite(prediction)).item()
    with pytest.raises(ValueError, match="requires video or audio"):
        head()


def test_seconds_to_num_frames_clamps_and_snaps_to_grid() -> None:
    assert seconds_to_num_frames(0.1, 24.0, 1.0, 20.0) == 25
    assert seconds_to_num_frames(5.0, 24.0, 1.0, 20.0) == 113
    assert seconds_to_num_frames(100.0, 24.0, 1.0, 20.0) == 473


@pytest.mark.parametrize(
    ("frame_rate", "minimum", "maximum"),
    [(0.0, 1.0, 20.0), (24.0, 0.0, 20.0), (24.0, 5.0, 2.0)],
)
def test_seconds_to_num_frames_rejects_invalid_bounds(
    frame_rate: float, minimum: float, maximum: float
) -> None:
    with pytest.raises(ValueError):
        seconds_to_num_frames(3.0, frame_rate, minimum, maximum)
