import mlx.core as mx

from mlx_video.models.ltx_2.generate import euler_ancestral_step


def test_euler_ancestral_step_is_reproducible() -> None:
    sample = mx.ones((1, 2, 3), dtype=mx.float32)
    denoised = mx.zeros_like(sample)

    first, first_key = euler_ancestral_step(
        sample, denoised, sigma=1.0, sigma_next=0.5, key=mx.random.key(7)
    )
    second, second_key = euler_ancestral_step(
        sample, denoised, sigma=1.0, sigma_next=0.5, key=mx.random.key(7)
    )

    assert mx.array_equal(first, second).item()
    assert mx.array_equal(first_key, second_key).item()
    assert not mx.array_equal(first, sample).item()


def test_euler_ancestral_final_step_returns_denoised_without_consuming_key() -> None:
    sample = mx.ones((1, 2), dtype=mx.float32)
    denoised = mx.full_like(sample, 3.0)
    key = mx.random.key(9)

    actual, actual_key = euler_ancestral_step(
        sample, denoised, sigma=0.5, sigma_next=0.0, key=key
    )

    assert mx.array_equal(actual, denoised).item()
    assert mx.array_equal(actual_key, key).item()
