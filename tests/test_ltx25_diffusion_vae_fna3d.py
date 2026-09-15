import mlx.core as mx
import pytest

from mlx_video.models.ltx_2.diffusion_vae.fna3d import (
    neighborhood_attention_3d,
    neighborhood_attention_3d_reference,
)


@pytest.mark.parametrize("dtype,atol", [(mx.float32, 1e-4), (mx.bfloat16, 3e-2)])
def test_metal_fna3d_matches_reference_at_boundaries(dtype, atol):
    mx.random.seed(7)
    shape = (1, 4, 4, 4, 2, 8)
    query = mx.random.normal(shape).astype(dtype)
    key = mx.random.normal(shape).astype(dtype)
    value = mx.random.normal(shape).astype(dtype)
    expected = neighborhood_attention_3d_reference(query, key, value, (3, 3, 3))
    actual = neighborhood_attention_3d(query, key, value, (3, 3, 3))
    mx.eval(actual)
    assert mx.allclose(actual, expected, atol=atol, rtol=atol)


def test_metal_fna3d_supports_ltx_kernel_and_head_dimension():
    mx.random.seed(11)
    shape = (1, 11, 11, 11, 1, 64)
    query = mx.random.normal(shape).astype(mx.bfloat16)
    key = mx.random.normal(shape).astype(mx.bfloat16)
    value = mx.random.normal(shape).astype(mx.bfloat16)
    output = neighborhood_attention_3d(query, key, value, (11, 11, 11))
    mx.eval(output)
    assert output.shape == shape
    assert mx.all(mx.isfinite(output))


def test_simd_head_dim_64_kernel_matches_reference():
    mx.random.seed(19)
    shape = (1, 4, 4, 4, 1, 64)
    query = mx.random.normal(shape).astype(mx.bfloat16)
    key = mx.random.normal(shape).astype(mx.bfloat16)
    value = mx.random.normal(shape).astype(mx.bfloat16)
    expected = neighborhood_attention_3d_reference(query, key, value, (3, 3, 3))
    actual = neighborhood_attention_3d(query, key, value, (3, 3, 3))
    mx.eval(actual)
    assert mx.allclose(actual, expected, atol=3e-2, rtol=3e-2)


def test_fna3d_rejects_invalid_shapes():
    tensor = mx.zeros((1, 3, 3, 3, 1, 8))
    with pytest.raises(ValueError, match="positive odd"):
        neighborhood_attention_3d(tensor, tensor, tensor, (2, 3, 3))
    with pytest.raises(ValueError, match="exceeds"):
        neighborhood_attention_3d(tensor, tensor, tensor, (5, 3, 3))
