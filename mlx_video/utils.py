"""Utility functions for MLX Video."""

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from functools import partial
from pathlib import Path
from huggingface_hub import snapshot_download

def get_model_path(model_repo: str) -> Path:
    """Get or download model path. Supports HuggingFace repos and local paths.
    
    Args:
        model_repo: Either a HuggingFace repo ID (e.g., 'Lightricks/LTX-2') 
                   or a local filesystem path.
    
    Returns:
        Path to the model directory.
    """
    # Check if it's a local path first
    path = Path(model_repo)
    if path.exists() and path.is_dir():
        return path
    
    # Otherwise treat as HuggingFace repo
    try:
        return Path(snapshot_download(repo_id=model_repo, local_files_only=True))
    except Exception:
        print(f"Downloading model from {model_repo}...")
        return Path(snapshot_download(
            repo_id=model_repo,
            local_files_only=False,
            resume_download=True,
            allow_patterns=["*.safetensors", "*.json"],
        ))


@partial(mx.compile, shapeless=True)
def rms_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    return mx.fast.rms_norm(x, mx.ones((x.shape[-1],)), eps)



@partial(mx.compile, shapeless=True)
def to_denoised(
    noisy: mx.array,
    velocity: mx.array,
    sigma: mx.array | float
) -> mx.array:
    """Convert velocity prediction to denoised output.

    Given noisy input x_t and velocity prediction v, compute denoised x_0:
    x_0 = x_t - sigma * v

    Args:
        noisy: Noisy input tensor x_t
        velocity: Velocity prediction v
        sigma: Noise level (scalar or per-sample)

    Returns:
        Denoised tensor x_0
    """
    if isinstance(sigma, (int, float)):
        return noisy - sigma * velocity
    else:
        # sigma is per-sample
        while sigma.ndim < velocity.ndim:
            sigma = mx.expand_dims(sigma, axis=-1)
        return noisy - sigma * velocity


def repeat_interleave(x: mx.array, repeats: int, axis: int = -1) -> mx.array:
    """Repeat elements of tensor along an axis, similar to torch.repeat_interleave.

    Args:
        x: Input tensor
        repeats: Number of repetitions for each element
        axis: The axis along which to repeat values

    Returns:
        Tensor with repeated values
    """
    # Handle negative axis
    if axis < 0:
        axis = x.ndim + axis

    # Get shape
    shape = list(x.shape)

    # Expand dims, repeat, then reshape
    x = mx.expand_dims(x, axis=axis + 1)

    # Create tile pattern
    tile_pattern = [1] * x.ndim
    tile_pattern[axis + 1] = repeats

    x = mx.tile(x, tile_pattern)

    # Reshape to merge the repeated dimension
    new_shape = shape.copy()
    new_shape[axis] *= repeats

    return mx.reshape(x, new_shape)


class PixelNorm(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return x / mx.sqrt(mx.mean(x * x, axis=1, keepdims=True) + self.eps)


def get_timestep_embedding(
    timesteps: mx.array,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1.0,
    scale: float = 1.0,
    max_period: int = 10000,
) -> mx.array:
    """Create sinusoidal timestep embeddings.

    Args:
        timesteps: 1D tensor of timesteps
        embedding_dim: Dimension of the embeddings to create
        flip_sin_to_cos: If True, flip sin and cos ordering
        downscale_freq_shift: Frequency shift factor
        scale: Scale factor for timesteps
        max_period: Maximum period for the sinusoids

    Returns:
        Tensor of shape (len(timesteps), embedding_dim)
    """
    assert timesteps.ndim == 1, "Timesteps should be 1D"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * mx.arange(0, half_dim, dtype=mx.float32)
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = mx.exp(exponent)
    emb = (timesteps[:, None].astype(mx.float32) * scale) * emb[None, :]

    # Compute sin and cos embeddings
    if flip_sin_to_cos:
        emb = mx.concatenate([mx.cos(emb), mx.sin(emb)], axis=-1)
    else:
        emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)

    # Zero pad if odd embedding dimension
    if embedding_dim % 2 == 1:
        emb = mx.pad(emb, [(0, 0), (0, 1)])

    return emb
