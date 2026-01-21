import math
from typing import Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from functools import partial
from pathlib import Path
from huggingface_hub import snapshot_download
from PIL import Image

def get_model_path(model_repo: str):
    """Get or download model path.
    
    Args:
        model_repo: Either a Hugging Face repo ID (e.g., 'namespace/repo_name')
                   or a local path to the model directory.
    
    Returns:
        Path to the model directory.
    """
    local_path = Path(model_repo)
    if local_path.exists() and local_path.is_dir():
        return local_path
    
    try:
        return Path(snapshot_download(repo_id=model_repo, local_files_only=True))
    except Exception:
        print("Downloading LTX-2 model weights...")
        return Path(snapshot_download(
            repo_id=model_repo,
            local_files_only=False,
            resume_download=True,
            allow_patterns=["*.safetensors", "*.json"],
        ))

def apply_quantization(model: nn.Module, weights: mx.array, quantization: dict):
    if quantization is not None:
        def get_class_predicate(p, m):
            # Handle custom per layer quantizations
            if p in quantization:
                return quantization[p]
            if not hasattr(m, "to_quantized"):
                return False
            # Skip layers not divisible by 64
            if hasattr(m, "weight") and m.weight.shape[0] % 64 != 0:
                return False
            # Handle legacy models which may not have everything quantized
            return f"{p}.scales" in weights

        nn.quantize(
            model,
            group_size=quantization["group_size"],
            bits=quantization["bits"],
            mode=quantization.get("mode", "affine"),
            class_predicate=get_class_predicate,
        )

@partial(mx.compile, shapeless=True)  
def rms_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    return mx.fast.rms_norm(x, mx.ones((x.shape[-1],), dtype=x.dtype), eps)



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
        # Convert to array with matching dtype to avoid float32 promotion
        sigma_arr = mx.array(sigma, dtype=velocity.dtype)
        return noisy - sigma_arr * velocity
    else:
        # sigma is per-sample - ensure dtype matches
        sigma = sigma.astype(velocity.dtype)
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


def load_image(
    image_path: Union[str, Path],
    height: Optional[int] = None,
    width: Optional[int] = None,
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """Load and preprocess an image for I2V conditioning.

    Args:
        image_path: Path to the image file
        height: Target height (must be divisible by 32). If None, uses original.
        width: Target width (must be divisible by 32). If None, uses original.

    Returns:
        Image tensor of shape (H, W, 3) in range [0, 1]
    """
    image = Image.open(image_path).convert("RGB")

    # Resize if dimensions specified
    if height is not None and width is not None:
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    elif height is not None or width is not None:
        # If only one dimension specified, resize preserving aspect ratio
        orig_w, orig_h = image.size
        if height is not None:
            scale = height / orig_h
            new_w = int(orig_w * scale)
            new_w = (new_w // 32) * 32  # Round to nearest 32
            image = image.resize((new_w, height), Image.Resampling.LANCZOS)
        else:
            scale = width / orig_w
            new_h = int(orig_h * scale)
            new_h = (new_h // 32) * 32  # Round to nearest 32
            image = image.resize((width, new_h), Image.Resampling.LANCZOS)
    else:
        # Round to nearest 32
        orig_w, orig_h = image.size
        new_w = (orig_w // 32) * 32
        new_h = (orig_h // 32) * 32
        if new_w != orig_w or new_h != orig_h:
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Convert to numpy then MLX
    image_np = np.array(image).astype(np.float32) / 255.0
    return mx.array(image_np, dtype=dtype)


def resize_image_aspect_ratio(
    image: mx.array,
    long_side: int = 512,
) -> mx.array:
    """Resize image preserving aspect ratio, making long side = long_side.

    Args:
        image: Image tensor of shape (H, W, 3)
        long_side: Target size for the longer dimension

    Returns:
        Resized image tensor
    """
    h, w = image.shape[:2]

    if h > w:
        new_h = long_side
        new_w = int(w * long_side / h)
    else:
        new_w = long_side
        new_h = int(h * long_side / w)

    # Round to nearest 32
    new_h = (new_h // 32) * 32
    new_w = (new_w // 32) * 32

    # Use PIL for high-quality resize
    image_np = np.array(image)
    if image_np.max() <= 1.0:
        image_np = (image_np * 255).astype(np.uint8)
    pil_image = Image.fromarray(image_np)
    pil_image = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    return mx.array(np.array(pil_image).astype(np.float32) / 255.0)


def prepare_image_for_encoding(
    image: mx.array,
    target_height: int,
    target_width: int,
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """Prepare image for VAE encoding by resizing and normalizing.

    Args:
        image: Image tensor of shape (H, W, 3) in range [0, 1]
        target_height: Target height for the video
        target_width: Target width for the video

    Returns:
        Image tensor ready for encoding, shape (1, 3, 1, H, W) in range [-1, 1]
    """
    h, w = image.shape[:2]

    # Resize if needed
    if h != target_height or w != target_width:
        image_np = np.array(image)
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)
        pil_image = pil_image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        image = mx.array(np.array(pil_image).astype(np.float32) / 255.0)

    # Normalize to [-1, 1]
    image = image * 2.0 - 1.0

    # Convert to (B, C, 1, H, W)
    image = mx.transpose(image, (2, 0, 1))  # (3, H, W)
    image = mx.expand_dims(image, axis=0)  # (1, 3, H, W)
    image = mx.expand_dims(image, axis=2)  # (1, 3, 1, H, W)

    return image.astype(dtype)
