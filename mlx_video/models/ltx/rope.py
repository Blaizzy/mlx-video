
import math
from typing import List, Optional, Tuple

import mlx.core as mx

from mlx_video.models.ltx.config import LTXRopeType


def apply_rotary_emb(
    input_tensor: mx.array,
    freqs_cis: Tuple[mx.array, mx.array],
    rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
) -> mx.array:
    """Apply rotary position embeddings to input tensor.

    Args:
        input_tensor: Input tensor to apply RoPE to
        freqs_cis: Tuple of (cos_freqs, sin_freqs)
        rope_type: Type of RoPE to apply (INTERLEAVED or SPLIT)

    Returns:
        Tensor with rotary embeddings applied
    """
    if rope_type == LTXRopeType.INTERLEAVED:
        return apply_interleaved_rotary_emb(input_tensor, freqs_cis[0], freqs_cis[1])
    elif rope_type == LTXRopeType.SPLIT:
        return apply_split_rotary_emb(input_tensor, freqs_cis[0], freqs_cis[1])
    else:
        raise ValueError(f"Invalid rope type: {rope_type}")


def apply_interleaved_rotary_emb(
    input_tensor: mx.array,
    cos_freqs: mx.array,
    sin_freqs: mx.array,
) -> mx.array:
    """Apply interleaved rotary embeddings.

    Pairs adjacent dimensions and applies rotation.
    Pattern: [x0, x1, x2, x3, ...] -> rotate pairs (x0,x1), (x2,x3), ...

    Args:
        input_tensor: Input tensor of shape (..., dim)
        cos_freqs: Cosine frequencies
        sin_freqs: Sine frequencies

    Returns:
        Tensor with interleaved rotary embeddings applied
    """
    # Compute in float32 for better precision
    input_dtype = input_tensor.dtype
    input_tensor = input_tensor.astype(mx.float32)
    cos_freqs = cos_freqs.astype(mx.float32)
    sin_freqs = sin_freqs.astype(mx.float32)

    # Reshape to pair adjacent dimensions: (..., dim) -> (..., dim/2, 2)
    shape = input_tensor.shape
    input_tensor = mx.reshape(input_tensor, shape[:-1] + (shape[-1] // 2, 2))

    # Extract pairs
    t1 = input_tensor[..., 0]  # Even indices
    t2 = input_tensor[..., 1]  # Odd indices

    # Apply rotation: (-t2, t1) pattern
    t_rot = mx.stack([-t2, t1], axis=-1)

    # Flatten back: (..., dim/2, 2) -> (..., dim)
    input_tensor = mx.reshape(input_tensor, shape)
    t_rot = mx.reshape(t_rot, shape)

    # Apply rotary embeddings
    out = input_tensor * cos_freqs + t_rot * sin_freqs

    return out.astype(input_dtype)


def rotate_half_interleaved(x: mx.array) -> mx.array:
    """Rotate for interleaved RoPE: [x0, x1, x2, x3] -> [-x1, x0, -x3, x2].

    PyTorch equivalent:
        t_dup = rearrange(x, "... (d r) -> ... d r", r=2)
        t1, t2 = t_dup.unbind(dim=-1)
        t_dup = torch.stack((-t2, t1), dim=-1)
        return rearrange(t_dup, "... d r -> ... (d r)")
    """
    # x: (..., dim) where dim is even
    x_even = x[..., 0::2]  # [x0, x2, x4, ...]
    x_odd = x[..., 1::2]   # [x1, x3, x5, ...]
    # Stack: [[-x1, x0], [-x3, x2], ...] then flatten to [-x1, x0, -x3, x2, ...]
    rotated = mx.stack([-x_odd, x_even], axis=-1)
    return mx.reshape(rotated, x.shape)

def apply_rotary_emb_1d(
    q: mx.array,
    k: mx.array,
    freqs_cis: mx.array,
) -> Tuple[mx.array, mx.array]:
    """Apply 1D rotary embeddings using precomputed frequencies (interleaved)."""
    # freqs_cis: (1, seq_len, num_heads, head_dim, 2) where [..., 0] = cos, [..., 1] = sin
    cos = freqs_cis[..., 0]  # (1, seq_len, num_heads, head_dim)
    sin = freqs_cis[..., 1]

    # q, k: (batch, seq_len, num_heads, head_dim)
    # Interleaved RoPE: pairs of adjacent dims rotate together
    q_r = q * cos + rotate_half_interleaved(q) * sin
    k_r = k * cos + rotate_half_interleaved(k) * sin

    return q_r, k_r


def apply_split_rotary_emb(
    input_tensor: mx.array,
    cos_freqs: mx.array,
    sin_freqs: mx.array,
) -> mx.array:
    """Apply split rotary embeddings.

    Splits dimensions into two halves and applies rotation.
    Pattern: split into first half and second half

    Args:
        input_tensor: Input tensor
        cos_freqs: Cosine frequencies of shape (B, H, T, D//2)
        sin_freqs: Sine frequencies of shape (B, H, T, D//2)

    Returns:
        Tensor with split rotary embeddings applied
    """
    input_dtype = input_tensor.dtype
    needs_reshape = False
    original_shape = input_tensor.shape

    # Handle dimension mismatch
    if input_tensor.ndim != 4 and cos_freqs.ndim == 4:
        b, h, t, _ = cos_freqs.shape
        # Reshape from (B, T, H*D) to (B, H, T, D)
        input_tensor = mx.reshape(input_tensor, (b, t, h, -1))
        input_tensor = mx.swapaxes(input_tensor, 1, 2)
        needs_reshape = True

    # Cast to float32 for computation precision
    input_tensor = input_tensor.astype(mx.float32)
    cos_freqs = cos_freqs.astype(mx.float32)
    sin_freqs = sin_freqs.astype(mx.float32)

    # Split into two halves: (..., dim) -> (..., 2, dim//2)
    dim = input_tensor.shape[-1]
    split_input = mx.reshape(input_tensor, input_tensor.shape[:-1] + (2, dim // 2))

    # Get first and second halves
    first_half = split_input[..., 0, :]  # (..., dim//2)
    second_half = split_input[..., 1, :]  # (..., dim//2)

    # Apply cosine to both halves
    output_first = first_half * cos_freqs
    output_second = second_half * cos_freqs

    # Apply sine cross-terms (addcmul pattern)
    output_first = output_first - sin_freqs * second_half
    output_second = output_second + sin_freqs * first_half

    # Stack back together
    output = mx.stack([output_first, output_second], axis=-2)

    # Flatten: (..., 2, dim//2) -> (..., dim)
    output = mx.reshape(output, input_tensor.shape)

    if needs_reshape:
        # Reshape back: (B, H, T, D) -> (B, T, H*D)
        b, h, t, d = output.shape
        output = mx.swapaxes(output, 1, 2)
        output = mx.reshape(output, (b, t, h * d))

    return output.astype(input_dtype)


def generate_freq_grid(
    positional_embedding_theta: float,
    positional_embedding_max_pos_count: int,
    inner_dim: int,
) -> mx.array:
    """Generate frequency grid for RoPE.

    Args:
        positional_embedding_theta: Base theta value
        positional_embedding_max_pos_count: Number of position dimensions
        inner_dim: Inner dimension of the model

    Returns:
        Frequency indices tensor
    """
    theta = positional_embedding_theta
    start = 1.0
    end = theta

    n_elem = 2 * positional_embedding_max_pos_count

    # Compute logarithmic spacing
    log_start = math.log(start) / math.log(theta)
    log_end = math.log(end) / math.log(theta)

    num_indices = inner_dim // n_elem
    if num_indices == 0:
        num_indices = 1

    # Create linearly spaced values in log space
    lin_space = mx.linspace(log_start, log_end, num_indices)

    # Compute power indices
    pow_indices = mx.power(theta, lin_space)

    # Scale by pi/2
    return pow_indices * (math.pi / 2)


def get_fractional_positions(
    indices_grid: mx.array,
    max_pos: List[int],
) -> mx.array:
    """Convert indices to fractional positions.

    Args:
        indices_grid: Grid of position indices of shape (B, n_pos_dims, ...)
        max_pos: Maximum position for each dimension

    Returns:
        Fractional positions in range [-1, 1] after scaling
    """
    n_pos_dims = indices_grid.shape[1]
    assert n_pos_dims == len(max_pos), (
        f"Number of position dimensions ({n_pos_dims}) must match max_pos length ({len(max_pos)})"
    )

    # Divide each dimension by its max position
    fractional_positions = []
    for i in range(n_pos_dims):
        frac = indices_grid[:, i] / max_pos[i]
        fractional_positions.append(frac)

    return mx.stack(fractional_positions, axis=-1)


def generate_freqs(
    indices: mx.array,
    indices_grid: mx.array,
    max_pos: List[int],
    use_middle_indices_grid: bool,
) -> mx.array:
    """Generate frequencies from indices and position grid.

    Args:
        indices: Frequency indices
        indices_grid: Position indices grid
        max_pos: Maximum positions per dimension
        use_middle_indices_grid: Whether to use middle of index ranges

    Returns:
        Frequency tensor
    """
    # Handle middle indices grid
    if use_middle_indices_grid:
        # indices_grid shape: (B, n_dims, T, 2) where last dim is [start, end]
        assert len(indices_grid.shape) == 4
        assert indices_grid.shape[-1] == 2
        indices_grid_start = indices_grid[..., 0]
        indices_grid_end = indices_grid[..., 1]
        indices_grid = (indices_grid_start + indices_grid_end) / 2.0
    elif len(indices_grid.shape) == 4:
        indices_grid = indices_grid[..., 0]

    # Get fractional positions
    fractional_positions = get_fractional_positions(indices_grid, max_pos)

    # Compute frequencies
    # fractional_positions: (B, T, n_dims)
    # indices: (inner_dim // n_elem,)
    # Result: (B, T, inner_dim // n_elem * n_dims)

    # Scale fractional positions to [-1, 1]
    scaled_positions = fractional_positions * 2 - 1  # (B, T, n_dims)

    # Outer product with indices
    # (B, T, n_dims, 1) * (1, 1, 1, n_indices) -> (B, T, n_dims, n_indices)
    freqs = mx.expand_dims(scaled_positions, axis=-1) * mx.expand_dims(
        mx.expand_dims(mx.expand_dims(indices, axis=0), axis=0), axis=0
    )

    # Transpose and flatten: (B, T, n_dims, n_indices) -> (B, T, n_indices * n_dims)
    freqs = mx.swapaxes(freqs, -1, -2)  # (B, T, n_indices, n_dims)
    freqs = mx.reshape(freqs, freqs.shape[:-2] + (-1,))

    return freqs


def split_freqs_cis(
    freqs: mx.array,
    pad_size: int,
    num_attention_heads: int,
) -> Tuple[mx.array, mx.array]:
    """Prepare cos/sin frequencies for split RoPE.

    Args:
        freqs: Frequency tensor
        pad_size: Padding size for dimension alignment
        num_attention_heads: Number of attention heads

    Returns:
        Tuple of (cos_freq, sin_freq) with shape (B, H, T, D//2)
    """
    cos_freq = mx.cos(freqs)
    sin_freq = mx.sin(freqs)

    # Add padding if needed
    if pad_size != 0:
        cos_padding = mx.ones_like(cos_freq[:, :, :pad_size])
        sin_padding = mx.zeros_like(sin_freq[:, :, :pad_size])

        cos_freq = mx.concatenate([cos_padding, cos_freq], axis=-1)
        sin_freq = mx.concatenate([sin_padding, sin_freq], axis=-1)

    # Reshape for multi-head attention
    b, t = cos_freq.shape[0], cos_freq.shape[1]

    cos_freq = mx.reshape(cos_freq, (b, t, num_attention_heads, -1))
    sin_freq = mx.reshape(sin_freq, (b, t, num_attention_heads, -1))

    # Swap axes: (B, T, H, D//2) -> (B, H, T, D//2)
    cos_freq = mx.swapaxes(cos_freq, 1, 2)
    sin_freq = mx.swapaxes(sin_freq, 1, 2)

    return cos_freq, sin_freq


def interleaved_freqs_cis(
    freqs: mx.array,
    pad_size: int,
) -> Tuple[mx.array, mx.array]:
    """Prepare cos/sin frequencies for interleaved RoPE.

    Args:
        freqs: Frequency tensor of shape (B, T, dim//2)
        pad_size: Padding size for dimension alignment

    Returns:
        Tuple of (cos_freq, sin_freq) with shape (B, T, dim)
    """
    # Compute cos and sin
    cos_freq = mx.cos(freqs)
    sin_freq = mx.sin(freqs)

    # Repeat interleave: each element repeated twice
    # (B, T, D) -> (B, T, 2*D) with pattern [c0, c0, c1, c1, ...]
    cos_freq = mx.repeat(cos_freq, 2, axis=-1)
    sin_freq = mx.repeat(sin_freq, 2, axis=-1)

    # Add padding if needed
    if pad_size != 0:
        cos_padding = mx.ones_like(cos_freq[:, :, :pad_size])
        sin_padding = mx.zeros_like(sin_freq[:, :, :pad_size])
        cos_freq = mx.concatenate([cos_padding, cos_freq], axis=-1)
        sin_freq = mx.concatenate([sin_padding, sin_freq], axis=-1)

    return cos_freq, sin_freq


def precompute_freqs_cis(
    indices_grid: mx.array,
    dim: int,
    theta: float = 10000.0,
    max_pos: Optional[List[int]] = None,
    use_middle_indices_grid: bool = False,
    num_attention_heads: int = 32,
    rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
    double_precision: bool = False,
) -> Tuple[mx.array, mx.array]:
    """Precompute RoPE frequencies.

    Args:
        indices_grid: Position indices grid
        dim: Dimension for RoPE
        theta: Base theta value for frequency computation
        max_pos: Maximum position per dimension
        use_middle_indices_grid: Whether to use middle indices
        num_attention_heads: Number of attention heads
        rope_type: Type of RoPE (INTERLEAVED or SPLIT)
        double_precision: If True, compute frequencies in float64 for higher precision

    Returns:
        Tuple of (cos_freq, sin_freq) tensors
    """
    if max_pos is None:
        max_pos = [20, 2048, 2048]


    if double_precision:
        return _precompute_freqs_cis_double_precision(
            indices_grid, dim, theta, max_pos, use_middle_indices_grid,
            num_attention_heads, rope_type
        )

    # Generate frequency indices
    indices = generate_freq_grid(theta, indices_grid.shape[1], dim)

    # Generate frequencies
    freqs = generate_freqs(indices, indices_grid, max_pos, use_middle_indices_grid)

    # Prepare cos/sin based on rope type
    if rope_type == LTXRopeType.SPLIT:
        expected_freqs = dim // 2
        current_freqs = freqs.shape[-1]
        pad_size = expected_freqs - current_freqs
        cos_freq, sin_freq = split_freqs_cis(freqs, pad_size, num_attention_heads)
    else:
        # Interleaved
        n_elem = 2 * indices_grid.shape[1]
        cos_freq, sin_freq = interleaved_freqs_cis(freqs, dim % n_elem)

    return cos_freq, sin_freq


def _precompute_freqs_cis_double_precision(
    indices_grid: mx.array,
    dim: int,
    theta: float,
    max_pos: List[int],
    use_middle_indices_grid: bool,
    num_attention_heads: int,
    rope_type: LTXRopeType,
) -> Tuple[mx.array, mx.array]:
    """Compute RoPE frequencies with higher precision using float32.

    This version stays entirely in MLX/GPU, avoiding expensive NumPy round-trips.
    Uses float32 for computation precision (sufficient for RoPE).
    """
    # Warn if positions are bfloat16 - this causes quality degradation
    if indices_grid.dtype == mx.bfloat16:
        import warnings
        warnings.warn(
            "Position grid has dtype bfloat16, which causes precision loss in RoPE. "
            "Use float32 for position grids to avoid quality degradation.",
            UserWarning,
            stacklevel=2
        )

    # Cast to float32 for computation (stay on GPU, no NumPy/CPU conversion)
    indices_grid_f32 = indices_grid.astype(mx.float32)

    n_pos_dims = indices_grid_f32.shape[1]
    n_elem = 2 * n_pos_dims

    # Compute log-spaced frequencies in float32
    log_start = math.log(1.0) / math.log(theta)
    log_end = math.log(theta) / math.log(theta)
    num_indices = dim // n_elem
    if num_indices == 0:
        num_indices = 1

    lin_space = mx.linspace(log_start, log_end, num_indices)
    freq_indices = mx.power(mx.array(theta, dtype=mx.float32), lin_space) * (math.pi / 2)

    # Handle middle indices grid
    # Input shape: (B, n_dims, T, 2) for middle indices or (B, n_dims, T, 1) otherwise
    if use_middle_indices_grid:
        assert len(indices_grid_f32.shape) == 4
        assert indices_grid_f32.shape[-1] == 2
        indices_grid_start = indices_grid_f32[..., 0]
        indices_grid_end = indices_grid_f32[..., 1]
        indices_grid_f32 = (indices_grid_start + indices_grid_end) / 2.0
    elif len(indices_grid_f32.shape) == 4:
        indices_grid_f32 = indices_grid_f32[..., 0]
    # After handling: indices_grid_f32 shape is (B, n_dims, T)

    # Get fractional positions: (B, n_dims, T) -> (B, T, n_dims)
    # Compute fractional positions for each dimension
    fractional_list = []
    for i in range(n_pos_dims):
        frac = indices_grid_f32[:, i, :] / max_pos[i]  # (B, T)
        fractional_list.append(frac)

    # Stack: (B, T, n_dims)
    fractional_positions = mx.stack(fractional_list, axis=-1)

    # Scale to [-1, 1]
    scaled_positions = fractional_positions * 2 - 1

    # Compute frequencies: outer product
    # scaled_positions: (B, T, n_dims) -> (B, T, n_dims, 1)
    # freq_indices: (num_indices,) -> (1, 1, 1, num_indices)
    freqs = mx.expand_dims(scaled_positions, axis=-1) * mx.reshape(freq_indices, (1, 1, 1, -1))
    # freqs: (B, T, n_dims, num_indices)

    # Transpose and flatten: (B, T, n_dims, num_indices) -> (B, T, num_indices, n_dims) -> (B, T, num_indices * n_dims)
    freqs = mx.swapaxes(freqs, -1, -2)
    freqs = mx.reshape(freqs, (freqs.shape[0], freqs.shape[1], -1))

    # Compute cos/sin
    cos_freq = mx.cos(freqs)
    sin_freq = mx.sin(freqs)

    # Prepare based on rope type
    if rope_type == LTXRopeType.SPLIT:
        expected_freqs = dim // 2
        current_freqs = cos_freq.shape[-1]
        pad_size = expected_freqs - current_freqs

        # Add padding
        if pad_size > 0:
            cos_padding = mx.ones((*cos_freq.shape[:-1], pad_size), dtype=mx.float32)
            sin_padding = mx.zeros((*sin_freq.shape[:-1], pad_size), dtype=mx.float32)
            cos_freq = mx.concatenate([cos_padding, cos_freq], axis=-1)
            sin_freq = mx.concatenate([sin_padding, sin_freq], axis=-1)

        # Reshape for multi-head attention: (B, T, dim//2) -> (B, H, T, dim//2//H)
        b, t = cos_freq.shape[0], cos_freq.shape[1]
        cos_freq = mx.reshape(cos_freq, (b, t, num_attention_heads, -1))
        sin_freq = mx.reshape(sin_freq, (b, t, num_attention_heads, -1))
        cos_freq = mx.swapaxes(cos_freq, 1, 2)
        sin_freq = mx.swapaxes(sin_freq, 1, 2)
    else:
        # Interleaved
        cos_freq = mx.repeat(cos_freq, 2, axis=-1)
        sin_freq = mx.repeat(sin_freq, 2, axis=-1)

        pad_size = dim % n_elem
        if pad_size > 0:
            cos_padding = mx.ones((*cos_freq.shape[:-1], pad_size), dtype=mx.float32)
            sin_padding = mx.zeros((*sin_freq.shape[:-1], pad_size), dtype=mx.float32)
            cos_freq = mx.concatenate([cos_padding, cos_freq], axis=-1)
            sin_freq = mx.concatenate([sin_padding, sin_freq], axis=-1)

    return cos_freq, sin_freq
