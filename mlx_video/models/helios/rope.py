import mlx.core as mx
import numpy as np


def helios_rope_params(
    rope_dim: tuple = (44, 42, 42),
    theta: float = 10000.0,
    max_seq_len: int = 1024,
) -> tuple:
    """Precompute per-dimension RoPE frequency bases for Helios.

    Unlike Wan which uses a single frequency table split by dimension,
    Helios computes separate frequency bases for each spatial/temporal
    dimension using that dimension's size as the denominator.

    Returns:
        (freqs_t, freqs_h, freqs_w) each of shape [max_seq_len, d_i//2, 2]
    """
    results = []
    for d in rope_dim:
        base = 1.0 / np.power(
            theta, np.arange(0, d, 2, dtype=np.float64) / d
        )
        positions = np.arange(max_seq_len, dtype=np.float64)
        freqs = positions[:, None] * base[None, :]
        cos_f = np.cos(freqs).astype(np.float32)
        sin_f = np.sin(freqs).astype(np.float32)
        results.append(mx.array(np.stack([cos_f, sin_f], axis=-1)))
    return tuple(results)


def _rope_compute_5d(
    frame_indices: mx.array,
    height: int,
    width: int,
    freqs: tuple,
    dtype: type = mx.float32,
) -> mx.array:
    """Compute RoPE frequencies as a 5D tensor [F, H, W, half_d, 2].

    Args:
        frame_indices: 1D array of temporal frame indices [F]
        height: Spatial height grid size
        width: Spatial width grid size
        freqs: (freqs_t, freqs_h, freqs_w) from helios_rope_params()
        dtype: Output dtype

    Returns:
        [F, H, W, half_d, 2] where half_d = d_t + d_h + d_w
    """
    freqs_t, freqs_h, freqs_w = freqs
    if freqs_t.dtype != dtype:
        freqs_t = freqs_t.astype(dtype)
        freqs_h = freqs_h.astype(dtype)
        freqs_w = freqs_w.astype(dtype)

    f = frame_indices.shape[0]
    d_t, d_h, d_w = freqs_t.shape[1], freqs_h.shape[1], freqs_w.shape[1]

    ft = mx.broadcast_to(
        freqs_t[frame_indices].reshape(f, 1, 1, d_t, 2), (f, height, width, d_t, 2)
    )
    fh = mx.broadcast_to(
        freqs_h[:height].reshape(1, height, 1, d_h, 2), (f, height, width, d_h, 2)
    )
    fw = mx.broadcast_to(
        freqs_w[:width].reshape(1, 1, width, d_w, 2), (f, height, width, d_w, 2)
    )

    return mx.concatenate([ft, fh, fw], axis=3)


def _rope_pad_and_downsample(rope_5d: mx.array, kernel: tuple) -> mx.array:
    """Downsample a 5D RoPE tensor by averaging over kernel-sized blocks.

    Equivalent to: pad_for_3d_conv + center_down_sample_3d (avg_pool3d) from
    the reference implementation.

    Args:
        rope_5d: [F, H, W, half_d, 2]
        kernel: (kt, kh, kw) downsampling factors

    Returns:
        [F//kt, H//kh, W//kw, half_d, 2]
    """
    f, h, w, d, c = rope_5d.shape
    kt, kh, kw = kernel

    # Replicate-pad to make divisible
    pad_t = (kt - (f % kt)) % kt
    pad_h = (kh - (h % kh)) % kh
    pad_w = (kw - (w % kw)) % kw
    if pad_t > 0:
        rope_5d = mx.concatenate(
            [rope_5d, mx.repeat(rope_5d[-1:], pad_t, axis=0)], axis=0
        )
    if pad_h > 0:
        rope_5d = mx.concatenate(
            [rope_5d, mx.repeat(rope_5d[:, -1:], pad_h, axis=1)], axis=1
        )
    if pad_w > 0:
        rope_5d = mx.concatenate(
            [rope_5d, mx.repeat(rope_5d[:, :, -1:], pad_w, axis=2)], axis=2
        )
    f, h, w = rope_5d.shape[:3]

    # Reshape and average (avg_pool3d equivalent)
    rope_5d = rope_5d.reshape(f // kt, kt, h // kh, kh, w // kw, kw, d, c)
    rope_5d = rope_5d.mean(axis=(1, 3, 5))
    return rope_5d


def _flatten_rope_5d(rope_5d: mx.array) -> tuple:
    """Flatten 5D RoPE to (cos, sin) each [seq_len, 1, half_d].

    Args:
        rope_5d: [F, H, W, half_d, 2]

    Returns:
        (cos_f, sin_f) each [F*H*W, 1, half_d]
    """
    f, h, w, d, _ = rope_5d.shape
    flat = rope_5d.reshape(f * h * w, 1, d, 2)
    return flat[..., 0], flat[..., 1]


def helios_rope_apply(
    x: mx.array,
    frame_indices: mx.array,
    grid_size: tuple,
    freqs: tuple,
    precomputed_cos_sin: tuple | None = None,
) -> mx.array:
    """Apply 3-way factorized RoPE to Q or K tensor for Helios.

    Args:
        x: Shape [B, L, num_heads, head_dim]
        frame_indices: Frame indices for this chunk, shape [F] (auto-regressive offset)
        grid_size: (F, H, W) spatial/temporal grid for current chunk
        freqs: (freqs_t, freqs_h, freqs_w) from helios_rope_params()
        precomputed_cos_sin: Optional (cos_f, sin_f) for constant grids
    """
    b, s, n, d = x.shape
    half_d = d // 2
    f, h, w = grid_size
    seq_len = f * h * w

    if precomputed_cos_sin is not None:
        cos_f, sin_f = precomputed_cos_sin
    else:
        rope_5d = _rope_compute_5d(frame_indices, h, w, freqs, dtype=x.dtype)
        cos_f, sin_f = _flatten_rope_5d(rope_5d)

    x_seq = x[:, :seq_len].reshape(b, seq_len, n, half_d, 2)
    x_real = x_seq[..., 0]
    x_imag = x_seq[..., 1]
    out_real = x_real * cos_f - x_imag * sin_f
    out_imag = x_real * sin_f + x_imag * cos_f
    x_rotated = mx.stack([out_real, out_imag], axis=-1).reshape(
        b, seq_len, n, d
    )
    if seq_len < s:
        x_rotated = mx.concatenate([x_rotated, x[:, seq_len:]], axis=1)
    return x_rotated


def helios_rope_precompute_cos_sin(
    frame_indices: mx.array,
    grid_size: tuple,
    freqs: tuple,
    dtype: type = mx.float32,
) -> tuple:
    """Precompute cos/sin for a constant grid. Call once before denoising loop.

    Args:
        frame_indices: 1D array [F] of temporal frame indices
        grid_size: (F, H, W)
        freqs: (freqs_t, freqs_h, freqs_w)

    Returns:
        (cos_f, sin_f) each [seq_len, 1, half_d]
    """
    f, h, w = grid_size
    rope_5d = _rope_compute_5d(frame_indices, h, w, freqs, dtype=dtype)
    return _flatten_rope_5d(rope_5d)


def helios_rope_precompute_history(
    frame_indices: mx.array,
    spatial_h: int,
    spatial_w: int,
    freqs: tuple,
    downsample_kernel: tuple | None = None,
    dtype: type = mx.float32,
) -> tuple:
    """Precompute cos/sin for history tokens, with optional spatial downsampling.

    This matches the reference approach: compute RoPE at the short-history
    spatial resolution, then avg-pool downsample for mid/long scales.

    Args:
        frame_indices: 1D array [F] of temporal frame indices
        spatial_h: Height at the base (short) spatial resolution
        spatial_w: Width at the base (short) spatial resolution
        freqs: (freqs_t, freqs_h, freqs_w)
        downsample_kernel: (kt, kh, kw) for mid/long. None for short.
        dtype: Output dtype

    Returns:
        (cos_f, sin_f) each [seq_len, 1, half_d]
    """
    rope_5d = _rope_compute_5d(frame_indices, spatial_h, spatial_w, freqs, dtype=dtype)
    if downsample_kernel is not None:
        rope_5d = _rope_pad_and_downsample(rope_5d, downsample_kernel)
    return _flatten_rope_5d(rope_5d)
