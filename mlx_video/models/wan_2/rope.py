
import mlx.core as mx
import numpy as np


def rope_params(max_seq_len: int, dim: int, theta: float = 10000.0) -> mx.array:
    """Precompute RoPE frequency parameters as complex numbers.

    Returns:
        Complex frequency tensor of shape [max_seq_len, dim // 2].
    """
    assert dim % 2 == 0
    freqs = (
        np.arange(max_seq_len, dtype=np.float64)[:, None]
        * (
            1.0
            / np.power(
                theta,
                np.arange(0, dim, 2, dtype=np.float64) / dim,
            )
        )[None, :]
    )
    # Store as (cos, sin) pairs: shape [max_seq_len, dim // 2, 2]
    cos_freqs = np.cos(freqs).astype(np.float32)
    sin_freqs = np.sin(freqs).astype(np.float32)
    return mx.array(np.stack([cos_freqs, sin_freqs], axis=-1))


def rope_cos_sin_at_positions(
    positions,
    dim: int,
    theta: float = 10000.0,
    dtype: mx.Dtype = mx.float32,
) -> tuple:
    """Compute (cos, sin) RoPE frequencies at arbitrary (possibly negative) positions.

    Args:
        positions: 1-D mx.array or Python list of positions.
        dim: per-axis RoPE frequency dim (must be even). Each pair (cos, sin)
            corresponds to a single "complex" element, so returned tensors have
            width dim // 2.
        theta: RoPE theta base (same as rope_params).
        dtype: output dtype.

    Returns:
        (cos, sin) each of shape (N, dim // 2).
    """
    assert dim % 2 == 0
    if not isinstance(positions, mx.array):
        positions = mx.array(list(positions), dtype=mx.float32)
    else:
        positions = positions.astype(mx.float32)

    # Compute inverse frequencies in float64 (as in rope_params) for precision.
    inv_freq_np = 1.0 / np.power(
        theta, np.arange(0, dim, 2, dtype=np.float64) / dim
    )
    inv_freq = mx.array(inv_freq_np.astype(np.float32))
    angles = positions[:, None] * inv_freq[None, :]  # (N, dim // 2)
    return mx.cos(angles).astype(dtype), mx.sin(angles).astype(dtype)


def rope_precompute_cos_sin_segments(
    segments,
    freqs: mx.array,
    dtype: mx.Dtype = mx.float32,
    theta: float = 10000.0,
) -> tuple:
    """Precompute (cos, sin) for a heterogeneous multi-segment sequence.

    Used for S2V, where the sequence is
        [noise (T=[0..F-1] x [0..H-1] x [0..W-1]),
         ref   (T=[t_ref]      x [0..H-1] x [0..W-1]),
         motion_post   (T=[-1] x [0..H-1]     x [0..W-1]),
         motion_2x     (T=[-3] x [0..H_2x-1]  x [0..W_2x-1]),
         motion_4x     (T=[-19..-16] x [0..H_4x-1] x [0..W_4x-1])]

    Each segment carries its own list of temporal indices (may be negative and
    non-contiguous) and an (H, W) spatial grid. Spatial indices start at 0.

    Args:
        segments: list of dicts with keys::
            "t_indices" : Sequence[int] or mx.array — temporal positions.
            "h"         : int             — height patches for this segment.
            "w"         : int             — width  patches for this segment.
            Optional::
            "h_start"   : int             — spatial-height offset (default 0).
            "w_start"   : int             — spatial-width  offset (default 0).
        freqs: precomputed (max_seq_len, half_d, 2) freqs table — only the
            shape's half_d is used to derive the (d_t, d_h, d_w) axis split.
        dtype: output dtype for cos/sin.
        theta: RoPE theta base (must match rope_params).

    Returns:
        (cos_f, sin_f) each of shape (total_seq, 1, half_d) where
        total_seq = sum(len(t_indices_i) * h_i * w_i over segments).
    """
    half_d = freqs.shape[1]
    d_t = half_d - 2 * (half_d // 3)
    d_h = half_d // 3
    d_w = half_d // 3

    all_cos = []
    all_sin = []

    for seg in segments:
        t_indices = seg["t_indices"]
        h = int(seg["h"])
        w = int(seg["w"])
        h_start = int(seg.get("h_start", 0))
        w_start = int(seg.get("w_start", 0))

        if not isinstance(t_indices, mx.array):
            t_indices_arr = mx.array(list(t_indices), dtype=mx.float32)
        else:
            t_indices_arr = t_indices.astype(mx.float32)
        F_seg = int(t_indices_arr.shape[0])

        # Temporal cos/sin for arbitrary (possibly negative) positions.
        cos_t_1d, sin_t_1d = rope_cos_sin_at_positions(
            t_indices_arr, d_t * 2, theta=theta, dtype=dtype
        )  # (F_seg, d_t)

        # Spatial: rows h_start..h_start+h-1 (non-negative in practice).
        h_positions = mx.arange(h_start, h_start + h, dtype=mx.float32)
        w_positions = mx.arange(w_start, w_start + w, dtype=mx.float32)
        cos_h_1d, sin_h_1d = rope_cos_sin_at_positions(
            h_positions, d_h * 2, theta=theta, dtype=dtype
        )  # (h, d_h)
        cos_w_1d, sin_w_1d = rope_cos_sin_at_positions(
            w_positions, d_w * 2, theta=theta, dtype=dtype
        )  # (w, d_w)

        # Broadcast each axis to (F_seg, h, w, *) and concat along last dim.
        cos_t = mx.broadcast_to(
            cos_t_1d.reshape(F_seg, 1, 1, d_t), (F_seg, h, w, d_t)
        )
        cos_h = mx.broadcast_to(
            cos_h_1d.reshape(1, h, 1, d_h), (F_seg, h, w, d_h)
        )
        cos_w = mx.broadcast_to(
            cos_w_1d.reshape(1, 1, w, d_w), (F_seg, h, w, d_w)
        )
        cos_seg = mx.concatenate([cos_t, cos_h, cos_w], axis=-1)
        cos_seg = cos_seg.reshape(F_seg * h * w, 1, half_d)

        sin_t = mx.broadcast_to(
            sin_t_1d.reshape(F_seg, 1, 1, d_t), (F_seg, h, w, d_t)
        )
        sin_h = mx.broadcast_to(
            sin_h_1d.reshape(1, h, 1, d_h), (F_seg, h, w, d_h)
        )
        sin_w = mx.broadcast_to(
            sin_w_1d.reshape(1, 1, w, d_w), (F_seg, h, w, d_w)
        )
        sin_seg = mx.concatenate([sin_t, sin_h, sin_w], axis=-1)
        sin_seg = sin_seg.reshape(F_seg * h * w, 1, half_d)

        all_cos.append(cos_seg)
        all_sin.append(sin_seg)

    cos_f = mx.concatenate(all_cos, axis=0) if len(all_cos) > 1 else all_cos[0]
    sin_f = mx.concatenate(all_sin, axis=0) if len(all_sin) > 1 else all_sin[0]
    return cos_f, sin_f


def rope_apply(
    x: mx.array,
    grid_sizes: list,
    freqs: mx.array,
    precomputed_cos_sin: tuple | None = None,
) -> mx.array:
    """Apply 3-way factorized RoPE to Q or K tensor.

    Args:
        x: Shape [B, L, num_heads, head_dim]
        grid_sizes: List of (F, H, W) tuples per batch element
        freqs: Precomputed cos/sin, shape [1024, d//2, 2] split into 3 parts
        precomputed_cos_sin: Optional (cos, sin) from rope_precompute_cos_sin()
    """
    b, s, n, d = x.shape
    half_d = d // 2

    if precomputed_cos_sin is not None:
        cos_f, sin_f = precomputed_cos_sin
        # For plain T2V/I2V rope_cos_sin covers exactly F*H*W tokens; for S2V
        # multi-segment rope it covers the full concatenated [noise, ref, motion]
        # sequence. Derive seq_len from cos_f so both cases work.
        seq_len = int(cos_f.shape[0])
        # Check if all batch elements have the same grid (common for CFG B=2)
        f0, h0, w0 = grid_sizes[0]
        all_same_grid = (
            all(grid_sizes[i] == grid_sizes[0] for i in range(1, b)) if b > 1 else True
        )

        if all_same_grid:
            # Vectorized path: apply RoPE to all batch elements at once
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
        else:
            # Per-element path for mixed grid sizes
            outputs = []
            for i in range(b):
                f, h, w = grid_sizes[i]
                sl = f * h * w
                x_i = x[i, :sl].reshape(sl, n, half_d, 2)
                x_real = x_i[..., 0]
                x_imag = x_i[..., 1]
                out_real = x_real * cos_f - x_imag * sin_f
                out_imag = x_real * sin_f + x_imag * cos_f
                x_rotated = mx.stack([out_real, out_imag], axis=-1).reshape(sl, n, d)
                if sl < s:
                    x_rotated = mx.concatenate([x_rotated, x[i, sl:]], axis=0)
                outputs.append(x_rotated)
            return mx.stack(outputs)

    # Cast freqs to input dtype to prevent float32 promotion cascade
    if freqs.dtype != x.dtype:
        freqs = freqs.astype(x.dtype)

    # Split frequency dimensions: temporal gets more capacity
    d_t = half_d - 2 * (half_d // 3)
    d_h = half_d // 3
    d_w = half_d // 3

    # Split freqs along dim axis
    freqs_t = freqs[:, :d_t]  # [1024, d_t, 2]
    freqs_h = freqs[:, d_t : d_t + d_h]  # [1024, d_h, 2]
    freqs_w = freqs[:, d_t + d_h : d_t + d_h + d_w]  # [1024, d_w, 2]

    outputs = []
    for i in range(b):
        f, h, w = grid_sizes[i]
        seq_len = f * h * w

        # Reshape x to pairs for rotation: [seq_len, n, half_d, 2]
        x_i = x[i, :seq_len].reshape(seq_len, n, half_d, 2)

        # Build per-position frequencies by expanding along grid dims
        # temporal: [f,1,1,d_t,2] -> [f,h,w,d_t,2]
        ft = mx.broadcast_to(freqs_t[:f].reshape(f, 1, 1, d_t, 2), (f, h, w, d_t, 2))
        # height: [1,h,1,d_h,2] -> [f,h,w,d_h,2]
        fh = mx.broadcast_to(freqs_h[:h].reshape(1, h, 1, d_h, 2), (f, h, w, d_h, 2))
        # width: [1,1,w,d_w,2] -> [f,h,w,d_w,2]
        fw = mx.broadcast_to(freqs_w[:w].reshape(1, 1, w, d_w, 2), (f, h, w, d_w, 2))

        # Concatenate: [f*h*w, half_d, 2]
        freqs_i = mx.concatenate([ft, fh, fw], axis=3).reshape(seq_len, 1, half_d, 2)

        # Apply rotation: (a + bi) * (cos + sin*i) = (a*cos - b*sin) + (a*sin + b*cos)i
        cos_f = freqs_i[..., 0]  # [seq_len, 1, half_d]
        sin_f = freqs_i[..., 1]  # [seq_len, 1, half_d]

        x_real = x_i[..., 0]  # [seq_len, n, half_d]
        x_imag = x_i[..., 1]  # [seq_len, n, half_d]

        out_real = x_real * cos_f - x_imag * sin_f
        out_imag = x_real * sin_f + x_imag * cos_f

        # Interleave back: [seq_len, n, half_d, 2] -> [seq_len, n, d]
        x_rotated = mx.stack([out_real, out_imag], axis=-1).reshape(seq_len, n, d)

        # Handle padding: keep non-rotated tokens after seq_len
        if seq_len < s:
            x_rotated = mx.concatenate([x_rotated, x[i, seq_len:]], axis=0)

        outputs.append(x_rotated)

    return mx.stack(outputs)


def rope_precompute_cos_sin(
    grid_sizes: list, freqs: mx.array, dtype: type = mx.float32
) -> tuple:
    """Precompute cos/sin frequency tensors for constant grid sizes.

    Call once before the diffusion loop. Pass result as precomputed_cos_sin
    to rope_apply to skip per-step broadcast/concat.

    Args:
        grid_sizes: List of (F, H, W) tuples (must be same for all batch elements)
        freqs: Precomputed frequencies [1024, d//2, 2]
        dtype: Target dtype for the output tensors

    Returns:
        (cos_f, sin_f) each [seq_len, 1, half_d]
    """
    if freqs.dtype != dtype:
        freqs = freqs.astype(dtype)

    f, h, w = grid_sizes[0]
    seq_len = f * h * w
    half_d = freqs.shape[1]

    d_t = half_d - 2 * (half_d // 3)
    d_h = half_d // 3
    d_w = half_d // 3

    freqs_t = freqs[:, :d_t]
    freqs_h = freqs[:, d_t : d_t + d_h]
    freqs_w = freqs[:, d_t + d_h : d_t + d_h + d_w]

    ft = mx.broadcast_to(freqs_t[:f].reshape(f, 1, 1, d_t, 2), (f, h, w, d_t, 2))
    fh = mx.broadcast_to(freqs_h[:h].reshape(1, h, 1, d_h, 2), (f, h, w, d_h, 2))
    fw = mx.broadcast_to(freqs_w[:w].reshape(1, 1, w, d_w, 2), (f, h, w, d_w, 2))

    freqs_i = mx.concatenate([ft, fh, fw], axis=3).reshape(seq_len, 1, half_d, 2)
    return freqs_i[..., 0], freqs_i[..., 1]
