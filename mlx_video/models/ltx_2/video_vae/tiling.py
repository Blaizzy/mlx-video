"""VAE Tiling Configuration for decoding large videos.

Implements spatial and temporal tiling with trapezoidal blending masks
to decode large videos without running out of memory.

Default configuration (from PyTorch):
- Spatial: 512px tiles with 64px overlap
- Temporal: 64 frames with 24 frame overlap
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import mlx.core as mx


def compute_trapezoidal_mask_1d(
    length: int,
    ramp_left: int,
    ramp_right: int,
    left_starts_from_0: bool = False,
) -> mx.array:
    """Generate a 1D trapezoidal blending mask with linear ramps.

    Args:
        length: Output length of the mask.
        ramp_left: Fade-in length on the left.
        ramp_right: Fade-out length on the right.
        left_starts_from_0: Whether the ramp starts from 0 or first non-zero value.
            Useful for temporal tiles where the first tile is causal.

    Returns:
        A 1D array of shape (length,) with values in [0, 1].
    """
    if length <= 0:
        raise ValueError("Mask length must be positive.")

    ramp_left = max(0, min(ramp_left, length))
    ramp_right = max(0, min(ramp_right, length))

    # Start with ones
    mask = [1.0] * length

    # Apply left ramp (fade in)
    if ramp_left > 0:
        interval_length = ramp_left + 1 if left_starts_from_0 else ramp_left + 2
        # Create fade_in values using linspace logic
        fade_in_full = [i / (interval_length - 1) for i in range(interval_length)]
        fade_in = fade_in_full[:-1]  # Remove last element
        if not left_starts_from_0:
            fade_in = fade_in[1:]  # Remove first element too
        for i in range(min(ramp_left, len(fade_in))):
            mask[i] *= fade_in[i]

    # Apply right ramp (fade out)
    if ramp_right > 0:
        # Create fade_out: linspace(1, 0, ramp_right + 2)[1:-1]
        fade_out = [
            (ramp_right + 1 - i) / (ramp_right + 1) for i in range(1, ramp_right + 1)
        ]
        for i in range(ramp_right):
            mask[length - ramp_right + i] *= fade_out[i]

    return mx.clip(mx.array(mask), 0, 1)


@dataclass(frozen=True)
class SpatialTilingConfig:
    """Configuration for dividing each frame into spatial tiles with optional overlap."""

    tile_size_in_pixels: int
    tile_overlap_in_pixels: int = 0

    def __post_init__(self) -> None:
        if self.tile_size_in_pixels < 64:
            raise ValueError(
                f"tile_size_in_pixels must be at least 64, got {self.tile_size_in_pixels}"
            )
        if self.tile_size_in_pixels % 32 != 0:
            raise ValueError(
                f"tile_size_in_pixels must be divisible by 32, got {self.tile_size_in_pixels}"
            )
        if self.tile_overlap_in_pixels % 32 != 0:
            raise ValueError(
                f"tile_overlap_in_pixels must be divisible by 32, got {self.tile_overlap_in_pixels}"
            )
        if self.tile_overlap_in_pixels >= self.tile_size_in_pixels:
            raise ValueError(
                f"Overlap must be less than tile size, got {self.tile_overlap_in_pixels} and {self.tile_size_in_pixels}"
            )


@dataclass(frozen=True)
class TemporalTilingConfig:
    """Configuration for dividing a video into temporal tiles."""

    tile_size_in_frames: int
    tile_overlap_in_frames: int = 0

    def __post_init__(self) -> None:
        if self.tile_size_in_frames < 16:
            raise ValueError(
                f"tile_size_in_frames must be at least 16, got {self.tile_size_in_frames}"
            )
        if self.tile_size_in_frames % 8 != 0:
            raise ValueError(
                f"tile_size_in_frames must be divisible by 8, got {self.tile_size_in_frames}"
            )
        if self.tile_overlap_in_frames % 8 != 0:
            raise ValueError(
                f"tile_overlap_in_frames must be divisible by 8, got {self.tile_overlap_in_frames}"
            )
        if self.tile_overlap_in_frames >= self.tile_size_in_frames:
            raise ValueError(
                f"Overlap must be less than tile size, got {self.tile_overlap_in_frames} and {self.tile_size_in_frames}"
            )


@dataclass(frozen=True)
class TilingConfig:
    """Configuration for splitting video into tiles with optional overlap."""

    spatial_config: Optional[SpatialTilingConfig] = None
    temporal_config: Optional[TemporalTilingConfig] = None

    @classmethod
    def default(cls) -> "TilingConfig":
        """Default tiling: 512px spatial, 64 frame temporal."""
        return cls(
            spatial_config=SpatialTilingConfig(
                tile_size_in_pixels=512, tile_overlap_in_pixels=64
            ),
            temporal_config=TemporalTilingConfig(
                tile_size_in_frames=64, tile_overlap_in_frames=24
            ),
        )

    @classmethod
    def spatial_only(cls, tile_size: int = 512, overlap: int = 64) -> "TilingConfig":
        """Spatial tiling only (for short videos with large resolution)."""
        return cls(
            spatial_config=SpatialTilingConfig(
                tile_size_in_pixels=tile_size, tile_overlap_in_pixels=overlap
            ),
            temporal_config=None,
        )

    @classmethod
    def temporal_only(cls, tile_size: int = 64, overlap: int = 24) -> "TilingConfig":
        """Temporal tiling only (for long videos with small resolution)."""
        return cls(
            spatial_config=None,
            temporal_config=TemporalTilingConfig(
                tile_size_in_frames=tile_size, tile_overlap_in_frames=overlap
            ),
        )

    @classmethod
    def aggressive(cls) -> "TilingConfig":
        """Aggressive tiling for very large videos (smaller tiles, much lower memory)."""
        return cls(
            spatial_config=SpatialTilingConfig(
                tile_size_in_pixels=256, tile_overlap_in_pixels=64
            ),
            temporal_config=TemporalTilingConfig(
                tile_size_in_frames=32, tile_overlap_in_frames=8
            ),
        )

    @classmethod
    def conservative(cls) -> "TilingConfig":
        """Conservative tiling (larger tiles, less memory savings but faster)."""
        return cls(
            spatial_config=SpatialTilingConfig(
                tile_size_in_pixels=768, tile_overlap_in_pixels=64
            ),
            temporal_config=TemporalTilingConfig(
                tile_size_in_frames=96, tile_overlap_in_frames=24
            ),
        )

    @classmethod
    def auto(
        cls,
        height: int,
        width: int,
        num_frames: int,
        spatial_threshold: int = 512,
        temporal_threshold: int = 65,
    ) -> Optional["TilingConfig"]:
        """Automatically determine tiling config based on video dimensions.

        Uses PyTorch's default tiling (512px spatial, 64f temporal) which provides
        enough context for CausalConv3d and sufficient overlap for clean blending.

        Args:
            height: Video height in pixels
            width: Video width in pixels
            num_frames: Number of video frames
            spatial_threshold: Enable spatial tiling if either dimension exceeds this
            temporal_threshold: Enable temporal tiling if frames exceed this

        Returns:
            TilingConfig if tiling is needed, None otherwise
        """
        needs_spatial = height > spatial_threshold or width > spatial_threshold
        needs_temporal = num_frames > temporal_threshold

        if not needs_spatial and not needs_temporal:
            return None

        # Use the same defaults as PyTorch (512px spatial, 64f temporal).
        # Smaller tiles cause quality degradation because CausalConv3d needs
        # sufficient temporal context and overlap for clean blending.
        spatial_config = None
        temporal_config = None

        if needs_spatial:
            spatial_config = SpatialTilingConfig(
                tile_size_in_pixels=512, tile_overlap_in_pixels=64
            )

        if needs_temporal:
            # 32 frames (not 64): with 512px spatial tiles, 64-frame temporal
            # tiles push the Wan VAE Resample Conv2d input past ~2e9 elements,
            # where MLX's Metal conv2d silently returns all zeros — tiles decode
            # as flat gray with no error. 32-frame tiles keep every conv input
            # under the threshold. See ml-explore/mlx#3979.
            temporal_config = TemporalTilingConfig(
                tile_size_in_frames=32, tile_overlap_in_frames=24
            )

        return cls(spatial_config=spatial_config, temporal_config=temporal_config)


@dataclass
class DimensionIntervals:
    """Intervals for splitting a single dimension."""

    starts: List[int]
    ends: List[int]
    left_ramps: List[int]
    right_ramps: List[int]


def split_in_spatial(
    size: int, overlap: int, dimension_size: int
) -> DimensionIntervals:
    """Split a spatial dimension into intervals."""
    if dimension_size <= size:
        return DimensionIntervals(
            starts=[0], ends=[dimension_size], left_ramps=[0], right_ramps=[0]
        )

    amount = (dimension_size + size - 2 * overlap - 1) // (size - overlap)
    starts = [i * (size - overlap) for i in range(amount)]
    ends = [start + size for start in starts]
    ends[-1] = dimension_size
    left_ramps = [0] + [overlap] * (amount - 1)
    right_ramps = [overlap] * (amount - 1) + [0]

    return DimensionIntervals(
        starts=starts, ends=ends, left_ramps=left_ramps, right_ramps=right_ramps
    )


def split_in_temporal(
    size: int, overlap: int, dimension_size: int
) -> DimensionIntervals:
    """Split a temporal dimension into intervals with causal adjustment."""
    if dimension_size <= size:
        return DimensionIntervals(
            starts=[0], ends=[dimension_size], left_ramps=[0], right_ramps=[0]
        )

    # Start with spatial split
    intervals = split_in_spatial(size, overlap, dimension_size)

    # Adjust for temporal: starts[1:] -= 1, left_ramps[1:] += 1
    starts = intervals.starts.copy()
    left_ramps = intervals.left_ramps.copy()

    for i in range(1, len(starts)):
        starts[i] = starts[i] - 1
        left_ramps[i] = left_ramps[i] + 1

    return DimensionIntervals(
        starts=starts,
        ends=intervals.ends,
        left_ramps=left_ramps,
        right_ramps=intervals.right_ramps,
    )


def map_temporal_slice(
    begin: int, end: int, left_ramp: int, right_ramp: int, scale: int
) -> Tuple[slice, mx.array]:
    """Map temporal latent interval to output coordinates and mask."""
    start = begin * scale
    stop = 1 + (end - 1) * scale
    left_ramp_scaled = 1 + (left_ramp - 1) * scale if left_ramp > 0 else 0
    right_ramp_scaled = right_ramp * scale

    mask = compute_trapezoidal_mask_1d(
        stop - start, left_ramp_scaled, right_ramp_scaled, True
    )
    return slice(start, stop), mask


def map_spatial_slice(
    begin: int, end: int, left_ramp: int, right_ramp: int, scale: int
) -> Tuple[slice, mx.array]:
    """Map spatial latent interval to output coordinates and mask."""
    start = begin * scale
    stop = end * scale
    left_ramp_scaled = left_ramp * scale
    right_ramp_scaled = right_ramp * scale

    mask = compute_trapezoidal_mask_1d(
        stop - start, left_ramp_scaled, right_ramp_scaled, False
    )
    return slice(start, stop), mask


def decode_with_tiling(
    decoder_fn,
    latents: mx.array,
    tiling_config: TilingConfig,
    spatial_scale: int = 32,
    temporal_scale: int = 8,
    causal: bool = False,
    timestep: Optional[mx.array] = None,
    chunked_conv: bool = False,
    on_frames_ready: Optional[Callable[[mx.array, int], None]] = None,
) -> mx.array:
    """Decode latents using tiling to reduce memory usage.

    Args:
        decoder_fn: Decoder function to call for each tile.
        latents: Input latents of shape (B, C, F, H, W).
        tiling_config: Tiling configuration.
        spatial_scale: Spatial scale factor (32 for LTX VAE: 8x upsample + 4x unpatchify).
        temporal_scale: Temporal scale factor (8 for LTX VAE).
        causal: Whether to use causal convolutions.
        timestep: Optional timestep for conditioning.
        chunked_conv: Whether to use chunked conv mode for upsampling (reduces memory).
        on_frames_ready: Optional callback called with (frames, start_idx) when frames are finalized.
            frames: Tensor of shape (B, 3, num_frames, H, W) with finalized RGB frames.
            start_idx: Starting frame index in the full video.

    Returns:
        Decoded video.
    """
    import gc

    b, c, f_latent, h_latent, w_latent = latents.shape

    # Compute output shape
    out_f = 1 + (f_latent - 1) * temporal_scale
    out_h = h_latent * spatial_scale
    out_w = w_latent * spatial_scale

    # Get tile size and overlap in latent space
    if tiling_config.spatial_config is not None:
        s_cfg = tiling_config.spatial_config
        spatial_tile_size = s_cfg.tile_size_in_pixels // spatial_scale
        spatial_overlap = s_cfg.tile_overlap_in_pixels // spatial_scale
    else:
        spatial_tile_size = max(h_latent, w_latent)
        spatial_overlap = 0

    if tiling_config.temporal_config is not None:
        t_cfg = tiling_config.temporal_config
        temporal_tile_size = t_cfg.tile_size_in_frames // temporal_scale
        temporal_overlap = t_cfg.tile_overlap_in_frames // temporal_scale
    else:
        temporal_tile_size = f_latent
        temporal_overlap = 0

    # Compute intervals for each dimension
    temporal_intervals = split_in_temporal(
        temporal_tile_size, temporal_overlap, f_latent
    )
    height_intervals = split_in_spatial(spatial_tile_size, spatial_overlap, h_latent)
    width_intervals = split_in_spatial(spatial_tile_size, spatial_overlap, w_latent)

    num_t_tiles = len(temporal_intervals.starts)
    num_h_tiles = len(height_intervals.starts)
    num_w_tiles = len(width_intervals.starts)
    total_tiles = num_t_tiles * num_h_tiles * num_w_tiles

    # Initialize output and weight accumulator
    # Use float32 for accumulation to avoid precision issues
    output = mx.zeros((b, 3, out_f, out_h, out_w), dtype=mx.float32)
    weights = mx.zeros((b, 1, out_f, out_h, out_w), dtype=mx.float32)
    mx.eval(output, weights)

    tile_idx = 0
    for t_idx in range(num_t_tiles):
        t_start = temporal_intervals.starts[t_idx]
        t_end = temporal_intervals.ends[t_idx]
        t_left = temporal_intervals.left_ramps[t_idx]
        t_right = temporal_intervals.right_ramps[t_idx]

        # Map temporal coordinates
        out_t_slice, t_mask = map_temporal_slice(
            t_start, t_end, t_left, t_right, temporal_scale
        )

        for h_idx in range(num_h_tiles):
            h_start = height_intervals.starts[h_idx]
            h_end = height_intervals.ends[h_idx]
            h_left = height_intervals.left_ramps[h_idx]
            h_right = height_intervals.right_ramps[h_idx]

            # Map height coordinates
            out_h_slice, h_mask = map_spatial_slice(
                h_start, h_end, h_left, h_right, spatial_scale
            )

            for w_idx in range(num_w_tiles):
                w_start = width_intervals.starts[w_idx]
                w_end = width_intervals.ends[w_idx]
                w_left = width_intervals.left_ramps[w_idx]
                w_right = width_intervals.right_ramps[w_idx]

                # Map width coordinates
                out_w_slice, w_mask = map_spatial_slice(
                    w_start, w_end, w_left, w_right, spatial_scale
                )

                # Extract tile latents (small slice)
                tile_latents = latents[
                    :, :, t_start:t_end, h_start:h_end, w_start:w_end
                ]

                # Decode tile
                tile_output = decoder_fn(
                    tile_latents,
                    causal=causal,
                    timestep=timestep,
                    debug=False,
                    chunked_conv=chunked_conv,
                )
                mx.eval(tile_output)

                # Clear tile_latents reference
                del tile_latents

                # Get actual decoded dimensions
                _, _, decoded_t, decoded_h, decoded_w = tile_output.shape
                expected_t = out_t_slice.stop - out_t_slice.start
                expected_h = out_h_slice.stop - out_h_slice.start
                expected_w = out_w_slice.stop - out_w_slice.start

                # Handle potential size mismatches (use minimum)
                actual_t = min(decoded_t, expected_t)
                actual_h = min(decoded_h, expected_h)
                actual_w = min(decoded_w, expected_w)

                # Build blend mask
                t_mask_slice = t_mask[:actual_t] if len(t_mask) > actual_t else t_mask
                h_mask_slice = h_mask[:actual_h] if len(h_mask) > actual_h else h_mask
                w_mask_slice = w_mask[:actual_w] if len(w_mask) > actual_w else w_mask

                blend_mask = (
                    t_mask_slice.reshape(1, 1, -1, 1, 1)
                    * h_mask_slice.reshape(1, 1, 1, -1, 1)
                    * w_mask_slice.reshape(1, 1, 1, 1, -1)
                )

                # Slice tile output to match
                tile_output_slice = tile_output[
                    :, :, :actual_t, :actual_h, :actual_w
                ].astype(mx.float32)

                # Clear full tile_output
                del tile_output

                # Compute output coordinates
                t_out_start = out_t_slice.start
                t_out_end = t_out_start + actual_t
                h_out_start = out_h_slice.start
                h_out_end = h_out_start + actual_h
                w_out_start = out_w_slice.start
                w_out_end = w_out_start + actual_w

                # Use direct slice assignment (MLX supports this)
                # Weighted accumulation
                weighted_tile = tile_output_slice * blend_mask

                # Update output using slice assignment
                output[
                    :,
                    :,
                    t_out_start:t_out_end,
                    h_out_start:h_out_end,
                    w_out_start:w_out_end,
                ] = (
                    output[
                        :,
                        :,
                        t_out_start:t_out_end,
                        h_out_start:h_out_end,
                        w_out_start:w_out_end,
                    ]
                    + weighted_tile
                )
                weights[
                    :,
                    :,
                    t_out_start:t_out_end,
                    h_out_start:h_out_end,
                    w_out_start:w_out_end,
                ] = (
                    weights[
                        :,
                        :,
                        t_out_start:t_out_end,
                        h_out_start:h_out_end,
                        w_out_start:w_out_end,
                    ]
                    + blend_mask
                )

                # Force evaluation to free memory
                mx.eval(output, weights)

                # Clean up tile-specific arrays
                del tile_output_slice, weighted_tile, blend_mask
                del t_mask_slice, h_mask_slice, w_mask_slice

                tile_idx += 1

                # Periodic garbage collection and cache clearing
                if tile_idx % 4 == 0:
                    gc.collect()
                    try:
                        mx.clear_cache()
                    except Exception:
                        pass  # May not be available on all platforms

        # After completing all spatial tiles for this temporal tile,
        # check if any frames are now finalized (no future tiles will contribute)
        if on_frames_ready is not None and num_t_tiles > 1:
            # Determine the finalized frame boundary
            # Frames before the start of the next tile's output region are finalized
            if t_idx < num_t_tiles - 1:
                # Next tile starts at temporal_intervals.starts[t_idx + 1]
                next_tile_start_latent = temporal_intervals.starts[t_idx + 1]
                # Map to output frame index (first frame of next tile's contribution)
                if next_tile_start_latent == 0:
                    next_tile_start_out = 0
                else:
                    next_tile_start_out = (
                        1 + (next_tile_start_latent - 1) * temporal_scale
                    )

                # We need to track how many frames we've already emitted
                if not hasattr(decode_with_tiling, "_emitted_frames"):
                    decode_with_tiling._emitted_frames = 0
                emitted = decode_with_tiling._emitted_frames

                if next_tile_start_out > emitted:
                    # Normalize and emit frames [emitted, next_tile_start_out)
                    finalized_weights = weights[:, :, emitted:next_tile_start_out, :, :]
                    finalized_weights = mx.maximum(finalized_weights, 1e-8)
                    finalized_output = (
                        output[:, :, emitted:next_tile_start_out, :, :]
                        / finalized_weights
                    )
                    finalized_output = finalized_output.astype(latents.dtype)
                    mx.eval(finalized_output)

                    on_frames_ready(finalized_output, emitted)
                    decode_with_tiling._emitted_frames = next_tile_start_out

                    del finalized_output, finalized_weights
                    gc.collect()

    # Normalize by weights
    weights = mx.maximum(weights, 1e-8)
    output = output / weights
    mx.eval(output)

    # Emit remaining frames if callback provided
    if on_frames_ready is not None:
        emitted = getattr(decode_with_tiling, "_emitted_frames", 0)
        if emitted < out_f:
            remaining_output = output[:, :, emitted:, :, :].astype(latents.dtype)
            mx.eval(remaining_output)
            on_frames_ready(remaining_output, emitted)
            del remaining_output

    # Reset emitted frames counter for next call
    if hasattr(decode_with_tiling, "_emitted_frames"):
        del decode_with_tiling._emitted_frames

    # Clean up weights
    del weights
    gc.collect()

    # Convert back to original dtype if needed
    return output.astype(latents.dtype)
