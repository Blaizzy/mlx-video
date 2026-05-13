"""Sampling operations for Video VAE (upsampling/downsampling)."""

from typing import Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from mlx_video.models.ltx_2.video_vae.convolution import CausalConv3d, PaddingModeType


class SpaceToDepthDownsample(nn.Module):
    """Space-to-depth downsampling with 3x3 conv and skip connection.

    PyTorch-compatible implementation:
    1. Apply 3x3 conv: in_channels -> out_channels // prod(stride)
    2. Space-to-depth on conv output: channels * prod(stride)
    3. Space-to-depth on input with group averaging for skip connection
    4. Add skip connection
    """

    def __init__(
        self,
        dims: int,
        in_channels: int,
        out_channels: int,
        stride: Union[int, Tuple[int, int, int]],
        spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
    ):
        super().__init__()

        if isinstance(stride, int):
            stride = (stride, stride, stride)

        self.stride = stride
        self.dims = dims
        self.out_channels = out_channels

        # Calculate channels
        multiplier = stride[0] * stride[1] * stride[2]
        self.group_size = in_channels * multiplier // out_channels
        conv_out_channels = out_channels // multiplier

        # 3x3 convolution (not 1x1)
        self.conv = CausalConv3d(
            in_channels=in_channels,
            out_channels=conv_out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            spatial_padding_mode=spatial_padding_mode,
        )

    def _space_to_depth(self, x: mx.array) -> mx.array:
        """Rearrange: b c (d p1) (h p2) (w p3) -> b (c p1 p2 p3) d h w"""
        b, c, d, h, w = x.shape
        st, sh, sw = self.stride

        # Reshape to group spatial elements
        x = mx.reshape(x, (b, c, d // st, st, h // sh, sh, w // sw, sw))

        # Permute: (B, C, D', st, H', sh, W', sw) -> (B, C, st, sh, sw, D', H', W')
        x = mx.transpose(x, (0, 1, 3, 5, 7, 2, 4, 6))

        # Reshape to combine channels
        new_c = c * st * sh * sw
        new_d = d // st
        new_h = h // sh
        new_w = w // sw
        x = mx.reshape(x, (b, new_c, new_d, new_h, new_w))

        return x

    def __call__(self, x: mx.array, causal: bool = True) -> mx.array:
        b, c, d, h, w = x.shape
        st, sh, sw = self.stride

        # Temporal padding for causal mode
        if st == 2:
            # Duplicate first frame for padding
            x = mx.concatenate([x[:, :, :1, :, :], x], axis=2)
            d = d + 1

        # Pad if necessary to make dimensions divisible by stride
        pad_d = (st - d % st) % st
        pad_h = (sh - h % sh) % sh
        pad_w = (sw - w % sw) % sw

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x = mx.pad(x, [(0, 0), (0, 0), (0, pad_d), (0, pad_h), (0, pad_w)])

        # Skip connection: space-to-depth on input, then group mean
        x_in = self._space_to_depth(x)
        # Derive actual out_channels from the conv weight to handle config/weight mismatches
        # (e.g. LTX-2.3 weights have different channel counts than the config implies)
        st, sh, sw = self.stride
        actual_conv_out = self.conv.conv.weight.shape[0]
        actual_out_channels = actual_conv_out * st * sh * sw
        b2, c2, d2, h2, w2 = x_in.shape
        actual_group_size = c2 // actual_out_channels
        x_in = mx.reshape(x_in, (b2, actual_out_channels, actual_group_size, d2, h2, w2))
        x_in = mx.mean(x_in, axis=2)  # (b, actual_out_channels, d, h, w)

        # Conv branch: apply conv then space-to-depth
        x_conv = self.conv(x, causal=causal)
        x_conv = self._space_to_depth(x_conv)

        # Add skip connection
        return x_conv + x_in


class DepthToSpaceUpsample(nn.Module):

    def __init__(
        self,
        dims: int,
        in_channels: int,
        stride: Union[int, Tuple[int, int, int]],
        residual: bool = False,
        out_channels_reduction_factor: int = 1,
        spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
    ):

        super().__init__()

        if isinstance(stride, int):
            stride = (stride, stride, stride)

        self.stride = stride
        self.dims = dims
        self.residual = residual
        self.out_channels_reduction_factor = out_channels_reduction_factor

        # Calculate output channels
        multiplier = stride[0] * stride[1] * stride[2]
        out_channels = in_channels // out_channels_reduction_factor
        self.out_channels = out_channels

        # 3x3x3 convolution to prepare channels for unpacking (matches PyTorch)
        self.conv = CausalConv3d(
            in_channels=in_channels,
            out_channels=out_channels * multiplier,
            kernel_size=3,
            stride=1,
            padding=1,
            spatial_padding_mode=spatial_padding_mode,
        )

    def _depth_to_space(self, x: mx.array) -> mx.array:
        b, c_packed, d, h, w = x.shape
        st, sh, sw = self.stride
        c = c_packed // (st * sh * sw)

        # (B, C*st*sh*sw, D, H, W) -> (B, C, st, sh, sw, D, H, W)
        x = mx.reshape(x, (b, c, st, sh, sw, d, h, w))

        # (B, C, st, sh, sw, D, H, W) -> (B, C, D, st, H, sh, W, sw)
        x = mx.transpose(x, (0, 1, 5, 2, 6, 3, 7, 4))

        # (B, C, D, st, H, sh, W, sw) -> (B, C, D*st, H*sh, W*sw)
        x = mx.reshape(x, (b, c, d * st, h * sh, w * sw))

        return x

    def __call__(
        self, x: mx.array, causal: bool = True, chunked_conv: bool = False
    ) -> mx.array:

        b, c, d, h, w = x.shape
        st, sh, sw = self.stride

        # Compute residual path if enabled
        x_residual = None
        if self.residual:
            # Reshape input: treat channels as spatial factors
            # "b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)"
            x_residual = self._depth_to_space(x)

            # Tile channels to match output (PyTorch .repeat() tiles, not element-repeat!)
            # num_repeat = prod(stride) / out_channels_reduction_factor
            num_repeat = (st * sh * sw) // self.out_channels_reduction_factor
            x_residual = mx.tile(x_residual, (1, num_repeat, 1, 1, 1))

            # Remove first temporal frame if temporal upsampling
            if st > 1:
                x_residual = x_residual[:, :, 1:, :, :]

        # Use chunked mode for large tensors to reduce peak memory
        if chunked_conv and d > 4:
            x = self._chunked_conv_depth_to_space(x, causal)
        else:
            # Apply conv
            x = self.conv(x, causal=causal)
            # Depth to space rearrangement
            x = self._depth_to_space(x)

        # Remove first frame for causal temporal upsampling
        if st > 1:
            x = x[:, :, 1:, :, :]

        # Add residual
        if self.residual and x_residual is not None:
            x = x + x_residual

        return x

    def _chunked_conv_depth_to_space(
        self, x: mx.array, causal: bool = True
    ) -> mx.array:
        """Chunked conv + depth_to_space that processes in temporal chunks.

        This reduces peak memory by avoiding the full high-channel intermediate tensor.
        Instead of materializing (B, 4096, D, H, W), we process temporal chunks and
        immediately apply depth_to_space.

        Args:
            x: Input tensor of shape (B, C, D, H, W)
            causal: Whether to use causal convolutions

        Returns:
            Output tensor after conv + depth_to_space
        """
        b, c, d, h, w = x.shape
        st, sh, sw = self.stride
        out_c = self.out_channels

        # Output dimensions
        out_d = d * st
        out_h = h * sh
        out_w = w * sw

        # Chunk size in temporal dimension (process 4 frames at a time)
        chunk_size = 4
        kernel_t = 3  # Temporal kernel size

        # For causal conv, we need (kernel_t - 1) frames of padding at the start
        # For non-causal, we need (kernel_t - 1) // 2 on each side
        if causal:
            # Pad start with first frame repeated
            pad_start = kernel_t - 1
            pad_end = 0
        else:
            pad_start = (kernel_t - 1) // 2
            pad_end = (kernel_t - 1) // 2

        # Allocate output
        outputs = []

        # Process in chunks with overlap for conv kernel
        t_pos = 0
        while t_pos < d:
            t_end = min(t_pos + chunk_size, d)

            # Calculate input range with padding for kernel
            in_start = max(0, t_pos - pad_start)
            in_end = min(d, t_end + pad_end)

            # Extract chunk
            chunk = x[:, :, in_start:in_end, :, :]

            # Apply conv to chunk
            chunk_conv = self.conv(chunk, causal=causal)

            # Apply depth_to_space
            chunk_out = self._depth_to_space(chunk_conv)

            # Calculate valid output range (excluding padding effects)
            # Each input frame produces st output frames
            out_start = (t_pos - in_start) * st
            out_end = out_start + (t_end - t_pos) * st

            # Extract valid portion
            chunk_out = chunk_out[:, :, out_start:out_end, :, :]

            outputs.append(chunk_out)

            # Evaluate to free intermediate memory
            mx.eval(outputs[-1])

            t_pos = t_end

        # Concatenate all chunks
        if len(outputs) == 1:
            return outputs[0]
        return mx.concatenate(outputs, axis=2)
