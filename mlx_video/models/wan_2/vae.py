"""3D VAE Decoder for Wan2.1/2.2 (compression 4×8×8).

Module structure mirrors original PyTorch checkpoint key hierarchy
so weights load directly without key sanitization.
"""

import mlx.core as mx
import mlx.nn as nn

CACHE_T = 2

# Per-channel normalization statistics for z_dim=16
VAE_MEAN = [
    -0.7571,
    -0.7089,
    -0.9113,
    0.1075,
    -0.1745,
    0.9653,
    -0.1517,
    1.5508,
    0.4134,
    -0.0715,
    0.5517,
    -0.3632,
    -0.1922,
    -0.9497,
    0.2503,
    -0.2921,
]
VAE_STD = [
    2.8184,
    1.4541,
    2.3275,
    2.6558,
    1.2196,
    1.7708,
    2.6052,
    2.0743,
    3.2687,
    2.1526,
    2.8652,
    1.5579,
    1.6382,
    1.1253,
    2.8251,
    1.9160,
]


class CausalConv3d(nn.Module):
    """3D convolution with causal temporal padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple,
        stride: int | tuple = 1,
        padding: int | tuple = 0,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)

        self.kernel_size = kernel_size
        self.stride = stride
        # Causal padding: match reference formula dilation*(k-1) + (1-stride)
        # With dilation=1: k-stride (pads left only, no future context)
        self._causal_pad_t = kernel_size[0] - stride[0]
        self._pad_h = padding[1]
        self._pad_w = padding[2]

        # MLX Conv3d: weight shape [O, D, H, W, I]
        self.weight = mx.zeros(
            (out_channels, kernel_size[0], kernel_size[1], kernel_size[2], in_channels)
        )
        self.bias = mx.zeros((out_channels,))

    def __call__(self, x: mx.array, cache_x: mx.array = None) -> mx.array:
        """x: [B, C, T, H, W] (channel-first)"""
        b, c, t, h, w = x.shape

        causal_pad = self._causal_pad_t
        if cache_x is not None and causal_pad > 0:
            x = mx.concatenate([cache_x, x], axis=2)
            causal_pad = max(0, causal_pad - cache_x.shape[2])

        if causal_pad > 0:
            pad_t = mx.zeros((b, c, causal_pad, h, w), dtype=x.dtype)
            x = mx.concatenate([pad_t, x], axis=2)

        if self._pad_h > 0 or self._pad_w > 0:
            x = mx.pad(
                x,
                [
                    (0, 0),
                    (0, 0),
                    (0, 0),
                    (self._pad_h, self._pad_h),
                    (self._pad_w, self._pad_w),
                ],
            )

        x = x.transpose(0, 2, 3, 4, 1)  # [B, T, H, W, C]
        out = self._conv3d(x)
        return out.transpose(0, 4, 1, 2, 3)  # [B, O, T', H', W']

    def _conv3d(self, x: mx.array) -> mx.array:
        """3D conv via sliding window + 2D conv per time step.
        x: [B, T, H, W, C_in] -> [B, T_out, H_out, W_out, C_out]
        """
        b, t, h, w, c_in = x.shape
        kt, kh, kw = self.kernel_size
        st, sh, sw = self.stride
        t_out = (t - kt) // st + 1

        # Pre-reshape weight: [O, D, H, W, I] -> [O, H, W, D*I]
        w_2d = self.weight.transpose(0, 2, 3, 1, 4).reshape(
            self.weight.shape[0], kh, kw, kt * c_in
        )
        outputs = []
        for t_i in range(t_out):
            t_start = t_i * st
            window = x[:, t_start : t_start + kt]
            window = window.transpose(0, 2, 3, 1, 4).reshape(b, h, w, kt * c_in)
            out_2d = mx.conv2d(window, w_2d, stride=(sh, sw)) + self.bias
            outputs.append(out_2d)
        return mx.stack(outputs, axis=1)


class RMS_norm(nn.Module):
    """Channel-first L2 normalization matching original Wan VAE.

    Uses F.normalize (L2 norm) with learned scale, equivalent to RMS norm.
    images=True: gamma shape (dim, 1, 1) for 4D (per-frame) input.
    images=False: gamma shape (dim, 1, 1, 1) for 5D video input.
    """

    def __init__(self, dim: int, channel_first: bool = True, images: bool = True):
        super().__init__()
        self.channel_first = channel_first
        self.scale = dim**0.5
        if channel_first:
            broadcastable = (1, 1) if images else (1, 1, 1)
            self.gamma = mx.ones((dim, *broadcastable))
        else:
            self.gamma = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        norm_dim = 1 if self.channel_first else -1
        # L2 normalize along channel dim (matches F.normalize)
        norm = mx.sqrt(
            mx.clip(
                mx.sum(x * x, axis=norm_dim, keepdims=True), a_min=1e-12, a_max=None
            )
        )
        return (x / norm) * self.scale * self.gamma


class ResidualBlock(nn.Module):
    """Residual block with causal 3D convolutions.

    Uses `residual` list with None gaps to match original PyTorch
    nn.Sequential indices: [0]=norm, [1]=SiLU, [2]=conv, [3]=norm,
    [4]=SiLU, [5]=Dropout, [6]=conv. Only indices 0,2,3,6 have params.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.residual = [
            RMS_norm(in_dim, images=False),  # [0]
            None,  # [1] SiLU
            CausalConv3d(in_dim, out_dim, 3, padding=1),  # [2]
            RMS_norm(out_dim, images=False),  # [3]
            None,  # [4] SiLU
            None,  # [5] Dropout
            CausalConv3d(out_dim, out_dim, 3, padding=1),  # [6]
        ]
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else None

    def __call__(self, x: mx.array, feat_cache=None, feat_idx=None) -> mx.array:
        h = x if self.shortcut is None else self.shortcut(x)

        if feat_cache is not None:
            # First conv: norm -> silu -> [cache] -> conv
            x = nn.silu(self.residual[0](x))
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:]
            if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
                cache_x = mx.concatenate([feat_cache[idx][:, :, -1:], cache_x], axis=2)
            x = self.residual[2](x, cache_x=feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1

            # Second conv: norm -> silu -> [cache] -> conv
            x = nn.silu(self.residual[3](x))
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:]
            if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
                cache_x = mx.concatenate([feat_cache[idx][:, :, -1:], cache_x], axis=2)
            x = self.residual[6](x, cache_x=feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = nn.silu(self.residual[0](x))
            x = self.residual[2](x)
            x = nn.silu(self.residual[3](x))
            x = self.residual[6](x)

        return x + h


class AttentionBlock(nn.Module):
    """Single-head spatial self-attention."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = RMS_norm(dim, images=True)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def __call__(self, x: mx.array) -> mx.array:
        """x: [B, C, T, H, W]"""
        identity = x
        b, c, t, h, w = x.shape

        # [B,C,T,H,W] -> [B,T,C,H,W] -> [BT,C,H,W] -> norm -> [BT,H,W,C]
        x = x.transpose(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.norm(x)
        x = x.transpose(0, 2, 3, 1)  # [BT, H, W, C]

        qkv = self.to_qkv(x)  # [BT, H, W, 3C]
        qkv = qkv.reshape(b * t, h * w, 3, c).transpose(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q[:, None, :, :]  # [BT, 1, HW, C]
        k = k[:, None, :, :]
        v = v[:, None, :, :]
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=c**-0.5)
        out = out.squeeze(1).reshape(b * t, h, w, c)  # [BT, H, W, C]

        out = self.proj(out)  # [BT, H, W, C]
        out = out.reshape(b, t, h, w, c).transpose(0, 4, 1, 2, 3)  # [B, C, T, H, W]
        return out + identity


class Resample(nn.Module):
    """Resample block matching original Wan VAE structure.

    Supports both upsampling (decoder) and downsampling (encoder).
    Uses list-based param storage to match original nn.Sequential key hierarchy.
    """

    def __init__(self, dim: int, mode: str):
        super().__init__()
        assert mode in ("upsample2d", "upsample3d", "downsample2d", "downsample3d")
        self.mode = mode
        self.dim = dim

        if mode.startswith("upsample"):
            # resample.0 = Upsample (no params), resample.1 = Conv2d
            self.resample = [None, nn.Conv2d(dim, dim // 2, 3, padding=1)]
            if mode == "upsample3d":
                self.time_conv = CausalConv3d(
                    dim, dim * 2, (3, 1, 1), padding=(1, 0, 0)
                )
        else:
            # resample.0 = ZeroPad2d (no params), resample.1 = Conv2d(stride=2)
            self.resample = [None, nn.Conv2d(dim, dim, 3, stride=2)]
            if mode == "downsample3d":
                self.time_conv = CausalConv3d(
                    dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0)
                )

    def __call__(self, x: mx.array, feat_cache=None, feat_idx=None) -> mx.array:
        """x: [B, C, T, H, W]"""
        b, c, t, h, w = x.shape

        if self.mode == "upsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    # First chunk: 'Rep' sentinel — skip time_conv so the
                    # first latent frame is not temporally doubled (reference
                    # Wan2.1 causal mapping: T latents -> 1 + (T-1)*4 frames)
                    feat_cache[idx] = "Rep"
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -CACHE_T:]
                    was_rep = isinstance(feat_cache[idx], str)
                    if cache_x.shape[2] < 2 and not was_rep:
                        cache_x = mx.concatenate(
                            [feat_cache[idx][:, :, -1:], cache_x], axis=2
                        )
                    elif cache_x.shape[2] < 2 and was_rep:
                        cache_x = mx.concatenate(
                            [mx.zeros_like(cache_x), cache_x], axis=2
                        )
                    if was_rep:
                        x_t = self.time_conv(x)
                    else:
                        x_t = self.time_conv(x, cache_x=feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
                    x_t = x_t.reshape(b, 2, c, t, h, w)
                    x = mx.stack([x_t[:, 0], x_t[:, 1]], axis=3).reshape(
                        b, c, t * 2, h, w
                    )
                    t = t * 2
            else:
                # Temporal upsample via learned conv
                x_t = self.time_conv(x)  # [B, 2C, T, H, W]
                x_t = x_t.reshape(b, 2, c, t, h, w)
                x = mx.stack([x_t[:, 0], x_t[:, 1]], axis=3).reshape(b, c, t * 2, h, w)
                t = t * 2

        if self.mode.startswith("upsample"):
            # Per-frame spatial upsample: nearest 2x + Conv2d
            x = x.transpose(0, 2, 3, 4, 1).reshape(b * t, h, w, c)  # [BT, H, W, C]
            x = mx.repeat(x, 2, axis=1)
            x = mx.repeat(x, 2, axis=2)
            x = self.resample[1](x)  # Conv2d [BT, 2H, 2W, C//2]
            c_out = x.shape[-1]
            return x.reshape(b, t, h * 2, w * 2, c_out).transpose(0, 4, 1, 2, 3)
        else:
            # Per-frame spatial downsample: ZeroPad(0,1,0,1) + Conv2d(stride=2)
            x = x.transpose(0, 2, 3, 4, 1).reshape(b * t, h, w, c)  # [BT, H, W, C]
            x = mx.pad(x, [(0, 0), (0, 1), (0, 1), (0, 0)])  # ZeroPad2d(0,1,0,1)
            x = self.resample[1](x)  # Conv2d stride=2
            c_out = x.shape[-1]
            h_out, w_out = x.shape[1], x.shape[2]
            x = x.reshape(b, t, h_out, w_out, c_out).transpose(0, 4, 1, 2, 3)

            if self.mode == "downsample3d":
                if feat_cache is not None:
                    idx = feat_idx[0]
                    if feat_cache[idx] is None:
                        # First chunk: save x, skip time_conv
                        feat_cache[idx] = x
                        feat_idx[0] += 1
                    else:
                        # Subsequent chunks: use cached frame as temporal context
                        cache_x = x[:, :, -1:]
                        x = self.time_conv(x, cache_x=feat_cache[idx][:, :, -1:])
                        feat_cache[idx] = cache_x
                        feat_idx[0] += 1
                else:
                    x = self.time_conv(x)
            return x


class Decoder3d(nn.Module):
    """3D VAE Decoder matching Wan2.1 architecture.

    Uses flat `middle` and `upsamples` lists to match original
    PyTorch nn.Sequential weight key hierarchy.
    """

    def __init__(
        self,
        dim: int = 96,
        z_dim: int = 16,
        dim_mult: list = None,
        num_res_blocks: int = 2,
        temporal_upsample: list = None,
    ):
        super().__init__()
        if dim_mult is None:
            dim_mult = [1, 2, 4, 4]
        if temporal_upsample is None:
            temporal_upsample = [True, True, False]

        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]

        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        # Middle: [ResBlock, AttentionBlock, ResBlock]
        self.middle = [
            ResidualBlock(dims[0], dims[0]),
            AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0]),
        ]

        # Flat upsample list matching original nn.Sequential indexing
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if i in (1, 2, 3):
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim))
                in_dim = out_dim
            if i != len(dim_mult) - 1:
                mode = "upsample3d" if temporal_upsample[i] else "upsample2d"
                upsamples.append(Resample(out_dim, mode=mode))
        self.upsamples = upsamples

        # Output head: [RMS_norm, SiLU (no params), CausalConv3d]
        self.head = [
            RMS_norm(dims[-1], images=False),  # [0]
            None,  # [1] SiLU
            CausalConv3d(dims[-1], 3, 3, padding=1),  # [2]
        ]

    def __call__(self, x: mx.array, feat_cache=None, feat_idx=None) -> mx.array:
        """x: [B, z_dim, T, H, W] -> [B, 3, T_out, H_out, W_out]"""
        if feat_cache is not None:
            # conv1 with caching
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:]
            if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
                cache_x = mx.concatenate([feat_cache[idx][:, :, -1:], cache_x], axis=2)
            x = self.conv1(x, cache_x=feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        for layer in self.middle:
            if feat_cache is not None and isinstance(layer, ResidualBlock):
                x = layer(x, feat_cache=feat_cache, feat_idx=feat_idx)
            else:
                x = layer(x)

        for layer in self.upsamples:
            if feat_cache is not None and isinstance(layer, (ResidualBlock, Resample)):
                x = layer(x, feat_cache=feat_cache, feat_idx=feat_idx)
            else:
                x = layer(x)

        if feat_cache is not None:
            # Head: norm -> silu -> [cache] -> conv
            x = nn.silu(self.head[0](x))
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:]
            if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
                cache_x = mx.concatenate([feat_cache[idx][:, :, -1:], cache_x], axis=2)
            x = self.head[2](x, cache_x=feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = nn.silu(self.head[0](x))
            x = self.head[2](x)
        return x


class Encoder3d(nn.Module):
    """3D VAE Encoder matching Wan2.1 architecture.

    Mirror of Decoder3d with downsampling instead of upsampling.
    Uses flat lists to match original PyTorch nn.Sequential weight key hierarchy.
    """

    def __init__(
        self,
        dim: int = 96,
        z_dim: int = 16,
        dim_mult: list = None,
        num_res_blocks: int = 2,
        temporal_downsample: list = None,
    ):
        super().__init__()
        if dim_mult is None:
            dim_mult = [1, 2, 4, 4]
        if temporal_downsample is None:
            temporal_downsample = [False, True, True]

        dims = [dim * u for u in [1] + dim_mult]

        self.conv1 = CausalConv3d(3, dims[0], 3, padding=1)

        # Flat downsample list matching original nn.Sequential indexing
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim))
                in_dim = out_dim
            if i != len(dim_mult) - 1:
                mode = "downsample3d" if temporal_downsample[i] else "downsample2d"
                downsamples.append(Resample(out_dim, mode=mode))
        self.downsamples = downsamples

        # Middle: [ResBlock, AttentionBlock, ResBlock]
        self.middle = [
            ResidualBlock(dims[-1], dims[-1]),
            AttentionBlock(dims[-1]),
            ResidualBlock(dims[-1], dims[-1]),
        ]

        # Output head: [RMS_norm, SiLU (no params), CausalConv3d]
        self.head = [
            RMS_norm(dims[-1], images=False),
            None,  # SiLU
            CausalConv3d(dims[-1], z_dim, 3, padding=1),
        ]

    def __call__(self, x: mx.array, feat_cache=None, feat_idx=None) -> mx.array:
        """x: [B, 3, T, H, W] -> [B, z_dim, T_lat, H_lat, W_lat]"""
        if feat_cache is not None:
            # conv1 with caching
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:]
            if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
                cache_x = mx.concatenate([feat_cache[idx][:, :, -1:], cache_x], axis=2)
            x = self.conv1(x, cache_x=feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        for layer in self.downsamples:
            if feat_cache is not None and isinstance(layer, (ResidualBlock, Resample)):
                x = layer(x, feat_cache=feat_cache, feat_idx=feat_idx)
            else:
                x = layer(x)

        for layer in self.middle:
            if feat_cache is not None and isinstance(layer, ResidualBlock):
                x = layer(x, feat_cache=feat_cache, feat_idx=feat_idx)
            else:
                x = layer(x)

        if feat_cache is not None:
            # Head: norm -> silu -> [cache] -> conv
            x = nn.silu(self.head[0](x))
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:]
            if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
                cache_x = mx.concatenate([feat_cache[idx][:, :, -1:], cache_x], axis=2)
            x = self.head[2](x, cache_x=feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = nn.silu(self.head[0](x))
            x = self.head[2](x)

        return x


class WanVAE(nn.Module):
    """Wan2.1 VAE wrapper with per-channel normalization.

    Supports both encode (for I2V) and decode (for all models).
    """

    def __init__(self, z_dim: int = 16, encoder: bool = False):
        super().__init__()
        self.z_dim = z_dim
        self.mean = mx.array(VAE_MEAN)
        self.std = mx.array(VAE_STD)
        self.inv_std = 1.0 / self.std

        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(dim=96, z_dim=z_dim)

        if encoder:
            self.encoder = Encoder3d(dim=96, z_dim=z_dim * 2)
            self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)

    def encode(self, x: mx.array) -> mx.array:
        """Encode video to normalized latent using chunked encoding.

        Uses chunked encoding with temporal caching to match reference behavior.
        First frame encoded alone, then 4-frame chunks with cached context.

        Args:
            x: Video [B, 3, T, H, W] in [-1, 1]

        Returns:
            Normalized latent [B, z_dim, T_lat, H_lat, W_lat]
        """
        # Count cacheable CausalConv3d slots in encoder
        num_slots = self._count_encoder_cache_slots()
        feat_cache = [None] * num_slots

        t = x.shape[2]
        num_chunks = 1 + (t - 1) // 4

        out = None
        for i in range(num_chunks):
            feat_idx = [0]
            if i == 0:
                chunk = x[:, :, :1]
            else:
                chunk = x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i]

            chunk_out = self.encoder(chunk, feat_cache=feat_cache, feat_idx=feat_idx)

            if out is None:
                out = chunk_out
            else:
                out = mx.concatenate([out, chunk_out], axis=2)

        mu, _ = mx.split(self.conv1(out), 2, axis=1)

        # Normalize: (mu - mean) * inv_std
        mean = self.mean.reshape(1, -1, 1, 1, 1)
        inv_std = self.inv_std.reshape(1, -1, 1, 1, 1)
        return (mu - mean) * inv_std

    def _count_encoder_cache_slots(self) -> int:
        """Count CausalConv3d that participate in chunked encoding cache."""
        count = 1  # encoder.conv1
        for layer in self.encoder.downsamples:
            if isinstance(layer, ResidualBlock):
                count += 2  # two convs in residual path
            elif isinstance(layer, Resample) and layer.mode == "downsample3d":
                count += 1  # time_conv
        for layer in self.encoder.middle:
            if isinstance(layer, ResidualBlock):
                count += 2
        count += 1  # encoder.head CausalConv3d
        return count

    def decode(self, z: mx.array) -> mx.array:
        """Decode latent to video using chunked decoding.

        Decodes one latent frame at a time with temporal feature caching to
        match reference behavior: T latent frames produce 1 + (T-1)*4 output
        frames (the first latent frame is not temporally upsampled). Memory
        stays flat in T.

        Args:
            z: Normalized latent [B, z_dim, T, H, W]

        Returns:
            Video [B, 3, 1 + (T-1)*4, H_out, W_out] clamped to [-1, 1]
        """
        mean = self.mean.reshape(1, -1, 1, 1, 1)
        inv_std = self.inv_std.reshape(1, -1, 1, 1, 1)
        z = z / inv_std + mean

        x = self.conv2(z)

        num_slots = self._count_decoder_cache_slots()
        feat_cache = [None] * num_slots
        out = None
        for i in range(x.shape[2]):
            feat_idx = [0]
            chunk = self.decoder(x[:, :, i : i + 1], feat_cache=feat_cache, feat_idx=feat_idx)
            out = chunk if out is None else mx.concatenate([out, chunk], axis=2)
        return mx.clip(out, -1, 1)

    def _count_decoder_cache_slots(self) -> int:
        """Count CausalConv3d that participate in chunked decoding cache."""
        count = 1  # decoder.conv1
        for layer in self.decoder.middle:
            if isinstance(layer, ResidualBlock):
                count += 2
        for layer in self.decoder.upsamples:
            if isinstance(layer, ResidualBlock):
                count += 2
            elif isinstance(layer, Resample) and layer.mode == "upsample3d":
                count += 1  # time_conv
        count += 1  # decoder.head CausalConv3d
        return count

    def decode_tiled(self, z: mx.array, tiling_config=None) -> mx.array:
        """Decode latent to video using tiling to reduce memory usage.

        Splits the latent tensor into overlapping spatial/temporal tiles,
        decodes each tile independently, and blends them with trapezoidal
        masks. Reuses the LTX-2 tiling infrastructure.

        Args:
            z: Normalized latent [B, z_dim, T, H, W]
            tiling_config: Optional TilingConfig. If None, uses default.

        Returns:
            Video [B, 3, T_out, H_out, W_out] clamped to [-1, 1]
        """
        from mlx_video.models.wan_2.tiling import TilingConfig, decode_with_tiling

        if tiling_config is None:
            tiling_config = TilingConfig.default()

        # Check if tiling is actually needed
        _, _, f, h, w = z.shape
        needs_tiling = False
        if tiling_config.spatial_config is not None:
            s_tile = tiling_config.spatial_config.tile_size_in_pixels // 8
            if h > s_tile or w > s_tile:
                needs_tiling = True
        if tiling_config.temporal_config is not None:
            t_tile = tiling_config.temporal_config.tile_size_in_frames // 4
            if f > t_tile:
                needs_tiling = True

        if not needs_tiling:
            return self.decode(z)

        # Denormalize once (small tensor), then tile the denormalized latents
        mean = self.mean.reshape(1, -1, 1, 1, 1)
        inv_std = self.inv_std.reshape(1, -1, 1, 1, 1)
        z_denorm = z / inv_std + mean

        def tile_decode(tile_latents, **kwargs):
            x = self.conv2(tile_latents)
            feat_cache = [None] * self._count_decoder_cache_slots()
            out = None
            for i in range(x.shape[2]):
                feat_idx = [0]
                chunk = self.decoder(
                    x[:, :, i : i + 1], feat_cache=feat_cache, feat_idx=feat_idx
                )
                out = chunk if out is None else mx.concatenate([out, chunk], axis=2)
            return mx.clip(out, -1, 1)

        return decode_with_tiling(
            decoder_fn=tile_decode,
            latents=z_denorm,
            tiling_config=tiling_config,
            spatial_scale=8,  # 3× spatial 2× upsamples = 8×
            temporal_scale=4,  # 2× temporal upsamples × 2 = 4×
            causal_temporal=True,  # Wan2.1 causal mapping: T -> 1 + (T-1)*4
        )
