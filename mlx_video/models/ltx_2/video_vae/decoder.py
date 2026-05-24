"""Video VAE Decoder for LTX-2 with timestep conditioning.

Architecture (from PyTorch weights):
- conv_in: 128 -> 1024
- up_blocks.0: 5 ResBlocks at 1024 (with timestep)
- up_blocks.1: Conv 1024 -> 4096, depth2space -> 512, upscale 2x
- up_blocks.2: 5 ResBlocks at 512 (with timestep)
- up_blocks.3: Conv 512 -> 2048, depth2space -> 256, upscale 2x
- up_blocks.4: 5 ResBlocks at 256 (with timestep)
- up_blocks.5: Conv 256 -> 1024, depth2space -> 128, upscale 2x
- up_blocks.6: 5 ResBlocks at 128 (with timestep)
- pixel_norm + timestep modulation (last_scale_shift_table)
- conv_out: 128 -> 48
- unpatchify: 48 -> 3 with patch_size=4
"""

import math
from pathlib import Path
from typing import Dict, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_video.models.ltx_2.video_vae.convolution import CausalConv3d, PaddingModeType
from mlx_video.models.ltx_2.video_vae.ops import PerChannelStatistics, unpatchify
from mlx_video.models.ltx_2.video_vae.sampling import DepthToSpaceUpsample
from mlx_video.models.ltx_2.video_vae.tiling import TilingConfig, decode_with_tiling


def get_timestep_embedding(
    timesteps: mx.array,
    embedding_dim: int,
    flip_sin_to_cos: bool = True,
    downscale_freq_shift: float = 0,
    scale: float = 1,
    max_period: int = 10000,
) -> mx.array:
    """Create sinusoidal timestep embeddings."""
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * mx.arange(0, half_dim, dtype=mx.float32)
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = mx.exp(exponent)
    emb = timesteps[:, None].astype(mx.float32) * emb[None, :]
    emb = scale * emb

    emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)

    if flip_sin_to_cos:
        emb = mx.concatenate([emb[:, half_dim:], emb[:, :half_dim]], axis=-1)

    if embedding_dim % 2 == 1:
        emb = mx.pad(emb, [(0, 0), (0, 1)])

    return emb


class TimestepEmbedding(nn.Module):
    """MLP for timestep embedding."""

    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)
        self.act = nn.SiLU()

    def __call__(self, sample: mx.array) -> mx.array:
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


class PixArtAlphaTimestepEmbedder(nn.Module):
    """Combined timestep embedding (sinusoidal + MLP)."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256, time_embed_dim=embedding_dim
        )

    def __call__(
        self, timestep: mx.array, hidden_dtype: mx.Dtype = mx.float32
    ) -> mx.array:
        timesteps_proj = get_timestep_embedding(
            timestep, embedding_dim=256, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        timesteps_emb = self.timestep_embedder(timesteps_proj.astype(hidden_dtype))
        return timesteps_emb


class ResnetBlock3DSimple(nn.Module):
    """ResNet block with optional timestep conditioning.

    Weight keys: conv1.conv, conv2.conv, scale_shift_table
    """

    def __init__(
        self,
        channels: int,
        spatial_padding_mode: PaddingModeType = PaddingModeType.REFLECT,
        timestep_conditioning: bool = False,
    ):
        super().__init__()
        self.timestep_conditioning = timestep_conditioning

        # Nested conv structure to match PyTorch naming: conv1.conv.weight
        self.conv1 = self._make_conv_wrapper(channels, channels, spatial_padding_mode)
        self.conv2 = self._make_conv_wrapper(channels, channels, spatial_padding_mode)

        self.act = nn.SiLU()

        # Scale-shift table for timestep conditioning: [shift1, scale1, shift2, scale2]
        if timestep_conditioning:
            self.scale_shift_table = mx.zeros((4, channels))

    def _make_conv_wrapper(self, in_ch, out_ch, padding_mode):
        """Create a wrapper object with a 'conv' attribute to match PyTorch naming."""

        class ConvWrapper(nn.Module):
            def __init__(self_inner):
                super().__init__()
                self_inner.conv = CausalConv3d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    spatial_padding_mode=padding_mode,
                )

            def __call__(self_inner, x, causal=False):
                return self_inner.conv(x, causal=causal)

        return ConvWrapper()

    def pixel_norm(self, x: mx.array, eps: float = 1e-8) -> mx.array:
        """Apply pixel normalization."""
        return x / mx.sqrt(mx.mean(x**2, axis=1, keepdims=True) + eps)

    def __call__(
        self,
        x: mx.array,
        causal: bool = False,
        timestep_embed: Optional[mx.array] = None,
    ) -> mx.array:
        residual = x
        batch_size = x.shape[0]

        # Block 1 with optional timestep conditioning
        x = self.pixel_norm(x)

        if self.timestep_conditioning and timestep_embed is not None:
            # scale_shift_table: (4, C), timestep_embed: (B, 4*C, 1, 1, 1)
            # Combine table with timestep embedding
            ada_values = self.scale_shift_table[
                None, :, :, None, None, None
            ]  # (1, 4, C, 1, 1, 1)
            # Reshape timestep_embed from (B, 4*C, 1, 1, 1) to (B, 4, C, 1, 1, 1)
            channels = self.scale_shift_table.shape[1]
            ts_reshaped = timestep_embed.reshape(batch_size, 4, channels, 1, 1, 1)
            ada_values = ada_values + ts_reshaped

            shift1 = ada_values[:, 0]  # (B, C, 1, 1, 1)
            scale1 = ada_values[:, 1]
            shift2 = ada_values[:, 2]
            scale2 = ada_values[:, 3]

            x = x * (1 + scale1) + shift1

        x = self.act(x)
        x = self.conv1(x, causal=causal)

        # Block 2 with optional timestep conditioning
        x = self.pixel_norm(x)

        if self.timestep_conditioning and timestep_embed is not None:
            x = x * (1 + scale2) + shift2

        x = self.act(x)
        x = self.conv2(x, causal=causal)

        return x + residual


class ResBlockGroup(nn.Module):
    """Group of ResNet blocks with shared timestep embedding.

    PyTorch naming: res_blocks.0, res_blocks.1, etc.
    """

    def __init__(
        self,
        channels: int,
        num_layers: int = 5,
        spatial_padding_mode: PaddingModeType = PaddingModeType.REFLECT,
        timestep_conditioning: bool = False,
    ):
        super().__init__()
        self.timestep_conditioning = timestep_conditioning

        # Time embedder for this block group: embed_dim = 4 * channels
        if timestep_conditioning:
            self.time_embedder = PixArtAlphaTimestepEmbedder(embedding_dim=channels * 4)

        # Use dict with int keys for MLX to track parameters properly
        self.res_blocks = {
            i: ResnetBlock3DSimple(
                channels,
                spatial_padding_mode,
                timestep_conditioning=timestep_conditioning,
            )
            for i in range(num_layers)
        }

    def __call__(
        self,
        x: mx.array,
        causal: bool = False,
        timestep: Optional[mx.array] = None,
    ) -> mx.array:
        timestep_embed = None

        if self.timestep_conditioning and timestep is not None:
            batch_size = x.shape[0]
            timestep_embed = self.time_embedder(
                timestep.flatten(), hidden_dtype=x.dtype
            )
            # Reshape to (B, 4*C, 1, 1, 1) for broadcasting
            timestep_embed = timestep_embed.reshape(batch_size, -1, 1, 1, 1)

        for res_block in self.res_blocks.values():
            x = res_block(x, causal=causal, timestep_embed=timestep_embed)
        return x


class LTX2VideoDecoder(nn.Module):
    """LTX-2 Video VAE Decoder with timestep conditioning.

    Architecture:
    - conv_in: 128 -> 1024
    - up_blocks.0: 5 ResBlocks at 1024 (with timestep)
    - up_blocks.1: Upsampler 1024 -> 512
    - up_blocks.2: 5 ResBlocks at 512 (with timestep)
    - up_blocks.3: Upsampler 512 -> 256
    - up_blocks.4: 5 ResBlocks at 256 (with timestep)
    - up_blocks.5: Upsampler 256 -> 128
    - up_blocks.6: 5 ResBlocks at 128 (with timestep)
    - conv_out: 128 -> 48 (3 * 4^2 for patch_size=4)
    """

    # Block definitions: ("res", channels, num_layers) or ("d2s", in_channels, reduction, stride)
    # stride is (D, H, W) tuple
    DEFAULT_BLOCKS = [
        ("res", 1024, 5),
        ("d2s", 1024, 2, (2, 2, 2)),
        ("res", 512, 5),
        ("d2s", 512, 2, (2, 2, 2)),
        ("res", 256, 5),
        ("d2s", 256, 2, (2, 2, 2)),
        ("res", 128, 5),
    ]

    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 3,
        patch_size: int = 4,
        num_layers_per_block: int = 5,
        spatial_padding_mode: PaddingModeType = PaddingModeType.REFLECT,
        timestep_conditioning: bool = True,
        decoder_blocks: list = None,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.timestep_conditioning = timestep_conditioning

        # Decode parameters (configurable via constructor)
        self.decode_noise_scale = 0.025  # Set to 0.0 to disable noise
        self.decode_timestep = 0.05

        # Per-channel statistics for denormalization (loaded from weights)
        self.per_channel_statistics = PerChannelStatistics(latent_channels=in_channels)

        blocks = decoder_blocks or self.DEFAULT_BLOCKS
        first_ch = blocks[0][1]
        last_ch = blocks[-1][1]

        # Initial conv: in_channels -> first block channels
        class ConvInWrapper(nn.Module):
            def __init__(self_inner):
                super().__init__()
                self_inner.conv = CausalConv3d(
                    in_channels=in_channels,
                    out_channels=first_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    spatial_padding_mode=spatial_padding_mode,
                )

            def __call__(self_inner, x, causal=False):
                return self_inner.conv(x, causal=causal)

        self.conv_in = ConvInWrapper()

        # Build up blocks from config
        self.up_blocks = {}
        for idx, block_def in enumerate(blocks):
            block_type = block_def[0]
            ch = block_def[1]
            if block_type == "res":
                num_layers = (
                    block_def[2] if len(block_def) > 2 else num_layers_per_block
                )
                self.up_blocks[idx] = ResBlockGroup(
                    ch, num_layers, spatial_padding_mode, timestep_conditioning
                )
            elif block_type == "d2s":
                reduction = block_def[2] if len(block_def) > 2 else 2
                stride = block_def[3] if len(block_def) > 3 else (2, 2, 2)
                residual = block_def[4] if len(block_def) > 4 else True
                self.up_blocks[idx] = DepthToSpaceUpsample(
                    dims=3,
                    in_channels=ch,
                    stride=stride,
                    residual=residual,
                    out_channels_reduction_factor=reduction,
                    spatial_padding_mode=spatial_padding_mode,
                )

        final_out_channels = out_channels * patch_size * patch_size

        class ConvOutWrapper(nn.Module):
            def __init__(self_inner):
                super().__init__()
                self_inner.conv = CausalConv3d(
                    in_channels=last_ch,
                    out_channels=final_out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    spatial_padding_mode=spatial_padding_mode,
                )

            def __call__(self_inner, x, causal=False):
                return self_inner.conv(x, causal=causal)

        self.conv_out = ConvOutWrapper()

        self.act = nn.SiLU()

        if timestep_conditioning:
            self.timestep_scale_multiplier = mx.array(1000.0)
            self.last_time_embedder = PixArtAlphaTimestepEmbedder(
                embedding_dim=last_ch * 2
            )
            self.last_scale_shift_table = mx.zeros((2, last_ch))

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        # Build decoder weights dict with key remapping
        sanitized = {}
        if "per_channel_statistics.mean" in weights:
            return weights
        for key, value in weights.items():
            new_key = key

            if not key.startswith("vae.") or key.startswith("vae.encoder."):
                continue

            if key.startswith("vae.per_channel_statistics."):
                # Map per-channel statistics (use exact key matching)
                if key == "vae.per_channel_statistics.mean-of-means":
                    new_key = "per_channel_statistics.mean"
                elif key == "vae.per_channel_statistics.std-of-means":
                    new_key = "per_channel_statistics.std"
                else:
                    continue  # Skip other statistics keys

            if key.startswith("vae.decoder."):
                new_key = key.replace("vae.decoder.", "")

            # Handle Conv3d weight transpose: (O, I, D, H, W) -> (O, D, H, W, I)
            if ".conv.weight" in key and value.ndim == 5:
                value = mx.transpose(value, (0, 2, 3, 4, 1))

            if ".conv.bias" in key:
                pass  # bias doesn't need transpose

            if ".conv.weight" in new_key or ".conv.bias" in new_key:

                if (
                    ".conv.conv.weight" not in new_key
                    and ".conv.conv.bias" not in new_key
                ):
                    new_key = new_key.replace(".conv.weight", ".conv.conv.weight")
                    new_key = new_key.replace(".conv.bias", ".conv.conv.bias")

            sanitized[new_key] = value
        return sanitized

    @classmethod
    def from_pretrained(
        cls, model_path: Path, strict: bool = True
    ) -> "LTX2VideoDecoder":
        """Load a pretrained decoder from a directory with config.json and weights.

        Args:
            model_path: Path to directory containing config.json and safetensors files,
                        or path to a single safetensors file.
            strict: Whether to require all weight keys to match.

        Returns:
            Loaded LTX2VideoDecoder instance
        """
        import json

        model_path = Path(model_path)
        config_dict = {}
        weight_files = sorted(model_path.glob("*.safetensors"))
        if (
            not weight_files
            and model_path.name == "decoder"
            and model_path.parent.exists()
        ):
            parent_weight_files = sorted(model_path.parent.glob("*.safetensors"))
            if parent_weight_files:
                model_path = model_path.parent
                weight_files = parent_weight_files

        # Load config from directory
        config_path = model_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config_dict = json.load(f)

        # Load weights from directory
        if not weight_files:
            raise FileNotFoundError(f"No safetensors files found in {model_path}")
        weights = {}
        for wf in weight_files:
            weights.update(mx.load(str(wf)))

        # Infer block structure from weights
        decoder_blocks = cls._infer_blocks(weights)

        # Determine spatial padding mode from config
        spatial_padding_mode_str = config_dict.get(
            "spatial_padding_mode",
            config_dict.get("decoder_spatial_padding_mode", "reflect"),
        )
        spatial_padding_mode = PaddingModeType(spatial_padding_mode_str)

        model = cls(
            timestep_conditioning=config_dict.get("timestep_conditioning", False),
            decoder_blocks=decoder_blocks,
            spatial_padding_mode=spatial_padding_mode,
        )
        weights = model.sanitize(weights)
        model.load_weights(list(weights.items()), strict=strict)
        return model

    @staticmethod
    def _infer_blocks(weights: dict) -> list:
        """Infer decoder block structure from weight keys."""
        block_indices = set()
        for k in weights:
            if "up_blocks." in k:
                idx_str = k.split("up_blocks.")[1].split(".")[0]
                if idx_str.isdigit():
                    block_indices.add(int(idx_str))

        if not block_indices:
            return None

        # First pass: collect block info
        raw_blocks = []
        for idx in sorted(block_indices):
            has_conv = any(f"up_blocks.{idx}.conv." in k for k in weights)
            res_indices = set()
            for k in weights:
                prefix = f"up_blocks.{idx}.res_blocks."
                if prefix in k:
                    res_idx = k.split(prefix)[1].split(".")[0]
                    if res_idx.isdigit():
                        res_indices.add(int(res_idx))

            if has_conv and not res_indices:
                # D2S block - get conv shape
                for k, v in weights.items():
                    if f"up_blocks.{idx}.conv." in k and "weight" in k:
                        in_ch = v.shape[-1] if v.ndim == 5 else v.shape[1]
                        conv_out_ch = v.shape[0]
                        raw_blocks.append(("d2s", in_ch, conv_out_ch))
                        break
            elif res_indices:
                num_res = max(res_indices) + 1
                for k, v in weights.items():
                    if f"up_blocks.{idx}.res_blocks.0.conv1" in k and "weight" in k:
                        ch = v.shape[0]
                        raw_blocks.append(("res", ch, num_res))
                        break

        # Second pass: determine d2s strides using the channel progression
        # For each d2s block, the next res block tells us the expected output channels
        blocks = []
        d2s_strides = []
        for i, block in enumerate(raw_blocks):
            if block[0] == "res":
                blocks.append(block)
            elif block[0] == "d2s":
                in_ch, conv_out_ch = block[1], block[2]
                # Find next res block's channels
                next_ch = None
                for j in range(i + 1, len(raw_blocks)):
                    if raw_blocks[j][0] == "res":
                        next_ch = raw_blocks[j][1]
                        break

                if next_ch is None:
                    next_ch = in_ch // 2  # fallback

                # out_ch = in_ch // reduction
                reduction = in_ch // next_ch if next_ch > 0 else 2

                # conv_out = next_ch * multiplier → multiplier = conv_out / next_ch
                multiplier = conv_out_ch // next_ch if next_ch > 0 else 8

                # Determine stride from multiplier
                if multiplier == 8:
                    stride = (2, 2, 2)
                elif multiplier == 4:
                    stride = (1, 2, 2)
                elif multiplier == 2:
                    stride = (2, 1, 1)
                else:
                    stride = (2, 2, 2)

                d2s_strides.append(stride)
                blocks.append(("d2s", in_ch, reduction, stride))

        if not blocks:
            return None

        # Determine residual flag: LTX-2 has uniform (2,2,2) strides with reduction=2 → residual=True
        # LTX-2.3 has mixed strides or reduction=1 → residual=False
        has_mixed_strides = len(set(d2s_strides)) > 1
        has_non_standard_reduction = any(b[2] != 2 for b in blocks if b[0] == "d2s")
        use_residual = not has_mixed_strides and not has_non_standard_reduction

        # Apply residual flag to all d2s blocks
        final_blocks = []
        for block in blocks:
            if block[0] == "d2s":
                final_blocks.append(("d2s", block[1], block[2], block[3], use_residual))
            else:
                final_blocks.append(block)

        return final_blocks

    def pixel_norm(self, x: mx.array, eps: float = 1e-8) -> mx.array:
        """Apply pixel normalization."""
        return x / mx.sqrt(mx.mean(x**2, axis=1, keepdims=True) + eps)

    def __call__(
        self,
        sample: mx.array,
        causal: bool = False,
        timestep: Optional[mx.array] = None,
        debug: bool = False,
        chunked_conv: bool = False,
    ) -> mx.array:

        batch_size = sample.shape[0]

        # Add noise if timestep conditioning is enabled
        if self.timestep_conditioning:
            noise = mx.random.normal(sample.shape) * self.decode_noise_scale
            sample = noise + (1.0 - self.decode_noise_scale) * sample

        sample = self.per_channel_statistics.un_normalize(sample)

        if timestep is None and self.timestep_conditioning:
            timestep = mx.full((batch_size,), self.decode_timestep)

        scaled_timestep = None
        if self.timestep_conditioning and timestep is not None:
            scaled_timestep = timestep * self.timestep_scale_multiplier

        x = self.conv_in(sample, causal=causal)

        for i, block in self.up_blocks.items():
            if isinstance(block, ResBlockGroup):
                x = block(x, causal=causal, timestep=scaled_timestep)
            elif isinstance(block, DepthToSpaceUpsample):
                x = block(x, causal=causal, chunked_conv=chunked_conv)
            else:
                x = block(x, causal=causal)

        x = self.pixel_norm(x)

        if self.timestep_conditioning and scaled_timestep is not None:
            embedded_timestep = self.last_time_embedder(
                scaled_timestep.flatten(), hidden_dtype=x.dtype
            )
            embedded_timestep = embedded_timestep.reshape(batch_size, -1, 1, 1, 1)

            ada_values = self.last_scale_shift_table[
                None, :, :, None, None, None
            ]  # (1, 2, 128, 1, 1, 1)
            ts_reshaped = embedded_timestep.reshape(batch_size, 2, 128, 1, 1, 1)
            ada_values = ada_values + ts_reshaped

            shift = ada_values[:, 0]  # (B, 128, 1, 1, 1)
            scale = ada_values[:, 1]

            x = x * (1 + scale) + shift

        x = self.act(x)

        x = self.conv_out(x, causal=causal)

        # Unpatchify: (B, 48, F', H', W') -> (B, 3, F, H*4, W*4)
        x = unpatchify(x, patch_size_hw=self.patch_size, patch_size_t=1)

        return x

    def decode_tiled(
        self,
        sample: mx.array,
        tiling_config: Optional[TilingConfig] = None,
        tiling_mode: str = "auto",
        causal: bool = False,
        timestep: Optional[mx.array] = None,
        debug: bool = False,
        on_frames_ready: Optional[callable] = None,
    ) -> mx.array:
        """Decode latents using tiling to reduce memory usage.

        This method is useful for decoding large videos that would otherwise
        cause out-of-memory errors. It divides the latents into tiles,
        decodes each tile separately, and blends them together.

        Args:
            sample: Input latents of shape (B, C, F, H, W).
            tiling_config: Tiling configuration. If None, uses TilingConfig.default().
            causal: Whether to use causal convolutions.
            timestep: Optional timestep for conditioning.
            debug: Whether to print debug info.

        Returns:
            Decoded video of shape (B, 3, F*8, H*8, W*8).
        """
        if tiling_config is None:
            tiling_config = TilingConfig.default()

        # Check if tiling is actually needed
        _, _, f, h, w = sample.shape
        needs_spatial_tiling = False
        needs_temporal_tiling = False

        # Spatial scale is 32 (8x VAE upsample + 4x unpatchify)
        # Temporal scale is 8
        spatial_scale = 32
        temporal_scale = 8

        if tiling_config.spatial_config is not None:
            s_cfg = tiling_config.spatial_config
            tile_size_latent = s_cfg.tile_size_in_pixels // spatial_scale
            if h > tile_size_latent or w > tile_size_latent:
                needs_spatial_tiling = True

        if tiling_config.temporal_config is not None:
            t_cfg = tiling_config.temporal_config
            tile_size_latent = t_cfg.tile_size_in_frames // temporal_scale
            if f > tile_size_latent:
                needs_temporal_tiling = True

        # Auto-enable chunked conv for modes where it helps (larger tiles)
        # Chunked conv reduces memory by processing conv+depth_to_space in temporal chunks
        use_chunked_conv = tiling_mode in (
            "conservative",
            "none",
            "auto",
            "default",
            "spatial",
        )

        if not needs_spatial_tiling and not needs_temporal_tiling:
            # No tiling needed, use regular decode
            return self(
                sample,
                causal=causal,
                timestep=timestep,
                debug=debug,
                chunked_conv=use_chunked_conv,
            )

        return decode_with_tiling(
            decoder_fn=self,
            latents=sample,
            tiling_config=tiling_config,
            spatial_scale=32,  # VAE spatial: 8x upsampling + 4x unpatchify = 32x
            temporal_scale=8,  # VAE temporal upsampling factor
            causal=causal,
            timestep=timestep,
            chunked_conv=use_chunked_conv,
            on_frames_ready=on_frames_ready,
        )


# Backward-compatible alias
VideoDecoder = LTX2VideoDecoder
