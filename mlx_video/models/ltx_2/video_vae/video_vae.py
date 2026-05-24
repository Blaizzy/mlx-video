"""Video VAE Encoder and Decoder for LTX-2."""

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from mlx_video.models.ltx_2.video_vae.convolution import CausalConv3d, PaddingModeType
from mlx_video.models.ltx_2.video_vae.ops import (
    PerChannelStatistics,
    patchify,
    unpatchify,
)
from mlx_video.models.ltx_2.video_vae.resnet import (
    NormLayerType,
    ResnetBlock3D,
    UNetMidBlock3D,
)
from mlx_video.models.ltx_2.video_vae.sampling import (
    DepthToSpaceUpsample,
    SpaceToDepthDownsample,
)
from mlx_video.utils import PixelNorm


class LogVarianceType(Enum):
    """Log variance mode for VAE."""

    PER_CHANNEL = "per_channel"
    UNIFORM = "uniform"
    CONSTANT = "constant"
    NONE = "none"


def _make_encoder_block(
    block_name: str,
    block_config: Dict[str, Any],
    in_channels: int,
    convolution_dimensions: int,
    norm_layer: NormLayerType,
    norm_num_groups: int,
    spatial_padding_mode: PaddingModeType,
) -> Tuple[nn.Module, int]:
    """Create an encoder block.

    Args:
        block_name: Type of block
        block_config: Block configuration
        in_channels: Input channels
        convolution_dimensions: Number of dimensions
        norm_layer: Normalization layer type
        norm_num_groups: Number of groups for group norm
        spatial_padding_mode: Padding mode

    Returns:
        Tuple of (block, output_channels)
    """
    out_channels = in_channels

    if block_name == "res_x":
        block = UNetMidBlock3D(
            dims=convolution_dimensions,
            in_channels=in_channels,
            num_layers=block_config["num_layers"],
            resnet_eps=1e-6,
            resnet_groups=norm_num_groups,
            norm_layer=norm_layer,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "res_x_y":
        out_channels = in_channels * block_config.get("multiplier", 2)
        block = ResnetBlock3D(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            eps=1e-6,
            groups=norm_num_groups,
            norm_layer=norm_layer,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_time":
        block = CausalConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=(2, 1, 1),
            padding=1,
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_space":
        block = CausalConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=(1, 2, 2),
            padding=1,
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_all":
        block = CausalConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=(2, 2, 2),
            padding=1,
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_all_x_y":
        out_channels = in_channels * block_config.get("multiplier", 2)
        block = CausalConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=(2, 2, 2),
            padding=1,
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_all_res":
        out_channels = in_channels * block_config.get("multiplier", 2)
        block = SpaceToDepthDownsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=(2, 2, 2),
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_space_res":
        out_channels = in_channels * block_config.get("multiplier", 2)
        block = SpaceToDepthDownsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=(1, 2, 2),
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_time_res":
        out_channels = in_channels * block_config.get("multiplier", 2)
        block = SpaceToDepthDownsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=(2, 1, 1),
            spatial_padding_mode=spatial_padding_mode,
        )
    else:
        raise ValueError(f"Unknown encoder block: {block_name}")

    return block, out_channels


def _make_decoder_block(
    block_name: str,
    block_config: Dict[str, Any],
    in_channels: int,
    convolution_dimensions: int,
    norm_layer: NormLayerType,
    timestep_conditioning: bool,
    norm_num_groups: int,
    spatial_padding_mode: PaddingModeType,
) -> Tuple[nn.Module, int]:
    """Create a decoder block."""
    out_channels = in_channels

    if block_name == "res_x":
        block = UNetMidBlock3D(
            dims=convolution_dimensions,
            in_channels=in_channels,
            num_layers=block_config["num_layers"],
            resnet_eps=1e-6,
            resnet_groups=norm_num_groups,
            norm_layer=norm_layer,
            inject_noise=block_config.get("inject_noise", False),
            timestep_conditioning=timestep_conditioning,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "res_x_y":
        out_channels = in_channels // block_config.get("multiplier", 2)
        block = ResnetBlock3D(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            eps=1e-6,
            groups=norm_num_groups,
            norm_layer=norm_layer,
            inject_noise=block_config.get("inject_noise", False),
            timestep_conditioning=False,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_time":
        block = DepthToSpaceUpsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            stride=(2, 1, 1),
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_space":
        block = DepthToSpaceUpsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            stride=(1, 2, 2),
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_all":
        out_channels = in_channels // block_config.get("multiplier", 1)
        block = DepthToSpaceUpsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            stride=(2, 2, 2),
            residual=block_config.get("residual", False),
            out_channels_reduction_factor=block_config.get("multiplier", 1),
            spatial_padding_mode=spatial_padding_mode,
        )
    else:
        raise ValueError(f"Unknown decoder block: {block_name}")

    return block, out_channels


class VideoEncoder(nn.Module):

    _DEFAULT_NORM_NUM_GROUPS = 32

    def __init__(self, config: "VideoEncoderModelConfig"):
        """Initialize VideoEncoder from config.

        Args:
            config: VideoEncoderModelConfig with encoder parameters
        """
        super().__init__()

        self.patch_size = config.patch_size
        self.norm_layer = config.norm_layer
        self.latent_channels = config.out_channels
        self.latent_log_var = config.latent_log_var
        self._norm_num_groups = self._DEFAULT_NORM_NUM_GROUPS

        encoder_blocks = config.encoder_blocks if config.encoder_blocks else []
        encoder_spatial_padding_mode = config.encoder_spatial_padding_mode

        # Per-channel statistics for normalizing latents
        self.per_channel_statistics = PerChannelStatistics(
            latent_channels=config.out_channels
        )

        # After patchify, channels increase by patch_size^2
        in_channels = config.in_channels * config.patch_size**2
        feature_channels = config.out_channels

        # Initial convolution
        self.conv_in = CausalConv3d(
            in_channels=in_channels,
            out_channels=feature_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            causal=True,
            spatial_padding_mode=encoder_spatial_padding_mode,
        )

        # Build encoder blocks
        # Use dict with int keys for MLX to track parameters (lists are NOT tracked)
        self.down_blocks = {}
        for idx, (block_name, block_params) in enumerate(encoder_blocks):
            block_config = (
                {"num_layers": block_params}
                if isinstance(block_params, int)
                else block_params
            )

            block, feature_channels = _make_encoder_block(
                block_name=block_name,
                block_config=block_config,
                in_channels=feature_channels,
                convolution_dimensions=config.convolution_dimensions,
                norm_layer=config.norm_layer,
                norm_num_groups=self._norm_num_groups,
                spatial_padding_mode=encoder_spatial_padding_mode,
            )
            self.down_blocks[idx] = block

        # Output normalization and convolution
        if config.norm_layer == NormLayerType.GROUP_NORM:
            self.conv_norm_out = nn.GroupNorm(
                num_groups=self._norm_num_groups,
                dims=feature_channels,
                eps=1e-6,
            )
        elif config.norm_layer == NormLayerType.PIXEL_NORM:
            self.conv_norm_out = PixelNorm()

        self.conv_act = nn.SiLU()

        # Calculate output convolution channels
        conv_out_channels = config.out_channels
        if config.latent_log_var == LogVarianceType.PER_CHANNEL:
            conv_out_channels *= 2
        elif config.latent_log_var in {
            LogVarianceType.UNIFORM,
            LogVarianceType.CONSTANT,
        }:
            conv_out_channels += 1

        self.conv_out = CausalConv3d(
            in_channels=feature_channels,
            out_channels=conv_out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            causal=True,
            spatial_padding_mode=encoder_spatial_padding_mode,
        )

    def __call__(self, sample: mx.array) -> mx.array:
        """Encode video to latent representation.

        Args:
            sample: Input video of shape (B, C, F, H, W).
                    F must be 1 + 8*k (e.g., 1, 9, 17, 25, 33...)

        Returns:
            Normalized latent means of shape (B, 128, F', H', W')
        """
        # Validate frame count
        frames_count = sample.shape[2]
        if ((frames_count - 1) % 8) != 0:
            raise ValueError(
                "Invalid number of frames: Encode input must have 1 + 8 * x frames "
                f"(e.g., 1, 9, 17, ...). Got {frames_count} frames."
            )

        # Initial patchify
        sample = patchify(sample, patch_size_hw=self.patch_size, patch_size_t=1)
        sample = self.conv_in(sample, causal=True)

        # Process through encoder blocks
        for i in range(len(self.down_blocks)):
            down_block = self.down_blocks[i]
            if isinstance(down_block, (UNetMidBlock3D, ResnetBlock3D)):
                sample = down_block(sample, causal=True)
            else:
                sample = down_block(sample, causal=True)

        # Output processing
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample, causal=True)

        # Handle log variance modes
        if self.latent_log_var == LogVarianceType.UNIFORM:
            means = sample[:, :-1, ...]
            logvar = sample[:, -1:, ...]
            num_channels = means.shape[1]
            repeated_logvar = mx.tile(logvar, (1, num_channels, 1, 1, 1))
            sample = mx.concatenate([means, repeated_logvar], axis=1)
        elif self.latent_log_var == LogVarianceType.CONSTANT:
            sample = sample[:, :-1, ...]
            approx_ln_0 = -30
            sample = mx.concatenate(
                [
                    sample,
                    mx.full_like(sample, approx_ln_0),
                ],
                axis=1,
            )

        # Split into means and logvar, normalize means
        means = sample[:, : self.latent_channels, ...]
        return self.per_channel_statistics.normalize(means)

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Sanitize VAE encoder weights from PyTorch format to MLX format."""
        sanitized = {}
        if "per_channel_statistics.mean" in weights:
            return weights

        for key, value in weights.items():
            new_key = key

            if "position_ids" in key:
                continue

            # Only process VAE encoder weights
            if not key.startswith("vae."):
                continue

            # Handle per-channel statistics
            if "vae.per_channel_statistics" in key:
                if key == "vae.per_channel_statistics.mean-of-means":
                    new_key = "per_channel_statistics.mean"
                elif key == "vae.per_channel_statistics.std-of-means":
                    new_key = "per_channel_statistics.std"
                else:
                    continue
            elif key.startswith("vae.encoder."):
                new_key = key.replace("vae.encoder.", "")
            else:
                continue

            # Conv3d: PyTorch (O, I, D, H, W) -> MLX (O, D, H, W, I)
            if "conv" in new_key.lower() and "weight" in new_key and value.ndim == 5:
                value = mx.transpose(value, (0, 2, 3, 4, 1))

            # Conv2d: PyTorch (O, I, H, W) -> MLX (O, H, W, I)
            if "conv" in new_key.lower() and "weight" in new_key and value.ndim == 4:
                value = mx.transpose(value, (0, 2, 3, 1))

            sanitized[new_key] = value
        return sanitized

    @classmethod
    def from_pretrained(cls, model_path: Path) -> "VideoEncoder":
        """Load a pretrained VideoEncoder from a directory with weights and config.

        Args:
            model_path: Path to directory containing safetensors weights and config.json

        Returns:
            Loaded VideoEncoder instance
        """
        import json

        from mlx_video.models.ltx_2.config import VideoEncoderModelConfig

        model_path = Path(model_path)
        weight_files = sorted(model_path.glob("*.safetensors"))
        if (
            not weight_files
            and model_path.name == "encoder"
            and model_path.parent.exists()
        ):
            parent_weight_files = sorted(model_path.parent.glob("*.safetensors"))
            if parent_weight_files:
                model_path = model_path.parent
                weight_files = parent_weight_files

        # Load config
        config_path = model_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config_dict = json.load(f)
            config = VideoEncoderModelConfig.from_dict(config_dict)
        else:
            config = VideoEncoderModelConfig()

        # Load weights
        if not weight_files:
            if model_path.is_file():
                weights = mx.load(str(model_path))
            else:
                raise FileNotFoundError(f"No safetensors files found in {model_path}")
        else:
            weights = {}
            for wf in weight_files:
                weights.update(mx.load(str(wf)))

        # Create model, sanitize and load weights
        model = cls(config)
        weights = model.sanitize(weights)
        model.load_weights(list(weights.items()), strict=False)
        return model


class VideoDecoder(nn.Module):

    _DEFAULT_NORM_NUM_GROUPS = 32

    def __init__(
        self,
        convolution_dimensions: int = 3,
        in_channels: int = 128,
        out_channels: int = 3,
        decoder_blocks: List[Tuple[str, Any]] = None,
        patch_size: int = 4,
        norm_layer: NormLayerType = NormLayerType.PIXEL_NORM,
        causal: bool = False,
        timestep_conditioning: bool = False,
        decoder_spatial_padding_mode: PaddingModeType = PaddingModeType.REFLECT,
    ):
        """Initialize VideoDecoder.

        Args:
            convolution_dimensions: Number of dimensions
            in_channels: Input latent channels
            out_channels: Output channels (3 for RGB)
            decoder_blocks: List of (block_name, config) tuples
            patch_size: Spatial patch size
            norm_layer: Normalization layer type
            causal: Whether to use causal convolutions
            timestep_conditioning: Whether to use timestep conditioning
            decoder_spatial_padding_mode: Padding mode
        """
        super().__init__()

        if decoder_blocks is None:
            decoder_blocks = []

        self.patch_size = patch_size
        out_channels = out_channels * patch_size**2
        self.causal = causal
        self.timestep_conditioning = timestep_conditioning
        self._norm_num_groups = self._DEFAULT_NORM_NUM_GROUPS

        # Per-channel statistics for denormalizing latents
        self.per_channel_statistics = PerChannelStatistics(latent_channels=in_channels)

        # Noise and timestep parameters
        self.decode_noise_scale = 0.025
        self.decode_timestep = 0.05

        # Compute initial feature channels
        feature_channels = in_channels
        for block_name, block_params in list(reversed(decoder_blocks)):
            block_config = block_params if isinstance(block_params, dict) else {}
            if block_name == "res_x_y":
                feature_channels = feature_channels * block_config.get("multiplier", 2)
            if block_name == "compress_all":
                feature_channels = feature_channels * block_config.get("multiplier", 1)

        # Initial convolution
        self.conv_in = CausalConv3d(
            in_channels=in_channels,
            out_channels=feature_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            causal=True,
            spatial_padding_mode=decoder_spatial_padding_mode,
        )

        # Build decoder blocks (reversed order)
        # Use dict with int keys for MLX to track parameters (lists are NOT tracked)
        self.up_blocks = {}
        for idx, (block_name, block_params) in enumerate(reversed(decoder_blocks)):
            block_config = (
                {"num_layers": block_params}
                if isinstance(block_params, int)
                else block_params
            )

            block, feature_channels = _make_decoder_block(
                block_name=block_name,
                block_config=block_config,
                in_channels=feature_channels,
                convolution_dimensions=convolution_dimensions,
                norm_layer=norm_layer,
                timestep_conditioning=timestep_conditioning,
                norm_num_groups=self._norm_num_groups,
                spatial_padding_mode=decoder_spatial_padding_mode,
            )
            self.up_blocks[idx] = block

        # Output normalization
        if norm_layer == NormLayerType.GROUP_NORM:
            self.conv_norm_out = nn.GroupNorm(
                num_groups=self._norm_num_groups,
                dims=feature_channels,
                eps=1e-6,
            )
        elif norm_layer == NormLayerType.PIXEL_NORM:
            self.conv_norm_out = PixelNorm()

        self.conv_act = nn.SiLU()
        self.conv_out = CausalConv3d(
            in_channels=feature_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            causal=True,
            spatial_padding_mode=decoder_spatial_padding_mode,
        )

    def __call__(
        self,
        sample: mx.array,
        timestep: Optional[mx.array] = None,
    ) -> mx.array:
        """Decode latent to video.

        Args:
            sample: Latent tensor of shape (B, 128, F', H', W')
            timestep: Optional timestep for conditioning

        Returns:
            Decoded video of shape (B, 3, F, H, W)
        """
        batch_size = sample.shape[0]

        # Add noise if timestep conditioning is enabled
        if self.timestep_conditioning:
            noise = mx.random.normal(sample.shape) * self.decode_noise_scale
            sample = noise + (1.0 - self.decode_noise_scale) * sample

        # Denormalize latents
        sample = self.per_channel_statistics.un_normalize(sample)

        # Use default timestep if not provided
        if timestep is None and self.timestep_conditioning:
            timestep = mx.full((batch_size,), self.decode_timestep)

        # Initial convolution
        sample = self.conv_in(sample, causal=self.causal)

        # Process through decoder blocks
        for i in range(len(self.up_blocks)):
            up_block = self.up_blocks[i]
            if isinstance(up_block, UNetMidBlock3D):
                sample = up_block(sample, causal=self.causal)
            elif isinstance(up_block, ResnetBlock3D):
                sample = up_block(sample, causal=self.causal)
            else:
                sample = up_block(sample, causal=self.causal)

        # Output processing
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample, causal=self.causal)

        # Unpatchify to restore spatial resolution
        sample = unpatchify(sample, patch_size_hw=self.patch_size, patch_size_t=1)

        return sample
