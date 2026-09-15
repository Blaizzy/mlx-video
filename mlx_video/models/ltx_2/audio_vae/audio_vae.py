"""Audio VAE encoder and decoder for LTX-2."""

from pathlib import Path
from typing import Dict

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.base import check_array_shape

from ..config import AudioDecoderModelConfig, AudioEncoderModelConfig, CausalityAxis
from .attention import AttentionType, make_attn
from .causal_conv_2d import make_conv2d
from .downsample import build_downsampling_path
from .normalization import NormType, build_normalization_layer
from .ops import AudioLatentShape, AudioPatchifier, PerChannelStatistics
from .resnet import ResnetBlock
from .upsample import build_upsampling_path

LATENT_DOWNSAMPLE_FACTOR = 4


def build_mid_block(
    channels: int,
    temb_channels: int,
    dropout: float,
    norm_type: NormType,
    causality_axis: CausalityAxis,
    attn_type: AttentionType,
    add_attention: bool,
) -> dict:
    """Build the middle block with two ResNet blocks and optional attention."""
    mid = {}
    mid["block_1"] = ResnetBlock(
        in_channels=channels,
        out_channels=channels,
        temb_channels=temb_channels,
        dropout=dropout,
        norm_type=norm_type,
        causality_axis=causality_axis,
    )
    mid["attn_1"] = (
        make_attn(channels, attn_type=attn_type, norm_type=norm_type)
        if add_attention
        else None
    )
    mid["block_2"] = ResnetBlock(
        in_channels=channels,
        out_channels=channels,
        temb_channels=temb_channels,
        dropout=dropout,
        norm_type=norm_type,
        causality_axis=causality_axis,
    )
    return mid


def run_mid_block(mid: dict, features: mx.array) -> mx.array:
    """Run features through the middle block."""
    features = mid["block_1"](features, temb=None)
    if mid["attn_1"] is not None:
        features = mid["attn_1"](features)
    return mid["block_2"](features, temb=None)


class AudioEncoder(nn.Module):
    """Encoder that compresses audio spectrograms into latent representations."""

    def __init__(self, config: AudioEncoderModelConfig) -> None:
        super().__init__()

        self.per_channel_statistics = PerChannelStatistics(latent_channels=config.ch)
        self.sample_rate = config.sample_rate
        self.mel_hop_length = config.mel_hop_length
        self.is_causal = config.is_causal
        self.mel_bins = config.mel_bins

        self.patchifier = AudioPatchifier(
            patch_size=1,
            audio_latent_downsample_factor=LATENT_DOWNSAMPLE_FACTOR,
            sample_rate=config.sample_rate,
            hop_length=config.mel_hop_length,
            is_causal=config.is_causal,
        )

        self.ch = config.ch
        self.temb_ch = 0
        self.num_resolutions = len(config.ch_mult)
        self.num_res_blocks = config.num_res_blocks
        self.resolution = config.resolution
        self.in_channels = config.in_channels
        self.z_channels = config.z_channels
        self.double_z = config.double_z
        self.norm_type = config.norm_type
        self.causality_axis = config.causality_axis
        self.attn_type = config.attn_type

        self.conv_in = make_conv2d(
            config.in_channels,
            self.ch,
            kernel_size=3,
            stride=1,
            causality_axis=self.causality_axis,
        )

        self.down, block_in = build_downsampling_path(
            ch=config.ch,
            ch_mult=config.ch_mult,
            num_resolutions=self.num_resolutions,
            num_res_blocks=config.num_res_blocks,
            resolution=config.resolution,
            temb_channels=self.temb_ch,
            dropout=config.dropout,
            norm_type=self.norm_type,
            causality_axis=self.causality_axis,
            attn_type=self.attn_type,
            attn_resolutions=config.attn_resolutions or set(),
            resamp_with_conv=config.resamp_with_conv,
        )

        self.mid = build_mid_block(
            channels=block_in,
            temb_channels=self.temb_ch,
            dropout=config.dropout,
            norm_type=self.norm_type,
            causality_axis=self.causality_axis,
            attn_type=self.attn_type,
            add_attention=config.mid_block_add_attention,
        )

        self.norm_out = build_normalization_layer(block_in, normtype=self.norm_type)
        out_channels = 2 * config.z_channels if config.double_z else config.z_channels
        self.conv_out = make_conv2d(
            block_in,
            out_channels,
            kernel_size=3,
            stride=1,
            causality_axis=self.causality_axis,
        )

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Sanitize audio encoder weights from PyTorch format."""
        sanitized = {}
        for key, value in weights.items():
            new_key = key
            if key.startswith("audio_vae.encoder."):
                new_key = key.replace("audio_vae.encoder.", "")
            elif key.startswith("encoder."):
                new_key = key.replace("encoder.", "")
            elif key.startswith("audio_vae.per_channel_statistics."):
                if "mean-of-means" in key:
                    new_key = "per_channel_statistics.mean_of_means"
                elif "std-of-means" in key:
                    new_key = "per_channel_statistics.std_of_means"
                else:
                    continue
            elif "per_channel_statistics" in key:
                if "mean-of-means" in key or "latents_mean" in key:
                    new_key = "per_channel_statistics.mean_of_means"
                elif "std-of-means" in key or "latents_std" in key:
                    new_key = "per_channel_statistics.std_of_means"
                else:
                    continue
            elif key == "latents_mean":
                new_key = "per_channel_statistics.mean_of_means"
            elif key == "latents_std":
                new_key = "per_channel_statistics.std_of_means"
            else:
                continue

            if "conv" in new_key.lower() and "weight" in new_key and value.ndim == 4:
                value = (
                    value
                    if check_array_shape(value)
                    else mx.transpose(value, (0, 2, 3, 1))
                )

            sanitized[new_key] = value
        return sanitized

    @classmethod
    def from_pretrained(cls, model_path: Path) -> "AudioEncoder":
        """Load audio encoder from pretrained weights."""
        import json
        from safetensors import safe_open

        from mlx_video.models.ltx_2.config import AudioEncoderModelConfig

        model_path = Path(model_path)
        if model_path.is_file():
            with safe_open(model_path, framework="numpy") as f:
                metadata = f.metadata() or {}
            config_dict = json.loads(metadata["config"])["audio_vae"]["model"][
                "params"
            ]["ddconfig"]
            weights = mx.load(str(model_path))
        else:
            config_dict = json.load(open(model_path / "config.json"))
            weights = mx.load(str(model_path / "model.safetensors"))
        config = AudioEncoderModelConfig.from_dict(config_dict)
        encoder = cls(config)
        if model_path.is_file():
            weights = encoder.sanitize(weights)
        encoder.load_weights(list(weights.items()), strict=True)
        return encoder

    def __call__(self, spectrogram: mx.array) -> mx.array:
        """Encode audio spectrogram into normalized latent representation.

        Args:
            spectrogram: (B, C, T, F) PyTorch format or (B, T, F, C) MLX format.
        Returns:
            Normalized latent (B, T', F', z_channels) in MLX channels-last format.
        """
        if spectrogram.ndim == 4 and spectrogram.shape[1] == self.in_channels:
            spectrogram = mx.transpose(spectrogram, (0, 2, 3, 1))

        h = self.conv_in(spectrogram)
        h = self._run_downsampling_path(h)
        h = run_mid_block(self.mid, h)
        h = self._finalize_output(h)
        return self._normalize_latents(h)

    def _run_downsampling_path(self, h: mx.array) -> mx.array:
        for level in range(self.num_resolutions):
            stage = self.down[level]
            for block_idx in range(self.num_res_blocks):
                h = stage["block"][block_idx](h, temb=None)
                if block_idx in stage["attn"]:
                    h = stage["attn"][block_idx](h)
            if level != self.num_resolutions - 1 and "downsample" in stage:
                h = stage["downsample"](h)
        return h

    def _finalize_output(self, h: mx.array) -> mx.array:
        h = self.norm_out(h)
        h = nn.silu(h)
        return self.conv_out(h)

    def _normalize_latents(self, h: mx.array) -> mx.array:
        """Normalize encoder output using per-channel statistics.

        Takes first half of channels (mean) when double_z=True,
        then patchifies, normalizes, and unpatchifies.
        """
        # h shape: (B, T', F', 2*z_channels) in MLX format
        z_channels = self.z_channels
        means = h[..., :z_channels]

        latent_shape = AudioLatentShape(
            batch=means.shape[0],
            channels=means.shape[3],
            frames=means.shape[1],
            mel_bins=means.shape[2],
        )

        patched = self.patchifier.patchify(means)
        normalized = self.per_channel_statistics.normalize(patched)
        return self.patchifier.unpatchify(normalized, latent_shape)


class AudioDecoder(nn.Module):
    """
    Symmetric decoder that reconstructs audio spectrograms from latent features.
    The decoder mirrors the encoder structure with configurable channel multipliers,
    attention resolutions, and causal convolutions.
    """

    def __init__(
        self,
        config: AudioDecoderModelConfig,
    ) -> None:
        """
        Initialize the AudioDecoder.
        Args:
            ch: Base number of feature channels
            out_ch: Number of output channels (2 for stereo)
            ch_mult: Multiplicative factors for channels at each resolution
            num_res_blocks: Number of residual blocks per resolution
            attn_resolutions: Resolutions at which to apply attention
            resolution: Input spatial resolution
            z_channels: Number of latent channels
            norm_type: Normalization type
            causality_axis: Axis for causal convolutions
            dropout: Dropout probability
            mid_block_add_attention: Whether to add attention in middle block
            sample_rate: Audio sample rate
            mel_hop_length: Hop length for mel spectrogram
            is_causal: Whether to use causal convolutions
            mel_bins: Number of mel frequency bins
        """
        super().__init__()

        # Per-channel statistics for denormalizing latents
        # Uses ch (base channel count) to match the patchified latent dimension
        # Input latent shape: (B, z_channels, T, latent_mel_bins) = (B, 8, T, 16)
        # After patchify: (B, T, z_channels * latent_mel_bins) = (B, T, 128)
        # ch=128 matches this dimension, so use ch for per_channel_statistics
        self.per_channel_statistics = PerChannelStatistics(latent_channels=config.ch)
        self.sample_rate = config.sample_rate
        self.mel_hop_length = config.mel_hop_length
        self.is_causal = config.is_causal
        self.mel_bins = config.mel_bins

        self.patchifier = AudioPatchifier(
            patch_size=1,
            audio_latent_downsample_factor=LATENT_DOWNSAMPLE_FACTOR,
            sample_rate=config.sample_rate,
            hop_length=config.mel_hop_length,
            is_causal=config.is_causal,
        )

        self.ch = config.ch
        self.temb_ch = 0
        self.num_resolutions = len(config.ch_mult)
        self.num_res_blocks = config.num_res_blocks
        self.resolution = config.resolution
        self.out_ch = config.out_ch
        self.give_pre_end = config.give_pre_end
        self.tanh_out = config.tanh_out
        self.norm_type = config.norm_type
        self.z_channels = config.z_channels
        self.channel_multipliers = config.ch_mult
        self.attn_resolutions = config.attn_resolutions
        self.causality_axis = config.causality_axis
        self.attn_type = config.attn_type

        base_block_channels = config.ch * self.channel_multipliers[-1]
        base_resolution = config.resolution // (2 ** (self.num_resolutions - 1))
        self.z_shape = (1, config.z_channels, base_resolution, base_resolution)

        self.conv_in = make_conv2d(
            config.z_channels,
            base_block_channels,
            kernel_size=3,
            stride=1,
            causality_axis=self.causality_axis,
        )

        self.mid = build_mid_block(
            channels=base_block_channels,
            temb_channels=self.temb_ch,
            dropout=config.dropout,
            norm_type=self.norm_type,
            causality_axis=self.causality_axis,
            attn_type=self.attn_type,
            add_attention=config.mid_block_add_attention,
        )

        self.up, final_block_channels = build_upsampling_path(
            ch=config.ch,
            ch_mult=config.ch_mult,
            num_resolutions=self.num_resolutions,
            num_res_blocks=config.num_res_blocks,
            resolution=config.resolution,
            temb_channels=self.temb_ch,
            dropout=config.dropout,
            norm_type=self.norm_type,
            causality_axis=self.causality_axis,
            attn_type=self.attn_type,
            attn_resolutions=config.attn_resolutions,
            resamp_with_conv=config.resamp_with_conv,
            initial_block_channels=base_block_channels,
        )

        self.norm_out = build_normalization_layer(
            final_block_channels, normtype=self.norm_type
        )
        self.conv_out = make_conv2d(
            final_block_channels,
            config.out_ch,
            kernel_size=3,
            stride=1,
            causality_axis=self.causality_axis,
        )

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Sanitize audio VAE weight names from PyTorch format to MLX format.

        Args:
            weights: Dictionary of weights with PyTorch naming

        Returns:
            Dictionary with MLX-compatible naming for audio VAE decoder
        """
        sanitized = {}

        for key, value in weights.items():
            new_key = key

            # Handle audio_vae.decoder weights
            if key.startswith("audio_vae.decoder."):
                new_key = key.replace("audio_vae.decoder.", "")
            elif key.startswith("audio_vae.per_channel_statistics."):
                # Map per-channel statistics
                if "mean-of-means" in key:
                    new_key = "per_channel_statistics.mean_of_means"
                elif "std-of-means" in key:
                    new_key = "per_channel_statistics.std_of_means"
                else:
                    continue  # Skip other statistics keys
            else:
                continue  # Skip non-decoder keys

            # Handle Conv2d weight shape conversion
            # PyTorch: (out_channels, in_channels, H, W)
            # MLX: (out_channels, H, W, in_channels)
            if "conv" in new_key.lower() and "weight" in new_key and value.ndim == 4:
                value = (
                    value
                    if check_array_shape(value)
                    else mx.transpose(value, (0, 2, 3, 1))
                )

            sanitized[new_key] = value

        return sanitized

    @classmethod
    def from_pretrained(cls, model_path: Path) -> "AudioDecoder":
        """Load audio VAE decoder from pretrained model."""
        import json
        from safetensors import safe_open

        from mlx_video.models.ltx_2.config import AudioDecoderModelConfig

        model_path = Path(model_path)
        if model_path.is_file():
            with safe_open(model_path, framework="numpy") as f:
                metadata = f.metadata() or {}
            config_dict = json.loads(metadata["config"])["audio_vae"]["model"][
                "params"
            ]["ddconfig"]
            weights = mx.load(str(model_path))
        else:
            config_dict = json.load(open(model_path / "config.json"))
            weights = mx.load(str(model_path / "model.safetensors"))
        config = AudioDecoderModelConfig.from_dict(config_dict)
        decoder = cls(config)
        if model_path.is_file():
            weights = decoder.sanitize(weights)
        decoder.load_weights(list(weights.items()), strict=True)
        return decoder

    def __call__(self, sample: mx.array) -> mx.array:
        """
        Decode latent features back to audio spectrograms.
        Args:
            sample: Encoded latent representation of shape (B, H, W, C) in MLX format
                    or (B, C, H, W) in PyTorch format (will be transposed)
        Returns:
            Reconstructed audio spectrogram
        """
        # Handle input format - if channels are in dim 1, transpose to channels-last
        if sample.shape[1] == self.z_channels and sample.ndim == 4:
            # PyTorch format (B, C, H, W) -> MLX format (B, H, W, C)
            sample = mx.transpose(sample, (0, 2, 3, 1))

        sample, target_shape = self._denormalize_latents(sample)

        h = self.conv_in(sample)
        h = run_mid_block(self.mid, h)
        h = self._run_upsampling_path(h)
        h = self._finalize_output(h)

        return self._adjust_output_shape(h, target_shape)

    def _denormalize_latents(
        self, sample: mx.array
    ) -> tuple[mx.array, AudioLatentShape]:
        """Denormalize latents using per-channel statistics."""
        # sample shape: (B, H, W, C) in MLX format
        latent_shape = AudioLatentShape(
            batch=sample.shape[0],
            channels=sample.shape[3],  # channels last
            frames=sample.shape[1],  # height = frames
            mel_bins=sample.shape[2],  # width = mel_bins
        )

        sample_patched = self.patchifier.patchify(sample)
        sample_denormalized = self.per_channel_statistics.un_normalize(sample_patched)
        sample = self.patchifier.unpatchify(sample_denormalized, latent_shape)

        target_frames = latent_shape.frames * LATENT_DOWNSAMPLE_FACTOR
        if self.causality_axis != CausalityAxis.NONE:
            target_frames = max(target_frames - (LATENT_DOWNSAMPLE_FACTOR - 1), 1)

        target_shape = AudioLatentShape(
            batch=latent_shape.batch,
            channels=self.out_ch,
            frames=target_frames,
            mel_bins=(
                self.mel_bins if self.mel_bins is not None else latent_shape.mel_bins
            ),
        )

        return sample, target_shape

    def _adjust_output_shape(
        self,
        decoded_output: mx.array,
        target_shape: AudioLatentShape,
    ) -> mx.array:
        """
        Adjust output shape to match target dimensions for variable-length audio.
        Args:
            decoded_output: Tensor of shape (B, H, W, C) in MLX format
            target_shape: AudioLatentShape describing target dimensions
        Returns:
            Tensor adjusted to match target_shape exactly
        """
        # Current output shape: (batch, frames, mel_bins, channels) in MLX format
        _, current_time, current_freq, _ = decoded_output.shape
        target_channels = target_shape.channels
        target_time = target_shape.frames
        target_freq = target_shape.mel_bins

        # Step 1: Crop first to avoid exceeding target dimensions
        decoded_output = decoded_output[
            :,
            : min(current_time, target_time),
            : min(current_freq, target_freq),
            :target_channels,
        ]

        # Step 2: Calculate padding needed for time and frequency dimensions
        time_padding_needed = target_time - decoded_output.shape[1]
        freq_padding_needed = target_freq - decoded_output.shape[2]

        # Step 3: Apply padding if needed
        if time_padding_needed > 0 or freq_padding_needed > 0:
            # MLX pad: [(before_0, after_0), ...]
            # For (B, H, W, C): H=time, W=freq
            padding = [
                (0, 0),  # batch
                (0, max(time_padding_needed, 0)),  # time
                (0, max(freq_padding_needed, 0)),  # freq
                (0, 0),  # channels
            ]
            decoded_output = mx.pad(decoded_output, padding)

        # Step 4: Final safety crop to ensure exact target shape
        decoded_output = decoded_output[:, :target_time, :target_freq, :target_channels]

        # Transpose back to PyTorch format (B, C, H, W) for vocoder compatibility
        decoded_output = mx.transpose(decoded_output, (0, 3, 1, 2))

        return decoded_output

    def _run_upsampling_path(self, h: mx.array) -> mx.array:
        """Run through upsampling path."""
        for level in reversed(range(self.num_resolutions)):
            stage = self.up[level]
            for block_idx in range(len(stage["block"])):
                h = stage["block"][block_idx](h, temb=None)
                if block_idx in stage["attn"]:
                    h = stage["attn"][block_idx](h)

            if level != 0 and "upsample" in stage:
                h = stage["upsample"](h)

        return h

    def _finalize_output(self, h: mx.array) -> mx.array:
        """Apply final normalization and convolution."""
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nn.silu(h)
        h = self.conv_out(h)
        return mx.tanh(h) if self.tanh_out else h


def decode_audio(
    latent: mx.array, audio_decoder: AudioDecoder, vocoder: "Vocoder"
) -> mx.array:
    """
    Decode an audio latent representation using the provided audio decoder and vocoder.
    Args:
        latent: Input audio latent tensor
        audio_decoder: Model to decode the latent to spectrogram
        vocoder: Model to convert spectrogram to audio waveform
    Returns:
        Decoded audio as a float tensor
    """
    decoded_audio = audio_decoder(latent)
    decoded_audio = vocoder(decoded_audio)
    # Remove batch dimension if present
    if decoded_audio.shape[0] == 1:
        decoded_audio = decoded_audio[0]
    return decoded_audio.astype(mx.float32)
