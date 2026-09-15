"""Vocoder for converting mel spectrograms to audio waveforms.

Supports:
- HiFi-GAN (LTX-2): ResBlock1 with LeakyReLU
- BigVGAN v2 (LTX-2.3): AMPBlock1 with Snake/SnakeBeta + anti-aliased resampling
- VocoderWithBWE (LTX-2.3): Base vocoder + bandwidth extension (16kHz -> 48kHz)
"""

import math
from pathlib import Path
from typing import Tuple

import mlx.core as mx
import mlx.nn as nn

from ..config import VocoderModelConfig
from .resnet import LRELU_SLOPE, ResBlock1, ResBlock2, leaky_relu


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


# ---------------------------------------------------------------------------
# Snake / SnakeBeta activations (BigVGAN v2)
# ---------------------------------------------------------------------------


class Snake(nn.Module):
    """Snake activation: x + (1/alpha) * sin^2(alpha * x)."""

    def __init__(self, in_features: int, alpha_logscale: bool = True) -> None:
        super().__init__()
        self.alpha_logscale = alpha_logscale
        self.alpha = (
            mx.zeros((in_features,)) if alpha_logscale else mx.ones((in_features,))
        )

    def __call__(self, x: mx.array) -> mx.array:
        # x: (N, L, C) in MLX format
        alpha = self.alpha  # (C,)
        if self.alpha_logscale:
            alpha = mx.exp(alpha)
        return x + (1.0 / (alpha + 1e-9)) * mx.power(mx.sin(x * alpha), 2)


class SnakeBeta(nn.Module):
    """SnakeBeta activation: x + (1/beta) * sin^2(alpha * x)."""

    def __init__(self, in_features: int, alpha_logscale: bool = True) -> None:
        super().__init__()
        self.alpha_logscale = alpha_logscale
        self.alpha = (
            mx.zeros((in_features,)) if alpha_logscale else mx.ones((in_features,))
        )
        self.beta = (
            mx.zeros((in_features,)) if alpha_logscale else mx.ones((in_features,))
        )

    def __call__(self, x: mx.array) -> mx.array:
        alpha = self.alpha
        beta = self.beta
        if self.alpha_logscale:
            alpha = mx.exp(alpha)
            beta = mx.exp(beta)
        return x + (1.0 / (beta + 1e-9)) * mx.power(mx.sin(x * alpha), 2)


# ---------------------------------------------------------------------------
# Anti-aliased resampling (Kaiser-sinc filters)
# ---------------------------------------------------------------------------


def _sinc(x: mx.array) -> mx.array:
    return mx.where(
        x == 0,
        mx.ones_like(x),
        mx.sin(mx.array(math.pi) * x) / (mx.array(math.pi) * x),
    )


def kaiser_sinc_filter1d(
    cutoff: float, half_width: float, kernel_size: int
) -> mx.array:
    """Compute a Kaiser-windowed sinc filter."""
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2
    delta_f = 4 * half_width
    amplitude = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if amplitude > 50.0:
        beta = 0.1102 * (amplitude - 8.7)
    elif amplitude >= 21.0:
        beta = 0.5842 * (amplitude - 21) ** 0.4 + 0.07886 * (amplitude - 21.0)
    else:
        beta = 0.0

    # Kaiser window - compute using scipy-compatible formula
    import numpy as np

    window = mx.array(np.kaiser(kernel_size, beta).astype(np.float32))

    if even:
        time = mx.arange(-half_size, half_size).astype(mx.float32) + 0.5
    else:
        time = mx.arange(kernel_size).astype(mx.float32) - half_size

    if cutoff == 0:
        filter_ = mx.zeros_like(time)
    else:
        filter_ = 2 * cutoff * window * _sinc(2 * cutoff * time)
        filter_ = filter_ / mx.sum(filter_)

    return filter_.reshape(1, 1, kernel_size)


def hann_sinc_filter1d(ratio: int) -> Tuple[mx.array, int, int, int]:
    """Compute a Hann-windowed sinc filter for upsampling (used by BWE resampler)."""
    import numpy as np

    rolloff = 0.99
    lowpass_filter_width = 6
    width = math.ceil(lowpass_filter_width / rolloff)
    kernel_size = 2 * width * ratio + 1
    pad = width
    pad_left = 2 * width * ratio
    pad_right = kernel_size - ratio

    time = (np.arange(kernel_size) / ratio - width) * rolloff
    time_clamped = np.clip(time, -lowpass_filter_width, lowpass_filter_width)
    window = np.cos(time_clamped * math.pi / lowpass_filter_width / 2) ** 2
    sinc_filter = np.sinc(time) * window * rolloff / ratio

    filter_ = mx.array(sinc_filter.astype(np.float32)).reshape(1, 1, kernel_size)
    return filter_, pad, pad_left, pad_right


class LowPassFilter1d(nn.Module):
    """Low-pass filter using depthwise convolution with Kaiser-sinc kernel."""

    def __init__(
        self,
        cutoff: float = 0.5,
        half_width: float = 0.6,
        stride: int = 1,
        kernel_size: int = 12,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        # Filter buffer - shape (1, 1, K) in PyTorch format, loaded from weights
        self.filter = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (N, L, C) in MLX format
        n, l, c = x.shape

        # Pad with edge values: replicate first/last value
        first = mx.repeat(x[:, :1, :], self.pad_left, axis=1)
        last = mx.repeat(x[:, -1:, :], self.pad_right, axis=1)
        x = mx.concatenate([first, x, last], axis=1)

        # Expand filter for depthwise conv: (1, 1, K) -> (C, K, 1) for groups=C
        # Filter is stored in PyTorch format (1, 1, K), need (C, K, 1) MLX format
        filt = self.filter.astype(x.dtype)  # (1, 1, K)
        filt = mx.transpose(filt, (0, 2, 1))  # (1, K, 1)
        filt = mx.repeat(filt, c, axis=0)  # (C, K, 1)

        # Transpose x for depthwise conv: (N, L, C) -> (N*C, L, 1) then conv
        x = mx.transpose(x, (0, 2, 1))  # (N, C, L)
        x = x.reshape(n * c, -1, 1)  # (N*C, L, 1)

        x = mx.conv1d(x, filt[:1], stride=self.stride, groups=1)  # (N*C, L', 1)

        x = x.reshape(n, c, -1)  # (N, C, L')
        x = mx.transpose(x, (0, 2, 1))  # (N, L', C)
        return x


class UpSample1d(nn.Module):
    """Anti-aliased upsampling using transposed convolution with sinc filter."""

    def __init__(
        self,
        ratio: int = 2,
        kernel_size: int = None,
        window_type: str = "kaiser",
    ) -> None:
        super().__init__()
        self.ratio = ratio
        self.stride = ratio

        if window_type == "hann":
            filt, self.pad, self.pad_left, self.pad_right = hann_sinc_filter1d(ratio)
            self.kernel_size = filt.shape[2]
            self.filter = filt
        else:
            self.kernel_size = (
                int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
            )
            self.pad = self.kernel_size // ratio - 1
            self.pad_left = (
                self.pad * self.stride + (self.kernel_size - self.stride) // 2
            )
            self.pad_right = (
                self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
            )
            self.filter = kaiser_sinc_filter1d(
                cutoff=0.5 / ratio,
                half_width=0.6 / ratio,
                kernel_size=self.kernel_size,
            )

    def __call__(self, x: mx.array) -> mx.array:
        # x: (N, L, C) in MLX format
        n, l, c = x.shape

        # Pad with edge values
        first = mx.repeat(x[:, :1, :], self.pad, axis=1)
        last = mx.repeat(x[:, -1:, :], self.pad, axis=1)
        x = mx.concatenate([first, x, last], axis=1)

        # Process per-channel via reshape: (N, L, C) -> (N*C, L, 1)
        x = mx.transpose(x, (0, 2, 1))  # (N, C, L)
        x = x.reshape(n * c, -1, 1)  # (N*C, L, 1)

        # Transposed conv for upsampling
        # Filter: (1, 1, K) PyTorch -> (1, K, 1) MLX format for conv_transpose1d
        filt = self.filter.astype(x.dtype)  # (1, 1, K)
        filt = mx.transpose(filt, (0, 2, 1))  # (1, K, 1)

        x = self.ratio * mx.conv_transpose1d(
            x, filt, stride=self.stride
        )  # (N*C, L', 1)

        # Trim padding
        x = x[:, self.pad_left : -self.pad_right, :]

        x = x.reshape(n, c, -1)  # (N, C, L')
        x = mx.transpose(x, (0, 2, 1))  # (N, L', C)
        return x


class DownSample1d(nn.Module):
    """Anti-aliased downsampling using low-pass filter."""

    def __init__(self, ratio: int = 2, kernel_size: int = None) -> None:
        super().__init__()
        self.ratio = ratio
        kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            stride=ratio,
            kernel_size=kernel_size,
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.lowpass(x)


class Activation1d(nn.Module):
    """Anti-aliased activation: upsample -> activate -> downsample."""

    def __init__(
        self,
        activation: nn.Module,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
    ) -> None:
        super().__init__()
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.upsample(x)
        x = self.act(x)
        return self.downsample(x)


# ---------------------------------------------------------------------------
# AMPBlock1 (BigVGAN v2 residual block)
# ---------------------------------------------------------------------------


class AMPBlock1(nn.Module):
    """BigVGAN v2 residual block with anti-aliased Snake activations."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: Tuple[int, int, int] = (1, 3, 5),
        activation: str = "snakebeta",
    ) -> None:
        super().__init__()
        act_cls = SnakeBeta if activation == "snakebeta" else Snake

        self.convs1 = {
            i: nn.Conv1d(
                channels,
                channels,
                kernel_size,
                stride=1,
                dilation=d,
                padding=get_padding(kernel_size, d),
            )
            for i, d in enumerate(dilation)
        }

        self.convs2 = {
            i: nn.Conv1d(
                channels,
                channels,
                kernel_size,
                stride=1,
                dilation=1,
                padding=get_padding(kernel_size, 1),
            )
            for i in range(len(dilation))
        }

        self.acts1 = {i: Activation1d(act_cls(channels)) for i in range(len(dilation))}
        self.acts2 = {i: Activation1d(act_cls(channels)) for i in range(len(dilation))}

    def __call__(self, x: mx.array) -> mx.array:
        for i in range(len(self.convs1)):
            xt = self.acts1[i](x)
            xt = self.convs1[i](xt)
            xt = self.acts2[i](xt)
            xt = self.convs2[i](xt)
            x = x + xt
        return x


# ---------------------------------------------------------------------------
# STFT and MelSTFT (for BWE)
# ---------------------------------------------------------------------------


class STFTFn(nn.Module):
    """STFT via conv1d with precomputed DFT x window bases (loaded from checkpoint)."""

    def __init__(self, filter_length: int, hop_length: int, win_length: int) -> None:
        super().__init__()
        self.hop_length = hop_length
        self.win_length = win_length
        n_freqs = filter_length // 2 + 1
        # Buffers loaded from checkpoint - PyTorch format (n_freqs*2, 1, filter_length)
        self.forward_basis = mx.zeros((n_freqs * 2, 1, filter_length))
        self.inverse_basis = mx.zeros((n_freqs * 2, 1, filter_length))

    def __call__(self, y: mx.array) -> Tuple[mx.array, mx.array]:
        """Compute magnitude and phase from waveform.

        Args:
            y: (B, T) waveform

        Returns:
            magnitude: (B, n_freqs, T_frames)
            phase: (B, n_freqs, T_frames)
        """
        if y.ndim == 2:
            y = mx.expand_dims(y, -1)  # (B, T, 1)

        left_pad = max(0, self.win_length - self.hop_length)
        if left_pad > 0:
            first = mx.repeat(y[:, :1, :], left_pad, axis=1)
            y = mx.concatenate([first, y], axis=1)

        # forward_basis: (514, 1, 512) PyTorch format -> (514, 512, 1) MLX
        basis = mx.transpose(
            self.forward_basis.astype(y.dtype), (0, 2, 1)
        )  # (514, K, 1)

        # Conv1d: (B, T, 1) * (514, K, 1) -> (B, T_frames, 514)
        spec = mx.conv1d(y, basis, stride=self.hop_length)

        # Split real and imaginary
        n_freqs = spec.shape[-1] // 2
        real = spec[..., :n_freqs]
        imag = spec[..., n_freqs:]

        magnitude = mx.sqrt(real**2 + imag**2)
        phase = mx.arctan2(imag.astype(mx.float32), real.astype(mx.float32)).astype(
            real.dtype
        )

        # Output: (B, T_frames, n_freqs) in MLX channels-last
        return magnitude, phase


class MelSTFT(nn.Module):
    """Causal log-mel spectrogram from precomputed STFT bases."""

    def __init__(
        self, filter_length: int, hop_length: int, win_length: int, n_mel_channels: int
    ) -> None:
        super().__init__()
        self.stft_fn = STFTFn(filter_length, hop_length, win_length)
        n_freqs = filter_length // 2 + 1
        self.mel_basis = mx.zeros((n_mel_channels, n_freqs))

    def mel_spectrogram(self, y: mx.array) -> mx.array:
        """Compute log-mel spectrogram.

        Args:
            y: (B, T) waveform

        Returns:
            log_mel: (B, n_mels, T_frames) in channels-first for compatibility
        """
        magnitude, phase = self.stft_fn(y)
        # magnitude: (B, T_frames, n_freqs)
        mel = (
            magnitude @ self.mel_basis.astype(magnitude.dtype).T
        )  # (B, T_frames, n_mels)
        log_mel = mx.log(mx.clip(mel, 1e-5, None))
        # Transpose to (B, n_mels, T_frames) for compatibility with vocoder input format
        return mx.transpose(log_mel, (0, 2, 1))


# ---------------------------------------------------------------------------
# Vocoder (supports both HiFi-GAN and BigVGAN v2)
# ---------------------------------------------------------------------------


class Vocoder(nn.Module):
    """Vocoder for mel-to-waveform synthesis.

    Supports resblock="1" (HiFi-GAN / LTX-2) and resblock="AMP1" (BigVGAN v2 / LTX-2.3).
    """

    def __init__(self, config: VocoderModelConfig) -> None:
        super().__init__()

        self.output_sampling_rate = config.output_sample_rate
        self.num_kernels = len(config.resblock_kernel_sizes)
        self.num_upsamples = len(config.upsample_rates)
        self.upsample_rates = config.upsample_rates
        self.is_amp = config.resblock == "AMP1"
        self.use_tanh_at_final = config.use_tanh_at_final
        self.apply_final_activation = config.apply_final_activation

        in_channels = 128 if config.stereo else 64
        self.conv_pre = nn.Conv1d(
            in_channels,
            config.upsample_initial_channel,
            kernel_size=7,
            stride=1,
            padding=3,
        )

        # Upsampling layers
        self.ups = {}
        for i, (stride, kernel_size) in enumerate(
            zip(config.upsample_rates, config.upsample_kernel_sizes)
        ):
            in_ch = config.upsample_initial_channel // (2**i)
            out_ch = config.upsample_initial_channel // (2 ** (i + 1))
            self.ups[i] = nn.ConvTranspose1d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - stride) // 2,
            )

        # Residual blocks
        if self.is_amp:
            self.resblocks = {}
            block_idx = 0
            for i in range(len(self.ups)):
                ch = config.upsample_initial_channel // (2 ** (i + 1))
                for kernel_size, dilations in zip(
                    config.resblock_kernel_sizes, config.resblock_dilation_sizes
                ):
                    self.resblocks[block_idx] = AMPBlock1(
                        ch,
                        kernel_size,
                        tuple(dilations),
                        activation=config.activation,
                    )
                    block_idx += 1
        else:
            resblock_class = ResBlock1 if config.resblock == "1" else ResBlock2
            self.resblocks = {}
            block_idx = 0
            for i in range(len(self.ups)):
                ch = config.upsample_initial_channel // (2 ** (i + 1))
                for kernel_size, dilations in zip(
                    config.resblock_kernel_sizes, config.resblock_dilation_sizes
                ):
                    self.resblocks[block_idx] = resblock_class(
                        ch, kernel_size, tuple(dilations)
                    )
                    block_idx += 1

        final_channels = config.upsample_initial_channel // (
            2 ** len(config.upsample_rates)
        )

        # Post-activation
        if self.is_amp:
            act_cls = SnakeBeta if config.activation == "snakebeta" else Snake
            self.act_post = Activation1d(act_cls(final_channels))

        # Final conv
        out_channels = 2 if config.stereo else 1
        self.conv_post = nn.Conv1d(
            final_channels,
            out_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            bias=config.use_bias_at_final,
        )

        self.upsample_factor = math.prod(config.upsample_rates)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Mel spectrogram (B, C, T, mel_bins) for stereo or (B, T, mel_bins) mono.

        Returns:
            Waveform (B, out_channels, T_audio) in channels-first format.
        """
        # (B, C, T, mel) -> (B, C, mel, T)
        x = mx.transpose(x, (0, 1, 3, 2))

        if x.ndim == 4:  # stereo: (B, 2, mel, T) -> (B, 2*mel, T)
            b, s, c, t = x.shape
            x = x.reshape(b, s * c, t)

        # Channels-first (B, C, T) -> channels-last (B, T, C) for MLX conv
        x = mx.transpose(x, (0, 2, 1))

        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            if not self.is_amp:
                x = leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)

            start = i * self.num_kernels
            end = start + self.num_kernels

            block_outputs = mx.stack(
                [self.resblocks[idx](x) for idx in range(start, end)],
                axis=0,
            )
            x = mx.mean(block_outputs, axis=0)

        if self.is_amp:
            x = self.act_post(x)
        else:
            x = nn.leaky_relu(x)

        x = self.conv_post(x)

        if self.apply_final_activation:
            x = mx.tanh(x) if self.use_tanh_at_final else mx.clip(x, -1, 1)

        # Back to channels-first (B, T, C) -> (B, C, T)
        x = mx.transpose(x, (0, 2, 1))
        return x


# ---------------------------------------------------------------------------
# VocoderWithBWE (Bandwidth Extension)
# ---------------------------------------------------------------------------


class VocoderWithBWE(nn.Module):
    """Vocoder + bandwidth extension upsampling (16kHz -> 48kHz).

    Chains a base vocoder with a BWE generator that predicts a residual
    added to a sinc-resampled skip connection.
    """

    def __init__(
        self,
        vocoder: Vocoder,
        bwe_generator: Vocoder,
        mel_stft: MelSTFT,
        input_sampling_rate: int = 16000,
        output_sampling_rate: int = 48000,
        hop_length: int = 80,
    ) -> None:
        super().__init__()
        self.vocoder = vocoder
        self.bwe_generator = bwe_generator
        self.mel_stft = mel_stft
        self.input_sampling_rate = input_sampling_rate
        self.output_sampling_rate = output_sampling_rate
        self.hop_length = hop_length
        # Hann-windowed sinc resampler (not stored in checkpoint)
        self.resampler = UpSample1d(
            ratio=output_sampling_rate // input_sampling_rate,
            window_type="hann",
        )

    @property
    def output_sample_rate(self) -> int:
        return self.output_sampling_rate

    def _compute_mel(self, audio: mx.array) -> mx.array:
        """Compute log-mel spectrogram from waveform.

        Args:
            audio: (B, C, T) waveform in channels-first

        Returns:
            mel: (B, C, n_mels, T_frames)
        """
        batch, n_channels, _ = audio.shape
        flat = audio.reshape(batch * n_channels, -1)  # (B*C, T)
        mel = self.mel_stft.mel_spectrogram(flat)  # (B*C, n_mels, T_frames)
        return mel.reshape(batch, n_channels, mel.shape[1], mel.shape[2])

    def __call__(self, mel_spec: mx.array) -> mx.array:
        """Run vocoder + BWE.

        Args:
            mel_spec: Mel spectrogram, same format as Vocoder.forward input.

        Returns:
            Waveform (B, out_channels, T_audio) at output_sampling_rate.
        """
        x = self.vocoder(mel_spec)  # (B, C, T) at input_sampling_rate
        _, _, length_low_rate = x.shape
        output_length = (
            length_low_rate * self.output_sampling_rate // self.input_sampling_rate
        )

        # Pad to hop_length multiple
        remainder = length_low_rate % self.hop_length
        if remainder != 0:
            pad_amount = self.hop_length - remainder
            x = mx.pad(x, [(0, 0), (0, 0), (0, pad_amount)])

        # Compute mel from vocoder output: (B, C, n_mels, T_frames)
        mel = self._compute_mel(x)

        # BWE expects (B, C, T_frames, mel_bins) -> transpose last two dims
        mel_for_bwe = mx.transpose(mel, (0, 1, 3, 2))  # (B, C, T_frames, n_mels)
        residual = self.bwe_generator(mel_for_bwe)  # (B, C, T_high)

        # Sinc upsample skip connection
        # resampler expects (N, L, C): transpose from (B, C, T) -> (B, T, C)
        x_for_resample = mx.transpose(x, (0, 2, 1))
        skip = self.resampler(x_for_resample)
        skip = mx.transpose(skip, (0, 2, 1))  # back to (B, C, T)

        return mx.clip(residual + skip, -1, 1)[..., :output_length]


# ---------------------------------------------------------------------------
# Factory / from_pretrained
# ---------------------------------------------------------------------------


def sanitize_split_vocoder_weights(weights: dict) -> dict:
    """Normalize the vocoder section of an LTX-2.5 audio checkpoint."""
    sanitized = {}
    for key, value in weights.items():
        if not key.startswith("vocoder."):
            continue
        new_key = key.removeprefix("vocoder.")
        if new_key.endswith(".weight") and value.ndim == 3:
            value = (
                mx.transpose(value, (1, 2, 0))
                if ".ups." in new_key
                else mx.transpose(value, (0, 2, 1))
            )
        sanitized[new_key] = value
    return sanitized


def load_vocoder(model_path: Path) -> nn.Module:
    """Load vocoder from pretrained model directory.

    Automatically detects whether to load a simple Vocoder or VocoderWithBWE.
    """
    import json
    from safetensors import safe_open

    model_path = Path(model_path)
    if model_path.is_file():
        with safe_open(model_path, framework="numpy") as f:
            metadata = f.metadata() or {}
        config_dict = json.loads(metadata["config"])["vocoder"]
        weights = sanitize_split_vocoder_weights(mx.load(str(model_path)))
        has_bwe = "bwe" in config_dict
    else:
        config_path = model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"No config.json found in {model_path}")
        with open(config_path) as f:
            config_dict = json.load(f)
        weights = mx.load(str(model_path / "model.safetensors"))
        has_bwe = config_dict.get("has_bwe_generator", False)

    if has_bwe:
        return _load_vocoder_with_bwe(config_dict, weights)
    else:
        config = VocoderModelConfig.from_dict(config_dict)
        model = Vocoder(config)
        model.load_weights(list(weights.items()), strict=True)
        return model


def _load_vocoder_with_bwe(config_dict: dict, weights: dict) -> VocoderWithBWE:
    """Load VocoderWithBWE from config and weights."""
    # Build vocoder from config
    vocoder_cfg = config_dict.get("vocoder", {})
    vocoder_config = VocoderModelConfig.from_dict(vocoder_cfg)
    vocoder = Vocoder(vocoder_config)

    # Build BWE generator from config
    bwe_cfg = config_dict.get("bwe", {})
    bwe_config = VocoderModelConfig.from_dict(bwe_cfg)
    bwe_config.apply_final_activation = False
    bwe_generator = Vocoder(bwe_config)

    # MelSTFT from weight shapes
    stft_basis = weights.get("mel_stft.stft_fn.forward_basis")
    filter_length = stft_basis.shape[2] if stft_basis is not None else 512
    mel_basis = weights.get("mel_stft.mel_basis")
    n_mel_channels = mel_basis.shape[0] if mel_basis is not None else 64

    hop_length = bwe_cfg.get("hop_length", 80)
    input_sr = bwe_cfg.get("input_sampling_rate", 16000)
    output_sr = bwe_cfg.get("output_sampling_rate", 48000)

    mel_stft = MelSTFT(
        filter_length=filter_length,
        hop_length=hop_length,
        win_length=filter_length,
        n_mel_channels=n_mel_channels,
    )

    model = VocoderWithBWE(
        vocoder=vocoder,
        bwe_generator=bwe_generator,
        mel_stft=mel_stft,
        input_sampling_rate=input_sr,
        output_sampling_rate=output_sr,
        hop_length=hop_length,
    )

    model.load_weights(list(weights.items()), strict=False)
    return model
