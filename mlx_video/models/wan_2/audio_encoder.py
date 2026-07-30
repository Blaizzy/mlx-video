"""Audio-conditioning encoder for Wan 2.2 S2V (Phase 2 real forward).

PyTorch reference (unavailable in sandbox — implemented from design doc §2.2
and the released ``Wan-AI/Wan2.2-S2V-14B`` state-dict layout):

    wan/modules/s2v/audio_utils.py :: CausalAudioEncoder
    wan/modules/s2v/auxi_blocks.py :: MotionEncoder_tc, CausalConv1d

State-dict layout (unchanged from PyTorch — upstream typo ``casual`` preserved):

    casual_audio_encoder.weights                       (1, 25, 1, 1)
    casual_audio_encoder.encoder.conv1_local.conv.w    (hd/4 * num_heads, 1024, 3)
    casual_audio_encoder.encoder.conv1_local.conv.b    (hd/4 * num_heads,)
    casual_audio_encoder.encoder.conv1_global.conv.w   (hd/4, 1024, 3)
    casual_audio_encoder.encoder.conv1_global.conv.b   (hd/4,)
    casual_audio_encoder.encoder.conv2.conv.w          (hd/2, hd/4, 3)      shared local/global
    casual_audio_encoder.encoder.conv2.conv.b          (hd/2,)
    casual_audio_encoder.encoder.conv3.conv.w          (hd,   hd/2, 3)      shared local/global
    casual_audio_encoder.encoder.conv3.conv.b          (hd,)
    casual_audio_encoder.encoder.final_linear.weight   (hd, hd)             global path
    casual_audio_encoder.encoder.final_linear.bias     (hd,)
    casual_audio_encoder.encoder.padding_tokens        (1, 1, 1, hd)        learnable pad token

Compute API assumption (TODO(verify) once reference source becomes available):

  * Two stride-2 convolutions reduce ``T_audio`` by a factor of 4.
  * For the released ``num_audio_token=4`` config, the caller feeds
    ``T_audio = num_audio_token * F_video`` so that after the stride-4 downsample
    each video frame has ``num_audio_token`` local audio tokens.
  * SiLU activation between conv layers (matches diffusers convention; the
    exact activation is not confirmed).
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# Low-level: CausalConv1d
# ---------------------------------------------------------------------------


class CausalConv1d(nn.Module):
    """Left-padded Conv1d that stores weights in PyTorch layout.

    Weight layout: ``(out_ch, in_ch, kernel)`` — matches
    ``casual_audio_encoder.encoder.conv*.conv.weight``. Forward transposes to
    MLX layout ``(out_ch, kernel, in_ch)`` for ``mx.conv1d``.

    Input:  ``(B, T, C_in)``
    Output: ``(B, T_out, C_out)`` where ``T_out == T // stride``.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        # PyTorch layout so state-dict loads directly.
        self.weight = mx.zeros((out_ch, in_ch, kernel_size), dtype=mx.float32)
        self.bias = mx.zeros((out_ch,), dtype=mx.float32)

    def __call__(self, x: mx.array, stride: int = 1) -> mx.array:
        # Left-pad along temporal axis for causal padding.
        pad = self.kernel_size - 1
        if pad > 0:
            pad_tensor = mx.zeros((x.shape[0], pad, x.shape[2]), dtype=x.dtype)
            x = mx.concatenate([pad_tensor, x], axis=1)
        # Transpose (out, in, k) -> (out, k, in) for mx.conv1d.
        w = self.weight.astype(x.dtype).transpose(0, 2, 1)
        y = mx.conv1d(x, w, stride=stride, padding=0)
        return y + self.bias.astype(y.dtype)


class _CausalConv1dModule(nn.Module):
    """Wraps CausalConv1d in a ``.conv`` sub-namespace to mirror PyTorch nesting
    (``casual_audio_encoder.encoder.conv1_local.conv.weight`` etc.)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        self.conv = CausalConv1d(in_ch, out_ch, kernel_size)

    def __call__(self, x: mx.array, stride: int = 1) -> mx.array:
        return self.conv(x, stride=stride)


# ---------------------------------------------------------------------------
# Mid-level: MotionEncoder_tc (3-conv stack + optional global path)
# ---------------------------------------------------------------------------


class MotionEncoder_tc(nn.Module):
    """Temporal audio-feature encoder used by CausalAudioEncoder.

    Args:
        in_dim: audio feature dim (1024 for wav2vec2-large-xls-r).
        hidden_dim: video model dim (5120 for S2V-14B).
        num_heads: audio tokens per video frame (4 in released S2V).
        need_global: build the global path (final_linear).

    Forward:
        x: (B, in_dim, T_audio)
        Returns:
            local:  (B, F_video, num_heads, hidden_dim)
            global: (B, F_video, 1, hidden_dim) if need_global else None
        where F_video = T_audio // 4 (two stride-2 convs).

    TODO(verify): activation between convs (assumed SiLU).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_heads: int,
        need_global: bool = True,
    ):
        super().__init__()
        assert hidden_dim % 4 == 0, "hidden_dim must be divisible by 4"
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.need_global = need_global

        # Local path: expand to num_heads * (hidden_dim // 4) channels.
        self.conv1_local = _CausalConv1dModule(
            in_dim, hidden_dim // 4 * num_heads, kernel_size=3
        )
        # Shared conv2/conv3 for both paths (per state-dict — only one of each).
        self.conv2 = _CausalConv1dModule(hidden_dim // 4, hidden_dim // 2, kernel_size=3)
        self.conv3 = _CausalConv1dModule(hidden_dim // 2, hidden_dim, kernel_size=3)

        # Activation between convs. TODO(verify): reference uses SiLU or GELU.
        self.act = nn.SiLU()

        if need_global:
            self.conv1_global = _CausalConv1dModule(
                in_dim, hidden_dim // 4, kernel_size=3
            )
            self.final_linear = nn.Linear(hidden_dim, hidden_dim)

        # Learnable padding token appended to the local output when the caller
        # needs an extra "silence" slot at the end of a clip. Kept as raw
        # parameter to match state-dict key exactly.
        self.padding_tokens = mx.zeros((1, 1, 1, hidden_dim), dtype=mx.float32)

    def _run_local(self, x_t: mx.array) -> mx.array:
        """Compute local audio tokens.

        x_t: (B, T_audio, in_dim)
        Returns: (B, F_video, num_heads, hidden_dim)
        """
        B, T, _ = x_t.shape
        hd = self.hidden_dim
        nh = self.num_heads
        # conv1_local outputs (B, T, num_heads * hd/4).
        y = self.conv1_local(x_t, stride=1)
        y = self.act(y)
        # Split num_heads into a batch-dim so conv2/conv3 act per-head.
        # (B, T, num_heads, hd/4)
        y = y.reshape(B, T, nh, hd // 4)
        # (B, num_heads, T, hd/4) -> (B*num_heads, T, hd/4)
        y = y.transpose(0, 2, 1, 3).reshape(B * nh, T, hd // 4)
        y = self.conv2(y, stride=2)  # (B*nh, T/2, hd/2)
        y = self.act(y)
        y = self.conv3(y, stride=2)  # (B*nh, T/4, hd)
        # Back to (B, F, num_heads, hd)
        F_out = y.shape[1]
        y = y.reshape(B, nh, F_out, hd).transpose(0, 2, 1, 3)
        return y

    def _run_global(self, x_t: mx.array) -> mx.array:
        """Compute global audio embedding.

        x_t: (B, T_audio, in_dim)
        Returns: (B, F_video, 1, hidden_dim)
        """
        B, T, _ = x_t.shape
        hd = self.hidden_dim
        y = self.conv1_global(x_t, stride=1)  # (B, T, hd/4)
        y = self.act(y)
        y = self.conv2(y, stride=2)  # (B, T/2, hd/2)
        y = self.act(y)
        y = self.conv3(y, stride=2)  # (B, T/4, hd)
        y = self.final_linear(y)  # (B, T/4, hd)
        return y[:, :, None, :]  # (B, F, 1, hd)

    def __call__(self, x: mx.array):
        # x: (B, in_dim, T_audio) — Conv1d-friendly channels-first layout in PT.
        # Transpose to (B, T_audio, in_dim) for MLX Conv1d.
        x_t = x.transpose(0, 2, 1)
        local = self._run_local(x_t)
        if self.need_global:
            glob = self._run_global(x_t)
            return local, glob
        return local


# ---------------------------------------------------------------------------
# Top-level: CausalAudioEncoder (weighted layer-sum + MotionEncoder_tc)
# ---------------------------------------------------------------------------


class CausalAudioEncoder(nn.Module):
    """Wav2vec2 hidden-state layer weighting + MotionEncoder_tc.

    Args:
        dim: audio feature dim (1024).
        num_layers: number of wav2vec2 hidden states summed (25 for xls-r).
        out_dim: video model dim (5120 for S2V-14B).
        num_token: audio tokens per video frame (4 for S2V-14B).
        need_global: also produce the global (AdaIN) embedding.

    Forward:
        features: (B, num_layers, dim, T_audio)
        Returns:
            local:  (B, F_video, num_token, out_dim)
            global: (B, F_video, 1, out_dim)  if need_global
    """

    def __init__(
        self,
        dim: int = 1024,
        num_layers: int = 25,
        out_dim: int = 5120,
        num_token: int = 4,
        need_global: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.num_token = num_token
        self.need_global = need_global

        # Learnable per-layer weighting. Initialised as-if uniform; the
        # softmax below turns any values into a probability distribution.
        self.weights = mx.full((1, num_layers, 1, 1), 0.01, dtype=mx.float32)
        self.encoder = MotionEncoder_tc(
            in_dim=dim,
            hidden_dim=out_dim,
            num_heads=num_token,
            need_global=need_global,
        )

    def __call__(self, features: mx.array):
        # features: (B, num_layers, dim, T_audio)
        # Softmax the layer weights along axis=1 to normalise contribution.
        w = mx.softmax(self.weights.astype(mx.float32), axis=1)
        # Weighted sum -> (B, dim, T_audio)
        x = (features.astype(mx.float32) * w).sum(axis=1)
        return self.encoder(x)


# ---------------------------------------------------------------------------
# Helper: wav2vec2 feature extraction (runs on the Mac, not in sandbox).
# ---------------------------------------------------------------------------


def extract_wav2vec_features(
    wav_path: str,
    num_video_frames: int,
    num_audio_token: int = 4,
    model_name: str = "facebook/wav2vec2-large-xlsr-53",
    target_sr: int = 16000,
) -> mx.array:
    """Extract wav2vec2 hidden states and align to the video frame axis.

    Runs a HuggingFace ``Wav2Vec2Model`` on CPU with ``output_hidden_states=True``,
    stacks the 25 hidden states into a single tensor, and interpolates the
    temporal axis to ``num_audio_token * num_video_frames`` so the downstream
    ``CausalAudioEncoder`` produces exactly ``num_video_frames`` output frames.

    Args:
        wav_path: path to a mono .wav file.
        num_video_frames: number of video frames the S2V model will generate.
        num_audio_token: audio tokens per video frame (4 for released S2V-14B).
        model_name: HF wav2vec2 checkpoint. TODO(verify): the S2V paper uses
            ``facebook/wav2vec2-large-xlsr-53`` (25 hidden states). Alternate
            candidates: ``TencentGameMate/chinese-wav2vec2-large`` for Chinese.
        target_sr: sample rate the wav2vec2 model expects (16 kHz).

    Returns:
        ``mx.array`` of shape ``(1, num_hidden_states, feature_dim, T_target)``
        where ``T_target = num_audio_token * num_video_frames`` and
        ``num_hidden_states`` typically equals 25 (24 transformer layers + 1
        conv output, matching the ``weights`` tensor of shape (1, 25, 1, 1)).

    Not runnable in the CI sandbox (no ``transformers`` / ``torch``, no audio
    files). Call this from the Mac side just before invoking S2V inference.
    """
    try:
        import numpy as np
        import torch
        import torch.nn.functional as F
        from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
    except ImportError as e:
        raise ImportError(
            "extract_wav2vec_features requires torch + transformers "
            "(install with `pip install torch transformers`)."
        ) from e

    try:
        import soundfile as sf
    except ImportError:
        try:
            import librosa  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Install `soundfile` or `librosa` to read audio files."
            ) from e
        wav, sr = librosa.load(wav_path, sr=target_sr, mono=True)
    else:
        wav, sr = sf.read(wav_path)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)  # mono
        if sr != target_sr:
            import librosa  # type: ignore

            wav = librosa.resample(wav.astype("float32"), orig_sr=sr, target_sr=target_sr)
            sr = target_sr

    fe = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name).eval()

    with torch.no_grad():
        inputs = fe(wav, sampling_rate=target_sr, return_tensors="pt")
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        # outputs.hidden_states: tuple of 25 tensors, each (1, T_wav, D)
        hs = torch.stack(outputs.hidden_states, dim=1)  # (1, 25, T_wav, D)
        # Reshape to (1, 25, D, T_wav) for interp along the time axis.
        hs = hs.permute(0, 1, 3, 2)  # (1, 25, D, T_wav)

        T_target = num_audio_token * num_video_frames
        # F.interpolate expects (N, C, L) — collapse the layer axis into channels.
        B, L, D, T_wav = hs.shape
        hs_flat = hs.reshape(B, L * D, T_wav)
        hs_flat = F.interpolate(hs_flat, size=T_target, mode="linear", align_corners=False)
        hs = hs_flat.reshape(B, L, D, T_target)

    return mx.array(hs.cpu().numpy())
