"""S2V-specific building blocks (Phase 2 real forward): AdaLN, AudioInjector,
FramePacker, CondEncoderProj, TrainableCondMask.

PyTorch reference (unavailable in sandbox — implemented from design doc §3
and the released ``Wan-AI/Wan2.2-S2V-14B`` state-dict layout):

    wan/modules/s2v/audio_utils.py :: AudioInjector_WAN, AudioCrossAttention
    wan/modules/s2v/motioner.py    :: FramePackMotioner
    diffusers.models.attention      :: AdaLayerNorm
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn

from .attention import WanCrossAttention, WanRMSNorm, _linear_dtype


# ---------------------------------------------------------------------------
# AdaLayerNorm — diffusers-style scale/shift modulation
# ---------------------------------------------------------------------------


class AdaLayerNorm(nn.Module):
    """Diffusers-style AdaLN with no internal LayerNorm.

    Weight layout (matches state-dict key ``.linear.weight/bias``):
        linear.weight: (2 * output_dim, embedding_dim)
        linear.bias:   (2 * output_dim,)

    Forward:
        x:    (..., output_dim)   already-normalised hidden states
        temb: (..., embedding_dim) conditioning embedding

    Returns:
        x * (1 + scale) + shift    where (scale, shift) = chunk(linear(SiLU(temb)))

    Design doc R5 explicitly says the released S2V variant uses no interior
    LayerNorm — it is applied *after* the parent module's attn norm
    (``adain_mode == "attn_norm"``).

    TODO(verify): whether the reference applies SiLU on temb before the linear.
    Diffusers ``AdaLayerNorm`` does; if the S2V variant skips it, remove the
    ``nn.silu`` call below.
    """

    def __init__(self, embedding_dim: int, output_dim: int, chunk_dim: int = 1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.chunk_dim = chunk_dim
        self.linear = nn.Linear(embedding_dim, 2 * output_dim)

    def __call__(self, x: mx.array, temb: mx.array) -> mx.array:
        # SiLU(temb) → Linear → (scale, shift)
        w_dtype = _linear_dtype(self.linear)
        temb_a = nn.silu(temb.astype(w_dtype))
        scale_shift = self.linear(temb_a)
        # chunk along last dim into (scale, shift), each of shape (..., output_dim)
        half = self.output_dim
        scale = scale_shift[..., :half]
        shift = scale_shift[..., half:]
        # Broadcast-safe: scale/shift may have fewer trailing tokens than x.
        return x * (1 + scale.astype(x.dtype)) + shift.astype(x.dtype)


# ---------------------------------------------------------------------------
# AudioCrossAttention — same signature as WanCrossAttention, no RoPE
# ---------------------------------------------------------------------------


class AudioCrossAttention(WanCrossAttention):
    """Cross-attention with Q from video, K/V from audio.

    Inherits :class:`WanCrossAttention` verbatim: identical q/k/v/o Linear and
    norm_q/norm_k shapes, and the base class's forward already does
    scaled-dot-product attention with *no RoPE on cross-attn* (design R2).
    We keep the class distinct only for state-dict namespacing.
    """

    # Forward inherited from WanCrossAttention:
    #     __call__(x, context, context_lens=None, kv_cache=None)


# ---------------------------------------------------------------------------
# AudioInjector — 12 cross-attn + AdaLN injectors, injected after selected blocks
# ---------------------------------------------------------------------------


class AudioInjector(nn.Module):
    """Container for the S2V audio-injection sublayers.

    Loaded keys (12 injectors, N=0..11 for released S2V-14B):
        audio_injector.injector.<N>.{q,k,v,o}.{weight,bias}
        audio_injector.injector.<N>.norm_{q,k}.weight
        audio_injector.injector_adain_layers.<N>.linear.{weight,bias}

    Insertion happens *after* the standard transformer block; each injected
    layer applies:
        1. Extract per-frame video slice: (B, F, N_tok, D)
        2. AdaLN modulate with audio_emb_global
        3. Cross-attend (Q=video, K/V=audio_emb)
        4. Residual-add back to the video slice, leaving ref/motion untouched.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        inject_layers: Sequence[int],
        enable_adain: bool = True,
        adain_dim: int | None = None,
        qk_norm: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.inject_layers = tuple(inject_layers)
        self.enable_adain = enable_adain
        self.adain_dim = adain_dim or dim

        # Map from transformer-block idx -> injector idx (0..len(inject_layers)-1).
        self.injected_block_id = {
            block_idx: i for i, block_idx in enumerate(self.inject_layers)
        }
        n = len(self.inject_layers)

        self.injector = [
            AudioCrossAttention(dim, num_heads, qk_norm=qk_norm, eps=eps)
            for _ in range(n)
        ]
        if enable_adain:
            self.injector_adain_layers = [
                AdaLayerNorm(embedding_dim=self.adain_dim, output_dim=dim, chunk_dim=1)
                for _ in range(n)
            ]

    def inject(
        self,
        hidden_states: mx.array,
        block_idx: int,
        audio_emb: mx.array,
        audio_emb_global: Optional[mx.array],
        F: int,
        N: int,
        seq_orig: int,
    ) -> mx.array:
        """Apply audio cross-attention + AdaLN residual to the video slice.

        Args:
            hidden_states: (B, seq_total, D). ``seq_total = seq_orig + ref + motion``.
            block_idx: current transformer block index; must be in
                ``self.injected_block_id`` else this is a no-op.
            audio_emb: (B, F, num_audio_token, D) local audio tokens.
            audio_emb_global: (B, F, 1, D) or None. Used for AdaLN.
            F: number of video frames.
            N: number of tokens per video frame.
            seq_orig: number of "noise" tokens = F * N (excluding ref/motion).

        Returns:
            hidden_states with the first ``seq_orig`` slice updated.
        """
        if block_idx not in self.injected_block_id:
            return hidden_states
        inj_idx = self.injected_block_id[block_idx]

        B, seq_total, D = hidden_states.shape
        assert seq_orig == F * N, (
            f"seq_orig ({seq_orig}) must equal F*N ({F}*{N}={F*N})"
        )
        # Extract video slice and reshape to (B*F, N, D) for per-frame attn.
        video_slice = hidden_states[:, :seq_orig, :]
        video_frame = video_slice.reshape(B, F, N, D).reshape(B * F, N, D)

        x = video_frame
        if self.enable_adain and audio_emb_global is not None:
            # audio_emb_global: (B, F, 1, D) -> (B*F, 1, D)
            temb = audio_emb_global.reshape(B * F, 1, D)
            adain = self.injector_adain_layers[inj_idx]
            x = adain(x, temb)

        # Cross-attention: (B*F, N, D) x (B*F, num_audio_token, D) -> (B*F, N, D)
        num_tok = audio_emb.shape[2]
        audio_ctx = audio_emb.reshape(B * F, num_tok, D)
        injector = self.injector[inj_idx]
        # WanCrossAttention.__call__(x, context, context_lens=None, kv_cache=None)
        residual = injector(x, audio_ctx)

        # Rearrange back and residual-add into the video slice only.
        residual = residual.reshape(B, F * N, D)
        new_video = video_slice + residual.astype(video_slice.dtype)
        if seq_total > seq_orig:
            tail = hidden_states[:, seq_orig:, :]
            return mx.concatenate([new_video, tail], axis=1)
        return new_video


# ---------------------------------------------------------------------------
# Conv3d-as-Linear helper (weight stored in raw PyTorch (O, I, D, H, W) layout).
# ---------------------------------------------------------------------------


class _Conv3dLinear(nn.Module):
    """Conv3d with stride == kernel treated as a per-patch Linear projection.

    Weight is stored in raw PyTorch layout (O, I, D, H, W) so state-dict loads
    directly without a sanitizer transpose. Forward reshapes the input to
    patches and multiplies by the flattened kernel.
    """

    def __init__(self, out_ch: int, in_ch: int, kernel: Sequence[int]):
        super().__init__()
        self.out_ch = out_ch
        self.in_ch = in_ch
        self.kernel = tuple(kernel)
        self.weight = mx.zeros((out_ch, in_ch, *self.kernel), dtype=mx.float32)
        self.bias = mx.zeros((out_ch,), dtype=mx.float32)

    def _patchify(self, x: mx.array):
        """Split (B, C, D, H, W) into non-overlapping (kd, kh, kw) patches.

        Returns:
            patches: (B, D', H', W', C * kd * kh * kw)
            grid: (D', H', W')
        """
        B, C, D, H, W = x.shape
        kd, kh, kw = self.kernel
        assert D % kd == 0 and H % kh == 0 and W % kw == 0, (
            f"input {(D,H,W)} not divisible by kernel {(kd,kh,kw)}"
        )
        d_out, h_out, w_out = D // kd, H // kh, W // kw
        x = x.reshape(B, C, d_out, kd, h_out, kh, w_out, kw)
        # (B, D', H', W', C, kd, kh, kw)
        x = x.transpose(0, 2, 4, 6, 1, 3, 5, 7)
        x = x.reshape(B, d_out, h_out, w_out, C * kd * kh * kw)
        return x, (d_out, h_out, w_out)

    def __call__(self, x: mx.array):
        """
        Args:
            x: (B, C_in, D, H, W)
        Returns:
            (B, D', H', W', C_out) where D' = D // kernel[0] etc.
        """
        patches, grid = self._patchify(x)  # (B, D', H', W', C*kd*kh*kw)
        # Flatten kernel weight: (O, I*kd*kh*kw)
        w = self.weight.astype(x.dtype).reshape(self.out_ch, -1)
        # Matmul: (..., C*k*k*k) @ (C*k*k*k, O)
        y = patches @ w.T + self.bias.astype(x.dtype)
        return y  # (B, D', H', W', C_out)


# ---------------------------------------------------------------------------
# FramePacker — 3-scale motion-history projection
# ---------------------------------------------------------------------------


class FramePacker(nn.Module):
    """Frame-Pack motion history compression (fine, medium, coarse temporal scales).

    Weight keys (released S2V-14B):
        frame_packer.proj.weight     (dim, 16, 1, 2, 2)   Conv3d fine
        frame_packer.proj_2x.weight  (dim, 16, 2, 4, 4)   Conv3d medium
        frame_packer.proj_4x.weight  (dim, 16, 4, 8, 8)   Conv3d coarse

    ``zip_frame_buckets = (1, 2, 16)`` means the last 1 latent frame is packed
    at fine scale, the preceding 2 at medium (2x downsample), and the earliest
    16 at coarse (4x downsample). Total motion-history length is 19 latent
    frames; each temporal token grid is projected independently and the
    resulting token lists are returned as three flat sequences.

    TODO(verify): the exact bucket ordering and drop-mode semantics
    (``framepack_drop_mode == "padd"`` — pad instead of drop when history is
    shorter than 19 latent frames). This implementation zero-pads the missing
    prefix and applies all three projections.
    """

    def __init__(
        self,
        inner_dim: int = 5120,
        in_channels: int = 16,
        zip_frame_buckets: Sequence[int] = (1, 2, 16),
        drop_mode: str = "padd",
    ):
        super().__init__()
        self.inner_dim = inner_dim
        self.in_channels = in_channels
        self.zip_frame_buckets = tuple(zip_frame_buckets)
        self.drop_mode = drop_mode

        # Kernels per bucket. Fine → (1,2,2); medium → (2,4,4); coarse → (4,8,8).
        self.proj = _Conv3dLinear(inner_dim, in_channels, (1, 2, 2))
        self.proj_2x = _Conv3dLinear(inner_dim, in_channels, (2, 4, 4))
        self.proj_4x = _Conv3dLinear(inner_dim, in_channels, (4, 8, 8))

    def pack_motion_frames(self, motion_latent: mx.array) -> List[mx.array]:
        """Split motion-history latent into three temporal buckets and project.

        Args:
            motion_latent: (B, C=16, F_motion, H_lat, W_lat)
                F_motion should equal ``sum(zip_frame_buckets) = 19``. If less,
                the missing prefix is zero-padded ("padd" drop mode). If more,
                the earliest frames are dropped.

        Returns:
            [tokens_coarse, tokens_medium, tokens_fine] — each (B, L_i, D)
            in reverse-chronological order (earliest first) so that the
            downstream RoPE assignment at "negative time" is straightforward.
        """
        B, C, F_m, H, W = motion_latent.shape
        total = sum(self.zip_frame_buckets)
        if F_m < total:
            pad = total - F_m
            pad_tensor = mx.zeros((B, C, pad, H, W), dtype=motion_latent.dtype)
            motion_latent = mx.concatenate([pad_tensor, motion_latent], axis=2)
        elif F_m > total:
            motion_latent = motion_latent[:, :, -total:, :, :]

        # Split chronologically: [coarse (earliest 16), medium (2), fine (last 1)]
        coarse_n, medium_n, fine_n = self.zip_frame_buckets[2], self.zip_frame_buckets[1], self.zip_frame_buckets[0]
        idx = 0
        coarse = motion_latent[:, :, idx : idx + coarse_n]; idx += coarse_n
        medium = motion_latent[:, :, idx : idx + medium_n]; idx += medium_n
        fine = motion_latent[:, :, idx : idx + fine_n]

        # Project each bucket → (B, D', H', W', dim), then flatten spatial+temporal.
        def _flatten(y):  # (B, D, H, W, dim) → (B, D*H*W, dim)
            B_, D_, H_, W_, C_ = y.shape
            return y.reshape(B_, D_ * H_ * W_, C_)

        toks_coarse = _flatten(self.proj_4x(coarse))
        toks_medium = _flatten(self.proj_2x(medium))
        toks_fine = _flatten(self.proj(fine))
        return [toks_coarse, toks_medium, toks_fine]


# Back-compat alias — Phase 1 test imports FramePackerStub.
FramePackerStub = FramePacker


# ---------------------------------------------------------------------------
# CondEncoderProj — pose/overlay conditioning Conv3d
# ---------------------------------------------------------------------------


class CondEncoderProj(nn.Module):
    """Pose/overlay conditioning projector — Conv3d in PyTorch, raw layout.

    State dict key: ``cond_encoder.{weight,bias}`` (flat, no ``.conv`` nesting).
    Weight shape: (dim, cond_dim=16, 1, 2, 2). Used to project a pose/overlay
    latent aligned with the video latent grid; its output is added into the
    patch-embedding sum during S2V forward.
    """

    def __init__(self, out_ch: int, in_ch: int, kernel: tuple = (1, 2, 2)):
        super().__init__()
        self.out_ch = out_ch
        self.in_ch = in_ch
        self.kernel = tuple(kernel)
        self.weight = mx.zeros((out_ch, in_ch, *self.kernel), dtype=mx.float32)
        self.bias = mx.zeros((out_ch,), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: (B, in_ch, F_lat, H_lat, W_lat)
        Returns:
            (B, F', H', W', out_ch) — same layout as `_Conv3dLinear`.
        """
        # Reuse the Conv3dLinear helper logic — inline copy to avoid the extra
        # nesting layer that would break the state-dict key.
        B, C, D, H, W = x.shape
        kd, kh, kw = self.kernel
        d_out, h_out, w_out = D // kd, H // kh, W // kw
        x = x.reshape(B, C, d_out, kd, h_out, kh, w_out, kw)
        x = x.transpose(0, 2, 4, 6, 1, 3, 5, 7)
        x = x.reshape(B, d_out, h_out, w_out, C * kd * kh * kw)
        w = self.weight.astype(x.dtype).reshape(self.out_ch, -1)
        return x @ w.T + self.bias.astype(x.dtype)


# ---------------------------------------------------------------------------
# TrainableCondMask — 3-way (noise/ref/motion) segment-id embedding
# ---------------------------------------------------------------------------


class TrainableCondMask(nn.Module):
    """``nn.Embedding(3, dim)`` — mask ids for {noise, ref, motion} tokens.

    Raw parameter storage so the state-dict key is exactly
    ``trainable_cond_mask.weight``.

    Segment IDs (per design doc):
        0 -> noise (denoise) tokens
        1 -> reference-image tokens
        2 -> motion-history tokens
    """

    def __init__(self, num_embeddings: int = 3, dim: int = 5120):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.dim = dim
        self.weight = mx.zeros((num_embeddings, dim), dtype=mx.float32)

    def __call__(self, ids: mx.array) -> mx.array:
        """Args: ids (int32) of any shape.  Returns: (..., dim) embedding."""
        return self.weight[ids]

    def segment_embedding(self, seg_id: int, length: int, dtype: mx.Dtype = mx.float32) -> mx.array:
        """Return ``(length, dim)`` — the same embedding row repeated ``length`` times."""
        row = self.weight[seg_id][None, :]  # (1, dim)
        return mx.broadcast_to(row, (length, self.dim)).astype(dtype)
