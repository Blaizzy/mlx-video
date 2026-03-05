"""Helios transformer backbone for MLX.

Implements the Helios 14B DiT with multi-scale history memory,
restricted self-attention, and 6-vector modulation per block.
"""

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .attention import (
    HeliosCrossAttention,
    HeliosLayerNorm,
    HeliosRMSNorm,
    HeliosSelfAttention,
    _linear_dtype,
)
from .config import HeliosModelConfig
from .rope import helios_rope_params, helios_rope_precompute_cos_sin, helios_rope_precompute_history


class HeliosFFN(nn.Module):
    """Gated feed-forward network with GELU(tanh) activation."""

    def __init__(self, dim: int, ffn_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, ffn_dim)
        self.act = nn.GELU(approx="tanh")
        self.fc2 = nn.Linear(ffn_dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        x_w = x.astype(_linear_dtype(self.fc1))
        return self.fc2(self.act(self.fc1(x_w)))


class HeliosTransformerBlock(nn.Module):
    """Helios transformer block: self-attn + cross-attn + FFN with 6-vector modulation."""

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: bool = True,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        restrict_self_attn: bool = False,
    ):
        super().__init__()

        # Self-attention
        self.norm1 = HeliosLayerNorm(dim, eps)
        self.self_attn = HeliosSelfAttention(
            dim, num_heads, qk_norm, eps,
            restrict_self_attn=restrict_self_attn,
        )

        # Cross-attention
        self.cross_attn = HeliosCrossAttention(dim, num_heads, qk_norm, eps)
        self.norm2 = (
            HeliosLayerNorm(dim, eps, elementwise_affine=True)
            if cross_attn_norm
            else None
        )

        # Feed-forward
        self.ffn = HeliosFFN(dim, ffn_dim)
        self.norm3 = HeliosLayerNorm(dim, eps)

        # 6-vector modulation table (scale/shift/gate for self-attn and FFN)
        self.scale_shift_table = (
            mx.random.normal((1, 6, dim)) * (dim**-0.5)
        ).astype(mx.float32)

        # Whether to separate history from current in cross-attention
        self.guidance_cross_attn = True

    def __call__(
        self,
        x: mx.array,
        encoder_hidden_states: mx.array,
        timestep_proj: mx.array,
        rotary_emb: tuple | None,
        original_context_length: int,
        frame_indices: mx.array | None = None,
        grid_size: tuple | None = None,
        freqs: tuple | None = None,
        cross_kv_cache: tuple | None = None,
    ) -> mx.array:
        w_dtype = _linear_dtype(self.self_attn.q)
        history_seq_len = x.shape[1] - original_context_length

        # Compute 6-vector modulation
        # timestep_proj: [B, L, 6, dim] (per-token) or [B, 6, dim] (global)
        if timestep_proj.ndim == 4:
            # [B, L, 6, dim] + [1, 1, 6, dim] → [B, L, 6, dim]
            mod = (self.scale_shift_table[None, :, :] + timestep_proj.astype(mx.float32))
            shift_msa = mod[:, :, 0].astype(w_dtype)
            scale_msa = mod[:, :, 1].astype(w_dtype)
            gate_msa = mod[:, :, 2].astype(w_dtype)
            c_shift = mod[:, :, 3].astype(w_dtype)
            c_scale = mod[:, :, 4].astype(w_dtype)
            c_gate = mod[:, :, 5].astype(w_dtype)
        else:
            # [B, 6, dim] + [1, 6, dim] → [B, 6, dim]
            mod = (self.scale_shift_table + timestep_proj.astype(mx.float32)).astype(w_dtype)
            shift_msa = mod[:, 0:1]
            scale_msa = mod[:, 1:2]
            gate_msa = mod[:, 2:3]
            c_shift = mod[:, 3:4]
            c_scale = mod[:, 4:5]
            c_gate = mod[:, 5:6]

        # 1. Self-attention
        norm_x = (self.norm1(x) * (1 + scale_msa) + shift_msa).astype(w_dtype)
        attn_out = self.self_attn(
            norm_x,
            frame_indices=frame_indices,
            grid_size=grid_size,
            freqs=freqs,
            rope_cos_sin=rotary_emb,
            original_context_length=original_context_length,
        )
        x = (x + attn_out * gate_msa)

        # 2. Cross-attention (history tokens skip cross-attention)
        if self.guidance_cross_attn and history_seq_len > 0:
            history_x, current_x = x[:, :history_seq_len], x[:, history_seq_len:]
            norm_current = self.norm2(current_x) if self.norm2 is not None else current_x
            cross_out = self.cross_attn(norm_current, encoder_hidden_states, kv_cache=cross_kv_cache)
            current_x = current_x + cross_out
            x = mx.concatenate([history_x, current_x], axis=1)
        else:
            norm_x = self.norm2(x) if self.norm2 is not None else x
            cross_out = self.cross_attn(norm_x, encoder_hidden_states, kv_cache=cross_kv_cache)
            x = x + cross_out

        # 3. Feed-forward
        norm_x = (self.norm3(x) * (1 + c_scale) + c_shift).astype(w_dtype)
        ff_out = self.ffn(norm_x)
        x = x + ff_out * c_gate

        return x


class HeliosModel(nn.Module):
    """Helios 14B diffusion backbone with multi-scale history memory."""

    def __init__(self, config: HeliosModelConfig):
        super().__init__()
        self.config = config
        dim = config.dim
        self.dim = dim
        self.num_heads = config.num_heads
        self.out_dim = config.out_dim
        self.patch_size = config.patch_size
        self.text_len = config.text_len
        self.freq_dim = config.freq_dim

        # Patch embedding (Conv3d as reshaped Linear)
        patch_dim = config.in_dim * math.prod(config.patch_size)
        self.patch_embedding = nn.Linear(patch_dim, dim)
        self._patch_size = config.patch_size

        # Multi-scale history patches (short/mid/long Conv3d as Linear)
        if config.has_multi_term_memory_patch:
            self.patch_short = nn.Linear(config.in_dim * 1 * 2 * 2, dim)
            self.patch_mid = nn.Linear(config.in_dim * 2 * 4 * 4, dim)
            self.patch_long = nn.Linear(config.in_dim * 4 * 8 * 8, dim)
        self.has_multi_term_memory_patch = config.has_multi_term_memory_patch

        # Text embedding (PixArtAlpha-style projection)
        self.text_embedding_0 = nn.Linear(config.text_dim, dim)
        self.text_embedding_act = nn.GELU(approx="tanh")
        self.text_embedding_1 = nn.Linear(dim, dim)

        # Time embedding (sinusoidal → MLP)
        self.time_embedding_0 = nn.Linear(config.freq_dim, dim)
        self.time_embedding_act = nn.SiLU()
        self.time_embedding_1 = nn.Linear(dim, dim)

        # Time projection for modulation (6x dim for scale/shift/gate)
        self.time_projection_act = nn.SiLU()
        self.time_projection = nn.Linear(dim, dim * 6)

        # Transformer blocks
        self.blocks = [
            HeliosTransformerBlock(
                dim=dim,
                ffn_dim=config.ffn_dim,
                num_heads=config.num_heads,
                qk_norm=config.qk_norm,
                cross_attn_norm=config.cross_attn_norm,
                eps=config.eps,
                restrict_self_attn=False,
            )
            for _ in range(config.num_layers)
        ]

        # Output norm and projection
        self.output_norm = HeliosLayerNorm(dim, config.eps)
        self.output_norm_table = (
            mx.random.normal((1, 2, dim)) * (dim**-0.5)
        ).astype(mx.float32)
        proj_dim = math.prod(config.patch_size) * config.out_dim
        self.proj_out = nn.Linear(dim, proj_dim)

        # RoPE frequencies
        self.rope_freqs = helios_rope_params(
            rope_dim=config.rope_dim,
            theta=config.rope_theta,
            max_seq_len=1024,
        )

        # Whether to zero out history timestep embedding
        self.zero_history_timestep = config.zero_history_timestep

        # Precompute sinusoidal inv_freq for time embedding
        half = config.freq_dim // 2
        self._inv_freq = mx.power(
            10000.0, -mx.arange(half).astype(mx.float32) / half
        )

    def _patchify(self, x: mx.array) -> tuple:
        """Convert video latent to patch embeddings.

        Args:
            x: Video latent [C, F, H, W]

        Returns:
            (patches, grid_size): patches [1, L, dim], grid_size (F', H', W')
        """
        c, f, h, w = x.shape
        pt, ph, pw = self._patch_size

        # Truncate to patch-aligned dims (matches Conv3d floor-division)
        f = (f // pt) * pt
        h = (h // ph) * ph
        w = (w // pw) * pw
        x = x[:, :f, :h, :w]

        f_out = f // pt
        h_out = h // ph
        w_out = w // pw

        x = x.reshape(c, f_out, pt, h_out, ph, w_out, pw)
        x = x.transpose(1, 3, 5, 0, 2, 4, 6)  # [F', H', W', C, pt, ph, pw]
        x = x.reshape(f_out * h_out * w_out, -1)

        patches = self.patch_embedding(x)
        patches = patches.astype(_linear_dtype(self.patch_embedding))
        return patches[None, :, :], (f_out, h_out, w_out)

    def _patchify_history(self, x: mx.array, scale: str) -> mx.array:
        """Patchify history latents at different scales.

        Args:
            x: History latent [C, F, H, W]
            scale: 'short' (1,2,2), 'mid' (2,4,4), or 'long' (4,8,8)
        """
        c, f, h, w = x.shape
        if scale == "short":
            kernel = (1, 2, 2)
            proj = self.patch_short
        elif scale == "mid":
            kernel = (2, 4, 4)
            proj = self.patch_mid
        else:
            kernel = (4, 8, 8)
            proj = self.patch_long

        kt, kh, kw = kernel

        # Pad to make divisible by kernel size
        pad_t = (kt - (f % kt)) % kt
        pad_h = (kh - (h % kh)) % kh
        pad_w = (kw - (w % kw)) % kw
        if pad_t > 0 or pad_h > 0 or pad_w > 0:
            # Replicate padding along each dimension
            if pad_t > 0:
                x = mx.concatenate([x, mx.repeat(x[:, -1:], pad_t, axis=1)], axis=1)
            if pad_h > 0:
                x = mx.concatenate([x, mx.repeat(x[:, :, -1:], pad_h, axis=2)], axis=2)
            if pad_w > 0:
                x = mx.concatenate([x, mx.repeat(x[:, :, :, -1:], pad_w, axis=3)], axis=3)
            c, f, h, w = x.shape

        f_out = f // kt
        h_out = h // kh
        w_out = w // kw

        x = x.reshape(c, f_out, kt, h_out, kh, w_out, kw)
        x = x.transpose(1, 3, 5, 0, 2, 4, 6)
        x = x.reshape(f_out * h_out * w_out, -1)

        patches = proj(x)
        patches = patches.astype(_linear_dtype(proj))
        return patches[None, :, :]  # [1, L, dim]

    def embed_text(self, context: list) -> mx.array:
        """Precompute text embeddings."""
        model_dtype = _linear_dtype(self.patch_embedding)
        context_padded = []
        for ctx in context:
            pad_len = self.text_len - ctx.shape[0]
            if pad_len > 0:
                ctx = mx.concatenate(
                    [ctx, mx.zeros((pad_len, ctx.shape[1]), dtype=ctx.dtype)],
                    axis=0,
                )
            context_padded.append(ctx)
        context_batch = mx.stack(context_padded)
        context_batch = self.text_embedding_1(
            self.text_embedding_act(self.text_embedding_0(context_batch))
        )
        return context_batch.astype(model_dtype)

    def prepare_cross_kv(self, context: mx.array) -> list:
        """Pre-compute cross-attention K/V caches."""
        return [block.cross_attn.prepare_kv(context) for block in self.blocks]

    def unpatchify(self, x: mx.array, grid_size: tuple) -> mx.array:
        """Reconstruct video from patch embeddings.

        Args:
            x: [B, L, out_dim * prod(patch_size)]
            grid_size: (F', H', W')

        Returns:
            [C, F, H, W]
        """
        c = self.out_dim
        pt, ph, pw = self.patch_size
        f, h, w = grid_size
        seq_len = f * h * w

        u = x[0, :seq_len]
        u = u.reshape(f, h, w, pt, ph, pw, c)
        u = u.transpose(6, 0, 3, 1, 4, 2, 5)
        return u.reshape(c, f * pt, h * ph, w * pw)

    def __call__(
        self,
        latents: mx.array,
        timestep: mx.array,
        encoder_hidden_states: mx.array,
        frame_indices: mx.array | None = None,
        history_short: mx.array | None = None,
        history_mid: mx.array | None = None,
        history_long: mx.array | None = None,
        history_short_indices: mx.array | None = None,
        history_mid_indices: mx.array | None = None,
        history_long_indices: mx.array | None = None,
        cross_kv_caches: list | None = None,
    ) -> mx.array:
        """Forward pass through the Helios transformer.

        Args:
            latents: Current chunk latent [C, F, H, W]
            timestep: Scalar diffusion timestep
            encoder_hidden_states: Text embeddings [B, text_len, dim]
            frame_indices: Frame indices for current chunk [F']
            history_short/mid/long: History latents at different scales
            history_*_indices: Frame indices for history at each scale
            cross_kv_caches: Pre-computed cross-attention K/V caches

        Returns:
            Predicted velocity/noise [C, F, H, W]
        """
        # 1. Patchify current latents
        hidden_states, grid_size = self._patchify(latents)
        f_out, h_out, w_out = grid_size
        current_seq_len = hidden_states.shape[1]

        if frame_indices is None:
            frame_indices = mx.arange(f_out)

        # 2. Compute RoPE for current chunk
        rope_cos_sin = helios_rope_precompute_cos_sin(
            frame_indices, grid_size, self.rope_freqs,
            dtype=hidden_states.dtype,
        )

        # 3. Process multi-scale history and prepend
        history_seq_len = 0
        if history_short is not None and self.has_multi_term_memory_patch:
            hist_s = self._patchify_history(history_short, "short")
            hist_m = self._patchify_history(history_mid, "mid")
            hist_l = self._patchify_history(history_long, "long")

            sh, sm, sl = hist_s.shape[1], hist_m.shape[1], hist_l.shape[1]

            # Short patch output spatial dims (kernel 1,2,2)
            c_s, f_s, h_s, w_s = history_short.shape
            hs_h = h_s // 2
            hs_w = w_s // 2
            hs_f = f_s  # temporal stride 1

            # RoPE for short history: compute at short output resolution
            rope_hist_s = helios_rope_precompute_history(
                history_short_indices, hs_h, hs_w, self.rope_freqs,
                downsample_kernel=None,
                dtype=hidden_states.dtype,
            )
            # RoPE for mid history: compute at short resolution, downsample (2,2,2)
            rope_hist_m = helios_rope_precompute_history(
                history_mid_indices, hs_h, hs_w, self.rope_freqs,
                downsample_kernel=(2, 2, 2),
                dtype=hidden_states.dtype,
            )
            # RoPE for long history: compute at short resolution, downsample (4,4,4)
            rope_hist_l = helios_rope_precompute_history(
                history_long_indices, hs_h, hs_w, self.rope_freqs,
                downsample_kernel=(4, 4, 4),
                dtype=hidden_states.dtype,
            )

            # Concatenate history: [long, mid, short, current]
            hidden_states = mx.concatenate(
                [hist_l, hist_m, hist_s, hidden_states], axis=1
            )
            history_seq_len = sl + sm + sh

            # Concatenate RoPE: match history ordering
            all_cos = mx.concatenate([
                rope_hist_l[0], rope_hist_m[0], rope_hist_s[0], rope_cos_sin[0]
            ], axis=0)
            all_sin = mx.concatenate([
                rope_hist_l[1], rope_hist_m[1], rope_hist_s[1], rope_cos_sin[1]
            ], axis=0)
            rope_cos_sin = (all_cos, all_sin)

        original_context_length = current_seq_len

        # 4. Time embedding
        t_emb = timestep.astype(mx.float32) * self._inv_freq
        t_emb = mx.concatenate([mx.cos(t_emb), mx.sin(t_emb)], axis=-1)
        if t_emb.ndim == 1:
            t_emb = t_emb[None, :]

        temb = self.time_embedding_1(
            self.time_embedding_act(self.time_embedding_0(t_emb))
        )
        timestep_proj = self.time_projection(
            self.time_projection_act(temb)
        )
        timestep_proj = timestep_proj.reshape(1, 6, -1)

        # Expand to per-token: [B, 6, L, dim]
        timestep_proj_expanded = mx.broadcast_to(
            timestep_proj[:, :, None, :],
            (1, 6, original_context_length, self.dim),
        )

        # Zero history timestep embedding (timestep=0 → sinusoidal [cos(0), sin(0)] = [1,...,1, 0,...,0])
        if self.zero_history_timestep and history_seq_len > 0:
            t0_emb = mx.array([0.0]) * self._inv_freq
            t0_emb = mx.concatenate([mx.cos(t0_emb), mx.sin(t0_emb)], axis=-1)
            if t0_emb.ndim == 1:
                t0_emb = t0_emb[None, :]
            temb_t0 = self.time_embedding_1(
                self.time_embedding_act(self.time_embedding_0(t0_emb))
            )
            tp_t0 = self.time_projection(
                self.time_projection_act(temb_t0)
            ).reshape(1, 6, -1)
            tp_t0_expanded = mx.broadcast_to(
                tp_t0[:, :, None, :],
                (1, 6, history_seq_len, self.dim),
            )
            timestep_proj_expanded = mx.concatenate(
                [tp_t0_expanded, timestep_proj_expanded], axis=2
            )

        # Permute to [B, L, 6, dim] for block consumption
        timestep_proj_expanded = timestep_proj_expanded.transpose(0, 2, 1, 3)

        # 5. Transformer blocks
        for i, block in enumerate(self.blocks):
            kv_cache = cross_kv_caches[i] if cross_kv_caches is not None else None
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                timestep_proj_expanded,
                rotary_emb=rope_cos_sin,
                original_context_length=original_context_length,
                frame_indices=frame_indices,
                grid_size=grid_size,
                freqs=self.rope_freqs,
                cross_kv_cache=kv_cache,
            )

        # 6. Output norm, projection & unpatchify (only current tokens)
        hidden_out = hidden_states[:, -original_context_length:]

        # Output modulation: temb is [B, 1, dim], expand to [B, L, dim]
        w_dtype = _linear_dtype(self.proj_out)
        temb_expanded = mx.broadcast_to(
            temb[:, None, :], (1, original_context_length, self.dim)
        )
        # scale_shift_table: [1, 2, dim] → [1, 1, 2, dim]
        # temb_expanded: [B, L, dim] → [B, L, 1, dim]
        mod_out = (self.output_norm_table[None, :, :] + temb_expanded[:, :, None, :]).astype(w_dtype)
        shift = mod_out[:, :, 0, :]  # [B, L, dim]
        scale = mod_out[:, :, 1, :]

        hidden_out = (self.output_norm(hidden_out) * (1 + scale) + shift).astype(w_dtype)
        hidden_out = self.proj_out(hidden_out)

        # Unpatchify
        output = self.unpatchify(hidden_out, grid_size)
        return output
