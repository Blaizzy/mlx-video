from dataclasses import dataclass, replace
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from mlx_video.models.ltx.config import LTXRopeType, TransformerConfig
from mlx_video.models.ltx.attention import Attention
from mlx_video.models.ltx.feed_forward import FeedForward
from mlx_video.utils import rms_norm


@dataclass(frozen=True)
class Modality:
    latent: mx.array
    timesteps: mx.array
    positions: mx.array
    context: mx.array
    enabled: bool = True
    context_mask: Optional[mx.array] = None
    # Optional precomputed positional embeddings (RoPE) to avoid recomputation
    positional_embeddings: Optional[Tuple[mx.array, mx.array]] = None


@dataclass(frozen=True)
class TransformerArgs:
    x: mx.array  
    context: mx.array  
    context_mask: Optional[mx.array] 
    timesteps: mx.array  
    embedded_timestep: mx.array  
    positional_embeddings: Tuple[mx.array, mx.array] 
    cross_positional_embeddings: Optional[Tuple[mx.array, mx.array]]  
    cross_scale_shift_timestep: Optional[mx.array]  
    cross_gate_timestep: Optional[mx.array]  
    enabled: bool


class BasicAVTransformerBlock(nn.Module):
    """Audio-Video transformer block with cross-modal attention.

    Supports video-only, audio-only, or combined audio-video processing
    with bidirectional cross-attention between modalities.
    """

    def __init__(
        self,
        idx: int,
        video: Optional[TransformerConfig] = None,
        audio: Optional[TransformerConfig] = None,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        norm_eps: float = 1e-6,
    ):
        """Initialize transformer block.

        Args:
            idx: Block index
            video: Video modality configuration
            audio: Audio modality configuration
            rope_type: Type of rotary position embedding
            norm_eps: Epsilon for normalization
        """
        super().__init__()

        self.idx = idx
        self.norm_eps = norm_eps

        # Video components
        if video is not None:
            self.attn1 = Attention(
                query_dim=video.dim,
                heads=video.heads,
                dim_head=video.d_head,
                context_dim=None,  # Self-attention
                rope_type=rope_type,
                norm_eps=norm_eps,
            )
            self.attn2 = Attention(
                query_dim=video.dim,
                context_dim=video.context_dim,
                heads=video.heads,
                dim_head=video.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
            )
            self.ff = FeedForward(video.dim, dim_out=video.dim)
            # 6 scale-shift parameters: 3 for attention, 3 for MLP
            self.scale_shift_table = mx.zeros((6, video.dim))

        # Audio components
        if audio is not None:
            self.audio_attn1 = Attention(
                query_dim=audio.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                context_dim=None,
                rope_type=rope_type,
                norm_eps=norm_eps,
            )
            self.audio_attn2 = Attention(
                query_dim=audio.dim,
                context_dim=audio.context_dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
            )
            self.audio_ff = FeedForward(audio.dim, dim_out=audio.dim)
            self.audio_scale_shift_table = mx.zeros((6, audio.dim))

        # Cross-modal attention (when both video and audio are enabled)
        if audio is not None and video is not None:
            # Audio-to-Video: Q from video, K/V from audio
            self.audio_to_video_attn = Attention(
                query_dim=video.dim,
                context_dim=audio.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
            )
            # Video-to-Audio: Q from audio, K/V from video
            self.video_to_audio_attn = Attention(
                query_dim=audio.dim,
                context_dim=video.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
            )
            # Scale-shift tables for cross-attention
            self.scale_shift_table_a2v_ca_audio = mx.zeros((5, audio.dim))
            self.scale_shift_table_a2v_ca_video = mx.zeros((5, video.dim))

    def get_ada_values(
        self,
        scale_shift_table: mx.array,
        batch_size: int,
        timestep: mx.array,
        indices: slice,
    ) -> Tuple[mx.array, ...]:
        """Get adaptive normalization values from scale-shift table.

        Args:
            scale_shift_table: Table of shape (num_params, dim)
            batch_size: Batch size
            timestep: Timestep embeddings of shape (B, 1, num_params * dim) or similar
            indices: Slice for which parameters to extract

        Returns:
            Tuple of scale-shift values
        """
        num_ada_params = scale_shift_table.shape[0]

        # scale_shift_table[indices]: (num_selected, dim)
        # Add batch and sequence dimensions: (1, 1, num_selected, dim)
        table_slice = scale_shift_table[indices]
        table_expanded = mx.expand_dims(mx.expand_dims(table_slice, axis=0), axis=0)

        # timestep: (B, seq, num_params * dim) -> reshape to (B, seq, num_params, dim)
        timestep_reshaped = mx.reshape(
            timestep,
            (batch_size, timestep.shape[1], num_ada_params, -1)
        )

        # Extract the relevant indices
        timestep_slice = timestep_reshaped[:, :, indices, :]

        # Add table values to timestep
        ada_values = table_expanded + timestep_slice

        # Unbind along the parameter dimension
        # Result: tuple of tensors, each of shape (B, seq, dim)
        num_sliced = ada_values.shape[2]
        result = tuple(ada_values[:, :, i, :] for i in range(num_sliced))

        return result

    def get_av_ca_ada_values(
        self,
        scale_shift_table: mx.array,
        batch_size: int,
        scale_shift_timestep: mx.array,
        gate_timestep: mx.array,
        num_scale_shift_values: int = 4,
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
        """Get adaptive values for cross-modal attention.

        Args:
            scale_shift_table: Table with 5 parameters (4 scale-shift + 1 gate)
            batch_size: Batch size
            scale_shift_timestep: Timestep for scale-shift
            gate_timestep: Timestep for gating
            num_scale_shift_values: Number of scale-shift values (default 4)

        Returns:
            Tuple of 5 tensors: (scale1, shift1, scale2, shift2, gate)
        """
        # Get scale-shift values
        scale_shift_ada = self.get_ada_values(
            scale_shift_table[:num_scale_shift_values, :],
            batch_size,
            scale_shift_timestep,
            slice(None, None),
        )

        # Get gate values
        gate_ada = self.get_ada_values(
            scale_shift_table[num_scale_shift_values:, :],
            batch_size,
            gate_timestep,
            slice(None, None),
        )

        # Squeeze the sequence dimension if it's 1
        scale_shift_squeezed = tuple(mx.squeeze(t, axis=1) if t.shape[1] == 1 else t for t in scale_shift_ada)
        gate_squeezed = tuple(mx.squeeze(t, axis=1) if t.shape[1] == 1 else t for t in gate_ada)

        return (*scale_shift_squeezed, *gate_squeezed)

    def __call__(
        self,
        video: Optional[TransformerArgs] = None,
        audio: Optional[TransformerArgs] = None,
    ) -> Tuple[Optional[TransformerArgs], Optional[TransformerArgs]]:
        """Forward pass through transformer block.

        Args:
            video: Video modality arguments
            audio: Audio modality arguments

        Returns:
            Tuple of (updated_video, updated_audio) TransformerArgs
        """
        batch_size = video.x.shape[0] if video is not None else audio.x.shape[0]

        vx = video.x if video is not None else None
        ax = audio.x if audio is not None else None

        # Check which modalities to run
        run_vx = video is not None and video.enabled and vx.size > 0
        run_ax = audio is not None and audio.enabled and ax.size > 0
        run_a2v = run_vx and (audio is not None and audio.enabled and ax.size > 0)
        run_v2a = run_ax and (video is not None and video.enabled and vx.size > 0)

        # Process video self-attention and cross-attention with text
        if run_vx:
            vshift_msa, vscale_msa, vgate_msa = self.get_ada_values(
                self.scale_shift_table, vx.shape[0], video.timesteps, slice(0, 3)
            )

            # Self-attention with RoPE
            norm_vx = rms_norm(vx, eps=self.norm_eps) * (1 + vscale_msa) + vshift_msa
            vx = vx + self.attn1(norm_vx, pe=video.positional_embeddings) * vgate_msa

            # Cross-attention with text context
            vx = vx + self.attn2(
                rms_norm(vx, eps=self.norm_eps),
                context=video.context,
                mask=video.context_mask,
            )

        # Process audio self-attention and cross-attention with text
        if run_ax:
            ashift_msa, ascale_msa, agate_msa = self.get_ada_values(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(0, 3)
            )

            # Self-attention with RoPE
            norm_ax = rms_norm(ax, eps=self.norm_eps) * (1 + ascale_msa) + ashift_msa
            ax = ax + self.audio_attn1(norm_ax, pe=audio.positional_embeddings) * agate_msa

            # Cross-attention with text context
            ax = ax + self.audio_attn2(
                rms_norm(ax, eps=self.norm_eps),
                context=audio.context,
                mask=audio.context_mask,
            )

        # Audio-Video cross-modal attention
        if run_a2v or run_v2a:
            vx_norm3 = rms_norm(vx, eps=self.norm_eps)
            ax_norm3 = rms_norm(ax, eps=self.norm_eps)

            # Get adaptive values for audio cross-attention
            (
                scale_ca_audio_a2v,
                shift_ca_audio_a2v,
                scale_ca_audio_v2a,
                shift_ca_audio_v2a,
                gate_out_v2a,
            ) = self.get_av_ca_ada_values(
                self.scale_shift_table_a2v_ca_audio,
                ax.shape[0],
                audio.cross_scale_shift_timestep,
                audio.cross_gate_timestep,
            )

            # Get adaptive values for video cross-attention
            (
                scale_ca_video_a2v,
                shift_ca_video_a2v,
                scale_ca_video_v2a,
                shift_ca_video_v2a,
                gate_out_a2v,
            ) = self.get_av_ca_ada_values(
                self.scale_shift_table_a2v_ca_video,
                vx.shape[0],
                video.cross_scale_shift_timestep,
                video.cross_gate_timestep,
            )

            # Audio-to-Video cross-attention
            if run_a2v:
                vx_scaled = vx_norm3 * (1 + scale_ca_video_a2v) + shift_ca_video_a2v
                ax_scaled = ax_norm3 * (1 + scale_ca_audio_a2v) + shift_ca_audio_a2v
                vx = vx + (
                    self.audio_to_video_attn(
                        vx_scaled,
                        context=ax_scaled,
                        pe=video.cross_positional_embeddings,
                        k_pe=audio.cross_positional_embeddings,
                    )
                    * gate_out_a2v
                )

            # Video-to-Audio cross-attention
            if run_v2a:
                ax_scaled = ax_norm3 * (1 + scale_ca_audio_v2a) + shift_ca_audio_v2a
                vx_scaled = vx_norm3 * (1 + scale_ca_video_v2a) + shift_ca_video_v2a
                ax = ax + (
                    self.video_to_audio_attn(
                        ax_scaled,
                        context=vx_scaled,
                        pe=audio.cross_positional_embeddings,
                        k_pe=video.cross_positional_embeddings,
                    )
                    * gate_out_v2a
                )

        # Process video feed-forward
        if run_vx:
            vshift_mlp, vscale_mlp, vgate_mlp = self.get_ada_values(
                self.scale_shift_table, vx.shape[0], video.timesteps, slice(3, None)
            )
            vx_scaled = rms_norm(vx, eps=self.norm_eps) * (1 + vscale_mlp) + vshift_mlp
            vx = vx + self.ff(vx_scaled) * vgate_mlp

        # Process audio feed-forward
        if run_ax:
            ashift_mlp, ascale_mlp, agate_mlp = self.get_ada_values(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(3, None)
            )
            ax_scaled = rms_norm(ax, eps=self.norm_eps) * (1 + ascale_mlp) + ashift_mlp
            ax = ax + self.audio_ff(ax_scaled) * agate_mlp

        # Return updated TransformerArgs
        video_out = replace(video, x=vx) if video is not None else None
        audio_out = replace(audio, x=ax) if audio is not None else None

        return video_out, audio_out
