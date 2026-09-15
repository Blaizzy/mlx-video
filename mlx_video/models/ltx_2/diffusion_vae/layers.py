"""Neighborhood-attention layers used by the LTX-2.5 diffusion VAE."""

from __future__ import annotations

import math

import mlx.core as mx
from mlx import nn

from mlx_video.models.ltx_2.diffusion_vae.fna3d import neighborhood_attention_3d
from mlx_video.models.ltx_2.diffusion_vae.joint_fna3d import (
    joint_neighborhood_attention_3d,
)


def default_rope_dim_split(head_dim: int) -> tuple[int, int, int]:
    if head_dim % 8:
        raise ValueError(f"head_dim must be divisible by 8, got {head_dim}")
    dim_t = (head_dim // 4) // 2 * 2
    dim_hw = (head_dim - dim_t) // 2
    if dim_hw % 2:
        dim_t -= 2
        dim_hw = (head_dim - dim_t) // 2
    return dim_t, dim_hw, dim_hw


def _rotate_axis(
    x: mx.array,
    *,
    axis: int,
    positions: mx.array | None = None,
    base: float = 10000.0,
) -> mx.array:
    dim = x.shape[-1]
    if positions is None:
        positions = mx.arange(x.shape[axis], dtype=mx.float32)
    inverse = mx.exp(-math.log(base) * mx.arange(0, dim, 2, dtype=mx.float32) / dim)
    shape = [1] * x.ndim
    shape[axis] = x.shape[axis]
    shape[-1] = dim // 2
    angles = mx.reshape(positions[:, None] * inverse[None], shape)
    pairs = mx.reshape(x.astype(mx.float32), (*x.shape[:-1], dim // 2, 2))
    even, odd = pairs[..., 0], pairs[..., 1]
    cosine, sine = mx.cos(angles), mx.sin(angles)
    rotated = mx.stack(
        [even * cosine - odd * sine, even * sine + odd * cosine], axis=-1
    )
    return mx.reshape(rotated, x.shape).astype(x.dtype)


def apply_absolute_rope(
    x: mx.array,
    split: tuple[int, int, int],
    *,
    temporal_positions: mx.array | None = None,
) -> mx.array:
    """Apply independent absolute RoPE chunks to T, H, and W."""
    dim_t, dim_h, _ = split
    return mx.concatenate(
        [
            _rotate_axis(x[..., :dim_t], axis=1, positions=temporal_positions),
            _rotate_axis(x[..., dim_t : dim_t + dim_h], axis=2),
            _rotate_axis(x[..., dim_t + dim_h :], axis=3),
        ],
        axis=-1,
    )


class NeighborhoodAttention3D(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: tuple[int, int, int],
        head_dim: int = 64,
    ) -> None:
        super().__init__()
        if dim % head_dim:
            raise ValueError(f"dim={dim} must be divisible by head_dim={head_dim}")
        self.dim = dim
        self.kernel_size = kernel_size
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.rope_dim_split = default_rope_dim_split(head_dim)
        self.to_q = nn.Linear(dim, dim, bias=True)
        self.to_k = nn.Linear(dim, dim, bias=True)
        self.to_v = nn.Linear(dim, dim, bias=True)
        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.proj = nn.Linear(dim, dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        query, key, value = self.project_qkv(x)
        output = neighborhood_attention_3d(
            query,
            key,
            value,
            self.kernel_size,
            scale=self.head_dim**-0.5,
        )
        return self.proj(mx.reshape(output, x.shape))

    def project_qkv(
        self, x: mx.array, *, temporal_positions: mx.array | None = None
    ) -> tuple[mx.array, mx.array, mx.array]:
        shape = (*x.shape[:-1], self.num_heads, self.head_dim)
        query = mx.reshape(self.to_q(x), shape)
        key = mx.reshape(self.to_k(x), shape)
        value = mx.reshape(self.to_v(x), shape)
        query = apply_absolute_rope(
            self.q_norm(query),
            self.rope_dim_split,
            temporal_positions=temporal_positions,
        )
        key = apply_absolute_rope(
            self.k_norm(key),
            self.rope_dim_split,
            temporal_positions=temporal_positions,
        )
        return query, key, value

    def forward_joint(
        self, x: mx.array, keyframes: mx.array, keyframe_times: mx.array
    ) -> tuple[mx.array, mx.array]:
        query, key, value = self.project_qkv(x)
        keyframe_query, keyframe_key, keyframe_value = self.project_qkv(
            keyframes, temporal_positions=keyframe_times
        )
        output, keyframe_output = joint_neighborhood_attention_3d(
            query,
            key,
            value,
            keyframe_query,
            keyframe_key,
            keyframe_value,
            keyframe_times,
            self.kernel_size,
            scale=self.head_dim**-0.5,
        )
        return self.proj(mx.reshape(output, x.shape)), self.proj(
            mx.reshape(keyframe_output, keyframes.shape)
        )


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w_down(nn.silu(self.w_gate(x)) * self.w_up(x))


class NABlock(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: tuple[int, int, int],
        head_dim: int = 64,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.attn = NeighborhoodAttention3D(dim, kernel_size, head_dim)
        self.norm2 = nn.RMSNorm(dim, eps=1e-6)
        hidden = (int(dim * mlp_ratio) + 15) // 16 * 16
        self.mlp = SwiGLU(dim, hidden)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))

    def forward_joint(
        self, x: mx.array, keyframes: mx.array, keyframe_times: mx.array
    ) -> tuple[mx.array, mx.array]:
        attention, keyframe_attention = self.attn.forward_joint(
            self.norm1(x), self.norm1(keyframes), keyframe_times
        )
        x = x + attention
        keyframes = keyframes + keyframe_attention
        return x + self.mlp(self.norm2(x)), keyframes + self.mlp(self.norm2(keyframes))


class LinearPixelShuffleUpsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        stride: tuple[int, int, int],
        reduction: int,
    ) -> None:
        super().__init__()
        self.stride = stride
        expanded = math.prod(stride) * in_channels // reduction
        self.out_channels = expanded // math.prod(stride)
        self.proj = nn.Linear(in_channels, expanded, bias=True)

    def __call__(self, x: mx.array, *, drop_leading_frame: bool = True) -> mx.array:
        x = self.proj(x)
        batch, frames, height, width, _ = x.shape
        stride_t, stride_h, stride_w = self.stride
        x = mx.reshape(
            x,
            (
                batch,
                frames,
                height,
                width,
                self.out_channels,
                stride_t,
                stride_h,
                stride_w,
            ),
        )
        x = mx.transpose(x, (0, 1, 5, 2, 6, 3, 7, 4))
        x = mx.reshape(
            x,
            (
                batch,
                frames * stride_t,
                height * stride_h,
                width * stride_w,
                self.out_channels,
            ),
        )
        if stride_t == 2 and drop_leading_frame:
            x = x[:, 1:]
        return x
