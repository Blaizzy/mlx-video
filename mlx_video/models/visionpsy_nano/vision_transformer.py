"""SigLIP2-base-patch16-512 vision encoder in MLX.

Matches the reference `vision_transformer.py` shipped with VisionPsy-Nano:
- 12 blocks, 768 hidden, 12 heads, 3072 MLP inner
- Fused `qkv_proj` linear (with bias) inside each block
- LayerNorm blocks (ln1/ln2), GELU-tanh MLP
- Learnable position embedding as a single parameter (no CLS token by default)
- Final `layer_norm`
"""
from __future__ import annotations

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .config import VisionPsyNanoConfig


class ViTPatchEmbeddings(nn.Module):
    """Conv2d patch embedding + learned absolute position embedding.

    In the reference PyTorch code, position_embedding is a bare `nn.Parameter`
    of shape `[1, num_patches, hidden]`. We mirror that here so the weight
    loader can drop the tensor in unchanged.
    """

    def __init__(self, cfg: VisionPsyNanoConfig):
        super().__init__()
        assert not cfg.vit_cls_flag, "VisionPsy-Nano ships without a CLS token"
        self.img_size = cfg.vit_img_size
        self.patch_size = cfg.vit_patch_size
        self.num_patches = cfg.vit_num_patches
        self.embd_dim = cfg.vit_hidden_dim

        # MLX Conv2d expects channels-last inputs (B,H,W,C) — the vision model
        # transposes patches into HWC before calling us.
        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=self.embd_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )
        self.position_embedding = mx.zeros((1, self.num_patches, self.embd_dim))

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, H, W, C=3) — channels-last, in [0, 1]
        x = self.conv(x)  # (B, H/p, W/p, embd)
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)
        return x + self.position_embedding


class ViTAttention(nn.Module):
    """Fused-QKV multi-head self-attention with biases (SigLIP style)."""

    def __init__(self, cfg: VisionPsyNanoConfig):
        super().__init__()
        self.n_heads = cfg.vit_n_heads
        self.embd_dim = cfg.vit_hidden_dim
        self.head_dim = cfg.vit_head_dim
        self.scale = self.head_dim ** -0.5

        self.qkv_proj = nn.Linear(self.embd_dim, 3 * self.embd_dim, bias=True)
        self.out_proj = nn.Linear(self.embd_dim, self.embd_dim, bias=True)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        # split along last dim into three C-wide chunks
        q, k, v = mx.split(qkv, 3, axis=-1)
        q = q.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.out_proj(y)


class ViTMLP(nn.Module):
    """FF block with GELU-tanh nonlinearity (SigLIP convention)."""

    def __init__(self, cfg: VisionPsyNanoConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.vit_hidden_dim, cfg.vit_inter_dim, bias=True)
        self.fc2 = nn.Linear(cfg.vit_inter_dim, cfg.vit_hidden_dim, bias=True)
        # Reference uses nn.GELU(approximate='tanh'); MLX uses gelu_approx.
        self.act = nn.GELU(approx="tanh")

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(self.act(self.fc1(x)))


class ViTBlock(nn.Module):
    def __init__(self, cfg: VisionPsyNanoConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.vit_hidden_dim, eps=cfg.vit_ln_eps)
        self.attn = ViTAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.vit_hidden_dim, eps=cfg.vit_ln_eps)
        self.mlp = ViTMLP(cfg)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ViT(nn.Module):
    """SigLIP2 ViT with learned absolute position embeddings."""

    def __init__(self, cfg: VisionPsyNanoConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embedding = ViTPatchEmbeddings(cfg)
        self.blocks = [ViTBlock(cfg) for _ in range(cfg.vit_n_blocks)]
        self.layer_norm = nn.LayerNorm(cfg.vit_hidden_dim, eps=cfg.vit_ln_eps)

    def __call__(self, pixel_values: mx.array) -> mx.array:
        """Run the vision encoder.

        pixel_values: (B, 3, H, W) channels-first, matching the PyTorch entry
        point. We transpose to channels-last for MLX Conv2d.
        """
        if pixel_values.ndim != 4:
            raise ValueError(
                f"pixel_values must be 4D (B,3,H,W); got shape {pixel_values.shape}"
            )
        if pixel_values.shape[1] != 3:
            raise ValueError(
                f"expected 3 channels in dim 1; got shape {pixel_values.shape}"
            )
        x = pixel_values.transpose(0, 2, 3, 1)  # (B, H, W, 3)
        x = self.patch_embedding(x)
        for block in self.blocks:
            x = block(x)
        return self.layer_norm(x)
