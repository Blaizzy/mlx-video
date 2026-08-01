"""SmolLM2-360M-Instruct decoder in MLX.

Mirrors the reference `language_model.py`:
- 32 blocks, 960 hidden, GQA 15/5 (head_dim = 64), rope_theta = 100_000
- RMSNorm pre-attn / pre-mlp
- SwiGLU MLP with a FUSED `gate_up_proj` linear (2 * inter_dim wide)
- Weight-tied lm_head with the embedding table
- No biases anywhere in attn / mlp

We support prefill from pre-computed embeddings (needed for the VLM path where
image tokens have been scattered in) and single-step decode with a per-layer
KV cache.
"""
from __future__ import annotations

from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

from .config import VisionPsyNanoConfig


class LMAttention(nn.Module):
    def __init__(self, cfg: VisionPsyNanoConfig):
        super().__init__()
        self.n_heads = cfg.lm_n_heads
        self.n_kv_heads = cfg.lm_n_kv_heads
        self.head_dim = cfg.lm_head_dim
        self.scale = self.head_dim ** -0.5

        dim = cfg.lm_hidden_dim
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)

        # Reference uses HuggingFace's "unrotated-half" RoPE: split q/k in half
        # along the feature dim, rotate both halves together. That's the
        # non-`traditional` layout in MLX's nn.RoPE.
        self.rope = nn.RoPE(
            self.head_dim,
            traditional=False,
            base=cfg.lm_re_base,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional["KVSlot"] = None,
    ) -> mx.array:
        B, T, _ = x.shape

        q = self.q_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        if cache is not None:
            offset = cache.offset
            q = self.rope(q, offset=offset)
            k = self.rope(k, offset=offset)
            k, v = cache.update(k, v)
        else:
            q = self.rope(q)
            k = self.rope(k)

        # MLX's SDPA handles GQA natively when q_heads != kv_heads.
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.out_proj(y)


class LMMLP(nn.Module):
    """SwiGLU with the fused gate/up projection kept as-is on disk."""

    def __init__(self, cfg: VisionPsyNanoConfig):
        super().__init__()
        self.inter_dim = cfg.lm_inter_dim
        self.gate_up_proj = nn.Linear(cfg.lm_hidden_dim, 2 * cfg.lm_inter_dim, bias=False)
        self.down_proj = nn.Linear(cfg.lm_inter_dim, cfg.lm_hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gu = self.gate_up_proj(x)
        gate, up = mx.split(gu, 2, axis=-1)
        return self.down_proj(nn.silu(gate) * up)


class LMBlock(nn.Module):
    def __init__(self, cfg: VisionPsyNanoConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.lm_hidden_dim, eps=cfg.lm_rms_eps)
        self.attn = LMAttention(cfg)
        self.norm2 = nn.RMSNorm(cfg.lm_hidden_dim, eps=cfg.lm_rms_eps)
        self.mlp = LMMLP(cfg)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional["KVSlot"] = None,
    ) -> mx.array:
        x = x + self.attn(self.norm1(x), mask=mask, cache=cache)
        x = x + self.mlp(self.norm2(x))
        return x


class KVSlot:
    """Per-layer growing KV cache."""

    __slots__ = ("keys", "values", "offset")

    def __init__(self):
        self.keys: Optional[mx.array] = None
        self.values: Optional[mx.array] = None
        self.offset: int = 0

    def update(self, k: mx.array, v: mx.array):
        if self.keys is None:
            self.keys = k
            self.values = v
        else:
            self.keys = mx.concatenate([self.keys, k], axis=2)
            self.values = mx.concatenate([self.values, v], axis=2)
        self.offset = self.keys.shape[2]
        return self.keys, self.values


class LanguageModel(nn.Module):
    """SmolLM2 decoder stack with a tied lm_head."""

    def __init__(self, cfg: VisionPsyNanoConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.lm_vocab_size, cfg.lm_hidden_dim)
        self.blocks = [LMBlock(cfg) for _ in range(cfg.lm_n_blocks)]
        self.norm = nn.RMSNorm(cfg.lm_hidden_dim, eps=cfg.lm_rms_eps)
        self.head = nn.Linear(cfg.lm_hidden_dim, cfg.lm_vocab_size, bias=False)

    def embed(self, input_ids: mx.array) -> mx.array:
        return self.token_embedding(input_ids)

    def __call__(
        self,
        inputs_embeds: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[List[KVSlot]] = None,
    ) -> mx.array:
        h = inputs_embeds

        # For prefill (T > 1, no cache) we build a causal mask ourselves; MLX's
        # nn.MultiHeadAttention has helpers for this but we're computing SDPA
        # directly, so pass the string mask down.
        if mask is None:
            B, T, _ = h.shape
            if T > 1:
                mask = "causal"

        if cache is None:
            cache = [None] * len(self.blocks)

        for blk, c in zip(self.blocks, cache):
            h = blk(h, mask=mask, cache=c)

        h = self.norm(h)
        return self.head(h)

    def make_cache(self) -> List[KVSlot]:
        return [KVSlot() for _ in range(len(self.blocks))]
