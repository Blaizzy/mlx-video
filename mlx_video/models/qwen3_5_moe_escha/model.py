"""Qwen3_5MoeEscha — end-to-end MLX forward.

Composes:

    embed_tokens (int8)
      → N × DecoderLayer(
            input_layernorm (RMSNorm)
            → GatedDeltaNet  OR  Qwen35MoeEschaAttention   (per layer_types[i])
            → residual +
            post_attention_layernorm (RMSNorm)
            → MoE block:
                router (fp16 gate → softmax → top-k → renorm)
                + naive-loop experts   (BLOCKED on codebooks — zero-fallback)
                + shared_expert (int8) * sigmoid(shared_expert_gate)
            → residual +
        )
      → final norm (RMSNorm)
      → lm_head (int8)

Design notes
------------
- Layer types come from `config.text_config.layer_types` (list of
  "linear_attention" / "full_attention" strings of length num_hidden_layers).
- The full-attention layers use Qwen3.5's `attn_output_gate=True`: q_proj
  doubles its output, splits into (queries, gate), and the SDPA output is
  multiplied by sigmoid(gate) before o_proj. Partial-rotary factor is 0.25
  so only the first 64 of 256 head dims get RoPE.
- The MoE experts are AQLM 2/3-bit and gated behind codebook extraction
  (parallel Linux track). Until the codebook file is on disk, the expert
  block returns zeros (with a one-shot warning). The shared expert is real
  int8 and runs on every token — so the model still emits well-defined
  logits, just with a big semantic hole.
- Cache is a heterogeneous list: `GDNCache` for linear-attn layers and a
  simple `KVCache` for full-attn layers. `layers` is exposed so `make_cache`
  can walk layer types.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from .gated_deltanet import GatedDeltaNet, GDNCache
from .eschamoe import DequantExpertLinear
from .moe import Qwen35MoeRouter, SharedExpertMLP, moe_forward_naive
from .quantization import Int8Linear
from .weight_loader import load_state, summarize
from .weight_loader_dequant import load_state_dequant, summarize_dequant


# --------------------------------------------------------------------------- #
# Simple KV cache for the full-attention layers.
# --------------------------------------------------------------------------- #


class KVCache:
    """Growing KV cache. Concatenates on the sequence axis on every step."""

    def __init__(self) -> None:
        self.keys: Optional[mx.array] = None
        self.values: Optional[mx.array] = None
        self.offset: int = 0

    def update_and_fetch(self, k: mx.array, v: mx.array) -> tuple[mx.array, mx.array]:
        if self.keys is None:
            self.keys, self.values = k, v
        else:
            self.keys = mx.concatenate([self.keys, k], axis=2)
            self.values = mx.concatenate([self.values, v], axis=2)
        self.offset = self.keys.shape[2]
        return self.keys, self.values


# --------------------------------------------------------------------------- #
# Full self-attention (Qwen3.5 attn_output_gate=True variant).
# --------------------------------------------------------------------------- #


class Qwen35MoeEschaAttention(nn.Module):
    def __init__(
        self,
        weights: dict[str, mx.array],
        *,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        partial_rotary_factor: float,
        rope_theta: float,
        rms_norm_eps: float,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.rot_dim = int(head_dim * partial_rotary_factor)

        self.q_proj = Int8Linear(
            weights["q_proj.weight_int8"], weights["q_proj.weight_scale"]
        )
        self.k_proj = Int8Linear(
            weights["k_proj.weight_int8"], weights["k_proj.weight_scale"]
        )
        self.v_proj = Int8Linear(
            weights["v_proj.weight_int8"], weights["v_proj.weight_scale"]
        )
        self.o_proj = Int8Linear(
            weights["o_proj.weight_int8"], weights["o_proj.weight_scale"]
        )

        # Per-head RMSNorm applied on the head_dim axis.
        self.q_norm_w = weights["q_norm.weight"].astype(mx.bfloat16)
        self.k_norm_w = weights["k_norm.weight"].astype(mx.bfloat16)
        self.rms_norm_eps = rms_norm_eps

        # Partial RoPE — first `rot_dim` dims only.
        self.rope = nn.RoPE(dims=self.rot_dim, traditional=False, base=rope_theta)

    def __call__(
        self,
        x: mx.array,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        B, S, _ = x.shape
        x = x.astype(mx.bfloat16)

        # q_proj outputs (num_heads * head_dim * 2) — split (q, gate).
        q_out = self.q_proj(x).reshape(B, S, self.num_heads, 2 * self.head_dim)
        q, gate = mx.split(q_out, 2, axis=-1)                 # each (B, S, H, D)
        gate = gate.reshape(B, S, self.num_heads * self.head_dim)

        k = self.k_proj(x).reshape(B, S, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, S, self.num_kv_heads, self.head_dim)

        # Per-head RMSNorm (head_dim axis).
        q = mx.fast.rms_norm(q, self.q_norm_w, self.rms_norm_eps)
        k = mx.fast.rms_norm(k, self.k_norm_w, self.rms_norm_eps)

        # Transpose to (B, H, S, D) for SDPA and RoPE.
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        if cache is not None:
            q = self.rope(q, offset=cache.offset)
            k = self.rope(k, offset=cache.offset)
            k, v = cache.update_and_fetch(k, v)
        else:
            q = self.rope(q)
            k = self.rope(k)

        mask = "causal" if S > 1 else None
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask
        )
        # (B, H, S, D) → (B, S, H*D)
        out = out.transpose(0, 2, 1, 3).reshape(B, S, -1)
        # attn_output_gate = True: gate the output before o_proj.
        return self.o_proj(out * mx.sigmoid(gate.astype(mx.float32)).astype(out.dtype))


# --------------------------------------------------------------------------- #
# MoE block — router + shared_expert + (zero-fallback) experts
# --------------------------------------------------------------------------- #

_EXPERTS_BLOCKED_WARNED = False


def _experts_zero_fallback(x: mx.array, *, hidden_size: int) -> mx.array:
    """Return zero expert output. Fires a ONE-SHOT loud warning explaining
    that the codebook file is missing and experts are effectively skipped."""
    global _EXPERTS_BLOCKED_WARNED
    if not _EXPERTS_BLOCKED_WARNED:
        warnings.warn(
            "Escha MoE experts are ZEROED — codebook file "
            "`codebooks/escha_codebooks_v1.npz` is missing. Router + shared "
            "expert still run, but token-routed expert contributions are 0. "
            "See docs/ESCHA_PORT_FEASIBILITY.md §5b for the extraction plan.",
            RuntimeWarning,
            stacklevel=2,
        )
        _EXPERTS_BLOCKED_WARNED = True
    return mx.zeros_like(x[..., :hidden_size])


class Qwen35MoeEschaMoEBlock(nn.Module):
    def __init__(
        self,
        weights: dict[str, mx.array],
        *,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        layer_idx: int | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.router = Qwen35MoeRouter(
            weights["gate.weight"], top_k=num_experts_per_tok
        )
        self.shared_expert = SharedExpertMLP(
            gate_proj=Int8Linear(
                weights["shared_expert.gate_proj.weight_int8"],
                weights["shared_expert.gate_proj.weight_scale"],
            ),
            up_proj=Int8Linear(
                weights["shared_expert.up_proj.weight_int8"],
                weights["shared_expert.up_proj.weight_scale"],
            ),
            down_proj=Int8Linear(
                weights["shared_expert.down_proj.weight_int8"],
                weights["shared_expert.down_proj.weight_scale"],
            ),
            gate_scalar_weight=weights["shared_expert_gate.weight"],
        )

        # Detect the dequant path: look for `experts.{E}.gate_up_proj.weight_dequant`.
        gu_keys = [k for k in weights.keys()
                   if k.startswith("experts.") and k.endswith(".gate_up_proj.weight_dequant")]
        dn_keys = [k for k in weights.keys()
                   if k.startswith("experts.") and k.endswith(".down_proj.weight_dequant")]
        self.dequant_mode = len(gu_keys) == num_experts and len(dn_keys) == num_experts

        if self.dequant_mode:
            # Build per-expert dequant linears. Layer key format is
            # "experts.{E}.gate_up_proj.weight_dequant" after prefix strip.
            self.experts_gate_up: list[DequantExpertLinear] = []
            self.experts_down: list[DequantExpertLinear] = []
            for e in range(num_experts):
                self.experts_gate_up.append(
                    DequantExpertLinear(weights[f"experts.{e}.gate_up_proj.weight_dequant"])
                )
                self.experts_down.append(
                    DequantExpertLinear(weights[f"experts.{e}.down_proj.weight_dequant"])
                )
            self._experts_zero = False
        else:
            self.experts_gate_up = []
            self.experts_down = []
            self._experts_zero = True

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, S, hidden). Router expects (T, hidden) flat.
        B, S, H = x.shape
        flat = x.reshape(B * S, H).astype(mx.bfloat16)

        if self.dequant_mode:
            # Real MoE forward via naive loop over experts.
            expert_out = moe_forward_naive(
                flat,
                self.router,
                experts_gate_up=lambda e, xe: self.experts_gate_up[e](xe),
                experts_down=lambda e, he: self.experts_down[e](he),
            )
        else:
            _ = self.router(flat)   # exercise for validation
            expert_out = _experts_zero_fallback(flat, hidden_size=H)  # (T, hidden)

        shared_out = self.shared_expert(flat)                     # (T, hidden)
        y = (expert_out + shared_out).reshape(B, S, H)
        return y


# --------------------------------------------------------------------------- #
# Decoder layer — dispatches to GDN or full attention.
# --------------------------------------------------------------------------- #


class Qwen35MoeEschaDecoderLayer(nn.Module):
    def __init__(
        self,
        weights: dict[str, mx.array],
        *,
        is_linear: bool,
        gdn_kwargs: dict[str, Any],
        attn_kwargs: dict[str, Any],
        moe_kwargs: dict[str, Any],
        rms_norm_eps: float,
        hidden_size: int,
        layer_idx: int = 0,
    ) -> None:
        super().__init__()
        self.is_linear = is_linear
        self.layer_idx = int(layer_idx)

        # Attention/GDN sub-module.
        if is_linear:
            la_w = _prefix_pick(weights, "linear_attn.")
            self.attn = GatedDeltaNet(la_w, rms_norm_eps=rms_norm_eps, **gdn_kwargs)
        else:
            sa_w = _prefix_pick(weights, "self_attn.")
            self.attn = Qwen35MoeEschaAttention(
                sa_w, rms_norm_eps=rms_norm_eps, **attn_kwargs
            )

        self.input_layernorm_w = weights["input_layernorm.weight"].astype(mx.bfloat16)
        self.post_attn_layernorm_w = weights["post_attention_layernorm.weight"].astype(
            mx.bfloat16
        )
        self.rms_norm_eps = rms_norm_eps

        mlp_w = _prefix_pick(weights, "mlp.")
        self.mlp = Qwen35MoeEschaMoEBlock(
            mlp_w, hidden_size=hidden_size, **moe_kwargs, layer_idx=self.layer_idx,
        )

    def __call__(self, x: mx.array, cache: Any = None) -> mx.array:
        h = mx.fast.rms_norm(x, self.input_layernorm_w, self.rms_norm_eps)
        if self.is_linear:
            r = self.attn(h, cache=cache)
        else:
            r = self.attn(h, cache=cache)
        x = x + r
        h2 = mx.fast.rms_norm(x, self.post_attn_layernorm_w, self.rms_norm_eps)
        return x + self.mlp(h2)


def _prefix_pick(weights: dict[str, mx.array], prefix: str) -> dict[str, mx.array]:
    out = {}
    for k, v in weights.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
    return out


# --------------------------------------------------------------------------- #
# Top-level model
# --------------------------------------------------------------------------- #


class Qwen35MoeEschaLanguageModel(nn.Module):
    def __init__(
        self,
        state: dict[str, mx.array],
        cfg: dict[str, Any],
    ) -> None:
        super().__init__()
        tc = cfg["text_config"]
        self.hidden_size = tc["hidden_size"]
        self.num_layers = tc["num_hidden_layers"]
        self.rms_norm_eps = tc["rms_norm_eps"]
        self.layer_types: list[str] = list(tc["layer_types"])
        assert len(self.layer_types) == self.num_layers

        # ---- embedding (int8) -------------------------------------------- #
        prefix = "model.language_model."
        self.embed = Int8Linear(
            state[prefix + "embed_tokens.weight_int8"],
            state[prefix + "embed_tokens.weight_scale"],
        )
        # Embed via lookup — treat as gather.
        self.embed_weight = self.embed.weight   # (vocab, hidden)

        # ---- final norm -------------------------------------------------- #
        self.final_norm_w = state[prefix + "norm.weight"].astype(mx.bfloat16)

        # ---- decoder layers --------------------------------------------- #
        gdn_kwargs = dict(
            hidden_size=tc["hidden_size"],
            num_v_heads=tc["linear_num_value_heads"],
            num_k_heads=tc["linear_num_key_heads"],
            head_k_dim=tc["linear_key_head_dim"],
            head_v_dim=tc["linear_value_head_dim"],
            conv_kernel_size=tc["linear_conv_kernel_dim"],
        )
        rope_theta = float(tc["rope_parameters"]["rope_theta"])
        attn_kwargs = dict(
            hidden_size=tc["hidden_size"],
            num_attention_heads=tc["num_attention_heads"],
            num_key_value_heads=tc["num_key_value_heads"],
            head_dim=tc["head_dim"],
            partial_rotary_factor=tc["partial_rotary_factor"],
            rope_theta=rope_theta,
        )
        moe_kwargs = dict(
            num_experts=tc["num_experts"],
            num_experts_per_tok=tc["num_experts_per_tok"],
        )

        self.layers: list[Qwen35MoeEschaDecoderLayer] = []
        for i in range(self.num_layers):
            lprefix = f"{prefix}layers.{i}."
            layer_w = _prefix_pick(state, lprefix)
            self.layers.append(
                Qwen35MoeEschaDecoderLayer(
                    layer_w,
                    is_linear=(self.layer_types[i] == "linear_attention"),
                    gdn_kwargs=gdn_kwargs,
                    attn_kwargs=attn_kwargs,
                    moe_kwargs=moe_kwargs,
                    rms_norm_eps=self.rms_norm_eps,
                    hidden_size=self.hidden_size,
                    layer_idx=i,
                )
            )

    def make_cache(self) -> list[Any]:
        return [GDNCache() if lt == "linear_attention" else KVCache()
                for lt in self.layer_types]

    def __call__(
        self,
        tokens: mx.array,
        cache: Optional[list[Any]] = None,
    ) -> mx.array:
        # Simple embed: gather rows of embed_weight.
        h = self.embed_weight[tokens]                          # (B, S, hidden)
        h = h.astype(mx.bfloat16)

        if cache is None:
            cache = [None] * self.num_layers
        for layer, c in zip(self.layers, cache):
            h = layer(h, cache=c)
        h = mx.fast.rms_norm(h, self.final_norm_w, self.rms_norm_eps)
        return h


class Qwen3_5MoeEschaForConditionalGeneration(nn.Module):
    """Top-level Escha-W2 model — text-only forward. Vision tower not ported yet."""

    def __init__(self, state: dict[str, mx.array], cfg: dict[str, Any]) -> None:
        super().__init__()
        self.config = cfg
        self.model = Qwen35MoeEschaLanguageModel(state, cfg)

        if cfg.get("tie_word_embeddings"):
            self.lm_head = None  # reuse embed_weight in forward
        else:
            self.lm_head = Int8Linear(
                state["lm_head.weight_int8"],
                state["lm_head.weight_scale"],
            )

    def __call__(
        self,
        tokens: mx.array,
        cache: Optional[list[Any]] = None,
    ) -> mx.array:
        h = self.model(tokens, cache=cache)                   # (B, S, hidden)
        if self.lm_head is None:
            return h @ self.model.embed_weight.T
        return self.lm_head(h)

    def make_cache(self) -> list[Any]:
        return self.model.make_cache()


# --------------------------------------------------------------------------- #
# Load helper
# --------------------------------------------------------------------------- #


def load_model(model_dir: str | Path) -> Qwen3_5MoeEschaForConditionalGeneration:
    """Load config + weights from a local Escha-W2 checkpoint directory."""
    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / "config.json").read_text())
    state = load_state(model_dir)
    print(summarize(state))  # noqa: T201 — dev diagnostic

    model = Qwen3_5MoeEschaForConditionalGeneration(state, cfg)
    return model


def load_model_dequant(
    orig_model_dir: str | Path,
    dequant_dir: str | Path,
) -> Qwen3_5MoeEschaForConditionalGeneration:
    """Load config + weights using the Option-B pre-composed dense M matrices.

    Args:
        orig_model_dir: path to the original Escha-W2 checkpoint (for attn/embed/norm/router).
        dequant_dir   : path to the layer_XX.safetensors files (MoE experts).
    """
    orig_model_dir = Path(orig_model_dir)
    cfg = json.loads((orig_model_dir / "config.json").read_text())
    state = load_state_dequant(orig_model_dir, dequant_dir)
    print(summarize_dequant(state))

    model = Qwen3_5MoeEschaForConditionalGeneration(state, cfg)
    return model
