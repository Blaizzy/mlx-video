"""Qwen3.5 MoE top-k router — Phase 1 correctness path.

Per config.json (Qwen3.6-35B-A3B-Escha-W2): num_experts=256, num_experts_per_tok=8,
moe_intermediate_size=512, shared_expert_intermediate_size=512.

This module implements ONLY the routing math + a naive scatter-gather that
loops over experts (correctness reference). A fused batched-expert call is
Phase 2. The Escha `PackedScaledExpertLinear` will slot in as the per-expert
projection once codebooks are available.

Architecture note (from config):
    - Gate is a plain Linear(hidden, num_experts) — weights at mlp.gate.weight (fp16, non-quantized).
    - Router weights use softmax then top-k, normalize the k gates back to sum-to-1.
    - Shared expert runs unconditionally on every token; its output is gated by
      sigmoid(shared_expert_gate(x)) and added to the MoE output (Qwen3-Next style).
"""

from __future__ import annotations

from typing import Callable

import mlx.core as mx
import mlx.nn as nn


class Qwen35MoeRouter(nn.Module):
    """Top-k router with softmax→topk→renormalize.

    Weights are loaded raw from `mlp.gate.weight` (fp16, shape [num_experts, hidden]).
    """

    def __init__(self, gate_weight: mx.array, top_k: int):
        super().__init__()
        # (num_experts, hidden). Cast to bf16 for numerical stability in softmax.
        self.gate = gate_weight.astype(mx.bfloat16)
        self.top_k = int(top_k)

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """Route tokens to experts.

        Args:
            x: (batch*seq, hidden) bf16 activations.

        Returns:
            (top_idx, top_gate):
                top_idx : int32 (T, top_k) — expert indices per token
                top_gate: bf16  (T, top_k) — renormalized routing weights
        """
        logits = x @ self.gate.T                             # (T, num_experts)
        probs = mx.softmax(logits.astype(mx.float32), axis=-1)
        # top-k with MLX: use argpartition-style pattern via argsort (MLX has topk).
        # mx.argpartition isn't exposed; use full sort — cheap for num_experts=256.
        idx = mx.argsort(-probs, axis=-1)[:, : self.top_k]   # (T, k) int32
        gathered = mx.take_along_axis(probs, idx, axis=-1)   # (T, k) fp32
        # Renormalize so per-token gates sum to 1.
        gathered = gathered / gathered.sum(axis=-1, keepdims=True)
        return idx.astype(mx.int32), gathered.astype(mx.bfloat16)


def moe_forward_naive(
    x: mx.array,
    router: Qwen35MoeRouter,
    experts_gate_up: Callable[[int, mx.array], mx.array],
    experts_down: Callable[[int, mx.array], mx.array],
) -> mx.array:
    """Reference MoE forward — loop over experts, gather tokens routed to each.

    Semantics (per Qwen3 MoE expert):
        gu = experts_gate_up(e, x_e)              # (n_e, 2 * moe_intermediate_size)
        gate, up = split(gu, 2, dim=-1)
        h = silu(gate) * up
        y_e = experts_down(e, h)                  # (n_e, hidden)

    This is intentionally simple — for Phase 1 correctness. Phase 2 will
    batch-decode all active experts in one fused kernel.
    """
    T = x.shape[0]
    hidden = x.shape[-1]
    top_idx, top_gate = router(x)                   # (T, k) each
    k = top_idx.shape[-1]

    # Flatten (T, k) into a linear routing table so we scatter once at the end.
    flat_idx_a = top_idx.reshape(-1)                # (T*k,)
    flat_gate_a = top_gate.reshape(-1)              # (T*k,)
    # Each row of x needs to be gathered k times (once per expert slot).
    token_ids_a = mx.arange(T).reshape(T, 1)
    token_ids_a = mx.broadcast_to(token_ids_a, (T, k)).reshape(-1)  # (T*k,)

    # Materialize the routing table on the CPU to avoid per-expert
    # host-device sync on 256 experts (naive reference path only).
    import numpy as np
    flat_idx_np = np.asarray(flat_idx_a).astype(np.int32)
    token_ids_np = np.asarray(token_ids_a).astype(np.int32)

    out = mx.zeros((T, hidden), dtype=x.dtype)
    # Loop only over experts that received at least one token.
    unique_experts = np.unique(flat_idx_np)
    for e_np in unique_experts:
        e = int(e_np)
        # Positions in flat_idx (length T*k) where this expert is routed.
        sel_np = np.where(flat_idx_np == e)[0].astype(np.int32)
        rows_np = token_ids_np[sel_np]
        sel = mx.array(sel_np)
        rows = mx.array(rows_np)
        gates = mx.take(flat_gate_a, sel, axis=0)   # (n_e,)
        x_e = mx.take(x, rows, axis=0)              # (n_e, hidden)
        gu = experts_gate_up(e, x_e)                # (n_e, 2*mi)
        half = gu.shape[-1] // 2
        gate, up = gu[..., :half], gu[..., half:]
        h = nn.silu(gate) * up                      # (n_e, mi)
        y_e = experts_down(e, h)                    # (n_e, hidden)
        y_e = y_e * gates.reshape(-1, 1)
        # Scatter-add back into out at row `rows`.
        out = out.at[rows].add(y_e)
    return out


class SharedExpertMLP(nn.Module):
    """Non-MoE shared expert (Qwen3-Next style).

    Layout from safetensors:
        mlp.shared_expert.gate_proj.weight_int8/scale  — (mi, hidden)
        mlp.shared_expert.up_proj.weight_int8/scale    — (mi, hidden)
        mlp.shared_expert.down_proj.weight_int8/scale  — (hidden, mi)
        mlp.shared_expert_gate.weight                  — (1, hidden)   fp16 scalar gate

    All int8 weights are pre-dequantized to bf16 at load (see Int8Linear).
    """

    def __init__(self, gate_proj, up_proj, down_proj, gate_scalar_weight: mx.array):
        super().__init__()
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj
        # (1, hidden) → treat as sigmoid-gated scalar per token.
        self.gate_scalar = gate_scalar_weight.astype(mx.bfloat16)

    def __call__(self, x: mx.array) -> mx.array:
        g = self.gate_proj(x)
        u = self.up_proj(x)
        h = nn.silu(g) * u
        y = self.down_proj(h)
        gate = mx.sigmoid((x @ self.gate_scalar.T).astype(mx.float32)).astype(y.dtype)
        return y * gate
