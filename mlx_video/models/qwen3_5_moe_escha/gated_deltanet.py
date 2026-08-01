"""Gated-DeltaNet (linear_attn) — MLX port for Escha-W2.

This is the linear-attention / Mamba-2-style SSM layer that fills 3 of every 4
decoder layers in Qwen3.6-35B-A3B-Escha (`full_attention_interval = 4`). It's
a bit-for-bit port of `mlx_lm.models.qwen3_next.Qwen3NextGatedDeltaNet`, but
adapted to Escha's *split* input-projection layout:

    Qwen3Next name          Escha safetensors keys
    -----------------       -----------------------------------------
    in_proj_qkvz.weight  →  in_proj_qkv.weight_int8 / weight_scale   (int8)
                            in_proj_z.weight_int8   / weight_scale   (int8)
    in_proj_ba.weight    →  in_proj_a.weight   (fp16, no quantization)
                            in_proj_b.weight   (fp16, no quantization)
    conv1d.weight        →  conv1d.weight      (fp16, shape (C, 1, K); needs moveaxis(2,1))
    A_log, dt_bias       →  A_log, dt_bias     (fp16)
    norm.weight          →  norm.weight        (fp16)
    out_proj.weight      →  out_proj.weight_int8 / weight_scale       (int8)

The split projections are the *only* structural change; the recurrence, the
depthwise causal conv, the RMSNorm-gated output and the head-count arithmetic
are identical to Qwen3Next.

Cache design
------------
Two-slot list (matches mlx-lm's `ArraysCache(size=2)` pattern so we can later
swap in without touching call-sites):

    cache[0]  →  conv_state  (B, K-1, conv_dim)     fp16/bf16
    cache[1]  →  ssm_state   (B, Hv, Dv, Dk)        fp16/bf16

`GDNCache` also carries `advance(N)` and `__len__` so a future generation loop
can hold a heterogeneous list of caches (one per layer, some full-attn, some
SSM) alongside standard KV caches.

Recurrence (ops-only reference — no Metal kernel yet)
-----------------------------------------------------
    beta      = sigmoid(b)                                   # (B, S, Hv)
    g         = exp(-exp(A_log) * softplus(a + dt_bias))     # (B, S, Hv)
    state     = state * g[..., None, None]                   # decay
    kv_mem    = sum(state * k[..., None, :], -1)             # (B, Hv, Dv)
    delta     = (v - kv_mem) * beta[..., None]               # (B, Hv, Dv)
    state     = state + k[..., None, :] * delta[..., None]
    y_t       = sum(state * q[..., None, :], -1)             # (B, Hv, Dv)

We loop `t = 0..S-1` for prompt prefill (chunked scan is a Phase-2 speedup;
correctness first).

Phase-1 correctness note: this module intentionally uses only public MLX ops
(no `mx.fast.metal_kernel`) so unit tests pass on both Metal and CPU-only
runs. Phase 2 will drop in `mlx_lm.models.gated_delta.gated_delta_kernel`
when Metal is available for a large prefill/generation speedup.
"""

from __future__ import annotations

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from .quantization import Int8Linear


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #


class GDNCache:
    """Two-slot state cache (conv_state, ssm_state) for one GDN layer.

    Mirrors the subset of `mlx_lm.models.cache.ArraysCache` we exercise so
    the layer's call-site is drop-in compatible if we later import that.
    """

    def __init__(self) -> None:
        self.cache: list[Optional[mx.array]] = [None, None]
        # `lengths` / `left_padding` are only used for padded-batch decoding
        # (right-padded prompts). Kept as `None` for the single-sequence path.
        self.lengths: Optional[mx.array] = None
        self.left_padding: Optional[mx.array] = None
        self.offset: int = 0

    def __getitem__(self, i: int) -> Optional[mx.array]:
        return self.cache[i]

    def __setitem__(self, i: int, v: mx.array) -> None:
        self.cache[i] = v

    def __len__(self) -> int:
        return len(self.cache)

    @property
    def state(self):
        return self.cache

    @state.setter
    def state(self, v):
        self.cache = v

    def advance(self, n: int) -> None:
        self.offset += int(n)
        if self.lengths is not None:
            self.lengths = self.lengths - n

    # No `make_mask` — SSM path does not consume attention masks in this port.


# --------------------------------------------------------------------------- #
# The gated_delta recurrence (ops-only; no Metal kernel).
# --------------------------------------------------------------------------- #


def _compute_g(A_log: mx.array, a: mx.array, dt_bias: mx.array) -> mx.array:
    """Per-step decay factor `g = exp(-exp(A_log) * softplus(a + dt_bias))`.

    Kept in float32 for numerical stability of the exp/softplus stack, then
    cast back to the input dtype so downstream matmul is bf16/fp16.
    """
    a_f32 = (a + dt_bias).astype(mx.float32)
    A_f32 = A_log.astype(mx.float32)
    g = mx.exp(-mx.exp(A_f32) * nn.softplus(a_f32))
    return g.astype(a.dtype)


def _gated_delta_step(
    q: mx.array,        # (B, Hv, Dk)
    k: mx.array,        # (B, Hv, Dk)
    v: mx.array,        # (B, Hv, Dv)
    g: mx.array,        # (B, Hv)
    beta: mx.array,     # (B, Hv)
    state: mx.array,    # (B, Hv, Dv, Dk)
) -> tuple[mx.array, mx.array]:
    """One recurrent SSM step. Semantically identical to
    `mlx_lm.models.gated_delta._gated_delta_step_ops` (scalar-gating path)."""
    # 1) Decay the memory
    decay = g[..., None, None]                              # (B, Hv, 1, 1)
    state = state * decay                                   # (B, Hv, Dv, Dk)
    # 2) Read out the current KV memory along the key axis
    kv_mem = (state * k[..., None, :]).sum(axis=-1)         # (B, Hv, Dv)
    # 3) Delta-rule residual write
    delta = (v - kv_mem) * beta[..., None]                  # (B, Hv, Dv)
    state = state + k[..., None, :] * delta[..., None]      # (B, Hv, Dv, Dk)
    # 4) Query the memory
    y = (state * q[..., None, :]).sum(axis=-1)              # (B, Hv, Dv)
    return y, state


def gated_delta_update(
    q: mx.array,            # (B, S, Hk, Dk)
    k: mx.array,            # (B, S, Hk, Dk)
    v: mx.array,            # (B, S, Hv, Dv)
    a: mx.array,            # (B, S, Hv)
    b: mx.array,            # (B, S, Hv)
    A_log: mx.array,        # (Hv,)
    dt_bias: mx.array,      # (Hv,)
    state: Optional[mx.array] = None,
) -> tuple[mx.array, mx.array]:
    """Prompt-prefill / single-step GDN recurrence in pure MLX ops.

    Handles the K/V head-count mismatch (`Hv >= Hk`, GQA-style) by repeating
    q and k along the head axis before the sequential scan.
    """
    B, S, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]

    beta = mx.sigmoid(b)                                    # (B, S, Hv)
    g = _compute_g(A_log, a, dt_bias)                       # (B, S, Hv)

    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=q.dtype)

    if Hv != Hk:
        rep = Hv // Hk
        q = mx.repeat(q, rep, axis=-2)                      # (B, S, Hv, Dk)
        k = mx.repeat(k, rep, axis=-2)

    ys: list[mx.array] = []
    for t in range(S):
        y, state = _gated_delta_step(
            q[:, t], k[:, t], v[:, t], g[:, t], beta[:, t], state
        )
        ys.append(y)
    y = mx.stack(ys, axis=1)                                # (B, S, Hv, Dv)
    return y, state


# --------------------------------------------------------------------------- #
# RMSNormGated (silu-gated post-norm — matches Qwen3NextRMSNormGated).
# --------------------------------------------------------------------------- #


class RMSNormGated(nn.Module):
    """`silu(gate) * RMSNorm(x)`. Both promoted to fp32 for the gate math,
    then cast back to the input dtype."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones(hidden_size)

    def __call__(self, x: mx.array, gate: mx.array | None = None) -> mx.array:
        y = mx.fast.rms_norm(x, self.weight, self.eps)
        if gate is None:
            return y.astype(x.dtype)
        g = nn.silu(gate.astype(mx.float32))
        return (g * y.astype(mx.float32)).astype(x.dtype)


# --------------------------------------------------------------------------- #
# The layer itself.
# --------------------------------------------------------------------------- #


class GatedDeltaNet(nn.Module):
    """MLX port of Qwen3.5-MoE-Escha's per-layer `linear_attn` module.

    Constructor signature accepts a pre-loaded flat weight dict keyed by the
    Escha safetensors names (with the `model.language_model.layers.{i}.linear_attn.`
    prefix already stripped). This mirrors how `moe.SharedExpertMLP` receives
    its projections and keeps `model.py` responsible for the top-level plumbing.
    """

    def __init__(
        self,
        weights: dict[str, mx.array],
        *,
        hidden_size: int,
        num_v_heads: int,
        num_k_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        conv_kernel_size: int,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        # ---- config ------------------------------------------------------ #
        self.hidden_size = int(hidden_size)
        self.num_v_heads = int(num_v_heads)
        self.num_k_heads = int(num_k_heads)
        self.head_k_dim = int(head_k_dim)
        self.head_v_dim = int(head_v_dim)
        self.conv_kernel_size = int(conv_kernel_size)
        self.rms_norm_eps = float(rms_norm_eps)

        if self.num_v_heads % self.num_k_heads != 0:
            raise ValueError(
                f"num_v_heads ({num_v_heads}) must be divisible by "
                f"num_k_heads ({num_k_heads})"
            )

        self.key_dim = self.head_k_dim * self.num_k_heads          # Hk*Dk
        self.value_dim = self.head_v_dim * self.num_v_heads        # Hv*Dv
        self.conv_dim = self.key_dim * 2 + self.value_dim          # qk + v

        # ---- input projections ------------------------------------------ #
        # in_proj_qkv (int8, out=8192, in=2048) — packs q + k + v per-k-head.
        self.in_proj_qkv = Int8Linear(
            weights["in_proj_qkv.weight_int8"],
            weights["in_proj_qkv.weight_scale"],
        )
        # in_proj_z (int8, out=4096, in=2048) — packs z per-k-head (2 v-slots each).
        self.in_proj_z = Int8Linear(
            weights["in_proj_z.weight_int8"],
            weights["in_proj_z.weight_scale"],
        )
        # in_proj_a / in_proj_b are fp16 unquantized: (32, 2048) each.
        self.in_proj_a_w = weights["in_proj_a.weight"].astype(mx.bfloat16)
        self.in_proj_b_w = weights["in_proj_b.weight"].astype(mx.bfloat16)

        # ---- depthwise causal conv1d ------------------------------------- #
        # Escha stores conv1d.weight as (C, 1, K); MLX nn.Conv1d expects
        # (C_out, K, C_in/groups) = (C, K, 1) for a depthwise conv.
        # (mlx-lm's Qwen3Next `sanitize` does the same moveaxis(2, 1).)
        conv_w = weights["conv1d.weight"]
        if conv_w.shape[-1] != 1:
            conv_w = mx.moveaxis(conv_w, 2, 1)              # (C, K, 1)
        assert conv_w.shape == (self.conv_dim, self.conv_kernel_size, 1), (
            f"conv1d weight shape {conv_w.shape} != "
            f"({self.conv_dim}, {self.conv_kernel_size}, 1)"
        )
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=0,
            bias=False,
        )
        # Overwrite the random-initialized weight with the loaded one.
        self.conv1d.weight = conv_w.astype(mx.bfloat16)

        # ---- SSM parameters --------------------------------------------- #
        # A_log and dt_bias are per-value-head scalars.
        self.A_log = weights["A_log"].astype(mx.float32)
        self.dt_bias = weights["dt_bias"].astype(mx.float32)

        # ---- gated post-norm on head_v_dim ------------------------------ #
        self.norm = RMSNormGated(self.head_v_dim, eps=self.rms_norm_eps)
        self.norm.weight = weights["norm.weight"].astype(mx.bfloat16)

        # ---- output projection ------------------------------------------ #
        self.out_proj = Int8Linear(
            weights["out_proj.weight_int8"],
            weights["out_proj.weight_scale"],
        )

    # ------------------------------------------------------------------ #
    # QKV/Z/BA unpacking — matches Qwen3Next.fix_query_key_value_ordering
    # but starts from Escha's *already-split* projection outputs.
    # ------------------------------------------------------------------ #

    def _unpack_qkvz(
        self,
        mixed_qkv: mx.array,   # (B, S, key_dim*2 + value_dim)
        mixed_z: mx.array,     # (B, S, value_dim)
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        """Reshape the split projection outputs into per-head q/k/v/z tensors."""
        B, S = mixed_qkv.shape[:2]
        nk, dk, nv, dv = (
            self.num_k_heads, self.head_k_dim, self.num_v_heads, self.head_v_dim,
        )
        rep = nv // nk    # value heads per key head

        # in_proj_qkv is laid out per-k-head as [q(dk), k(dk), v(rep*dv)] with
        # nk groups. Reshape → (B, S, nk, dk*2 + rep*dv) then split.
        qkv = mixed_qkv.reshape(B, S, nk, dk * 2 + rep * dv)
        q, k, v = mx.split(qkv, [dk, 2 * dk], axis=-1)
        # v: (B, S, nk, rep*dv) → (B, S, nv, dv)
        v = v.reshape(B, S, nv, dv)

        # in_proj_z is laid out per-k-head as z(rep*dv) with nk groups.
        # Reshape directly to (B, S, nv, dv).
        z = mixed_z.reshape(B, S, nk, rep * dv).reshape(B, S, nv, dv)
        return q, k, v, z

    def _unpack_ba(
        self,
        a_out: mx.array,   # (B, S, num_v_heads)
        b_out: mx.array,   # (B, S, num_v_heads)
    ) -> tuple[mx.array, mx.array]:
        """a/b are per-v-head scalars. Escha ships them already flat."""
        # Included as an explicit function so future GQA rewiring lives here.
        return a_out, b_out

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def __call__(
        self,
        inputs: mx.array,                                    # (B, S, hidden)
        mask: Optional[mx.array] = None,                     # unused (SSM path)
        cache: Optional[GDNCache] = None,
    ) -> mx.array:
        _ = mask   # SSM does not consume the attention mask in this port.
        B, S, _H = inputs.shape

        # 1) input projections (bf16 throughout).
        x = inputs.astype(mx.bfloat16)
        mixed_qkv = self.in_proj_qkv(x)                      # (B, S, key*2 + value)
        mixed_z = self.in_proj_z(x)                          # (B, S, value)
        a_out = x @ self.in_proj_a_w.T                       # (B, S, Hv)
        b_out = x @ self.in_proj_b_w.T                       # (B, S, Hv)

        q, k, v, z = self._unpack_qkvz(mixed_qkv, mixed_z)
        a, b = self._unpack_ba(a_out, b_out)

        # 2) prepend the cached (K-1) frames to mixed_qkv before the depthwise
        #    conv so the conv sees the correct autoregressive receptive field.
        mixed_qkv_flat = mx.concatenate(
            [q.reshape(B, S, -1), k.reshape(B, S, -1), v.reshape(B, S, -1)],
            axis=-1,
        )                                                    # (B, S, conv_dim)

        if cache is not None and cache[0] is not None:
            conv_state = cache[0]                            # (B, K-1, C)
        else:
            conv_state = mx.zeros(
                (B, self.conv_kernel_size - 1, self.conv_dim),
                dtype=mixed_qkv_flat.dtype,
            )
        conv_input = mx.concatenate([conv_state, mixed_qkv_flat], axis=1)

        # Update cache with the tail of the padded input.
        if cache is not None:
            n_keep = self.conv_kernel_size - 1
            cache[0] = conv_input[:, -n_keep:, :]

        # 3) depthwise causal conv1d + silu.
        conv_out = nn.silu(self.conv1d(conv_input))          # (B, S, C)

        # 4) split back into per-head q, k, v.
        q_c, k_c, v_c = mx.split(
            conv_out, [self.key_dim, 2 * self.key_dim], axis=-1
        )
        q_c = q_c.reshape(B, S, self.num_k_heads, self.head_k_dim)
        k_c = k_c.reshape(B, S, self.num_k_heads, self.head_k_dim)
        v_c = v_c.reshape(B, S, self.num_v_heads, self.head_v_dim)

        # 5) RMSNorm(q) and RMSNorm(k) with inverse-scale factors from Qwen3Next.
        inv_scale = self.head_k_dim ** -0.5
        q_c = (inv_scale ** 2) * mx.fast.rms_norm(q_c, None, 1e-6)
        k_c = inv_scale * mx.fast.rms_norm(k_c, None, 1e-6)

        # 6) SSM recurrence.
        state = cache[1] if cache is not None else None
        y, state = gated_delta_update(
            q_c, k_c, v_c, a, b, self.A_log, self.dt_bias, state
        )                                                    # (B, S, Hv, Dv)

        if cache is not None:
            cache[1] = state
            cache.advance(S)

        # 7) gated RMSNorm on head_v_dim, then out_proj.
        y = self.norm(y, z)                                  # (B, S, Hv, Dv)
        return self.out_proj(y.reshape(B, S, -1))
