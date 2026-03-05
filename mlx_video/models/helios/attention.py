import mlx.core as mx
import mlx.nn as nn

from .rope import helios_rope_apply


def _linear_dtype(layer) -> mx.Dtype:
    """Get the compute dtype of a linear layer, handling QuantizedLinear and LoRA wrappers."""
    inner = getattr(layer, "linear", layer)
    if isinstance(inner, nn.QuantizedLinear):
        return inner.scales.dtype
    return inner.weight.dtype


class HeliosRMSNorm(nn.Module):
    """RMS normalization with learnable scale."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


class HeliosLayerNorm(nn.Module):
    """LayerNorm computed in float32, with optional affine."""

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = False):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = mx.ones((dim,))
            self.bias = mx.zeros((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        if self.elementwise_affine:
            return mx.fast.layer_norm(x, self.weight, self.bias, self.eps)
        else:
            return mx.fast.layer_norm(x, None, None, self.eps)


class HeliosSelfAttention(nn.Module):
    """Self-attention with QK normalization, 3-way RoPE, and history restriction.

    When restrict_self_attn=True, the input sequence is split into history
    tokens (from previous chunks) and current tokens. History tokens attend
    only to other history tokens, while current tokens attend to all tokens.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qk_norm: bool = True,
        eps: float = 1e-6,
        restrict_self_attn: bool = False,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.restrict_self_attn = restrict_self_attn

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

        self.norm_q = HeliosRMSNorm(dim, eps=eps) if qk_norm else None
        self.norm_k = HeliosRMSNorm(dim, eps=eps) if qk_norm else None

    def __call__(
        self,
        x: mx.array,
        frame_indices: mx.array,
        grid_size: tuple,
        freqs: tuple,
        rope_cos_sin: tuple | None = None,
        original_context_length: int = 0,
    ) -> mx.array:
        b, s, _ = x.shape
        n, d = self.num_heads, self.head_dim
        history_seq_len = s - original_context_length

        w_dtype = _linear_dtype(self.q)
        x_w = x.astype(w_dtype)

        q = self.q(x_w)
        k = self.k(x_w)
        if self.norm_q is not None:
            q = self.norm_q(q)
        if self.norm_k is not None:
            k = self.norm_k(k)

        if self.restrict_self_attn and history_seq_len > 0:
            # Split history and current
            q_hist, q_curr = q[:, :history_seq_len], q[:, history_seq_len:]
            k_hist, k_curr = k[:, :history_seq_len], k[:, history_seq_len:]
            v_hist = self.v(x_w[:, :history_seq_len])
            v_curr = self.v(x_w[:, history_seq_len:])

            # Reshape to multi-head
            q_hist = q_hist.reshape(b, history_seq_len, n, d)
            k_hist = k_hist.reshape(b, history_seq_len, n, d)
            v_hist = v_hist.reshape(b, history_seq_len, n, d)

            q_curr = q_curr.reshape(b, original_context_length, n, d)
            k_curr = k_curr.reshape(b, original_context_length, n, d)
            v_curr = v_curr.reshape(b, original_context_length, n, d)

            # Apply RoPE to history and current separately
            # History RoPE uses history portion of frame_indices
            all_q = mx.concatenate([q_hist, q_curr], axis=1).reshape(b, s, n, d)
            all_k = mx.concatenate([k_hist, k_curr], axis=1).reshape(b, s, n, d)

            total_seq = history_seq_len + original_context_length
            all_q = helios_rope_apply(
                all_q.astype(mx.float32), frame_indices, (total_seq, 1, 1),
                freqs, precomputed_cos_sin=rope_cos_sin,
            ) if rope_cos_sin is not None else all_q
            all_k = helios_rope_apply(
                all_k.astype(mx.float32), frame_indices, (total_seq, 1, 1),
                freqs, precomputed_cos_sin=rope_cos_sin,
            ) if rope_cos_sin is not None else all_k

            # Actually we need to use the full precomputed rope since it covers
            # the concatenated history+current sequence.
            # Let me simplify: apply RoPE to the full concatenated sequence
            q_full = mx.concatenate([q_hist, q_curr], axis=1).reshape(b, s, n, d)
            k_full = mx.concatenate([k_hist, k_curr], axis=1).reshape(b, s, n, d)

            # RoPE is precomputed for the full (history + current) sequence
            if rope_cos_sin is not None:
                cos_f, sin_f = rope_cos_sin
                q_seq = q_full.reshape(b, s, n, d // 2, 2)
                q_real, q_imag = q_seq[..., 0], q_seq[..., 1]
                q_out_r = q_real * cos_f - q_imag * sin_f
                q_out_i = q_real * sin_f + q_imag * cos_f
                q_full = mx.stack([q_out_r, q_out_i], axis=-1).reshape(b, s, n, d)

                k_seq = k_full.reshape(b, s, n, d // 2, 2)
                k_real, k_imag = k_seq[..., 0], k_seq[..., 1]
                k_out_r = k_real * cos_f - k_imag * sin_f
                k_out_i = k_real * sin_f + k_imag * cos_f
                k_full = mx.stack([k_out_r, k_out_i], axis=-1).reshape(b, s, n, d)

            q_hist = q_full[:, :history_seq_len].astype(w_dtype)
            q_curr = q_full[:, history_seq_len:].astype(w_dtype)
            k_hist = k_full[:, :history_seq_len].astype(w_dtype)
            k_curr = k_full[:, history_seq_len:].astype(w_dtype)
            v_hist_h = v_hist
            v_curr_h = v_curr

            # History self-attention: history attends to history only
            q_h = q_hist.transpose(0, 2, 1, 3)
            k_h = k_hist.transpose(0, 2, 1, 3)
            v_h = v_hist_h.transpose(0, 2, 1, 3)
            hist_out = mx.fast.scaled_dot_product_attention(
                q_h, k_h, v_h, scale=self.scale
            )
            hist_out = hist_out.transpose(0, 2, 1, 3).reshape(b, history_seq_len, -1)

            # Current self-attention: current attends to history + current
            k_all = mx.concatenate([k_hist, k_curr], axis=1).transpose(0, 2, 1, 3)
            v_all = mx.concatenate([v_hist_h, v_curr_h], axis=1).transpose(0, 2, 1, 3)
            q_c = q_curr.transpose(0, 2, 1, 3)
            curr_out = mx.fast.scaled_dot_product_attention(
                q_c, k_all, v_all, scale=self.scale
            )
            curr_out = curr_out.transpose(0, 2, 1, 3).reshape(b, original_context_length, -1)

            out = mx.concatenate([hist_out, curr_out], axis=1)
        else:
            # Standard self-attention (no history)
            q = q.reshape(b, s, n, d)
            k = k.reshape(b, s, n, d)
            v = self.v(x_w).reshape(b, s, n, d)

            if rope_cos_sin is not None:
                cos_f, sin_f = rope_cos_sin
                q_seq = q.astype(mx.float32).reshape(b, s, n, d // 2, 2)
                q_real, q_imag = q_seq[..., 0], q_seq[..., 1]
                q = mx.stack([q_real * cos_f - q_imag * sin_f,
                               q_real * sin_f + q_imag * cos_f], axis=-1).reshape(b, s, n, d)

                k_seq = k.astype(mx.float32).reshape(b, s, n, d // 2, 2)
                k_real, k_imag = k_seq[..., 0], k_seq[..., 1]
                k = mx.stack([k_real * cos_f - k_imag * sin_f,
                               k_real * sin_f + k_imag * cos_f], axis=-1).reshape(b, s, n, d)

            q = q.astype(w_dtype).transpose(0, 2, 1, 3)
            k = k.astype(w_dtype).transpose(0, 2, 1, 3)
            v = v.transpose(0, 2, 1, 3)

            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
            out = out.transpose(0, 2, 1, 3).reshape(b, s, -1)

        return self.o(out)


class HeliosCrossAttention(nn.Module):
    """Cross-attention: Q from hidden states, K/V from text context."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qk_norm: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

        self.norm_q = HeliosRMSNorm(dim, eps=eps) if qk_norm else None
        self.norm_k = HeliosRMSNorm(dim, eps=eps) if qk_norm else None

    def prepare_kv(self, context: mx.array) -> tuple:
        """Pre-compute K and V projections for caching."""
        b = context.shape[0]
        n, d = self.num_heads, self.head_dim
        w_dtype = _linear_dtype(self.k)
        ctx = context.astype(w_dtype)
        k = self.k(ctx)
        if self.norm_k is not None:
            k = self.norm_k(k)
        k = k.reshape(b, -1, n, d).transpose(0, 2, 1, 3)
        v = self.v(ctx).reshape(b, -1, n, d).transpose(0, 2, 1, 3)
        return k, v

    def __call__(
        self,
        x: mx.array,
        context: mx.array,
        kv_cache: tuple | None = None,
    ) -> mx.array:
        b = x.shape[0]
        n, d = self.num_heads, self.head_dim

        w_dtype = _linear_dtype(self.q)
        q = self.q(x.astype(w_dtype))
        if self.norm_q is not None:
            q = self.norm_q(q)
        q = q.reshape(b, -1, n, d).transpose(0, 2, 1, 3)

        if kv_cache is not None:
            k, v = kv_cache
        else:
            ctx = context.astype(w_dtype)
            k = self.k(ctx)
            if self.norm_k is not None:
                k = self.norm_k(k)
            k = k.reshape(b, -1, n, d).transpose(0, 2, 1, 3)
            v = self.v(ctx).reshape(b, -1, n, d).transpose(0, 2, 1, 3)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(0, 2, 1, 3).reshape(b, -1, n * d)
        return self.o(out)
