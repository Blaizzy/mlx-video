import functools
import json
import logging
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.gemma3.config import TextConfig
from mlx_vlm.models.gemma3.language import Gemma3Model
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from mlx_video.utils import apply_quantization, rms_norm

# Path to system prompts
PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_system_prompt(prompt_name: str) -> str:
    """Load a system prompt from the prompts directory."""
    prompt_path = PROMPTS_DIR / prompt_name
    if prompt_path.exists():
        with open(prompt_path, "r") as f:
            return f.read()
    raise FileNotFoundError(f"System prompt not found: {prompt_path}")


class LanguageModel(nn.Module):

    def __init__(self, config: TextConfig):
        super().__init__()
        # Create config matching LTX-2 text encoder requirements
        self.config = config

        # Create the Gemma3Model from mlx-vlm
        self.model = Gemma3Model(self.config)

    def _create_causal_mask_with_padding(
        self,
        seq_len: int,
        attention_mask: Optional[mx.array],
        dtype: mx.Dtype,
    ) -> mx.array:

        causal_mask = mx.tril(mx.ones((seq_len, seq_len), dtype=mx.bool_))

        if attention_mask is not None:
            batch_size = attention_mask.shape[0]

            padding_mask = attention_mask.astype(mx.bool_)  # (batch, seq_len)
            combined = causal_mask[None, :, :] & padding_mask[:, None, :]
            min_val = (
                mx.finfo(dtype).min if dtype in (mx.float16, mx.bfloat16) else -1e9
            )
            mask = mx.where(
                combined,
                mx.zeros(combined.shape, dtype=dtype),
                mx.full(combined.shape, min_val, dtype=dtype),
            )
            return mask[:, None, :, :]
        else:
            # No padding mask, just causal
            min_val = (
                mx.finfo(dtype).min if dtype in (mx.float16, mx.bfloat16) else -1e9
            )
            mask = mx.where(
                causal_mask,
                mx.zeros((seq_len, seq_len), dtype=dtype),
                mx.full((seq_len, seq_len), min_val, dtype=dtype),
            )
            return mask[None, None, :, :]  # (1, 1, seq, seq)

    def __call__(
        self,
        inputs: mx.array,
        input_embeddings: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        output_hidden_states: bool = False,
        cache: Optional[List[mx.array]] = None,
    ) -> Tuple[mx.array, List[mx.array]]:
        """Forward pass returning hidden states.

        Args:
            input_ids: Input token IDs of shape (batch, seq_len)
            attention_mask: Optional attention mask (1 for valid, 0 for padding)
            output_hidden_states: Whether to return all hidden states

        Returns:
            Tuple of (final_hidden_states, list_of_all_hidden_states)
        """
        batch_size, seq_len = inputs.shape

        # Get embeddings
        h = (
            input_embeddings
            if input_embeddings is not None
            else self.model.embed_tokens(inputs)
        )

        # Apply Gemma scaling
        h *= mx.array(self.config.hidden_size**0.5, mx.bfloat16).astype(h.dtype)
        mx.eval(h)

        all_hidden_states = [h] if output_hidden_states else []

        # Set up cache (all None for non-cached inference)
        if cache is None:
            cache = [None] * len(self.model.layers)

        full_causal_mask = self._create_causal_mask_with_padding(
            seq_len, attention_mask, h.dtype
        )

        sliding_mask = full_causal_mask

        num_layers = len(self.model.layers)
        for i, layer in enumerate(self.model.layers):
            is_global = (
                i % self.config.sliding_window_pattern
                == self.config.sliding_window_pattern - 1
            )

            # Select appropriate mask for this layer
            if is_global:
                local_mask = full_causal_mask
            else:
                local_mask = sliding_mask

            h = layer(h, local_mask, cache[i])
            mx.eval(h)

            if output_hidden_states and i < num_layers - 1:
                all_hidden_states.append(h)

        # Apply final norm
        hidden_states = self.model.norm(h)
        mx.eval(hidden_states)

        # Append the final normalized output as the last hidden state
        if output_hidden_states:
            all_hidden_states.append(hidden_states)

            return hidden_states, all_hidden_states

        else:
            # Return logits
            return self.model.embed_tokens.as_linear(hidden_states)

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        prefix = "language_model."
        sanitized = {}
        for key, value in weights.items():
            if key.startswith(prefix):
                if hasattr(value, "dtype") and value.dtype == mx.float32:
                    sanitized[key[len(prefix) :]] = value.astype(mx.bfloat16)
                else:
                    sanitized[key[len(prefix) :]] = value
        return sanitized

    @property
    def layers(self) -> List[nn.Module]:
        return self.model.layers

    def make_cache(self):
        from mlx_vlm.models.cache import KVCache, RotatingKVCache

        caches = []
        for i in range(len(self.layers)):
            if (
                i % self.config.sliding_window_pattern
                == self.config.sliding_window_pattern - 1
            ):
                caches.append(KVCache())
            else:
                caches.append(RotatingKVCache(max_size=self.config.sliding_window))
        return caches

    @classmethod
    def from_pretrained(cls, model_path: str):
        import json

        weight_files = sorted(Path(model_path).glob("*.safetensors"))
        config_file = Path(model_path) / "config.json"
        config_dict = {}
        if config_file.exists():
            with open(config_file, "r") as f:
                config_dict = json.load(f)

            language_model = cls(
                config=TextConfig.from_dict(config_dict["text_config"])
            )
        else:
            raise ValueError(f"Config file not found at {model_path}")

        quantization = config_dict.get("quantization", None)
        weights = {}
        for i, wf in enumerate(weight_files):
            weights.update(mx.load(str(wf)))

        if hasattr(language_model, "sanitize"):
            weights = language_model.sanitize(weights=weights)

        apply_quantization(
            model=language_model, weights=weights, quantization=quantization
        )

        language_model.load_weights(list(weights.items()), strict=False)

        return language_model


class Gemma4LanguageModel(nn.Module):
    """Adapter exposing Gemma 4 hidden states through the existing LTX API."""

    def __init__(self, config: dict):
        super().__init__()
        from mlx_vlm.models.gemma4.language import LanguageModel as Gemma4Model
        from mlx_vlm.models.gemma4_unified.config import TextConfig as Gemma4Config

        self.config = Gemma4Config.from_dict(config["text_config"])
        self.model = Gemma4Model(self.config)

    def _attention_mask(self, attention_mask: mx.array, dtype: mx.Dtype) -> mx.array:
        seq_len = attention_mask.shape[1]
        causal = mx.tril(mx.ones((seq_len, seq_len), dtype=mx.bool_))
        allowed = causal[None, :, :] & attention_mask.astype(mx.bool_)[:, None, :]
        min_value = (
            mx.finfo(dtype).min if dtype in (mx.float16, mx.bfloat16) else -1e9
        )
        return mx.where(
            allowed[:, None, :, :],
            mx.zeros((1,), dtype=dtype),
            mx.array(min_value, dtype=dtype),
        )

    def __call__(
        self,
        inputs: mx.array,
        input_embeddings: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        output_hidden_states: bool = False,
        cache=None,
    ):
        inner = self.model.model
        hidden = (
            input_embeddings
            if input_embeddings is not None
            else inner.embed_tokens(inputs) * inner.embed_scale
        )
        states = [hidden] if output_hidden_states else []
        mask = (
            self._attention_mask(attention_mask, hidden.dtype)
            if attention_mask is not None
            else None
        )
        capture_ids = (
            list(range(len(inner.layers) - 1)) if output_hidden_states else None
        )
        output = inner(
            None,
            inputs_embeds=hidden,
            mask=mask,
            cache=cache,
            capture_layer_ids=capture_ids,
            hidden_sink=states if output_hidden_states else None,
        )
        if output_hidden_states:
            states.append(output)
            return output, states
        return self.model.logits_from_hidden(output)

    @classmethod
    def from_pretrained(cls, checkpoint_path: str | Path):
        from safetensors import safe_open

        checkpoint_path = Path(checkpoint_path)
        with safe_open(checkpoint_path, framework="numpy") as f:
            config = json.loads((f.metadata() or {})["gemma_config"])
        model = cls(config)
        weights = {
            key: value
            for key, value in mx.load(str(checkpoint_path)).items()
            if key.startswith("model.")
        }
        model.model.load_weights(list(weights.items()), strict=True)
        return model


def load_packed_tokenizer(checkpoint_path: str | Path):
    """Build the Hugging Face tokenizer embedded in an LTX-2.5 TE pack."""
    from safetensors import safe_open
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast

    with safe_open(checkpoint_path, framework="numpy") as f:
        tokenizer_json = f.get_tensor("tokenizer_json").tobytes()
        tokenizer_config = json.loads(
            f.get_tensor("hf_asset__tokenizer_config.json").tobytes()
        )
        chat_template = (
            f.get_tensor("hf_asset__chat_template.jinja").tobytes().decode()
        )

    ignored = {
        "tokenizer_class",
        "auto_map",
        "model_max_length",
        "backend",
        "is_local",
        "local_files_only",
        "processor_class",
        "added_tokens_decoder",
    }
    kwargs = {k: v for k, v in tokenizer_config.items() if k not in ignored}
    kwargs.setdefault("chat_template", chat_template)
    return PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer.from_buffer(tokenizer_json),
        model_max_length=1024,
        **kwargs,
    )


def enhance_prompt_gemma4(
    prompt: str,
    model_repo: str,
    image: str | None = None,
    max_tokens: int = 600,
    seed: int = 42,
    verbose: bool = True,
) -> str:
    """Enhance an LTX-2.5 prompt with a separate generative Gemma 4 model."""
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template

    model, processor = load(model_repo)
    prompt_name = (
        "gemma4_i2v_system_prompt.txt" if image else "gemma4_t2v_system_prompt.txt"
    )
    messages = [
        {"role": "system", "content": _load_system_prompt(prompt_name)},
        {"role": "user", "content": f"user prompt: {prompt}"},
    ]
    formatted = apply_chat_template(
        processor,
        model.config,
        messages,
        num_images=1 if image else 0,
    )
    mx.random.seed(seed)
    result = generate(
        model=model,
        processor=processor,
        prompt=formatted,
        image=image,
        max_tokens=max_tokens,
        temperature=0.0,
        verbose=verbose,
    )
    enhanced = re.sub(r"^[^\w\s]+", "", result.text.strip())
    del model
    mx.clear_cache()
    return enhanced


class ConnectorAttention(nn.Module):

    def __init__(
        self,
        dim: int = 3840,
        num_heads: int = 30,
        head_dim: int = 128,
        has_gate_logits: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim
        self.scale = 1.0 / math.sqrt(head_dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=True)
        self.to_k = nn.Linear(dim, inner_dim, bias=True)
        self.to_v = nn.Linear(dim, inner_dim, bias=True)
        self.to_out = nn.Linear(inner_dim, dim, bias=True)

        self.q_norm = nn.RMSNorm(inner_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(inner_dim, eps=1e-6)

        if has_gate_logits:
            self.to_gate_logits = nn.Linear(dim, num_heads, bias=True)

    def __call__(
        self,
        x: mx.array,
        attention_mask: Optional[mx.array] = None,
        pe: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        batch_size, seq_len, _ = x.shape

        # Compute per-head gate early (from original input)
        gate = None
        if hasattr(self, "to_gate_logits"):
            gate = 2.0 * mx.sigmoid(self.to_gate_logits(x))  # (B, seq, heads)

        # Project to Q, K, V
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # QK normalization on full inner_dim BEFORE reshape
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Reshape to (B, H, T, D) for SPLIT RoPE
        q = mx.reshape(
            q, (batch_size, seq_len, self.num_heads, self.head_dim)
        ).transpose(0, 2, 1, 3)
        k = mx.reshape(
            k, (batch_size, seq_len, self.num_heads, self.head_dim)
        ).transpose(0, 2, 1, 3)
        v = mx.reshape(
            v, (batch_size, seq_len, self.num_heads, self.head_dim)
        ).transpose(0, 2, 1, 3)

        if pe is not None:
            q = self._apply_split_rope(q, pe[0], pe[1])
            k = self._apply_split_rope(k, pe[0], pe[1])

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=None)
        out = out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)

        # Apply per-head gating
        if gate is not None:
            out = mx.reshape(out, (batch_size, seq_len, self.num_heads, self.head_dim))
            out = out * gate[..., None]
            out = mx.reshape(out, (batch_size, seq_len, -1))

        return self.to_out(out)

    def _apply_split_rope(
        self,
        x: mx.array,
        cos_freq: mx.array,
        sin_freq: mx.array,
    ) -> mx.array:
        """Apply SPLIT RoPE to input tensor.

        Args:
            x: Input tensor of shape (B, H, T, D)
            cos_freq: Cosine frequencies of shape (1, H, T, D//2)
            sin_freq: Sine frequencies of shape (1, H, T, D//2)

        Returns:
            Tensor with SPLIT rotary embeddings applied
        """
        input_dtype = x.dtype

        # Cast to float32 for precision, then cast back
        x = x.astype(mx.float32)
        cos_freq = cos_freq.astype(mx.float32)
        sin_freq = sin_freq.astype(mx.float32)

        # Split x into two halves: (B, H, T, D) -> two tensors of (B, H, T, D//2)
        half_dim = x.shape[-1] // 2
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]

        # Apply rotation: SPLIT pattern
        # out1 = x1 * cos - x2 * sin
        # out2 = x2 * cos + x1 * sin
        out1 = x1 * cos_freq - x2 * sin_freq
        out2 = x2 * cos_freq + x1 * sin_freq

        return mx.concatenate([out1, out2], axis=-1).astype(input_dtype)


class GEGLU(nn.Module):
    """GELU-gated linear unit."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return nn.gelu(self.proj(x))


class ConnectorFeedForward(nn.Module):

    def __init__(self, dim: int = 3840, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim * mult
        # Use explicit named attributes to match weight key structure (proj_in, proj_out)
        self.proj_in = nn.Linear(dim, inner_dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = nn.Linear(inner_dim, dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.gelu_approx(self.proj_in(x))
        x = self.dropout(x)
        x = self.proj_out(x)
        return x


class ConnectorTransformerBlock(nn.Module):

    def __init__(
        self,
        dim: int = 3840,
        num_heads: int = 30,
        head_dim: int = 128,
        has_gate_logits: bool = False,
    ):
        super().__init__()
        self.attn1 = ConnectorAttention(
            dim, num_heads, head_dim, has_gate_logits=has_gate_logits
        )
        self.ff = ConnectorFeedForward(dim)

    def __call__(
        self,
        x: mx.array,
        attention_mask: Optional[mx.array] = None,
        pe: Optional[mx.array] = None,
    ) -> mx.array:
        # Pre-norm + attention + residual
        norm_x = rms_norm(x)
        if norm_x.ndim == 4:
            norm_x = mx.squeeze(norm_x, axis=1)
        attn_out = self.attn1(norm_x, attention_mask, pe)
        x = x + attn_out
        if x.ndim == 4:
            x = mx.squeeze(x, axis=1)

        # Pre-norm + FFN + residual
        norm_x = rms_norm(x)
        ff_out = self.ff(norm_x)
        x = x + ff_out
        if x.ndim == 4:
            x = mx.squeeze(x, axis=1)

        return x


class Embeddings1DConnector(nn.Module):

    def __init__(
        self,
        dim: int = 3840,
        num_heads: int = 30,
        head_dim: int = 128,
        num_layers: int = 2,
        num_learnable_registers: int = 128,
        positional_embedding_theta: float = 10000.0,
        positional_embedding_max_pos: list = None,
        has_gate_logits: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_learnable_registers = num_learnable_registers
        self.positional_embedding_theta = positional_embedding_theta
        self.positional_embedding_max_pos = positional_embedding_max_pos or [1]

        self.transformer_1d_blocks = {
            i: ConnectorTransformerBlock(
                dim, num_heads, head_dim, has_gate_logits=has_gate_logits
            )
            for i in range(num_layers)
        }

        if num_learnable_registers > 0:
            self.learnable_registers = mx.zeros((num_learnable_registers, dim))

    def _precompute_freqs_cis(
        self, seq_len: int, dtype: mx.Dtype
    ) -> Tuple[mx.array, mx.array]:
        """Compute RoPE frequencies for connector (SPLIT type matching PyTorch).

        Returns tuple of (cos, sin) each with shape (1, num_heads, seq_len, head_dim//2).
        """

        import numpy as np

        dim = self.num_heads * self.head_dim  # inner_dim
        theta = self.positional_embedding_theta
        max_pos = self.positional_embedding_max_pos  # [1] = PyTorch default
        n_elem = 2 * len(max_pos)  # = 2

        start = 1.0
        end = theta
        num_indices = dim // n_elem

        # generate_freq_grid_np: compute indices in float64 then cast to float32
        # (matches PyTorch: double_precision_rope generates in np.float64,
        # but returns torch.float32)
        log_start = np.log(start) / np.log(theta)  # = 0
        log_end = np.log(end) / np.log(theta)  # = 1
        lin_space = np.linspace(log_start, log_end, num_indices, dtype=np.float64)
        indices = (np.power(theta, lin_space) * (np.pi / 2)).astype(np.float32)

        # generate_freqs: positions and freqs in float32 (matching PyTorch)
        positions = np.arange(seq_len, dtype=np.float32)
        fractional_positions = positions / max_pos[0]
        scaled_positions = fractional_positions * 2 - 1  # Shape: (seq_len,)

        # freqs = scaled_positions * indices (outer product) in float32
        freqs = scaled_positions[:, None] * indices[None, :]

        # split_freqs_cis: cos/sin in float32 (matching PyTorch)
        expected_freqs = dim // 2
        pad_size = expected_freqs - freqs.shape[-1]

        cos_freq = np.cos(freqs)  # (seq_len, num_indices)
        sin_freq = np.sin(freqs)

        if pad_size > 0:
            cos_padding = np.ones((seq_len, pad_size), dtype=np.float32)
            sin_padding = np.zeros((seq_len, pad_size), dtype=np.float32)
            cos_freq = np.concatenate([cos_padding, cos_freq], axis=-1)
            sin_freq = np.concatenate([sin_padding, sin_freq], axis=-1)

        # Reshape: (T, dim//2) -> (T, H, D//2) -> (1, H, T, D//2)
        cos_freq = cos_freq.reshape(seq_len, self.num_heads, self.head_dim // 2)
        sin_freq = sin_freq.reshape(seq_len, self.num_heads, self.head_dim // 2)

        cos_freq = np.transpose(cos_freq, (1, 0, 2))[np.newaxis, ...]
        sin_freq = np.transpose(sin_freq, (1, 0, 2))[np.newaxis, ...]

        # Convert to MLX
        cos_full = mx.array(cos_freq)
        sin_full = mx.array(sin_freq)

        return cos_full.astype(dtype), sin_full.astype(dtype)

    def _replace_padded_with_registers(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        batch_size, seq_len, dim = hidden_states.shape
        dtype = hidden_states.dtype

        # Binary mask: 1 for valid tokens, 0 for padded
        # attention_mask is additive: 0 for valid, large negative for padded
        mask_binary = (attention_mask.squeeze(1).squeeze(1) >= -9000.0).astype(
            mx.int32
        )  # (batch, seq)

        # Tile registers to match sequence length, cast to hidden_states dtype
        num_tiles = seq_len // self.num_learnable_registers
        registers = mx.tile(self.learnable_registers, (num_tiles, 1)).astype(
            dtype
        )  # (seq_len, dim)

        # Process each batch item (PyTorch uses advanced indexing)
        result_list = []
        for b in range(batch_size):
            mask_b = mask_binary[b]  # (seq,)
            hs_b = hidden_states[b]  # (seq, dim)

            # Count valid tokens
            num_valid = int(mx.sum(mask_b))

            # Extract valid tokens (where mask is 1)
            # Since we have left-padded input, valid tokens are at the end
            valid_tokens = hs_b[seq_len - num_valid :]  # (num_valid, dim)

            # Pad with zeros on the right to get back to seq_len
            pad_length = seq_len - num_valid
            if pad_length > 0:
                padding = mx.zeros((pad_length, dim), dtype=dtype)
                adjusted = mx.concatenate(
                    [valid_tokens, padding], axis=0
                )  # (seq_len, dim)
            else:
                adjusted = valid_tokens

            # Create flipped mask: 1s at front (where valid tokens now are), 0s at back
            flipped_mask = mx.concatenate(
                [
                    mx.ones((num_valid,), dtype=mx.int32),
                    mx.zeros((pad_length,), dtype=mx.int32),
                ],
                axis=0,
            )  # (seq,)

            # Combine: valid tokens at front, registers at back
            flipped_mask_expanded = flipped_mask[:, None].astype(dtype)  # (seq, 1)
            combined = (
                flipped_mask_expanded * adjusted
                + (1 - flipped_mask_expanded) * registers
            )
            result_list.append(combined)

        hidden_states = mx.stack(result_list, axis=0)  # (batch, seq, dim)

        # Reset attention mask to all zeros (no masking after register replacement)
        attention_mask = mx.zeros_like(attention_mask)

        return hidden_states, attention_mask

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        # Replace padded tokens with learnable registers
        if self.num_learnable_registers > 0 and attention_mask is not None:
            hidden_states, attention_mask = self._replace_padded_with_registers(
                hidden_states, attention_mask
            )

        # Compute RoPE frequencies
        seq_len = hidden_states.shape[1]
        freqs_cis = self._precompute_freqs_cis(seq_len, hidden_states.dtype)

        # Process through transformer blocks
        for i in range(len(self.transformer_1d_blocks)):
            hidden_states = self.transformer_1d_blocks[i](
                hidden_states, attention_mask, freqs_cis
            )

        # Final RMS norm
        hidden_states = rms_norm(hidden_states)

        return hidden_states, attention_mask


def norm_and_concat_hidden_states(
    hidden_states: List[mx.array],
    attention_mask: mx.array,
    padding_side: str = "left",
) -> mx.array:

    # Stack hidden states: (batch, seq, dim, num_layers)
    stacked = mx.stack(hidden_states, axis=-1)
    dtype = stacked.dtype
    b, t, d, num_layers = stacked.shape

    # Compute sequence lengths from attention mask
    sequence_lengths = mx.sum(attention_mask, axis=-1)  # (batch,)

    # Build mask based on padding side
    token_indices = mx.arange(t)[None, :]  # (1, T)

    if padding_side == "right":
        mask = token_indices < sequence_lengths[:, None]  # (B, T)
    else:  # left padding
        start_indices = t - sequence_lengths[:, None]  # (B, 1)
        mask = token_indices >= start_indices  # (B, T)

    mask = mask[:, :, None, None]  # (B, T, 1, 1)
    eps = mx.array(1e-6, dtype=dtype)

    # Compute masked mean per layer - ensure dtype consistency
    masked = mx.where(mask, stacked, mx.zeros_like(stacked))
    denom = (sequence_lengths * d).reshape(b, 1, 1, 1).astype(dtype)
    mean = mx.sum(masked, axis=(1, 2), keepdims=True) / (denom + eps)

    # Compute masked min/max per layer
    x_for_min = mx.where(
        mask, stacked, mx.full(stacked.shape, float("inf"), dtype=dtype)
    )
    x_for_max = mx.where(
        mask, stacked, mx.full(stacked.shape, float("-inf"), dtype=dtype)
    )
    x_min = mx.min(x_for_min, axis=(1, 2), keepdims=True)
    x_max = mx.max(x_for_max, axis=(1, 2), keepdims=True)
    range_val = x_max - x_min

    # Normalize: 8 * (x - mean) / range
    normed = 8 * (stacked - mean) / (range_val + eps)

    # Flatten layers into feature dimension: (B, T, D*L)
    normed = mx.reshape(normed, (b, t, -1))

    # Zero out padded positions
    mask_flat = mx.broadcast_to(mask[:, :, :, 0], (b, t, d * num_layers))
    normed = mx.where(mask_flat, normed, mx.zeros_like(normed))

    return normed


def norm_and_concat_per_token_rms(
    encoded_text: mx.array,
    attention_mask: mx.array,
) -> mx.array:
    """Per-token RMSNorm normalization for V2 feature extraction (LTX-2.3).

    Args:
        encoded_text: (B, T, D, L) stacked hidden states
        attention_mask: (B, T) binary mask (1=valid, 0=padding)

    Returns:
        (B, T, D*L) normalized tensor with padding zeroed out.
    """
    b, t, d, num_layers = encoded_text.shape
    dtype = encoded_text.dtype

    # Per-token RMSNorm across hidden dimension: variance = mean(x^2) over dim D
    variance = mx.mean(
        encoded_text.astype(mx.float32) ** 2, axis=2, keepdims=True
    )  # (B, T, 1, L)
    normed = encoded_text.astype(mx.float32) * mx.rsqrt(variance + 1e-6)
    normed = normed.astype(dtype)

    # Flatten layers: (B, T, D*L)
    normed = mx.reshape(normed, (b, t, d * num_layers))

    # Zero out padded positions
    mask_3d = attention_mask[:, :, None].astype(mx.bool_)  # (B, T, 1)
    normed = mx.where(mask_3d, normed, mx.zeros_like(normed))

    return normed


def _rescale_norm(x: mx.array, target_dim: int, source_dim: int) -> mx.array:
    """Rescale normalization: x * sqrt(target_dim / source_dim)."""
    return x * math.sqrt(target_dim / source_dim)


class GemmaFeaturesExtractor(nn.Module):
    """V1 feature extractor (LTX-2): 8 * (x - mean) / range normalization."""

    def __init__(
        self, input_dim: int = 188160, output_dim: int = 3840, bias: bool = False
    ):
        super().__init__()
        self.aggregate_embed = nn.Linear(input_dim, output_dim, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.aggregate_embed(x)


class GemmaFeaturesExtractorV2(nn.Module):
    """V2 feature extractor (LTX-2.3): per-token RMSNorm + rescale normalization."""

    def __init__(
        self,
        flat_dim: int,
        embedding_dim: int,
        video_output_dim: int,
        audio_output_dim: int,
        bias: bool = True,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim  # Gemma hidden_dim (3840), used for rescale
        self.video_aggregate_embed = nn.Linear(flat_dim, video_output_dim, bias=bias)
        self.audio_aggregate_embed = nn.Linear(flat_dim, audio_output_dim, bias=bias)

    def __call__(
        self,
        hidden_states: List[mx.array],
        attention_mask: mx.array,
        mode: str = "video",
    ) -> mx.array:
        """Extract features with per-token RMSNorm + rescale.

        Args:
            hidden_states: List of hidden states from all Gemma layers
            attention_mask: Binary attention mask (B, T)
            mode: "video" or "audio" to select which aggregate embed to use

        Returns:
            Projected features
        """
        # Stack hidden states: (B, T, D, L)
        encoded = mx.stack(hidden_states, axis=-1)

        # Per-token RMSNorm + flatten
        normed = norm_and_concat_per_token_rms(encoded, attention_mask)
        normed = normed.astype(encoded.dtype)

        if mode == "video":
            target_dim = self.video_aggregate_embed.weight.shape[0]
            return self.video_aggregate_embed(
                _rescale_norm(normed, target_dim, self.embedding_dim)
            )
        else:
            target_dim = self.audio_aggregate_embed.weight.shape[0]
            return self.audio_aggregate_embed(
                _rescale_norm(normed, target_dim, self.embedding_dim)
            )


class AudioEmbeddingsConnector(nn.Module):
    """Projects video embeddings to audio cross-attention dimension."""

    def __init__(self, input_dim: int = 3840, output_dim: int = 2048):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear(x)


class LTX2TextEncoder(nn.Module):

    def __init__(
        self,
        hidden_dim: int = 3840,
        audio_dim: int = 2048,
        num_layers: int = 49,  # 48 transformer layers + 1 embedding
        has_prompt_adaln: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.audio_dim = audio_dim
        self.num_layers = num_layers
        self.has_prompt_adaln = has_prompt_adaln
        self.language_model = None

        feature_input_dim = hidden_dim * num_layers

        if has_prompt_adaln:
            # LTX-2.3: V2 feature extractor with per-token RMSNorm + rescale
            video_output_dim = 4096
            audio_output_dim = 2048
            self.feature_extractor_v2 = GemmaFeaturesExtractorV2(
                flat_dim=feature_input_dim,  # 3840 * 49 = 188160 (concatenated)
                embedding_dim=hidden_dim,  # 3840 (Gemma hidden_dim, for rescale)
                video_output_dim=video_output_dim,
                audio_output_dim=audio_output_dim,
                bias=True,
            )

            # Deeper connectors with matching dims and gate_logits
            # connector_positional_embedding_max_pos=[4096] from LTX-2.3 safetensors
            # config (nested under config.transformer.connector_positional_embedding_max_pos)
            self.video_embeddings_connector = Embeddings1DConnector(
                dim=video_output_dim,
                num_heads=32,
                head_dim=128,
                num_layers=8,
                num_learnable_registers=128,
                positional_embedding_max_pos=[4096],
                has_gate_logits=True,
            )
            self.audio_embeddings_connector = Embeddings1DConnector(
                dim=audio_output_dim,
                num_heads=32,
                head_dim=64,
                num_layers=8,
                num_learnable_registers=128,
                positional_embedding_max_pos=[4096],
                has_gate_logits=True,
            )
        else:
            # LTX-2: shared feature extractor, 3840-dim connectors
            self.feature_extractor = GemmaFeaturesExtractor(
                feature_input_dim, hidden_dim
            )

            self.video_embeddings_connector = Embeddings1DConnector(
                dim=hidden_dim,
                num_heads=30,
                head_dim=128,
                num_layers=2,
                num_learnable_registers=128,
                positional_embedding_max_pos=[1],
            )
            self.audio_embeddings_connector = Embeddings1DConnector(
                dim=hidden_dim,
                num_heads=30,
                head_dim=128,
                num_layers=2,
                num_learnable_registers=128,
                positional_embedding_max_pos=[1],
            )

        self.processor = None

    def load(
        self,
        model_path: Optional[str] = None,
        text_encoder_path: Optional[str] = "google/gemma-3-12b-it",
    ):

        model_path = Path(model_path)
        text_encoder_path = Path(text_encoder_path)

        if text_encoder_path.is_file():
            self.language_model = Gemma4LanguageModel.from_pretrained(
                text_encoder_path
            )

            packed_weights = mx.load(str(text_encoder_path))
            projection_weights = {
                key.removeprefix("text_embedding_projection."): value
                for key, value in packed_weights.items()
                if key.startswith("text_embedding_projection.")
            }
            self._load_feature_extractors(projection_weights, is_reformatted=True)

            transformer_files = sorted(
                (model_path / "diffusion_models").glob(
                    "*distilled-transformer*.safetensors"
                )
            )
            if not transformer_files:
                raise FileNotFoundError(
                    f"No distilled transformer checkpoint found under {model_path}"
                )
            transformer_weights = mx.load(str(transformer_files[0]))
            self._load_connector(
                "video_embeddings_connector", transformer_weights, False
            )
            self._load_connector(
                "audio_embeddings_connector", transformer_weights, False
            )
            self.processor = load_packed_tokenizer(text_encoder_path)
            self.processor.padding_side = "left"
            print("LTX-2.5 Gemma 4 text encoder loaded successfully")
            return

        if Path(str(text_encoder_path)).joinpath("text_encoder").is_dir():
            text_encoder_path = str(Path(text_encoder_path) / "text_encoder")

        self.language_model = LanguageModel.from_pretrained(text_encoder_path)

        # Load transformer weights for feature extractor and connector.
        # These weights are stored differently depending on the repo format:
        # 1. Monolithic (Lightricks/LTX-2): single ltx-2-19b-*.safetensors at root
        #    with raw PyTorch key names (model.diffusion_model.* prefix)
        # 2. Reformatted (prince-canuma/LTX-2-distilled): separate text_projections/
        #    directory with pre-sanitized keys (no prefix, already renamed)
        transformer_weights = {}
        is_reformatted = False

        # Try reformatted layout first: text_projections/ subdirectory
        text_proj_dir = model_path / "text_projections"
        if text_proj_dir.is_dir():
            is_reformatted = True
            for sf in text_proj_dir.glob("*.safetensors"):
                transformer_weights.update(mx.load(str(sf)))

        # Fall back to monolithic layout: ltx-2-19*.safetensors at root
        if not transformer_weights:
            transformer_files = list(model_path.glob("ltx-2-19*.safetensors"))
            if transformer_files:
                transformer_weights = mx.load(str(transformer_files[0]))

        if transformer_weights:
            self._load_feature_extractors(transformer_weights, is_reformatted)
            self._load_connector(
                "video_embeddings_connector", transformer_weights, is_reformatted
            )
            self._load_connector(
                "audio_embeddings_connector", transformer_weights, is_reformatted
            )
        else:
            print(
                "WARNING: No transformer weights found for text projection connectors. "
                "Text conditioning will use uninitialized weights!"
            )

        # Load tokenizer
        from transformers import AutoTokenizer

        tokenizer_path = model_path / "tokenizer"
        if tokenizer_path.exists():
            self.processor = AutoTokenizer.from_pretrained(
                str(tokenizer_path), trust_remote_code=True
            )
        else:
            try:
                self.processor = AutoTokenizer.from_pretrained(
                    text_encoder_path, trust_remote_code=True
                )
            except Exception:
                self.processor = AutoTokenizer.from_pretrained(
                    "google/gemma-3-12b-it", trust_remote_code=True
                )
        # Set left padding to match official LTX-2 text encoder
        self.processor.padding_side = "left"

        print("Text encoder loaded successfully")

    def _load_feature_extractors(self, weights: dict, is_reformatted: bool):
        """Load feature extractor weights for both LTX-2 and LTX-2.3."""
        if self.has_prompt_adaln:
            # LTX-2.3: V2 feature extractor with separate video/audio aggregate embeds
            for attr, prefix in [
                ("video_aggregate_embed", "video_aggregate_embed"),
                ("audio_aggregate_embed", "audio_aggregate_embed"),
            ]:
                w_key = f"{prefix}.weight"
                b_key = f"{prefix}.bias"
                if w_key in weights:
                    submodule = getattr(self.feature_extractor_v2, attr)
                    submodule.weight = weights[w_key]
                    if b_key in weights:
                        submodule.bias = weights[b_key]
        else:
            # LTX-2: single aggregate_embed
            agg_key = (
                "aggregate_embed.weight"
                if is_reformatted
                else "text_embedding_projection.aggregate_embed.weight"
            )
            if agg_key in weights:
                self.feature_extractor.aggregate_embed.weight = weights[agg_key]

    def _load_connector(self, name: str, weights: dict, is_reformatted: bool):
        """Load a connector's weights (video or audio)."""
        connector = getattr(self, name)

        # Extract connector-specific weights
        connector_weights = {}
        if is_reformatted:
            prefix = f"{name}."
            for key, value in weights.items():
                if key.startswith(prefix):
                    connector_weights[key[len(prefix) :]] = value
        else:
            mono_prefix = f"model.diffusion_model.{name}."
            for key, value in weights.items():
                if key.startswith(mono_prefix):
                    connector_weights[key[len(mono_prefix) :]] = value

        if not connector_weights:
            return

        # Sanitize key names (only needed for monolithic/raw PyTorch keys)
        mapped = {}
        for key, value in connector_weights.items():
            new_key = key
            if not is_reformatted:
                new_key = new_key.replace(".ff.net.0.proj.", ".ff.proj_in.")
                new_key = new_key.replace(".ff.net.2.", ".ff.proj_out.")
                new_key = new_key.replace(".to_out.0.", ".to_out.")
            mapped[new_key] = value

        connector.load_weights(list(mapped.items()), strict=False)

        # Manually load learnable_registers (plain mx.array, not tracked as parameter)
        if "learnable_registers" in connector_weights:
            connector.learnable_registers = connector_weights["learnable_registers"]

    def encode(
        self,
        prompt: str,
        max_length: int = 1024,
        return_audio_embeddings: bool = True,
    ) -> Tuple[mx.array, mx.array]:
        """Encode text prompt to video and audio embeddings.

        Args:
            prompt: Text prompt to encode
            max_length: Maximum token length (default 1024 to match official PyTorch)
            return_audio_embeddings: If True, returns (video_emb, audio_emb).
                                     If False, returns (video_emb, attention_mask).

        Returns:
            Tuple of (video_embeddings, audio_embeddings) if return_audio_embeddings=True
            Tuple of (video_embeddings, attention_mask) otherwise
        """
        if self.processor is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        inputs = self.processor(
            prompt,
            return_tensors="np",
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )
        input_ids = mx.array(inputs["input_ids"])
        attention_mask = mx.array(inputs["attention_mask"])

        _, all_hidden_states = self.language_model(
            inputs=input_ids,
            input_embeddings=None,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        if self.has_prompt_adaln:
            # LTX-2.3: V2 feature extraction (per-token RMSNorm + rescale)
            video_features = self.feature_extractor_v2(
                all_hidden_states, attention_mask, mode="video"
            )
            additive_mask = (attention_mask - 1).astype(video_features.dtype)
            additive_mask = (
                additive_mask.reshape(attention_mask.shape[0], 1, 1, -1) * 1e9
            )

            video_embeddings, _ = self.video_embeddings_connector(
                video_features, additive_mask
            )

            if return_audio_embeddings:
                audio_features = self.feature_extractor_v2(
                    all_hidden_states, attention_mask, mode="audio"
                )
                audio_mask = (attention_mask - 1).astype(audio_features.dtype)
                audio_mask = audio_mask.reshape(attention_mask.shape[0], 1, 1, -1) * 1e9
                audio_embeddings, _ = self.audio_embeddings_connector(
                    audio_features, audio_mask
                )
                return video_embeddings, audio_embeddings
            else:
                return video_embeddings, attention_mask
        else:
            # LTX-2: V1 feature extraction (8 * (x - mean) / range)
            concat_hidden = norm_and_concat_hidden_states(
                all_hidden_states, attention_mask, padding_side="left"
            )

            video_features = self.feature_extractor(concat_hidden)
            additive_mask = (attention_mask - 1).astype(video_features.dtype)
            additive_mask = (
                additive_mask.reshape(attention_mask.shape[0], 1, 1, -1) * 1e9
            )

            video_embeddings, _ = self.video_embeddings_connector(
                video_features, additive_mask
            )

            if return_audio_embeddings:
                audio_embeddings, _ = self.audio_embeddings_connector(
                    video_features, additive_mask
                )
                return video_embeddings, audio_embeddings
            else:
                return video_embeddings, attention_mask

    def __call__(
        self,
        prompt: str,
        max_length: int = 1024,
        return_audio_embeddings: bool = True,
    ) -> Tuple[mx.array, mx.array]:
        """Encode text prompt.

        Args:
            prompt: Text prompt to encode
            max_length: Maximum token length (default 1024 to match official PyTorch)
            return_audio_embeddings: If True, returns (video_emb, audio_emb).
                                     If False, returns (video_emb, attention_mask).

        Returns:
            Tuple of embeddings based on return_audio_embeddings flag
        """
        return self.encode(prompt, max_length, return_audio_embeddings)

    @functools.cached_property
    def default_t2v_system_prompt(self) -> str:
        """Load the default T2V system prompt."""
        prompt_name = (
            "gemma4_t2v_system_prompt.txt"
            if isinstance(self.language_model, Gemma4LanguageModel)
            else "gemma_t2v_system_prompt.txt"
        )
        return _load_system_prompt(prompt_name)

    @functools.cached_property
    def default_i2v_system_prompt(self) -> str:
        """Load the default I2V system prompt."""
        prompt_name = (
            "gemma4_i2v_system_prompt.txt"
            if isinstance(self.language_model, Gemma4LanguageModel)
            else "gemma_i2v_system_prompt.txt"
        )
        return _load_system_prompt(prompt_name)

    def _clean_response(self, response: str) -> str:
        """Clean up the generated response."""
        # Remove leading/trailing whitespace
        response = response.strip()
        # Remove any leading punctuation
        response = re.sub(r"^[^\w\s]+", "", response)
        return response

    def _apply_chat_template(
        self,
        messages: List[Dict[str, str]],
    ) -> str:
        """Apply Gemma chat template to messages."""
        if isinstance(self.language_model, Gemma4LanguageModel):
            return self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        # Gemma 3 chat template format
        formatted = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                formatted += f"<start_of_turn>user\n{content}<end_of_turn>\n"
            elif role == "user":
                if isinstance(content, str):
                    formatted += f"<start_of_turn>user\n{content}<end_of_turn>\n"
                elif isinstance(content, list):
                    # Handle multimodal content (image + text)
                    text_parts = [c["text"] for c in content if c.get("type") == "text"]
                    formatted += (
                        f"<start_of_turn>user\n{' '.join(text_parts)}<end_of_turn>\n"
                    )
            elif role == "assistant":
                formatted += f"<start_of_turn>model\n{content}<end_of_turn>\n"
        # Add generation prompt
        formatted += "<start_of_turn>model\n"
        return formatted

    def enhance_t2v(
        self,
        prompt: str,
        max_tokens: int = 512,
        system_prompt: Optional[str] = None,
        seed: int = 42,
        verbose: bool = True,
        **kwargs,
    ) -> str:
        """Enhance a text prompt for T2V generation using mlx-lm.

        Args:
            prompt: The original user prompt
            max_new_tokens: Maximum number of tokens to generate
            system_prompt: Optional custom system prompt
            seed: Random seed for generation

        Returns:
            Enhanced prompt string
        """
        try:
            from mlx_lm import stream_generate
            from mlx_lm.sample_utils import make_logits_processors, make_sampler
        except ImportError:
            logging.warning(
                "mlx-lm not available for prompt enhancement. Using original prompt."
            )
            return prompt

        if self.processor is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        system_prompt = system_prompt or self.default_t2v_system_prompt

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"user prompt: {prompt}"},
        ]

        # Apply chat template
        formatted = self._apply_chat_template(messages)

        # Use mlx-lm generate with temperature sampling
        mx.random.seed(seed)

        # Tokenize
        inputs = self.processor(
            formatted,
            return_tensors="np",
            add_special_tokens=False,
        )
        input_ids = mx.array(inputs["input_ids"])

        sampler = make_sampler(
            kwargs.get("temperature", 0.7),
            kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
        )
        logits_processors = make_logits_processors(
            kwargs.get("logit_bias", None),
            kwargs.get("repetition_penalty", 1.3),
            kwargs.get("repetition_context_size", 20),
        )

        generated_token_count = 0
        generated_tokens = []
        console = Console()

        generator = stream_generate(
            self.language_model,
            tokenizer=self.processor,
            prompt=input_ids.squeeze(0),
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
        )

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
            disable=not verbose,
        )

        with progress:
            task = progress.add_task("[cyan]Generating[/]", total=max_tokens)

            for i, response in enumerate(generator):
                next_token = mx.array([response.token])
                input_ids = mx.concatenate([input_ids, next_token[None, :]], axis=1)
                generated_tokens.append(response.token)
                generated_token_count += 1
                progress.update(task, advance=1)

                if i % 50 == 0:
                    mx.clear_cache()

                # Check for EOS
                if response.token == 1 or response.token == 107:  # EOS tokens
                    progress.update(task, completed=max_tokens)
                    break

        mx.clear_cache()

        # Decode only the new tokens
        enhanced_prompt = self.processor.decode(
            generated_tokens, skip_special_tokens=True
        )

        enhanced_prompt = self._clean_response(enhanced_prompt)
        logging.info(f"Enhanced prompt: {enhanced_prompt}")

        return enhanced_prompt

    def enhance_i2v(
        self,
        prompt: str,
        image: Optional[mx.array] = None,
        max_new_tokens: int = 512,
        system_prompt: Optional[str] = None,
        seed: int = 42,
    ) -> str:
        """Enhance a text prompt for I2V generation.

        Args:
            prompt: The original user prompt
            image: Optional image tensor (not currently used)
            max_new_tokens: Maximum number of tokens to generate
            system_prompt: Optional custom system prompt
            seed: Random seed for generation

        Returns:
            Enhanced prompt string
        """
        # Use T2V enhancement with I2V system prompt
        return self.enhance_t2v(
            prompt,
            max_new_tokens=max_new_tokens,
            system_prompt=system_prompt or self.default_i2v_system_prompt,
            seed=seed,
        )


def load_text_encoder(model_path: str = "/tmp/ltx2") -> LTX2TextEncoder:
    encoder = LTX2TextEncoder()
    encoder.load(model_path=model_path)
    return encoder
