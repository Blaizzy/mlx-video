"""LTX-2.5 shot-duration prediction from text connector outputs."""

import json
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from safetensors import safe_open


class AttentionPooler(nn.Module):
    def __init__(
        self, hidden_dim: int = 256, num_queries: int = 1, num_heads: int = 4
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.query_tokens = mx.zeros((num_queries, hidden_dim))
        self.cross_attn = {
            "in_proj_weight": mx.zeros((hidden_dim * 3, hidden_dim)),
            "in_proj_bias": mx.zeros((hidden_dim * 3,)),
            "out_proj": nn.Linear(hidden_dim, hidden_dim),
        }

    def __call__(self, tokens: mx.array) -> mx.array:
        batch_size = tokens.shape[0]
        queries = mx.broadcast_to(
            self.query_tokens[None],
            (batch_size, self.query_tokens.shape[0], self.hidden_dim),
        )
        weight = self.cross_attn["in_proj_weight"]
        bias = self.cross_attn["in_proj_bias"]
        query_weight, key_weight, value_weight = mx.split(weight, 3, axis=0)
        query_bias, key_bias, value_bias = mx.split(bias, 3, axis=0)
        query = queries @ query_weight.T + query_bias
        key = tokens @ key_weight.T + key_bias
        value = tokens @ value_weight.T + value_bias

        head_dim = self.hidden_dim // self.num_heads

        def split_heads(x: mx.array) -> mx.array:
            return x.reshape(x.shape[0], x.shape[1], self.num_heads, head_dim).transpose(
                0, 2, 1, 3
            )

        pooled = mx.fast.scaled_dot_product_attention(
            split_heads(query),
            split_heads(key),
            split_heads(value),
            scale=1.0 / math.sqrt(head_dim),
        )
        pooled = pooled.transpose(0, 2, 1, 3).reshape(
            batch_size, queries.shape[1], self.hidden_dim
        )
        return self.cross_attn["out_proj"](pooled)


class DurationHead(nn.Module):
    def __init__(
        self,
        video_dim: int = 4096,
        audio_dim: int = 2048,
        hidden_dim: int = 256,
        num_queries: int = 1,
        num_heads: int = 4,
        mlp_hidden: int = 256,
    ) -> None:
        super().__init__()
        self.video_input_proj = nn.Linear(video_dim, hidden_dim)
        self.video_modality_emb = mx.zeros((hidden_dim,))
        self.audio_input_proj = nn.Linear(audio_dim, hidden_dim)
        self.audio_modality_emb = mx.zeros((hidden_dim,))
        self.attention_pooler = AttentionPooler(
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            num_heads=num_heads,
        )
        self.mlp_hidden = nn.Linear(hidden_dim * num_queries, mlp_hidden)
        self.mlp_out = nn.Linear(mlp_hidden, 1)

    def __call__(
        self,
        video_tokens: mx.array | None = None,
        audio_tokens: mx.array | None = None,
    ) -> mx.array:
        if video_tokens is None and audio_tokens is None:
            raise ValueError("DurationHead requires video or audio connector tokens")
        token_groups = []
        if video_tokens is not None:
            token_groups.append(
                self.video_input_proj(video_tokens) + self.video_modality_emb
            )
        if audio_tokens is not None:
            token_groups.append(
                self.audio_input_proj(audio_tokens) + self.audio_modality_emb
            )
        tokens = mx.concatenate(token_groups, axis=1)
        pooled = self.attention_pooler(tokens).reshape(tokens.shape[0], -1)
        hidden = nn.gelu_approx(self.mlp_hidden(pooled))
        return mx.exp(self.mlp_out(hidden).squeeze(-1))

    @classmethod
    def from_pretrained(cls, checkpoint_path: str | Path) -> "DurationHead":
        checkpoint_path = Path(checkpoint_path)
        with safe_open(checkpoint_path, framework="numpy") as f:
            metadata = f.metadata() or {}
        config = json.loads(metadata["config"])
        transformer = config.get("transformer", {})
        head_config = config.get("duration_head", {})
        model = cls(
            video_dim=transformer.get("cross_attention_dim", 4096),
            audio_dim=transformer.get("audio_cross_attention_dim", 2048),
            hidden_dim=head_config.get("pooler_hidden_dim", 256),
            num_queries=head_config.get("num_queries", 1),
            num_heads=head_config.get("num_pooler_heads", 4),
            mlp_hidden=head_config.get("mlp_hidden", 256),
        )
        weights = {
            key.removeprefix("duration_head."): value
            for key, value in mx.load(str(checkpoint_path)).items()
            if key.startswith("duration_head.")
        }
        model.load_weights(list(weights.items()), strict=True)
        return model


def seconds_to_num_frames(
    seconds: float,
    frame_rate: float,
    min_seconds: float = 1.0,
    max_seconds: float = 20.0,
) -> int:
    """Clamp seconds and floor the result onto the causal VAE's 8k+1 grid."""
    if frame_rate <= 0:
        raise ValueError(f"frame_rate must be positive, got {frame_rate}")
    if min_seconds <= 0 or max_seconds < min_seconds:
        raise ValueError(
            "duration bounds must satisfy 0 < min_seconds <= max_seconds, "
            f"got {min_seconds} and {max_seconds}"
        )
    min_frames = round(min_seconds * frame_rate)
    max_frames = round(max_seconds * frame_rate)
    raw_frames = max(min_frames, min(round(seconds * frame_rate), max_frames))
    frames = ((raw_frames - 1) // 8) * 8 + 1
    if frames < min_frames:
        frames = min(-(-(min_frames - 1) // 8) * 8 + 1, max_frames)
    return frames


def predict_num_frames(
    checkpoint_path: str | Path,
    video_tokens: mx.array | None,
    audio_tokens: mx.array | None,
    frame_rate: float,
    min_seconds: float = 1.0,
    max_seconds: float = 20.0,
) -> tuple[int, float]:
    head = DurationHead.from_pretrained(checkpoint_path)
    seconds_array = head(video_tokens, audio_tokens)
    mx.eval(seconds_array)
    if seconds_array.shape != (1,):
        raise ValueError(
            "DurationHead supports one prompt at a time, "
            f"got prediction shape {seconds_array.shape}"
        )
    seconds = float(seconds_array.item())
    return (
        seconds_to_num_frames(seconds, frame_rate, min_seconds, max_seconds),
        seconds,
    )
