"""Safetensors → MLX state-dict for Escha-W2.

Reads the sharded EschaLabs/Qwen3.6-35B-A3B-Escha-W2 checkpoint into the
minimal per-module tensor set the MLX port needs. Filters out fp32
`escha_s_in/s_out` (always ones — see feasibility doc §1a) and drops the
informational `escha_config` int32 arrays.

Public entry point: `load_state(model_dir: Path) -> dict[str, mx.array]`
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx


# Tensor-name suffixes we keep verbatim.
_KEEP_SUFFIXES = (
    "escha_code",
    "escha_rin",
    "escha_rout",
    "weight",
    "weight_int8",
    "weight_scale",
    "A_log",
    "dt_bias",
)
# Suffixes we drop (all-ones or informational).
_DROP_SUFFIXES = (
    "escha_s_in",
    "escha_s_out",
    "escha_bias",     # not present in Escha-W2 (bias-free Qwen3.5 MoE)
    "escha_config",
)


def load_state(model_dir: str | Path) -> dict[str, mx.array]:
    """Load all shards, filter, return a flat name→mx.array dict."""
    from safetensors import safe_open

    model_dir = Path(model_dir)
    idx = json.loads((model_dir / "model.safetensors.index.json").read_text())
    weight_map: dict[str, str] = idx["weight_map"]
    # Group by shard so we open each file exactly once.
    shard_to_names: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        shard_to_names.setdefault(shard, []).append(name)

    state: dict[str, mx.array] = {}
    dropped = 0
    for shard, names in shard_to_names.items():
        with safe_open(model_dir / shard, framework="numpy") as f:
            for name in names:
                suffix = name.rsplit(".", 1)[-1]
                if suffix in _DROP_SUFFIXES:
                    dropped += 1
                    continue
                if suffix not in _KEEP_SUFFIXES:
                    # Unknown suffix — surface loudly rather than silently drop.
                    raise KeyError(f"Unhandled tensor suffix {suffix!r} in {name}")
                tensor = f.get_tensor(name)
                state[name] = mx.array(tensor)
    return state


def summarize(state: dict[str, mx.array]) -> str:
    """One-line summary of what got loaded — for debugging."""
    from collections import Counter
    ctr = Counter(name.rsplit(".", 1)[-1] for name in state.keys())
    total_bytes = sum(v.nbytes for v in state.values())
    lines = [f"loaded {len(state)} tensors, {total_bytes / 1e9:.2f} GB total"]
    for suf, n in sorted(ctr.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {suf:20s} × {n}")
    return "\n".join(lines)
