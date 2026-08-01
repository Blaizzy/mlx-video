"""Weight loader for the Option-B dense-dequant repack.

Reads two sources:

1. The ORIGINAL Escha-W2 checkpoint directory (for attention int8, embeddings,
   layernorm weights, router weights, SSM params) — everything that is NOT
   an escha-quantized MoE expert projection.

2. The DEQUANT directory produced by `modal_dequant_all.py`, containing 40
   per-layer safetensors files with keys of the form
       layer_{L}.expert_{E}.{gate_up_proj|down_proj}.weight
   (bf16, shape (out_f, in_f)).

Returns a single flat state-dict where the escha_code/rin/rout tensors are
replaced by the composed M matrices, keyed under a stable convention:
    model.language_model.layers.{L}.mlp.experts.{E}.gate_up_proj.weight_dequant
    model.language_model.layers.{L}.mlp.experts.{E}.down_proj.weight_dequant
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx


# Suffixes to KEEP verbatim from the original checkpoint (non-escha weights).
_KEEP_ORIG_SUFFIXES = (
    "weight",
    "weight_int8",
    "weight_scale",
    "A_log",
    "dt_bias",
)
# Suffixes to DROP from the original checkpoint (replaced by dequant path).
_DROP_ORIG_SUFFIXES = (
    "escha_code",
    "escha_rin",
    "escha_rout",
    "escha_s_in",
    "escha_s_out",
    "escha_bias",
    "escha_config",
)


def load_state_dequant(
    orig_model_dir: str | Path,
    dequant_dir: str | Path,
) -> dict[str, mx.array]:
    """Assemble the flat state dict from the original + dequant repack.

    Args:
        orig_model_dir: local path to the original Escha-W2 checkpoint (contains
            model.safetensors.index.json + 3 shards + config.json + tokenizer.json).
        dequant_dir:    local path to the directory holding layer_XX.safetensors
            files produced by the Modal dequant sweep.

    Returns:
        Flat dict mapping tensor name → mx.array. Escha experts are exposed as
        `..gate_up_proj.weight_dequant` / `..down_proj.weight_dequant` in bf16.
    """
    orig_model_dir = Path(orig_model_dir)
    dequant_dir = Path(dequant_dir)

    # ---- 1. read the ORIGINAL (non-escha) tensors ----
    idx = json.loads((orig_model_dir / "model.safetensors.index.json").read_text())
    weight_map: dict[str, str] = idx["weight_map"]

    state: dict[str, mx.array] = {}
    shard_to_names: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        shard_to_names.setdefault(shard, []).append(name)

    dropped = 0
    for shard, names in shard_to_names.items():
        shard_state = mx.load(str(orig_model_dir / shard))
        for name in names:
            suffix = name.rsplit(".", 1)[-1]
            if suffix in _DROP_ORIG_SUFFIXES:
                dropped += 1
                continue
            if suffix not in _KEEP_ORIG_SUFFIXES:
                raise KeyError(f"Unhandled tensor suffix {suffix!r} in {name}")
            state[name] = shard_state[name]

    # ---- 2. read the DEQUANT per-layer files ----
    layer_files = sorted(dequant_dir.glob("layer_*.safetensors"))
    if not layer_files:
        raise FileNotFoundError(
            f"No layer_XX.safetensors files found under {dequant_dir}. "
            f"Run modal_dequant_all.py --mode dequant then download the /vol/dequant_v1/ contents."
        )

    prefix = "model.language_model.layers"
    for lf in layer_files:
        # layer index from filename
        L = int(lf.stem.split("_")[1])
        shard_state = mx.load(str(lf))
        for k, tensor in shard_state.items():
            # k = "layer_{L}.expert_{E}.{gate_up_proj|down_proj}.weight"
            parts = k.split(".")
            assert parts[0].startswith("layer_") and int(parts[0][6:]) == L
            assert parts[1].startswith("expert_"), f"unexpected key {k}"
            E = int(parts[1][7:])
            proj = parts[2]                              # gate_up_proj | down_proj
            assert parts[3] == "weight"
            new_name = f"{prefix}.{L}.mlp.experts.{E}.{proj}.weight_dequant"
            state[new_name] = tensor

    return state


def summarize_dequant(state: dict[str, mx.array]) -> str:
    from collections import Counter
    ctr = Counter(name.rsplit(".", 1)[-1] for name in state.keys())
    total_bytes = sum(v.nbytes for v in state.values())
    lines = [f"loaded {len(state)} tensors, {total_bytes / 1e9:.2f} GB total"]
    for suf, n in sorted(ctr.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {suf:20s} x {n}")
    return "\n".join(lines)
