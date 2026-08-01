"""Escha-W2 weight_loader real-model tests.

Skips when the local checkpoint is absent so this file is safe to run in CI.
When the checkpoint IS present, verifies the loader ingests all 3 shards,
filters correctly, and produces per-suffix counts consistent with a 40-layer
hybrid (linear_attn + self_attn) MoE model with 256 experts per layer.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


MODEL_DIR = Path.home() / "models" / "Qwen3.6-35B-A3B-Escha-W2"


@pytest.fixture(scope="module")
def loaded_state():
    if not (MODEL_DIR / "model.safetensors.index.json").exists():
        pytest.skip(f"Escha-W2 checkpoint not present at {MODEL_DIR}")
    from mlx_video.models.qwen3_5_moe_escha.weight_loader import load_state
    return load_state(MODEL_DIR)


def test_loader_total_bytes_matches_index(loaded_state):
    """Loaded bytes should roughly match index.json's total_size minus the
    dropped fp32 s_in/s_out (all-ones) and int32 escha_config metadata."""
    idx = json.loads((MODEL_DIR / "model.safetensors.index.json").read_text())
    total_disk = idx["metadata"]["total_size"]
    total_loaded = sum(v.nbytes for v in loaded_state.values())
    # We drop ~1-2 GB of s_in/s_out fp32 duplicates. Anything within 20 % is fine.
    assert 0.75 * total_disk <= total_loaded <= total_disk


def test_suffix_counts_match_arch(loaded_state):
    """40 layers, MoE experts on every layer (gate_up + down), hybrid attention.

    Expected per suffix (see config.json layer_types — 40 layers, every 4th is
    full self_attn, 30 are linear_attn):
      escha_code / escha_rin / escha_rout : 80 each (2 projections × 40 layers)
      A_log / dt_bias                     : 30 each (linear_attn layers only)
    """
    from collections import Counter
    ctr = Counter(name.rsplit(".", 1)[-1] for name in loaded_state)
    assert ctr["escha_code"] == 80
    assert ctr["escha_rin"] == 80
    assert ctr["escha_rout"] == 80
    assert ctr["A_log"] == 30
    assert ctr["dt_bias"] == 30
    # Int8 layers: 2 embed/head + per-layer (linear_attn: qkv/z/out=3; self_attn: q/k/v/o=4)
    # linear_attn count = 30, self_attn count = 10 → 3*30 + 4*10 + 2 = 132 for LM;
    # shared_expert has 3 projections (gate/up/down) per layer → +120 = 252.
    assert ctr["weight_int8"] == 252
    assert ctr["weight_scale"] == 252


def test_expert_code_shape_matches_config(loaded_state):
    """For layer 0 gate_up_proj: config says K=2, in_p=2048, out_p=1024, E=256.
    Expect escha_code shape [E, in_p/16, out_p/16, 16*K] = [256, 128, 64, 32].
    For down_proj: K=3, in_p=512, out_p=2048 → [256, 32, 128, 48].
    """
    gu = loaded_state["model.language_model.layers.0.mlp.experts.gate_up_proj.escha_code"]
    assert tuple(gu.shape) == (256, 128, 64, 32), gu.shape
    dn = loaded_state["model.language_model.layers.0.mlp.experts.down_proj.escha_code"]
    assert tuple(dn.shape) == (256, 32, 128, 48), dn.shape


def test_no_dropped_suffix_leaks(loaded_state):
    for name in loaded_state:
        suffix = name.rsplit(".", 1)[-1]
        assert suffix not in {"escha_s_in", "escha_s_out", "escha_config", "escha_bias"}
