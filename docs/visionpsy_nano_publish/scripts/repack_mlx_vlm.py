"""Repack VisionPsy-Nano for mlx-vlm compatibility.

- Renames tensors to standard prefixes:
    decoder.*        -> language_model.*
    vision_encoder.* -> vision_tower.*
    MP.*             -> multi_modal_projector.*
  (Drops decoder.rotary_embd.* — stale buffer, MLX RoPE recomputes freqs.)

- Rewrites config.json to the standard mlx-vlm layout with `text_config` and
  `vision_config` at the top level.

- Casts weights to bfloat16.

Writes to ~/models/VisionPsy-Nano-460M-MLX/ (Standard) and
~/models/VisionPsy-Nano-460M-Flash-MLX/ (Flash).
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file as torch_save_file


VARIANTS = {
    os.path.expanduser("~/models/VisionPsy-Nano-460M"):
        os.path.expanduser("~/models/VisionPsy-Nano-460M-MLX"),
    os.path.expanduser("~/models/VisionPsy-Nano-460M-Flash"):
        os.path.expanduser("~/models/VisionPsy-Nano-460M-Flash-MLX"),
}

# original prefix -> destination prefix
PREFIX_MAP = [
    ("decoder.",        "language_model."),
    ("vision_encoder.", "vision_tower."),
    ("MP.",             "multi_modal_projector."),
]

SIDECAR = {
    "chat_template.jinja",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "LICENSE",
    "ATTRIBUTIONS.md",
}


def rename_key(k: str) -> str | None:
    if k.startswith("decoder.rotary_embd."):
        return None
    for src, dst in PREFIX_MAP:
        if k.startswith(src):
            return dst + k[len(src):]
    return k


def rewrite_config(orig_cfg: dict) -> dict:
    """Build a mlx-vlm-style top-level config."""
    new = {
        "model_type": "visionpsy_nano",
        "architectures": ["VisionPsyNanoForConditionalGeneration"],
        "image_token_id": None,  # filled by processor; not required by loader
        "image_token_index": None,
        "pad_token_id": None,
        "eos_token_id": None,
        "torch_dtype": "bfloat16",
        # Retain provenance
        "original_hf_repo": orig_cfg.get("hf_repo_name"),
        "is_flash": orig_cfg.get("is_flash", False),
        # Multi-modal projector params
        "mp_image_token_length": orig_cfg["mp_image_token_length"],
        "mp_pixel_shuffle_factor": orig_cfg["mp_pixel_shuffle_factor"],
        # Extra tokens (row/col markers, image tokens)
        "vlm_extra_tokens": orig_cfg["vlm_extra_tokens"],
        # Text sub-config
        "text_config": {
            "model_type": "smollm2",
            "hidden_size":            orig_cfg["lm_hidden_dim"],
            "intermediate_size":      orig_cfg["lm_inter_dim"],
            "num_hidden_layers":      orig_cfg["lm_n_blocks"],
            "num_attention_heads":    orig_cfg["lm_n_heads"],
            "num_key_value_heads":    orig_cfg["lm_n_kv_heads"],
            "max_position_embeddings": orig_cfg["lm_max_position_embeddings"],
            "rope_theta":             orig_cfg["lm_re_base"],
            "rms_norm_eps":           orig_cfg["lm_rms_eps"],
            "tie_word_embeddings":    orig_cfg["lm_tie_weights"],
            "vocab_size":             orig_cfg["lm_vocab_size"],
            "base_vocab_size":        orig_cfg["lm_base_vocab_size"],
            "attention_scaling":      orig_cfg.get("lm_attn_scaling", 1.0),
            "hf_backbone":            orig_cfg["lm_model_type"],
        },
        # Vision sub-config
        "vision_config": {
            "model_type":         "siglip2_vision_model",
            "hidden_size":        orig_cfg["vit_hidden_dim"],
            "intermediate_size":  orig_cfg["vit_inter_dim"],
            "num_hidden_layers":  orig_cfg["vit_n_blocks"],
            "num_attention_heads": orig_cfg["vit_n_heads"],
            "image_size":         orig_cfg["vit_img_size"],
            "patch_size":         orig_cfg["vit_patch_size"],
            "layer_norm_eps":     orig_cfg["vit_ln_eps"],
            "cls_flag":           orig_cfg.get("vit_cls_flag", False),
            "max_img_size":       orig_cfg["max_img_size"],
            "hf_backbone":        orig_cfg["vit_model_type"],
        },
        # Preserve original keys under a nested block for round-tripping.
        "_original_config": orig_cfg,
    }
    return new


def convert_one(src_dir: Path, dst_dir: Path) -> None:
    print(f"\n=== {src_dir.name} -> {dst_dir.name} ===")
    dst_dir.mkdir(parents=True, exist_ok=True)

    # 1) Weights
    src_sft = src_dir / "model.safetensors"
    dst_sft = dst_dir / "model.safetensors"
    weights: dict[str, torch.Tensor] = {}
    dropped = renamed = kept = 0
    with safe_open(str(src_sft), framework="pt") as f:
        for k in f.keys():
            new_k = rename_key(k)
            if new_k is None:
                dropped += 1
                continue
            t = f.get_tensor(k)
            if t.is_floating_point():
                t = t.to(torch.bfloat16)
            weights[new_k] = t.contiguous()
            if new_k != k:
                renamed += 1
            else:
                kept += 1
    torch_save_file(weights, str(dst_sft))
    print(f"  {len(weights)} tensors  (renamed={renamed}, kept={kept}, dropped={dropped})")
    print(f"  weights {src_sft.stat().st_size / 1e9:.3f} GB -> "
          f"{dst_sft.stat().st_size / 1e9:.3f} GB")

    # 2) Config
    with open(src_dir / "config.json") as f:
        orig_cfg = json.load(f)
    new_cfg = rewrite_config(orig_cfg)
    with open(dst_dir / "config.json", "w") as f:
        json.dump(new_cfg, f, indent=2, ensure_ascii=False)

    # 3) Sidecar files
    for name in SIDECAR:
        p = src_dir / name
        if p.exists():
            shutil.copy2(p, dst_dir / name)


def main() -> None:
    for src, dst in VARIANTS.items():
        convert_one(Path(src), Path(dst))
    print("\nDone.")


if __name__ == "__main__":
    main()
