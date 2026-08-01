"""Config dataclass mirroring the VisionPsyNano HF config.

We keep a single flat dataclass (matching the reference `VLMConfig`) so we can
load either the Standard or the Flash variant from the on-disk `config.json`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_EXTRA_TOKENS: Dict[str, str] = {
    "image_token": "<|image|>",
    "global_image_token": "<|global_image|>",
    **{
        f"r{i}c{j}": f"<row_{i}_col_{j}>"
        for i in range(1, 9)
        for j in range(1, 9)
    },
}


@dataclass
class VisionPsyNanoConfig:
    # ViT (SigLIP2-base-patch16-512)
    vit_hidden_dim: int = 768
    vit_inter_dim: int = 3072
    vit_patch_size: int = 16
    vit_img_size: int = 512
    vit_n_heads: int = 12
    vit_n_blocks: int = 12
    vit_ln_eps: float = 1e-6
    vit_cls_flag: bool = False

    # LM (SmolLM2-360M-Instruct)
    lm_hidden_dim: int = 960
    lm_inter_dim: int = 2560
    lm_rms_eps: float = 1e-5
    lm_re_base: int = 100_000
    lm_max_position_embeddings: int = 8192
    lm_vocab_size: int = 49_218
    lm_n_heads: int = 15
    lm_n_kv_heads: int = 5
    lm_n_blocks: int = 32
    lm_tie_weights: bool = True

    # Modality projector
    mp_pixel_shuffle_factor: int = 4
    mp_image_token_length: int = 64

    # Preprocessor policy
    is_flash: bool = False
    max_img_size: int = 2048
    inference_max_img_size: Optional[int] = None
    resize_to_max_side_len: bool = True
    resize_min_side_len: Optional[int] = None

    # Extra tokens & chat template (copied from the HF config)
    vlm_extra_tokens: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_EXTRA_TOKENS))
    lm_tokenizer: str = "HuggingFaceTB/SmolLM2-360M-Instruct"
    lm_chat_template: Optional[str] = None

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "VisionPsyNanoConfig":
        """Build a config from a raw dict (HF-style)."""
        fields = cls.__dataclass_fields__
        payload = {k: v for k, v in raw.items() if k in fields}
        return cls(**payload)

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "VisionPsyNanoConfig":
        """Load config.json from a model directory."""
        p = Path(path)
        if p.is_dir():
            p = p / "config.json"
        with open(p, "r") as f:
            raw = json.load(f)
        return cls.from_dict(raw)

    @property
    def variant(self) -> str:
        return "flash" if self.is_flash else "standard"

    @property
    def lm_head_dim(self) -> int:
        assert self.lm_hidden_dim % self.lm_n_heads == 0
        return self.lm_hidden_dim // self.lm_n_heads

    @property
    def vit_head_dim(self) -> int:
        assert self.vit_hidden_dim % self.vit_n_heads == 0
        return self.vit_hidden_dim // self.vit_n_heads

    @property
    def vit_num_patches(self) -> int:
        return (self.vit_img_size // self.vit_patch_size) ** 2
