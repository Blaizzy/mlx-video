"""HuggingFace PretrainedConfig for VisionPsyNano."""
from __future__ import annotations

import os
from typing import Any, Optional

from transformers import PretrainedConfig


FLASH_MIN_SIDE_LEN = 512


def resolve_is_flash(
    *,
    is_flash: Optional[bool] = None,
    variant: Optional[str] = None,
    resize_to_max_side_len: Optional[bool] = None,
) -> bool:
    """Resolve Flash preprocess mode from explicit flag, legacy variant, or resize policy."""
    if is_flash is not None:
        return bool(is_flash)
    if variant is not None:
        v = str(variant).lower().strip()
        if v in ("flash", "nano-flash", "visionpsy-nano-flash"):
            return True
        if v in ("nano", "plain", "nano-plain", "visionpsy-nano"):
            return False
        raise ValueError(
            "legacy variant must be 'nano' or 'flash' "
            f"(or aliases); got {variant!r}. Prefer is_flash=True/False."
        )
    if resize_to_max_side_len is not None:
        return not bool(resize_to_max_side_len)
    return False


def apply_flash_preprocess(
    *,
    is_flash: bool,
    resize_to_max_side_len: Optional[bool] = None,
    resize_min_side_len: Optional[int] = None,
) -> tuple[bool, Optional[int]]:
    """Return (resize_to_max_side_len, resize_min_side_len) for the given Flash mode."""
    if resize_to_max_side_len is None:
        resize_to_max_side_len = not is_flash
    resize_to_max_side_len = bool(resize_to_max_side_len)
    if is_flash:
        resize_min_side_len = max(int(resize_min_side_len or 0), FLASH_MIN_SIDE_LEN)
    elif resize_to_max_side_len:
        resize_min_side_len = None
    return resize_to_max_side_len, resize_min_side_len


_DEFAULT_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)

_DEFAULT_EXTRA_TOKENS = {
    "image_token": "<|image|>",
    "global_image_token": "<|global_image|>",
    "r1c1": "<row_1_col_1>",
    "r1c2": "<row_1_col_2>",
    "r1c3": "<row_1_col_3>",
    "r1c4": "<row_1_col_4>",
    "r1c5": "<row_1_col_5>",
    "r1c6": "<row_1_col_6>",
    "r1c7": "<row_1_col_7>",
    "r1c8": "<row_1_col_8>",
    "r2c1": "<row_2_col_1>",
    "r2c2": "<row_2_col_2>",
    "r2c3": "<row_2_col_3>",
    "r2c4": "<row_2_col_4>",
    "r2c5": "<row_2_col_5>",
    "r2c6": "<row_2_col_6>",
    "r2c7": "<row_2_col_7>",
    "r2c8": "<row_2_col_8>",
    "r3c1": "<row_3_col_1>",
    "r3c2": "<row_3_col_2>",
    "r3c3": "<row_3_col_3>",
    "r3c4": "<row_3_col_4>",
    "r3c5": "<row_3_col_5>",
    "r3c6": "<row_3_col_6>",
    "r3c7": "<row_3_col_7>",
    "r3c8": "<row_3_col_8>",
    "r4c1": "<row_4_col_1>",
    "r4c2": "<row_4_col_2>",
    "r4c3": "<row_4_col_3>",
    "r4c4": "<row_4_col_4>",
    "r4c5": "<row_4_col_5>",
    "r4c6": "<row_4_col_6>",
    "r4c7": "<row_4_col_7>",
    "r4c8": "<row_4_col_8>",
    "r5c1": "<row_5_col_1>",
    "r5c2": "<row_5_col_2>",
    "r5c3": "<row_5_col_3>",
    "r5c4": "<row_5_col_4>",
    "r5c5": "<row_5_col_5>",
    "r5c6": "<row_5_col_6>",
    "r5c7": "<row_5_col_7>",
    "r5c8": "<row_5_col_8>",
    "r6c1": "<row_6_col_1>",
    "r6c2": "<row_6_col_2>",
    "r6c3": "<row_6_col_3>",
    "r6c4": "<row_6_col_4>",
    "r6c5": "<row_6_col_5>",
    "r6c6": "<row_6_col_6>",
    "r6c7": "<row_6_col_7>",
    "r6c8": "<row_6_col_8>",
    "r7c1": "<row_7_col_1>",
    "r7c2": "<row_7_col_2>",
    "r7c3": "<row_7_col_3>",
    "r7c4": "<row_7_col_4>",
    "r7c5": "<row_7_col_5>",
    "r7c6": "<row_7_col_6>",
    "r7c7": "<row_7_col_7>",
    "r7c8": "<row_7_col_8>",
    "r8c1": "<row_8_col_1>",
    "r8c2": "<row_8_col_2>",
    "r8c3": "<row_8_col_3>",
    "r8c4": "<row_8_col_4>",
    "r8c5": "<row_8_col_5>",
    "r8c6": "<row_8_col_6>",
    "r8c7": "<row_8_col_7>",
    "r8c8": "<row_8_col_8>",
}


class VisionPsyNanoConfig(PretrainedConfig):
    """Config for VisionPsyNano (``is_flash`` selects preprocess mode)."""

    model_type = "visionpsynano"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        is_flash: Optional[bool] = None,
        variant: Optional[str] = None,
        vit_hidden_dim: int = 768,
        vit_inter_dim: int = 3072,
        vit_patch_size: int = 16,
        vit_img_size: int = 512,
        vit_n_heads: int = 12,
        vit_dropout: float = 0.0,
        vit_n_blocks: int = 12,
        vit_ln_eps: float = 1e-6,
        vit_cls_flag: bool = False,
        vit_model_type: str = "google/siglip2-base-patch16-512",
        lm_hidden_dim: int = 960,
        lm_inter_dim: int = 2560,
        lm_rms_eps: float = 1e-5,
        lm_re_base: int = 100000,
        lm_max_position_embeddings: int = 8192,
        lm_base_vocab_size: int = 49152,
        extra_token_amount: int = 66,
        lm_vocab_size: Optional[int] = None,
        lm_n_heads: int = 15,
        lm_n_kv_heads: int = 5,
        lm_dropout: float = 0.0,
        lm_n_blocks: int = 32,
        lm_attn_scaling: float = 1.0,
        lm_max_length: int = 4096,
        lm_use_tokens: bool = False,
        lm_tie_weights: bool = True,
        lm_model_type: str = "HuggingFaceTB/SmolLM2-360M-Instruct",
        lm_tokenizer: str = "HuggingFaceTB/SmolLM2-360M-Instruct",
        lm_chat_template: str = _DEFAULT_CHAT_TEMPLATE,
        mp_pixel_shuffle_factor: int = 4,
        mp_image_token_length: int = 64,
        max_img_size: int = 2048,
        resize_to_max_side_len: Optional[bool] = None,
        resize_min_side_len: Optional[int] = None,
        inference_max_img_size: Optional[int] = None,
        vlm_extra_tokens: Optional[dict] = None,
        vlm_load_backbone_weights: bool = True,
        vlm_checkpoint_path: str = "checkpoints",
        hf_repo_name: str = "qvac/VisionPsy-Nano-460M",
        compile_inference: bool = True,
        compile_inference_mode: str = "reduce-overhead",
        cuda_graphs_cache_quantum: int = 128,
        eos_check_interval: int = 16,
        **kwargs: Any,
    ):
        is_flash = resolve_is_flash(
            is_flash=is_flash,
            variant=variant,
            resize_to_max_side_len=resize_to_max_side_len,
        )
        resize_to_max_side_len, resize_min_side_len = apply_flash_preprocess(
            is_flash=is_flash,
            resize_to_max_side_len=resize_to_max_side_len,
            resize_min_side_len=resize_min_side_len,
        )

        if lm_vocab_size is None:
            lm_vocab_size = lm_base_vocab_size + extra_token_amount

        self.is_flash = bool(is_flash)
        self.vit_hidden_dim = vit_hidden_dim
        self.vit_inter_dim = vit_inter_dim
        self.vit_patch_size = vit_patch_size
        self.vit_img_size = vit_img_size
        self.vit_n_heads = vit_n_heads
        self.vit_dropout = vit_dropout
        self.vit_n_blocks = vit_n_blocks
        self.vit_ln_eps = vit_ln_eps
        self.vit_cls_flag = vit_cls_flag
        self.vit_model_type = vit_model_type

        self.lm_hidden_dim = lm_hidden_dim
        self.lm_inter_dim = lm_inter_dim
        self.lm_rms_eps = lm_rms_eps
        self.lm_re_base = lm_re_base
        self.lm_max_position_embeddings = lm_max_position_embeddings
        self.lm_base_vocab_size = lm_base_vocab_size
        self.extra_token_amount = extra_token_amount
        self.lm_vocab_size = lm_vocab_size
        self.lm_n_heads = lm_n_heads
        self.lm_n_kv_heads = lm_n_kv_heads
        self.lm_dropout = lm_dropout
        self.lm_n_blocks = lm_n_blocks
        self.lm_attn_scaling = lm_attn_scaling
        self.lm_max_length = lm_max_length
        self.lm_use_tokens = lm_use_tokens
        self.lm_tie_weights = lm_tie_weights
        self.lm_model_type = lm_model_type
        self.lm_tokenizer = lm_tokenizer
        self.lm_chat_template = lm_chat_template

        self.mp_pixel_shuffle_factor = mp_pixel_shuffle_factor
        self.mp_image_token_length = mp_image_token_length
        self.max_img_size = max_img_size
        self.resize_to_max_side_len = bool(resize_to_max_side_len)
        self.resize_min_side_len = resize_min_side_len
        self.inference_max_img_size = inference_max_img_size
        self.vlm_extra_tokens = dict(vlm_extra_tokens or _DEFAULT_EXTRA_TOKENS)
        self.vlm_load_backbone_weights = vlm_load_backbone_weights
        self.vlm_checkpoint_path = vlm_checkpoint_path
        if self.is_flash and hf_repo_name == "qvac/VisionPsy-Nano-460M":
            self.hf_repo_name = "qvac/VisionPsy-Nano-460M-Flash"
        else:
            self.hf_repo_name = hf_repo_name

        self.compile_inference = compile_inference
        self.compile_inference_mode = compile_inference_mode
        self.cuda_graphs_cache_quantum = cuda_graphs_cache_quantum
        self.eos_check_interval = eos_check_interval

        kwargs.pop("text_config", None)
        kwargs.pop("vision_config", None)

        super().__init__(**kwargs)

    def get_text_config(self, decoder: bool = False, **kwargs):
        """VisionPsyNano is a single flat config (not text+vision composite)."""
        return self

    @property
    def variant(self) -> str:
        return "flash" if self.is_flash else "nano"

    def to_vlm_config(self):
        """Convert to the internal VLMConfig dataclass used by the core modules."""
        from dataclasses import fields as dc_fields

        try:
            from .vlm_config import VLMConfig
        except ImportError:
            from vlm_config import VLMConfig

        valid = {f.name for f in dc_fields(VLMConfig)}
        payload = {k: getattr(self, k) for k in valid if hasattr(self, k)}
        return VLMConfig(**payload)

    @classmethod
    def from_vlm_config(
        cls, cfg, *, is_flash: Optional[bool] = None, variant: Optional[str] = None
    ) -> "VisionPsyNanoConfig":
        from dataclasses import asdict

        data = asdict(cfg)
        data["is_flash"] = resolve_is_flash(
            is_flash=is_flash,
            variant=variant,
            resize_to_max_side_len=data.get("resize_to_max_side_len"),
        )
        data.pop("variant", None)
        return cls(**data)

    @classmethod
    def from_legacy_dict(
        cls,
        raw: dict,
        *,
        is_flash: Optional[bool] = None,
        variant: Optional[str] = None,
    ) -> "VisionPsyNanoConfig":
        """Load an existing VisionPsyNano / nanoVLM config.json (without model_type)."""
        raw = dict(raw)
        for key in (
            "model_type",
            "architectures",
            "auto_map",
            "transformers_version",
            "text_config",
            "vision_config",
        ):
            raw.pop(key, None)
        raw["is_flash"] = resolve_is_flash(
            is_flash=is_flash if is_flash is not None else raw.get("is_flash"),
            variant=variant if variant is not None else raw.get("variant"),
            resize_to_max_side_len=raw.get("resize_to_max_side_len"),
        )
        raw.pop("variant", None)
        return cls(**raw)
