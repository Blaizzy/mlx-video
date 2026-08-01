"""HuggingFace processor for VisionPsyNano."""
from __future__ import annotations

import json
import os
from typing import Any, List, Optional, Union

import torch
from PIL import Image
from transformers import AutoTokenizer, BatchFeature, ProcessorMixin
from transformers.utils import logging


try:
    from .configuration_visionpsynano import (
        VisionPsyNanoConfig,
        apply_flash_preprocess,
        resolve_is_flash,
    )
except ImportError:
    from configuration_visionpsynano import (
        VisionPsyNanoConfig,
        apply_flash_preprocess,
        resolve_is_flash,
    )

try:
    from .processors import get_image_processor, get_image_string, get_tokenizer
except ImportError as exc:
    raise ImportError(
        "Failed to import VisionPsyNano data processors. "
        "For Hub packages use scripts/package_hub_repo.py so imports are flattened."
    ) from exc

logger = logging.get_logger(__name__)


def _as_pil_list(images) -> List[Image.Image]:
    if images is None:
        return []
    if isinstance(images, Image.Image):
        return [images.convert("RGB")]
    if isinstance(images, (list, tuple)):
        out = []
        for im in images:
            if isinstance(im, Image.Image):
                out.append(im.convert("RGB"))
            else:
                raise TypeError(f"Expected PIL.Image, got {type(im)}")
        return out
    raise TypeError(f"Unsupported images type: {type(images)}")


def _extract_user_text(text: Any) -> str:
    """Normalize text / chat messages into a single user prompt string."""
    if text is None:
        return ""
    if isinstance(text, str):
        return text
    if isinstance(text, list) and text and isinstance(text[0], dict) and "role" in text[0]:
        parts = []
        for msg in text:
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, str):
                        parts.append(block)
                    elif isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))
        return " ".join(p for p in parts if p).strip()
    if isinstance(text, list) and all(isinstance(t, str) for t in text):
        if len(text) == 1:
            return text[0]
        raise ValueError("Batch text list with >1 items is not supported; call per sample.")
    raise TypeError(f"Unsupported text type: {type(text)}")


class VisionPsyNanoProcessor(ProcessorMixin):
    """Tokenizer + VisionPsyNano image preprocessing (default or Flash)."""

    attributes = ["tokenizer"]
    tokenizer_class = "AutoTokenizer"

    def __init__(self, tokenizer, chat_template=None, **kwargs):
        image_keys = {
            "is_flash",
            "variant",
            "vit_img_size",
            "max_img_size",
            "inference_max_img_size",
            "resize_to_max_side_len",
            "resize_min_side_len",
            "mp_image_token_length",
            "lm_max_position_embeddings",
            "vlm_extra_tokens",
            "lm_tokenizer",
            "lm_chat_template",
        }
        image_cfg = {k: kwargs.pop(k) for k in list(kwargs) if k in image_keys}
        super().__init__(tokenizer, chat_template=chat_template, **kwargs)

        flash = resolve_is_flash(
            is_flash=image_cfg.get("is_flash"),
            variant=image_cfg.get("variant"),
            resize_to_max_side_len=image_cfg.get("resize_to_max_side_len"),
        )
        resize_to_max, resize_min = apply_flash_preprocess(
            is_flash=flash,
            resize_to_max_side_len=image_cfg.get("resize_to_max_side_len"),
            resize_min_side_len=image_cfg.get("resize_min_side_len"),
        )
        self.is_flash = flash
        self.vit_img_size = int(image_cfg.get("vit_img_size", 512))
        self.max_img_size = int(image_cfg.get("max_img_size", 2048))
        self.inference_max_img_size = image_cfg.get("inference_max_img_size")
        self.resize_to_max_side_len = resize_to_max
        self.resize_min_side_len = resize_min
        self.mp_image_token_length = int(image_cfg.get("mp_image_token_length", 64))
        self.lm_max_position_embeddings = int(
            image_cfg.get("lm_max_position_embeddings", 8192)
        )
        self.vlm_extra_tokens = image_cfg.get("vlm_extra_tokens")
        self.lm_tokenizer = image_cfg.get("lm_tokenizer")
        self.lm_chat_template = image_cfg.get("lm_chat_template")

    def _effective_max_img_size(self) -> int:
        return int(self.inference_max_img_size or self.max_img_size)

    def _build_image_processor(self):
        return get_image_processor(
            self._effective_max_img_size(),
            self.vit_img_size,
            self.resize_to_max_side_len,
            self.resize_min_side_len,
        )

    def set_flash(self, enabled: bool = True) -> None:
        """Enable/disable Flash optimized image preprocessing in-place."""
        self.is_flash = bool(enabled)
        self.resize_to_max_side_len, self.resize_min_side_len = apply_flash_preprocess(
            is_flash=self.is_flash,
            resize_to_max_side_len=None,
            resize_min_side_len=self.resize_min_side_len if self.is_flash else None,
        )

    def __call__(
        self,
        images: Optional[Union[Image.Image, List[Image.Image]]] = None,
        text: Optional[Any] = None,
        return_tensors: Optional[str] = "pt",
        is_flash: Optional[bool] = None,
        **kwargs,
    ) -> BatchFeature:
        if is_flash is not None:
            self.set_flash(bool(is_flash))

        prompt = _extract_user_text(text)
        pil_images = _as_pil_list(images)

        image_processor = self._build_image_processor()
        processed_tensors = []
        ratios = []
        for img in pil_images:
            processed_image, splitted_image_ratio = image_processor(img)
            if (
                not hasattr(self.tokenizer, "global_image_token")
                and splitted_image_ratio[0] * splitted_image_ratio[1]
                == len(processed_image) - 1
            ):
                processed_image = processed_image[1:]
            processed_tensors.append(processed_image)
            ratios.append(splitted_image_ratio)

        if processed_tensors:
            image_string = get_image_string(
                self.tokenizer, ratios, self.mp_image_token_length
            )
            images_tensor = torch.cat(processed_tensors, dim=0)
        else:
            image_string = ""
            images_tensor = None

        messages = [{"role": "user", "content": image_string + prompt}]
        full_prompt = self.tokenizer.apply_chat_template(
            [messages], tokenize=False, add_generation_prompt=True
        )
        if isinstance(full_prompt, list):
            full_prompt = full_prompt[0]

        encoded = self.tokenizer(
            [full_prompt],
            return_tensors=return_tensors,
            padding=False,
            truncation=True,
            max_length=self.lm_max_position_embeddings,
            **{k: v for k, v in kwargs.items() if k in ("padding", "truncation", "max_length")},
        )

        data = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded.get("attention_mask"),
        }
        if images_tensor is not None:
            data["images"] = images_tensor
            data["pixel_values"] = images_tensor  # HF-friendly alias
        return BatchFeature(data=data, tensor_type=return_tensors)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def image_config_dict(self) -> dict:
        return {
            "processor_class": "VisionPsyNanoProcessor",
            "auto_map": {
                "AutoProcessor": "processing_visionpsynano.VisionPsyNanoProcessor",
            },
            "is_flash": self.is_flash,
            "vit_img_size": self.vit_img_size,
            "max_img_size": self.max_img_size,
            "inference_max_img_size": self.inference_max_img_size,
            "resize_to_max_side_len": self.resize_to_max_side_len,
            "resize_min_side_len": self.resize_min_side_len,
            "mp_image_token_length": self.mp_image_token_length,
            "lm_max_position_embeddings": self.lm_max_position_embeddings,
            "lm_tokenizer": self.lm_tokenizer,
            "lm_chat_template": self.lm_chat_template,
            "vlm_extra_tokens": self.vlm_extra_tokens,
        }

    def save_pretrained(self, save_directory: str, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        super().save_pretrained(save_directory, **kwargs)

        proc_cfg_path = os.path.join(save_directory, "processor_config.json")
        proc_cfg = {}
        if os.path.exists(proc_cfg_path):
            with open(proc_cfg_path) as f:
                proc_cfg = json.load(f)
        proc_cfg["processor_class"] = "VisionPsyNanoProcessor"
        proc_cfg["auto_map"] = {
            "AutoProcessor": "processing_visionpsynano.VisionPsyNanoProcessor",
        }
        with open(proc_cfg_path, "w") as f:
            json.dump(proc_cfg, f, indent=2)
            f.write("\n")

        with open(os.path.join(save_directory, "preprocessor_config.json"), "w") as f:
            json.dump(self.image_config_dict(), f, indent=2)
            f.write("\n")

    @staticmethod
    def _attach_extra_token_attrs(tokenizer, vlm_extra_tokens: Optional[dict]):
        """Restore named attrs (``image_token``, ``r1c1``, ...) used by image strings."""
        if not vlm_extra_tokens:
            return tokenizer
        for name, token in vlm_extra_tokens.items():
            if not hasattr(tokenizer, name):
                setattr(tokenizer, name, token)
        if hasattr(tokenizer, "image_token"):
            try:
                tokenizer.image_token_id = tokenizer.convert_tokens_to_ids(tokenizer.image_token)
            except Exception:
                pass
        return tokenizer

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        trust_remote_code = kwargs.pop("trust_remote_code", False)
        flash_override = kwargs.pop("is_flash", None)

        tokenizer = None
        prep = {}

        def _prep_from_cfg(cfg: VisionPsyNanoConfig) -> dict:
            return {
                "is_flash": cfg.is_flash,
                "vit_img_size": cfg.vit_img_size,
                "max_img_size": cfg.max_img_size,
                "inference_max_img_size": cfg.inference_max_img_size,
                "resize_to_max_side_len": cfg.resize_to_max_side_len,
                "resize_min_side_len": cfg.resize_min_side_len,
                "mp_image_token_length": cfg.mp_image_token_length,
                "lm_max_position_embeddings": cfg.lm_max_position_embeddings,
                "lm_tokenizer": cfg.lm_tokenizer,
                "lm_chat_template": cfg.lm_chat_template,
                "vlm_extra_tokens": cfg.vlm_extra_tokens,
            }

        resolved = pretrained_model_name_or_path
        prep_path = None
        if os.path.isdir(resolved):
            prep_path = os.path.join(resolved, "preprocessor_config.json")
            tok_files_present = any(
                os.path.exists(os.path.join(resolved, name))
                for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json")
            )
            if tok_files_present:
                tokenizer = AutoTokenizer.from_pretrained(
                    resolved, trust_remote_code=trust_remote_code, **kwargs
                )
            if prep_path and os.path.exists(prep_path):
                with open(prep_path) as f:
                    prep = json.load(f)
        else:
            from huggingface_hub import hf_hub_download, list_repo_files

            try:
                files = set(list_repo_files(resolved))
            except Exception:
                files = set()
            if "preprocessor_config.json" in files:
                prep_path = hf_hub_download(resolved, "preprocessor_config.json")
                with open(prep_path) as f:
                    prep = json.load(f)
            if any(n in files for n in ("tokenizer.json", "tokenizer_config.json", "vocab.json")):
                tokenizer = AutoTokenizer.from_pretrained(
                    resolved, trust_remote_code=trust_remote_code, **kwargs
                )

        if tokenizer is None:
            lm_tok = prep.get("lm_tokenizer")
            extra = prep.get("vlm_extra_tokens")
            chat = prep.get("lm_chat_template")
            if lm_tok is None:
                cfg = VisionPsyNanoConfig.from_pretrained(
                    pretrained_model_name_or_path, trust_remote_code=trust_remote_code
                )
                lm_tok = cfg.lm_tokenizer
                extra = cfg.vlm_extra_tokens
                chat = cfg.lm_chat_template
                if not prep:
                    prep = _prep_from_cfg(cfg)
            tokenizer = get_tokenizer(lm_tok, extra, chat)

        if not prep:
            try:
                cfg = VisionPsyNanoConfig.from_pretrained(
                    pretrained_model_name_or_path, trust_remote_code=trust_remote_code
                )
                prep = _prep_from_cfg(cfg)
            except Exception as e:
                logger.warning("Could not load VisionPsyNanoConfig for processor defaults: %s", e)

        if flash_override is not None:
            prep["is_flash"] = bool(flash_override)

        extra = prep.get("vlm_extra_tokens")
        if extra is None:
            try:
                cfg = VisionPsyNanoConfig.from_pretrained(
                    pretrained_model_name_or_path, trust_remote_code=trust_remote_code
                )
                extra = cfg.vlm_extra_tokens
                prep.setdefault("vlm_extra_tokens", extra)
            except Exception:
                pass
        cls._attach_extra_token_attrs(tokenizer, extra)

        chat_template = getattr(tokenizer, "chat_template", None) or prep.get("lm_chat_template")
        return cls(tokenizer, chat_template=chat_template, **{
            k: prep[k]
            for k in (
                "is_flash",
                "vit_img_size",
                "max_img_size",
                "inference_max_img_size",
                "resize_to_max_side_len",
                "resize_min_side_len",
                "mp_image_token_length",
                "lm_max_position_embeddings",
                "vlm_extra_tokens",
                "lm_tokenizer",
                "lm_chat_template",
            )
            if k in prep
        })

    @classmethod
    def from_config(cls, config: VisionPsyNanoConfig, tokenizer=None) -> "VisionPsyNanoProcessor":
        if tokenizer is None:
            tokenizer = get_tokenizer(
                config.lm_tokenizer, config.vlm_extra_tokens, config.lm_chat_template
            )
        return cls(
            tokenizer,
            chat_template=config.lm_chat_template,
            is_flash=config.is_flash,
            vit_img_size=config.vit_img_size,
            max_img_size=config.max_img_size,
            inference_max_img_size=config.inference_max_img_size,
            resize_to_max_side_len=config.resize_to_max_side_len,
            resize_min_side_len=config.resize_min_side_len,
            mp_image_token_length=config.mp_image_token_length,
            lm_max_position_embeddings=config.lm_max_position_embeddings,
            vlm_extra_tokens=config.vlm_extra_tokens,
            lm_tokenizer=config.lm_tokenizer,
            lm_chat_template=config.lm_chat_template,
        )


try:
    from transformers import AutoProcessor

    AutoProcessor.register(VisionPsyNanoConfig, VisionPsyNanoProcessor)
except Exception:
    pass
