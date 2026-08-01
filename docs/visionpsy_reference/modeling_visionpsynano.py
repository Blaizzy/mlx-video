"""HuggingFace PreTrainedModel wrapper for VisionPsyNano."""
from __future__ import annotations

import os
from typing import Optional

import torch
from transformers import PreTrainedModel


try:
    from .configuration_visionpsynano import VisionPsyNanoConfig
except ImportError:
    from configuration_visionpsynano import VisionPsyNanoConfig

try:
    from .processors import get_tokenizer
    from .language_model import LanguageModel
    from .modality_projector import ModalityProjector
    from .vision_language_model import VisionLanguageModel
    from .vision_transformer import ViT
except ImportError as exc:
    raise ImportError(
        "Failed to import VisionPsyNano core modules (models/, data/). "
        "For Hub packages use scripts/package_hub_repo.py so imports are flattened."
    ) from exc


class VisionPsyNanoForConditionalGeneration(PreTrainedModel):
    config_class = VisionPsyNanoConfig
    base_model_prefix = ""
    _no_split_modules = ["ViTBlock", "LanguageModelBlock", "ModalityProjector"]
    main_input_name = "input_ids"
    supports_gradient_checkpointing = False

    def __init__(self, config: VisionPsyNanoConfig):
        super().__init__(config)
        self.cfg = config.to_vlm_config()
        self.vision_encoder = ViT(self.cfg)
        self.decoder = LanguageModel(self.cfg)
        self.MP = ModalityProjector(self.cfg)
        self.tokenizer = get_tokenizer(
            self.cfg.lm_tokenizer,
            self.cfg.vlm_extra_tokens,
            self.cfg.lm_chat_template,
        )
        self._sync_image_token_id()
        self._compiled_decoder = None
        self._compiled_vision_encoder = None
        self.load_backbone = False
        self.post_init()
        self.tie_weights()

    def _sync_image_token_id(self) -> None:
        tok = self.tokenizer
        if hasattr(tok, "image_token"):
            tok.image_token_id = tok.convert_tokens_to_ids(tok.image_token)
        elif "<|image|>" in getattr(tok, "get_vocab", lambda: {})():
            tok.image_token = "<|image|>"
            tok.image_token_id = tok.convert_tokens_to_ids("<|image|>")

    def set_tokenizer(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self._sync_image_token_id()

    def _reinit_rotary_cache(self) -> None:
        re = getattr(self.decoder, "rotary_embd", None)
        if re is None or not hasattr(re, "extend_cache"):
            return
        cos = getattr(re, "cos_cached", None)
        if cos is None or getattr(cos, "is_meta", False) or cos.device.type == "meta":
            return
        length = int(getattr(re, "original_max_seq_len", None) or re.max_seq_len)
        re.extend_cache(max(length, 1))
        if bool(torch.isnan(re.cos_cached).any()) or bool(torch.isnan(re.sin_cached).any()):
            device = re.inv_freq.device
            positions = torch.arange(length, dtype=torch.float, device=device)
            freqs = positions.unsqueeze(-1) * re.inv_freq.unsqueeze(0)
            emb = torch.cat([freqs, freqs], dim=-1)
            re.cos_cached = emb.cos() * re.attention_scaling
            re.sin_cached = emb.sin() * re.attention_scaling
            re.original_max_seq_len = length

    def tie_weights(self, recompute_mapping: bool = True, **kwargs):
        _ = recompute_mapping, kwargs
        if not getattr(self.cfg, "lm_tie_weights", True):
            return
        head_w = self.decoder.head.weight
        emb_w = self.decoder.token_embedding.weight
        if head_w.data_ptr() != emb_w.data_ptr():
            with torch.no_grad():
                emb_w.copy_(head_w)
            self.decoder.head.weight = self.decoder.token_embedding.weight

    def _tie_weights(self):
        self.tie_weights()

    def _repair_legacy_checkpoint_weights(
        self, pretrained_model_name_or_path, *, revision: Optional[str] = None
    ) -> None:
        weights_path = None
        if os.path.isdir(pretrained_model_name_or_path):
            candidate = os.path.join(pretrained_model_name_or_path, "model.safetensors")
            if os.path.isfile(candidate):
                weights_path = candidate
        else:
            try:
                from huggingface_hub import hf_hub_download

                weights_path = hf_hub_download(
                    pretrained_model_name_or_path,
                    "model.safetensors",
                    revision=revision,
                )
            except Exception:
                return
        if not weights_path:
            return

        from safetensors import safe_open

        with safe_open(weights_path, framework="pt", device="cpu") as f:
            keys = set(f.keys())
            for i, block in enumerate(self.decoder.blocks):
                gate_key = f"decoder.blocks.{i}.mlp.gate_proj.weight"
                up_key = f"decoder.blocks.{i}.mlp.up_proj.weight"
                fused_key = f"decoder.blocks.{i}.mlp.gate_up_proj.weight"
                if gate_key in keys and up_key in keys and fused_key not in keys:
                    fused = torch.cat([f.get_tensor(gate_key), f.get_tensor(up_key)], dim=0)
                    param = block.mlp.gate_up_proj.weight
                    param.data.copy_(fused.to(device=param.device, dtype=param.dtype))
            if (
                "decoder.token_embedding.weight" not in keys
                and "decoder.head.weight" in keys
            ):
                head = f.get_tensor("decoder.head.weight")
                emb = self.decoder.token_embedding.weight
                emb.data.copy_(head.to(device=emb.device, dtype=emb.dtype))

    def _vision_encoder_for_prefill(self):
        return VisionLanguageModel._vision_encoder_for_prefill(self)

    def _decoder_for_decode(self):
        return VisionLanguageModel._decoder_for_decode(self)

    def _replace_img_tokens_with_embd(self, input_ids, token_embd, image_embd):
        return VisionLanguageModel._replace_img_tokens_with_embd(
            self, input_ids, token_embd, image_embd
        )

    def _process_images(self, images, device):
        return VisionLanguageModel._process_images(self, images, device)

    def apply_deploy_profile(self, device: Optional[torch.device] = None) -> None:
        try:
            from .runtime_profile import apply_deploy_profile
        except ImportError:
            from runtime_profile import apply_deploy_profile

        apply_deploy_profile(self, device or next(self.parameters()).device)

    def apply_eager_profile(self) -> None:
        try:
            from .runtime_profile import apply_eager_profile
        except ImportError:
            from runtime_profile import apply_eager_profile

        apply_eager_profile(self)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        images=None,
        pixel_values=None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        targets: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        images = images if images is not None else pixel_values
        targets = targets if targets is not None else labels
        return VisionLanguageModel.forward(
            self, input_ids, images, attention_mask=attention_mask, targets=targets
        )

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        images=None,
        pixel_values=None,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 5,
        top_k: int = 50,
        top_p: float = 0.9,
        temperature: float = 0.5,
        greedy: bool = False,
        **kwargs,
    ):
        images = images if images is not None else pixel_values
        _ = kwargs
        return VisionLanguageModel.generate(
            self,
            input_ids,
            images,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            greedy=greedy,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        model._repair_legacy_checkpoint_weights(
            pretrained_model_name_or_path, revision=kwargs.get("revision")
        )
        model.tie_weights()
        model._reinit_rotary_cache()
        try:
            from transformers import AutoTokenizer

            try:
                from .processing_visionpsynano import VisionPsyNanoProcessor
            except ImportError:
                from processing_visionpsynano import VisionPsyNanoProcessor

            tok = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=kwargs.get("trust_remote_code", False),
            )
            VisionPsyNanoProcessor._attach_extra_token_attrs(tok, model.config.vlm_extra_tokens)
            model.set_tokenizer(tok)
        except Exception:
            model._sync_image_token_id()
        return model

    @classmethod
    def from_legacy_pretrained(
        cls,
        repo_id_or_path: str,
        *,
        is_flash: Optional[bool] = None,
        variant: Optional[str] = None,
        revision: Optional[str] = None,
        **kwargs,
    ) -> "VisionPsyNanoForConditionalGeneration":
        import json

        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_model

        if os.path.isdir(repo_id_or_path):
            config_path = os.path.join(repo_id_or_path, "config.json")
            weights_path = os.path.join(repo_id_or_path, "model.safetensors")
        else:
            config_path = hf_hub_download(
                repo_id=repo_id_or_path, filename="config.json", revision=revision
            )
            weights_path = hf_hub_download(
                repo_id=repo_id_or_path, filename="model.safetensors", revision=revision
            )

        with open(config_path, "r") as f:
            raw = json.load(f)

        config = VisionPsyNanoConfig.from_legacy_dict(
            raw, is_flash=is_flash, variant=variant
        )
        model = cls(config)
        load_model(model, weights_path)
        model.tie_weights()
        model._reinit_rotary_cache()
        tok_dir = repo_id_or_path if os.path.isdir(repo_id_or_path) else os.path.dirname(config_path)
        if any(
            os.path.exists(os.path.join(tok_dir, n))
            for n in ("tokenizer.json", "tokenizer_config.json")
        ):
            from transformers import AutoTokenizer

            try:
                from .processing_visionpsynano import VisionPsyNanoProcessor
            except ImportError:
                from processing_visionpsynano import VisionPsyNanoProcessor

            tok = AutoTokenizer.from_pretrained(tok_dir)
            VisionPsyNanoProcessor._attach_extra_token_attrs(tok, config.vlm_extra_tokens)
            model.set_tokenizer(tok)
        return model


try:
    from transformers import AutoConfig, AutoModel, AutoModelForImageTextToText

    AutoConfig.register("visionpsynano", VisionPsyNanoConfig)
    AutoModel.register(VisionPsyNanoConfig, VisionPsyNanoForConditionalGeneration)
    AutoModelForImageTextToText.register(VisionPsyNanoConfig, VisionPsyNanoForConditionalGeneration)
except Exception:
    pass
