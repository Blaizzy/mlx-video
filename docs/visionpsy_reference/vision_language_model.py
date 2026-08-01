import json
import os
import tempfile
from dataclasses import asdict
from typing import Optional

from .model_utils import top_k_top_p_filtering
from .vision_transformer import ViT
from .language_model import LanguageModel
from .modality_projector import ModalityProjector
from .vlm_config import VLMConfig

from .processors import get_tokenizer

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_model, save_model

def _drop_if_all_attend(attention_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Drop all-ones masks (avoid decode length mismatch with a fixed prefill mask)."""
    if attention_mask is None:
        return None
    if bool(attention_mask.all()):
        return None
    return attention_mask

class VisionLanguageModel(nn.Module):
    def __init__(self, cfg: VLMConfig, load_backbone=True):
        super().__init__()
        self.cfg = cfg
        if load_backbone:
            print("Loading from backbone weights")
            self.vision_encoder = ViT.from_pretrained(cfg)
            self.decoder = LanguageModel.from_pretrained(cfg)
        else:
            self.vision_encoder = ViT(cfg)
            self.decoder = LanguageModel(cfg)
        self.MP = ModalityProjector(cfg)
        self.load_backbone = load_backbone
        self.tokenizer = get_tokenizer(cfg.lm_tokenizer, cfg.vlm_extra_tokens, cfg.lm_chat_template)

        self._compiled_decoder = None
        self._compiled_vision_encoder = None

    def _vision_encoder_for_prefill(self):
        if not getattr(self.cfg, "compile_inference", False):
            return self.vision_encoder
        if self._compiled_vision_encoder is None:
            self._compiled_vision_encoder = torch.compile(
                self.vision_encoder,
                mode=getattr(self.cfg, "compile_inference_mode", "default"),
                dynamic=True,
                fullgraph=False,
            )
        return self._compiled_vision_encoder

    def _decoder_for_decode(self):

        if not getattr(self.cfg, "compile_inference", False):
            return self.decoder
        if self._compiled_decoder is None:
            mode = getattr(self.cfg, "compile_inference_mode", "default")
            self._compiled_decoder = torch.compile(
                self.decoder,
                mode=mode,
                dynamic=(False if mode == "reduce-overhead" else True),
                fullgraph=False,
            )
        return self._compiled_decoder

    def _replace_img_tokens_with_embd(self, input_ids, token_embd, image_embd):
        image_token_id = getattr(self.tokenizer, "image_token_id", None)
        if image_token_id is None and hasattr(self.tokenizer, "image_token"):
            image_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.image_token)
        if image_token_id is None:
            image_token_id = self.tokenizer.convert_tokens_to_ids("<|image|>")
        mask = (input_ids == image_token_id).unsqueeze(-1)
        flat = image_embd.view(-1, image_embd.size(-1)).to(token_embd.dtype)
        token_embd.masked_scatter_(mask, flat)
        return token_embd

    def _process_images(self, images, device):
        if isinstance(images, list):
            if images and isinstance(images[0], list):
                images = [img for sublist in images for img in sublist]

            if not images:
                return None
            else:
                return torch.cat(images, dim=0).to(device)
        return images

    def forward(self, input_ids, images, attention_mask=None, targets=None):
        images_tensor = self._process_images(images, input_ids.device)
        token_embd = self.decoder.token_embedding(input_ids)

        if images_tensor is not None:
            image_embd = self.vision_encoder(images_tensor)
            image_embd = self.MP(image_embd)
            token_embd = self._replace_img_tokens_with_embd(input_ids, token_embd, image_embd)

        attention_mask = _drop_if_all_attend(attention_mask)

        logits, _ = self.decoder(token_embd, attention_mask=attention_mask)

        loss = None
        if targets is not None:
            logits = self.decoder.head(logits)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)

        return logits, loss

    @torch.inference_mode()
    def generate(self, input_ids, images, attention_mask=None, max_new_tokens=5, top_k=50, top_p=0.9, temperature=0.5, greedy=False):
        images_tensor = self._process_images(images, input_ids.device)
        token_embd = self.decoder.token_embedding(input_ids)

        if images_tensor is not None:
            vision_encoder_fn = self._vision_encoder_for_prefill()
            image_embd = vision_encoder_fn(images_tensor)
            image_embd = self.MP(image_embd)
            token_embd = self._replace_img_tokens_with_embd(input_ids, token_embd, image_embd)

        current_total_seq_len = token_embd.size(1)
        batch_size = input_ids.size(0)
        device = input_ids.device

        cache_max_length = current_total_seq_len + max(int(max_new_tokens), 1)
        head_dim = self.cfg.lm_hidden_dim // self.cfg.lm_n_heads
        use_cuda_graphs_pre = (
            getattr(self.cfg, "compile_inference", False)
            and getattr(self.cfg, "compile_inference_mode", "default") == "reduce-overhead"
            and device.type == "cuda"
        )
        if use_cuda_graphs_pre:
            quantum = int(getattr(self.cfg, "cuda_graphs_cache_quantum", 128))
            if quantum > 1:
                cache_max_length = ((cache_max_length + quantum - 1) // quantum) * quantum
        cache_shape = (batch_size, self.cfg.lm_n_kv_heads, cache_max_length, head_dim)
        cache_dtype = next(self.decoder.parameters()).dtype
        use_cuda_graphs = use_cuda_graphs_pre
        kv_cache_list = []
        for _ in range(self.cfg.lm_n_blocks):
            key_cache = torch.zeros(cache_shape, dtype=cache_dtype, device=device)
            value_cache = torch.zeros(cache_shape, dtype=cache_dtype, device=device)
            if use_cuda_graphs:
                torch._dynamo.mark_static_address(key_cache)
                torch._dynamo.mark_static_address(value_cache)
            kv_cache_list.append({
                "cache_max_length": cache_max_length,
                "key_cache": key_cache,
                "value_cache": value_cache,
            })

        attention_mask = _drop_if_all_attend(attention_mask)

        prefill_output, kv_cache_list = self.decoder(
            token_embd,
            attention_mask=attention_mask,
            kv_cache=kv_cache_list,
            start_pos=0,
        )

        last_token_output_from_prefill = prefill_output[:, -1, :]

        if not self.decoder.lm_use_tokens:
            current_logits = self.decoder.head(last_token_output_from_prefill)
        else:
            current_logits = last_token_output_from_prefill

        newly_generated_ids_list = []

        decode_decoder = self._decoder_for_decode()

        if use_cuda_graphs:
            start_pos_buf = torch.zeros((), dtype=torch.long, device=device)
            torch._dynamo.mark_static_address(start_pos_buf)
        else:
            start_pos_buf = None

        eos_id = self.tokenizer.eos_token_id
        eos_check_interval = int(getattr(self.cfg, "eos_check_interval", 16))
        if eos_id is None or eos_check_interval <= 0:
            eos_check_interval = 0

        for step_idx in range(max_new_tokens):
            if greedy:
                next_token_id = torch.argmax(current_logits, dim=-1, keepdim=True)
            else:
                filtered_logits = top_k_top_p_filtering(current_logits, top_k=top_k, top_p=top_p)
                probs = torch.softmax(filtered_logits / temperature, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)

            newly_generated_ids_list.append(next_token_id)

            if (
                eos_check_interval > 0
                and (step_idx + 1) % eos_check_interval == 0
                and bool((next_token_id == eos_id).all().item())
            ):
                break

            next_token_embed = self.decoder.token_embedding(next_token_id)

            current_token_start_pos = current_total_seq_len
            current_total_seq_len += 1

            if start_pos_buf is not None:
                start_pos_buf.fill_(current_token_start_pos)
                start_pos_to_pass = start_pos_buf
            else:
                start_pos_to_pass = current_token_start_pos

            decode_step_output, kv_cache_list = decode_decoder(
                next_token_embed,
                attention_mask=attention_mask,
                kv_cache=kv_cache_list,
                start_pos=start_pos_to_pass,
            )

            last_token_output = decode_step_output[:, -1, :]

            if not self.decoder.lm_use_tokens:
                current_logits = self.decoder.head(last_token_output)
            else:
                current_logits = last_token_output
        if not newly_generated_ids_list:
            return torch.empty((batch_size,0), dtype=torch.long, device=input_ids.device)

        generated_ids = torch.cat(newly_generated_ids_list, dim=1)

        if self.tokenizer.eos_token_id is not None and generated_ids.numel() > 0:
            seq_len = generated_ids.size(1)
            device = generated_ids.device

            eos_mask = (generated_ids == self.tokenizer.eos_token_id)

            col_indices_for_min = torch.arange(seq_len, device=device)
            masked_col_indices = torch.where(eos_mask, col_indices_for_min.unsqueeze(0).expand_as(generated_ids), seq_len + 1)

            first_eos_indices_values = torch.min(masked_col_indices, dim=1).values
            actual_first_eos_indices = torch.clamp(first_eos_indices_values, max=seq_len)

            col_indices_for_comparison = torch.arange(seq_len, device=device).unsqueeze(0).expand_as(generated_ids)
            replace_mask = col_indices_for_comparison > actual_first_eos_indices.unsqueeze(1)
            generated_ids[replace_mask] = self.tokenizer.eos_token_id
        return generated_ids

    @classmethod
    def from_pretrained(
        cls, repo_id_or_path: str, *, revision: Optional[str] = None
    ) -> "VisionLanguageModel":

        if os.path.exists(repo_id_or_path):
            config_path = os.path.join(repo_id_or_path, "config.json")
            weights_path = os.path.join(repo_id_or_path, "model.safetensors")

            if not os.path.exists(config_path):
                raise ValueError(
                    f"Config file not found at {config_path}. Please provide a valid path."
                )
            if not os.path.exists(weights_path):
                raise ValueError(
                    f"Weights file not found at {weights_path}. Please provide a valid path."
                )
        else:
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(
                repo_id=repo_id_or_path, filename="config.json", revision=revision
            )
            weights_path = hf_hub_download(
                repo_id=repo_id_or_path, filename="model.safetensors", revision=revision
            )

        with open(config_path, "r") as f:
            raw_cfg = json.load(f)
        from dataclasses import fields as dc_fields
        valid_keys = {fld.name for fld in dc_fields(VLMConfig)}
        cfg = VLMConfig(**{k: v for k, v in raw_cfg.items() if k in valid_keys})

        model = cls(cfg, load_backbone=False)
        model._saved_config_keys = set(raw_cfg.keys())

        load_model(model, weights_path)

        return model

    def save_pretrained(self, save_directory: str) -> None:

        os.makedirs(save_directory, exist_ok=True)

        with open(os.path.join(save_directory, "config.json"), "w") as f:
            f.write(json.dumps(asdict(self.cfg), indent=4))

        save_model(self, os.path.join(save_directory, "model.safetensors"))

    def push_to_hub(self, repo_id: str, private: bool = False) -> None:

        from huggingface_hub import create_repo, upload_folder

        repo_url = create_repo(repo_id=repo_id, private=private, exist_ok=True)
        repo_id = repo_url.repo_id
        print("Created repo: ", repo_url)

        with tempfile.TemporaryDirectory() as save_path:
            self.save_pretrained(save_path)

            with open(os.path.join(save_path, "README.md"), "w") as f:
                f.write(MODEL_CARD_TEMPLATE.format(repo_id=repo_id))

            return upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=save_path,
                commit_message="Upload VisionPsyNano using push_to_hub",
            )

MODEL_CARD_TEMPLATE = """
---
library_name: transformers
pipeline_tag: image-text-to-text
tags:
  - vision-language
  - multimodal
---

```python
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image

repo = "{repo_id}"
model = AutoModelForImageTextToText.from_pretrained(
    repo, trust_remote_code=True, dtype="auto"
).cuda().eval()
model.apply_deploy_profile(model.device)
processor = AutoProcessor.from_pretrained(repo, trust_remote_code=True)

inputs = processor(images=Image.open("image.jpg"), text="Describe the image.", return_tensors="pt")
inputs = {{k: v.cuda() if hasattr(v, "cuda") else v for k, v in inputs.items() if v is not None}}
inputs.pop("pixel_values", None)
out = model.generate(**inputs, max_new_tokens=64, greedy=True)
print(processor.batch_decode(out, skip_special_tokens=True)[0])
```
"""
