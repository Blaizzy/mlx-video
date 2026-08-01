---
license: apache-2.0
base_model: qvac/VisionPsy-Nano-460M
tags:
- mlx
- vlm
- vision-language-model
- apple-silicon
- siglip2
- smollm2
language:
- en
- zh
library_name: mlx
pipeline_tag: image-text-to-text
---

# VisionPsy-Nano-460M-MLX

MLX port of [**qvac/VisionPsy-Nano-460M**](https://huggingface.co/qvac/VisionPsy-Nano-460M), a compact 460M-parameter vision-language model from Tether AI Research, converted to run natively on Apple Silicon.

- **Architecture:** SigLIP2-base-patch16-512 vision encoder + pixel-shuffle modality projector + SmolLM2-360M-Instruct decoder
- **Parameters:** ~460M
- **Precision:** bfloat16 (~1.0 GB on disk, down from 2.0 GB fp32)
- **Runtime:** MLX on Apple Silicon (M-series)
- **License:** Apache-2.0

## Benchmarks (MLX bf16, M-series)

Measured across 7 images x 5 prompts, 64 max new tokens, greedy decode:

| Metric | Standard | Flash |
|---|---|---|
| Avg decode tok/s | 99 | 152 |
| Median decode tok/s | 90 | 157 |
| Avg peak GPU memory | 2.64 GB | 2.64 GB |
| Load time | ~0.4 s | ~0.7 s |

Per-prompt-type medians (Standard):

| Prompt type | Median tok/s | Example |
|---|---|---|
| Describe (EN, 1 sentence) | 158.7 | "A smiling man in a white lab coat gestures with his right hand..." |
| What text appears? | 90.3 | "OICOMELVANG" |
| Count objects/people | 40.0 | "There are 3 people in the image." |
| Main subject | 59.2 | "The main subject is a man wearing a white lab coat." |
| Describe (ZH) | 130.4 | "他說：\"OICOMELVANG, 25158\"" |

Full 70-run matrix (Standard + Flash) is at [github.com/KaedeTai/mlx-video/tree/visionpsy-mlx-port](https://github.com/KaedeTai/mlx-video/tree/visionpsy-mlx-port).

## Usage

This repo uses the mlx-vlm-style layout (`text_config` + `vision_config` top-level, `language_model.*` / `vision_tower.*` / `multi_modal_projector.*` tensor prefixes) so it slots cleanly into `mlx-vlm` once a `visionpsy_nano` handler lands there. Until then, load it via the MLX port bundled in **mlx-video** (branch `visionpsy-mlx-port`):

```bash
pip install mlx safetensors transformers pillow
git clone -b visionpsy-mlx-port https://github.com/KaedeTai/mlx-video.git
cd mlx-video
```

```python
from huggingface_hub import snapshot_download
from mlx_video.models.visionpsy_nano import load_visionpsy_nano
from mlx_video.models.visionpsy_nano.processor import load_processor
from PIL import Image

# Snapshot from HF (or point at your local folder)
path = snapshot_download("KaedeTai/VisionPsy-Nano-460M-MLX")

model, cfg = load_visionpsy_nano(path)
proc = load_processor(path, cfg=cfg)

img = Image.open("photo.jpg").convert("RGB")
batch = proc("Describe this image in one sentence.", image=img)

tokens = list(model.generate(
    batch["input_ids"],
    pixel_values=batch["pixel_values"],
    image_token_id=batch["image_token_id"],
    max_new_tokens=64,
    eos_token_id=proc.tokenizer.eos_token_id,
))
print(proc.decode(tokens, skip_special_tokens=True))
```

*Note:* the port's `load_visionpsy_nano` reads the repacked config via a compat shim; the original `_original_config` block is retained inside `config.json` for round-tripping.

## What changed vs the original

- **fp32 -> bf16.** Weights cast to bfloat16. Outputs verified byte-identical on greedy decode for both variants.
- **Prefix rename.** Tensor names moved from `decoder.*` / `vision_encoder.*` / `MP.*` to `language_model.*` / `vision_tower.*` / `multi_modal_projector.*` to match mlx-vlm conventions.
- **Config reshape.** Flat `lm_*` / `vit_*` keys refactored into nested `text_config` / `vision_config` blocks with standard field names (`hidden_size`, `num_hidden_layers`, etc.).
- **Stale buffers dropped.** `decoder.rotary_embd.*` buffers removed — MLX's `nn.RoPE` computes frequencies on the fly.

## Attribution

- **Original model:** Tether AI Research / QVAC — [qvac/VisionPsy-Nano-460M](https://huggingface.co/qvac/VisionPsy-Nano-460M) (Apache-2.0)
- **MLX port + weight repack:** [KaedeTai](https://huggingface.co/KaedeTai)
- **Base components:** SigLIP2 (Google), SmolLM2 (Hugging Face)

If you use these weights, please also cite the original QVAC release and the base model authors.

## See also

- Flash variant: [KaedeTai/VisionPsy-Nano-460M-Flash-MLX](https://huggingface.co/KaedeTai/VisionPsy-Nano-460M-Flash-MLX)
- Original release blog + benchmarks: [qvac/VisionPsy-Nano-460M](https://huggingface.co/qvac/VisionPsy-Nano-460M)
- MLX port source: [github.com/KaedeTai/mlx-video @ visionpsy-mlx-port](https://github.com/KaedeTai/mlx-video/tree/visionpsy-mlx-port)
