---
license: apache-2.0
base_model: qvac/VisionPsy-Nano-460M-Flash
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

# VisionPsy-Nano-460M-Flash-MLX

MLX port of [**qvac/VisionPsy-Nano-460M-Flash**](https://huggingface.co/qvac/VisionPsy-Nano-460M-Flash) — the latency-optimized sibling of VisionPsy-Nano-460M — converted to run natively on Apple Silicon.

- **Architecture:** SigLIP2-base-patch16-512 vision encoder + pixel-shuffle modality projector + SmolLM2-360M-Instruct decoder (same weights as Standard except the projector uses **fewer visual tokens per image**, ~7 tiles vs ~13 tiles at 512x512)
- **Parameters:** ~460M
- **Precision:** bfloat16 (~1.0 GB on disk, down from 2.0 GB fp32)
- **Runtime:** MLX on Apple Silicon (M-series)
- **License:** Apache-2.0

## Benchmarks (MLX bf16, M-series)

Measured across 7 images x 5 prompts, 64 max new tokens, greedy decode:

| Metric | Standard | Flash |
|---|---|---|
| Avg decode tok/s | 99 | **152** |
| Median decode tok/s | 90 | **157** |
| Avg peak GPU memory | 2.64 GB | 2.64 GB |
| Load time | ~0.4 s | ~0.7 s |

Per-prompt-type medians (Flash):

| Prompt type | Median tok/s | Example |
|---|---|---|
| Describe (EN, 1 sentence) | 198.5 | "A man in a white lab coat with the name 'De WARON' on it stands behind a desk..." |
| What text appears? | 96.9 | "De Wooning" |
| Count objects/people | 61.4 | "1" |
| Main subject | 170.1 | "The main subject is a man wearing a white lab coat with the name 'De Woon' and a logo on it." |
| Describe (ZH) | 221.2 | "这是一...話是柯尼特, 黃色胸部..." |

Flash gets the biggest speedups on text-heavy images (receipts: **~295 tok/s**, neon text: **~244 tok/s**) where fewer visual tokens still capture the content.

## Usage

Layout matches mlx-vlm conventions. Until `mlx-vlm` adds a `visionpsy_nano` handler, load via the MLX port at [KaedeTai/mlx-video @ visionpsy-mlx-port](https://github.com/KaedeTai/mlx-video/tree/visionpsy-mlx-port):

```python
from huggingface_hub import snapshot_download
from mlx_video.models.visionpsy_nano import load_visionpsy_nano
from mlx_video.models.visionpsy_nano.processor import load_processor
from PIL import Image

path = snapshot_download("KaedeTai/VisionPsy-Nano-460M-Flash-MLX")

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

## Attribution

- **Original model:** Tether AI Research / QVAC — [qvac/VisionPsy-Nano-460M-Flash](https://huggingface.co/qvac/VisionPsy-Nano-460M-Flash) (Apache-2.0)
- **MLX port + weight repack:** [KaedeTai](https://huggingface.co/KaedeTai)
- **Base components:** SigLIP2 (Google), SmolLM2 (Hugging Face)

## See also

- Standard variant: [KaedeTai/VisionPsy-Nano-460M-MLX](https://huggingface.co/KaedeTai/VisionPsy-Nano-460M-MLX)
- Original release blog + benchmarks: [qvac/VisionPsy-Nano-460M](https://huggingface.co/qvac/VisionPsy-Nano-460M)
- MLX port source: [github.com/KaedeTai/mlx-video @ visionpsy-mlx-port](https://github.com/KaedeTai/mlx-video/tree/visionpsy-mlx-port)
