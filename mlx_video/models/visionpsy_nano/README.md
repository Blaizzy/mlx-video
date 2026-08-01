# VisionPsy-Nano (MLX port)

MLX port of Tether QVAC's [VisionPsy-Nano-460M](https://huggingface.co/qvac/VisionPsy-Nano-460M)
vision-language model. Both variants are supported:

| variant  | HF repo                                    | preprocessor policy                                   |
| -------- | ------------------------------------------ | ----------------------------------------------------- |
| Standard | `qvac/VisionPsy-Nano-460M`                 | resize long side to `max_img_size` (2048), tile 512   |
| Flash    | `qvac/VisionPsy-Nano-460M-Flash`           | resize short side ≥ 512, cap long side at 2048        |

Architecture:
- **Vision:** SigLIP2-base-patch16-512 ViT (12 blocks, 768 hidden, 12 heads, fused QKV).
- **Projector:** single-layer 4x pixel-shuffle projector (12288 → 960, no bias).
- **Language:** SmolLM2-360M-Instruct decoder (32 blocks, 960 hidden, GQA 15/5, RoPE base 100k, tied lm_head, fused gate_up_proj).

## Usage

```python
import mlx.core as mx
from PIL import Image
from mlx_video.models.visionpsy_nano import load_visionpsy_nano
from mlx_video.models.visionpsy_nano.processor import load_processor

model, cfg = load_visionpsy_nano(
    "/path/to/VisionPsy-Nano-460M",
    dtype=mx.bfloat16,   # or fp16 / fp32
)
proc = load_processor("/path/to/VisionPsy-Nano-460M", cfg=cfg)

image = Image.open("your.png")
batch = proc("Describe this image in one sentence.", image=image)

tokens = list(model.generate(
    batch["input_ids"],
    pixel_values=batch["pixel_values"],
    image_token_id=batch["image_token_id"],
    max_new_tokens=64,
    eos_token_id=proc.tokenizer.eos_token_id,
))
print(proc.decode(tokens))
```

## Smoke test

```bash
python scripts/visionpsy_nano_smoke.py \
    --image docs/wan_s2v_phase3_notes/willy_portrait.png \
    --prompt "Describe this image in one sentence." \
    --variants standard,flash
```

## File layout

```
mlx_video/models/visionpsy_nano/
├── __init__.py               # public API
├── config.py                 # VisionPsyNanoConfig (loads config.json)
├── vision_transformer.py     # SigLIP2 ViT (fused QKV, no CLS)
├── language_model.py         # SmolLM2 decoder w/ growing KV cache
├── modality_projector.py     # 4x pixel-shuffle + linear
├── visionpsy_nano.py         # VisionPsyNano composition + greedy generate
├── processor.py              # PIL preprocess + chat-template wrapping
├── weight_loader.py          # safetensors -> MLX (bf16 cast, tie head)
└── README.md
tests/
└── test_visionpsy_nano_load.py
scripts/
└── visionpsy_nano_smoke.py
```

## Weight remapping

The PyTorch checkpoint keys are already very close to our MLX attribute names,
so the loader only has to do two things:

1. `vision_encoder.patch_embedding.conv.weight`: transpose from PyTorch's
   `(out, in, kH, kW)` to MLX's `(out, kH, kW, in)`.
2. Drop `decoder.rotary_embd.{inv_freq,cos_cached,sin_cached}` — MLX's
   `nn.RoPE` computes rotary frequencies on the fly.

The fused `decoder.blocks.{i}.mlp.gate_up_proj` and `vision_encoder.blocks.{i}.attn.qkv_proj`
tensors are kept fused; we split them at the last moment inside the forward
pass. This matches how the tensors are saved in the checkpoint.

## Serving with oMLX

oMLX 0.5.3 does NOT auto-detect `visionpsynano`-type models — its discovery
filters models by known `model_type`. Options to bridge:

1. Write an mlx-vlm-compatible config wrapper (this port already has all the
   pieces — the model dir just needs a config file with the split
   `text_config`/`vision_config` layout, plus a lightweight `mlx_vlm/models/`
   entry point).
2. Land an oMLX / mlx-vlm PR that registers the `visionpsynano` architecture.

Recommended: package as `mlx-community/VisionPsy-Nano-460M-MLX` following
mlx-vlm's SmolVLM layout so it becomes discoverable without a PR.
