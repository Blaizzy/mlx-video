# mlx-video

MLX-Video is the best package for inference and finetuning of Image-Video-Audio generation models on your Mac using MLX.

## Installation

### Option 1: Install with pip (requires git):
```bash
pip install git+https://github.com/Blaizzy/mlx-video.git
```

### Option 2: Install with uv (ultra-fast package manager, optional):
```bash
uv pip install git+https://github.com/Blaizzy/mlx-video.git
```

## Supported Models

- [**LTX-2 / LTX-2.5**](https://huggingface.co/Lightricks/LTX-2.5) — 19B and 22B video generation models from Lightricks
- [**Wan2.1**](https://github.com/Wan-Video/Wan2.1) — 1.3B / 14B parameter T2V models (single-model pipeline)
- [**Wan2.2**](https://github.com/Wan-Video/Wan2.2) — T2V-14B, TI2V-5B, and I2V-14B models (dual-model pipeline)

## Features

**LTX-2 / LTX-2.3 / LTX-2.5**
- Text-to-Video (T2V), Image-to-Video (I2V), Audio-to-Video (A2V)
- Audio-Video joint generation
- Prompt-driven multishot generation
- LTX-2.5 Diffusion Fidelity Rendering (DFR) with generated keyframes
- Experimental LTX-2.5 diffusion video VAE with fused Metal attention
- Multi-pipeline: distilled, dev, dev-two-stage, dev-two-stage-hq
- 2x spatial upscaling for images and videos
- Prompt enhancement via Gemma

**Wan2.1 / Wan2.2**
- Text-to-Video (T2V) — 1.3B and 14B models
- Image-to-Video (I2V) — 14B model
- Flow-matching diffusion with classifier-free guidance
- LoRA support (e.g. Wan2.2-Lightning for 4-step generation)

**General**
- Optimized for Apple Silicon using MLX

---

## LTX-2

### Text-to-Video Generation

```bash
# Text-to-Video (distilled, fastest)
uv run mlx_video.ltx_2.generate --prompt "Two dogs wearing sunglasses, cinematic, sunset" -n 97 --width 768

# LTX-2.5 distilled T2V (accept the gated model terms first)
uv run mlx_video.ltx_2.generate --model-repo Lightricks/LTX-2.5 \
    --prompt "Ocean waves at golden hour" -n 121 --width 768 --height 512

# Image-to-Video
uv run mlx_video.ltx_2.generate --prompt "A person dancing" --image photo.jpg

# Audio-to-Video
uv run mlx_video.ltx_2.generate --audio-file music.wav --prompt "A band playing music"

# Dev pipeline with CFG (higher quality)
uv run mlx_video.ltx_2.generate --pipeline dev --prompt "A cinematic scene" --cfg-scale 3.0

# Dev two-stage HQ (highest quality)
uv run mlx_video.ltx_2.generate --pipeline dev-two-stage-hq \
    --prompt "A cinematic scene of ocean waves at golden hour" \
    --model-repo prince-canuma/LTX-2-dev
```

<img src="https://github.com/Blaizzy/mlx-video/raw/main/examples/poodles.gif" width="512" alt="Poodles demo">

**Converting weights:**

Pre-converted weights are available on HuggingFace ([LTX-2-distilled](https://huggingface.co/prince-canuma/LTX-2-distilled), [LTX-2-dev](https://huggingface.co/prince-canuma/LTX-2-dev), [LTX-2.3-distilled](https://huggingface.co/prince-canuma/LTX-2.3-distilled), [LTX-2.3-dev](https://huggingface.co/prince-canuma/LTX-2.3-dev)), or convert from the original Lightricks checkpoint:


### LTX-2 CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--prompt`, `-p` | (required) | Text description of the video |
| `--height`, `-H` | 512 | Output height (must be divisible by 64) |
| `--width`, `-W` | 512 | Output width (must be divisible by 64) |
| `--num-frames`, `-n` | 100 | Number of frames |
| `--seed`, `-s` | 42 | Random seed for reproducibility |
| `--fps` | 24 | Frames per second |
| `--output`, `-o` | output.mp4 | Output video path |
| `--save-frames` | false | Save individual frames as images |
| `--model-repo` | Lightricks/LTX-2 | HuggingFace model repository |


---

## Wan2.1 / Wan2.2

Both [Wan2.1](https://github.com/Wan-Video/Wan2.1) and [Wan2.2](https://github.com/Wan-Video/Wan2.2) are text-to-video diffusion models built on a DiT (Diffusion Transformer) backbone with a T5 text encoder and 3D VAE.

### Step 0: Download and Convert Weights

See the dedicated Wan2.1/Wan2.2 [README.md](mlx_video/models/wan_2/README.md) for details.

### Step 1: Generate Video

```bash
# Wan2.1 — uses defaults from config (50 steps, shift=5.0, guide=5.0)
python -m mlx_video.wan_2.generate \
    --model-dir wan21_mlx \
    --prompt "A cat playing piano in a cozy room"

# Wan2.2 — uses defaults from config (40 steps, shift=12.0, guide=3.0,4.0)
python -m mlx_video.wan_2.generate \
    --model-dir wan22_mlx \
    --prompt "A cat playing piano in a cozy room"
```

With custom settings:

```bash
python -m mlx_video.wan_2.generate \
    --model-dir wan21_mlx \
    --prompt "Ocean waves at sunset, cinematic, 4K" \
    --negative-prompt "blurry, low quality" \
    --width 1280 \
    --height 720 \
    --num-frames 81 \
    --steps 50 \
    --guide-scale 5.0 \
    --shift 5.0 \
    --seed 42 \
    --output-path my_video.mp4
```

The pipeline auto-detects the model version from `config.json` and selects the right pipeline mode (single or dual model).

### Image-to-Video (I2V-14B)

```bash
python -m mlx_video.wan_2.generate \
    --model-dir wan22_i2v_mlx \
    --prompt "The camera slowly zooms in as the subject begins to move" \
    --image start.png \
    --num-frames 81 \
    --output-path my_video.mp4
```

### LoRA Support

LoRAs can be used with the `--lora-high` and `--lora-low` command line switches.

For example, using the distilled [Wan2.2-Lightning](https://huggingface.co/lightx2v/Wan2.2-Lightning) LoRA for 4-step generation:

```bash
python -m mlx_video.wan_2.generate \
    --model-dir /Volumes/SSD/Wan-AI/Wan2.2-T2V-A14B-MLX \
    --width 480 \
    --height 704 \
    --num-frames 41 \
    --prompt "Two dogs of the poodle breed sitting on a beach wearing sunglasses, nodding with their heads, close up, cinematic, sunset" \
    --steps 4 \
    --guide-scale 1 \
    --trim-first-frames 1 \
    --seed 2391784614 \
    --lora-high /Volumes/SSD/Wan-AI/lightx2v/Wan2.2-Lightning/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V2.0/high_noise_model.safetensors 1 \
    --lora-low /Volumes/SSD/Wan-AI/lightx2v/Wan2.2-Lightning/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V2.0/low_noise_model.safetensors 1
```

![Poodles](examples/poodles-wan.gif)

### Wan CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model-dir` | (required) | Path to converted MLX model directory |
| `--prompt` | (required) | Text description of the video |
| `--image` | `None` | Input image path (for I2V models) |
| `--negative-prompt` | `""` | Negative prompt for guidance |
| `--width` | 1280 | Video width |
| `--height` | 720 | Video height |
| `--num-frames` | 81 | Number of frames (must be 4n+1) |
| `--steps` | from config | Number of diffusion steps |
| `--guide-scale` | from config | Guidance scale: float or `low,high` pair |
| `--shift` | from config | Noise schedule shift |
| `--seed` | -1 (random) | Random seed for reproducibility |
| `--output-path` | `output.mp4` | Output video path |

---

## Requirements

- macOS with Apple Silicon
- Python >= 3.11
- MLX >= 0.22.0
- For weight conversion: PyTorch (`pip install torch`)

## License

MIT
