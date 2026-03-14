# mlx-video

MLX-Video is the best package for inference and finetuning of Image-Video-Audio generation models on your Mac using MLX.

## Installation

Install from source:

### Option 1: Install with pip (requires git):
```bash
pip install git+https://github.com/Blaizzy/mlx-video.git
```

### Option 2: Install with uv (ultra-fast package manager, optional):
```bash
uv pip install git+https://github.com/Blaizzy/mlx-video.git
```

## Model Downloads

Use the repo script to prefetch weights into the Hugging Face cache (so `snapshot_download` can find them locally):

```bash
./download.sh
```

You can also set a custom cache location:

```bash
./download.sh --hf-home /path/to/hf-cache
```

Supported models:

- [**LTX-2**](https://huggingface.co/Lightricks/LTX-Video) — 19B parameter video generation model from Lightricks
- [**Wan2.1**](https://github.com/Wan-Video/Wan2.1) — 1.3B / 14B parameter T2V models (single-model pipeline)
- [**Wan2.2**](https://github.com/Wan-Video/Wan2.2) — T2V-14B, TI2V-5B, and I2V-14B models (dual-model pipeline)

## Features

- Text-to-video generation with multiple model families
- LTX-2: Two-stage pipeline with 2x spatial upscaling
- Wan2.1/2.2: Flow-matching diffusion with classifier-free guidance
- Optimized for Apple Silicon using MLX

---

## LTX-2

> **ℹ️ Info:** Currently, only the distilled variant is supported. Full LTX-2 feature support is coming soon.

### Text-to-Video Generation

```bash
uv run mlx_video.generate --prompt "Two dogs of the poodle breed wearing sunglasses, close up, cinematic, sunset" -n 100 --width 768
```

<img src="https://github.com/Blaizzy/mlx-video/raw/main/examples/poodles.gif" width="512" alt="Poodles demo">

With custom settings:

```bash
python -m mlx_video.generate \
    --prompt "Ocean waves crashing on a beach at sunset" \
    --height 768 \
    --width 768 \
    --num-frames 65 \
    --seed 123 \
    --output my_video.mp4
```

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

### How It Works (LTX-2)

1. **Stage 1**: Generate at half resolution (e.g., 384×384) with 8 denoising steps
2. **Upsample**: 2× spatial upsampling via LatentUpsampler
3. **Stage 2**: Refine at full resolution (e.g., 768×768) with 3 denoising steps
4. **Decode**: VAE decoder converts latents to RGB video

---

## Wan2.1 / Wan2.2

Both [Wan2.1](https://github.com/Wan-Video/Wan2.1) and [Wan2.2](https://github.com/Wan-Video/Wan2.2) are text-to-video diffusion models built on a DiT (Diffusion Transformer) backbone with a T5 text encoder and 3D VAE. 

### Step 0: Download and Convert Weights

See the dedicated Wan2.1/Wan2.2 [README.md](mlx_video/models/wan/README.md) for details. 

### Step 1: Generate Video

```bash
# Wan2.1 — uses defaults from config (50 steps, shift=5.0, guide=5.0)
python -m mlx_video.generate_wan \
    --model-dir wan21_mlx \
    --prompt "A cat playing piano in a cozy room"

# Wan2.2 — uses defaults from config (40 steps, shift=12.0, guide=3.0,4.0)
python -m mlx_video.generate_wan \
    --model-dir wan22_mlx \
    --prompt "A cat playing piano in a cozy room"
```

With custom settings:

```bash
python -m mlx_video.generate_wan \
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

The pipeline auto-detects the model version from `config.json` and selects the right pipeline mode (single or dual model). You can also override any parameter via CLI flags.

#### Image-to-Video (I2V-14B)

```bash
# Generate video from an input image
python -m mlx_video.generate_wan \
    --model-dir wan22_i2v_mlx \
    --prompt "The camera slowly zooms in as the subject begins to move" \
    --image start.png \
    --num-frames 81 \
    --output-path my_video.mp4
```

The I2V-14B model encodes the input image through the Wan2.1 VAE encoder and uses channel concatenation (`y` tensor with 4 mask + 16 image latent channels) to condition generation on the first frame.

#### Generation CLI Options

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

## LoRA Support

LoRA's can be used with the `--lora-high` and `--lora-low` command line switches.

For example, for using the the distilled [Wan2.2-Lightning](https://huggingface.co/lightx2v/Wan2.2-Lightning) LoRA, use the following command. Lightning speeds up generation by using only 4 steps and a CFG scale of 1.

```bash
python -m mlx_video.generate_wan \
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

Which results in 
![Poodles](examples/poodles-wan.gif)

## Requirements

- macOS with Apple Silicon
- Python >= 3.11
- MLX >= 0.22.0
- For weight conversion: PyTorch (`pip install torch`)

## Project Structure

```
mlx_video/
├── generate.py              # LTX-2 generation pipeline
├── generate_wan.py          # Wan2.1/2.2 generation pipeline
├── convert.py               # LTX-2 weight conversion
├── convert_wan.py           # Wan weight conversion (PyTorch → MLX)
├── postprocess.py           # Video post-processing utilities
├── utils.py                 # Helper functions
└── models/
    ├── ltx/                 # LTX-2 model
    │   ├── ltx.py           # DiT transformer
    │   ├── config.py        # Configuration
    │   ├── transformer.py   # Transformer blocks
    │   ├── attention.py     # Multi-head attention with RoPE
    │   ├── text_encoder.py  # Gemma 3 text encoder
    │   ├── upsampler.py     # 2x spatial upsampler
    │   └── video_vae/       # VAE encoder/decoder
    └── wan/                 # Wan2.1/2.2 model
        ├── config.py        # Configuration (2.1 & 2.2 presets)
        ├── model.py         # WanModel (DiT transformer)
        ├── transformer.py   # Attention blocks with 6-element modulation
        ├── attention.py     # Self/cross attention with QK-norm
        ├── rope.py          # 3-way factorized RoPE
        ├── text_encoder.py  # T5 UMT5-XXL encoder
        ├── vae.py           # 3D causal VAE decoder
        └── scheduler.py     # Flow-matching Euler scheduler
```

## License

MIT
