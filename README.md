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

- [**LTX-2**](https://huggingface.co/Lightricks/LTX-Video) — 19B parameter video generation model from Lightricks
- [**Wan2.1**](https://github.com/Wan-Video/Wan2.1) — 1.3B / 14B parameter T2V models (single-model pipeline)
- [**Wan2.2**](https://github.com/Wan-Video/Wan2.2) — T2V-14B, TI2V-5B, and I2V-14B models (dual-model pipeline)

## Features

**LTX-2 / LTX-2.3**
- Text-to-Video (T2V), Image-to-Video (I2V), Audio-to-Video (A2V)
- Audio-Video joint generation
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

### Multi-Mac generation (CFG parallel)

With classifier-free guidance, every denoising step runs the transformer twice (prompt and negative prompt) as a batch of 2. `--cfg-parallel` runs each half on a different Mac with a batch of 1 and exchanges the predictions every step with `mx.distributed.all_gather` (about 8 MB per step at 832x480, 81 frames). Each Mac loads the full model, so it does not reduce memory; it reduces time per video.

Supported: single-model text-to-video (Wan2.1 T2V). Image-to-video and the Wan2.2 dual-expert models are rejected with an error for now. It requires guidance enabled (`--guide-scale` > 1), exactly 2 processes, and a fixed `--seed` (both Macs must start from the same noise).

Requirements:

- Two Macs with Thunderbolt 5 connected by a Thunderbolt 5 cable.
- macOS 26.2 or later, RDMA enabled once on each Mac from Recovery (`rdma_ctl enable` in Terminal, then reboot).
- The same Python environment with mlx-video installed and the converted model at the same path on both Macs.
- SSH access from the first Mac to the second.

See the [MLX distributed guide](https://ml-explore.github.io/mlx/build/html/usage/launching_distributed.html) for setting up JACCL/RDMA and the hostfile; `mlx.distributed_config` can generate it.

Example hostfile (`hosts.json`, JACCL over RDMA):

```json
[
  {"ssh": "mac1.local", "ips": ["169.254.x.1"], "rdma": [null, "rdma_en1"]},
  {"ssh": "mac2.local", "ips": ["169.254.x.2"], "rdma": ["rdma_en1", null]}
]
```

The RDMA device names come from `ibv_devices`. In our setup JACCL needed an IPv4 address on the Thunderbolt port itself (e.g. `en1`), not only on `bridge0`.

Launch command:

```bash
mlx.launch --backend jaccl --hostfile hosts.json --python "$(which python)" -- \
    -m mlx_video.models.wan_2.generate \
    --model-dir wan21_mlx \
    --prompt "A red fox walking through fresh snow in a pine forest" \
    --width 832 --height 480 --num-frames 81 \
    --seed 42 \
    --cfg-parallel \
    --output-path fox.mp4
```

Rank 0 decodes and saves the video; rank 1 exits after denoising. Use long-form flags after `--` (mlx.launch parses unknown short flags itself).

Results (Wan2.1 T2V 1.3B, 832x480, 81 frames, 50 steps, guide 5.0, two MacBook Pro M4 Pro 24 GB over Thunderbolt 5 with JACCL; one run per prompt):

| Prompt (seed) | 1 Mac | 2 Macs | Speedup (total) | Per step |
|---|---|---|---|---|
| Red fox in snow (42) | 97.5 min | 52.0 min | 1.87x | 111.2 s → 56.7 s (1.96x) |
| Paper boat in the rain (7) | 95.3 min | 51.4 min | 1.86x | 108.0 s → 56.0 s (1.93x) |

The output is not bit-identical to a single-Mac run (batch 1 vs batch 2 changes floating-point rounding, which accumulates over 50 steps) but is visually the same video (PSNR 36.9 and 37.3 dB against the single-Mac video). Text encoding and VAE decoding are not split, so the total speedup is a bit lower than the per-step one.

Note for contributors: with steps of ~55 s, a GPU command buffer that waits on the other Mac is killed by macOS ("GPU Timeout"). The implementation finishes GPU work first, exchanges on the CPU stream, and evaluates the result before queuing more GPU work.

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
| `--cfg-parallel` | off | Split classifier-free guidance across 2 Macs (launch with `mlx.launch`, needs `--seed`) |
| `--output-path` | `output.mp4` | Output video path |

---

## Requirements

- macOS with Apple Silicon
- Python >= 3.11
- MLX >= 0.22.0
- For weight conversion: PyTorch (`pip install torch`)

## License

MIT
