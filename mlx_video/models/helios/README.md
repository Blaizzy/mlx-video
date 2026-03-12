# Helios — Text-to-Video Generation on Apple Silicon

Helios is a 14B-parameter autoregressive video generation model that produces minute-scale, temporally coherent video. This implementation targets the **Helios-Distilled** variant for text-to-video generation on Apple Silicon via MLX.

## Quick Start

### 1. Convert Weights

Download the HuggingFace checkpoint and convert to MLX format:

```bash
python -m mlx_video.convert_helios \
    --checkpoint-dir /path/to/BestWishYsh/Helios-Distilled \
    --output-dir ./helios-mlx
```

With 4-bit quantization (~7 GB, fits 16 GB Macs):

```bash
python -m mlx_video.convert_helios \
    --checkpoint-dir /path/to/BestWishYsh/Helios-Distilled \
    --output-dir ./helios-mlx-4bit \
    --quantize --bits 4
```

Or quantize an existing MLX model (skips HF conversion):

```bash
python -m mlx_video.convert_helios \
    --checkpoint-dir ./helios-mlx \
    --output-dir ./helios-mlx-4bit \
    --quantize-only --bits 4
```

### 2. Generate Video

```bash
python -m mlx_video.generate_helios \
    --model-dir ./helios-mlx \
    --prompt "A golden retriever running through a sunlit meadow" \
    --output-path my_video.mp4
```

```bash
python -m mlx_video.generate_helios \
    --model-dir ../Helios-Distilled-MLX/ \
    --num-frames 330 \
    --seed 2391784614 \
    --prompt "Two dogs of the poodle breed sitting on a beach wearing sunglasses, nodding with their heads, close up, cinematic, sunset"
```

## CLI Options

| Flag | Default | Description |
|---|---|---|
| `--model-dir` | *(required)* | Path to converted MLX model directory |
| `--prompt` | *(required)* | Text prompt describing the video |
| `--width` | `640` | Video width in pixels (must be divisible by 64) |
| `--height` | `384` | Video height in pixels (must be divisible by 64) |
| `--num-frames` | `99` | Number of output frames (auto-rounded to multiple of 33) |
| `--pyramid-steps` | `2 2 2` | Steps per pyramid stage (3-stage progressive denoising) |
| `--amplify-first-chunk` | off | Double steps for first chunk (better quality) |
| `--guidance-scale` | `5.0` | CFG guidance scale (`1.0` = no guidance, `5.0` = default) |
| `--negative-prompt` | `""` | Negative prompt for classifier-free guidance |
| `--seed` | `-1` | Random seed (`-1` for random) |
| `--output-path` | `output_helios.mp4` | Output video file path |
| `--tiling` | `auto` | VAE tiling mode: `auto`, `none`, `default`, `aggressive`, `conservative` |

## How It Works

### Autoregressive Chunked Generation

Unlike single-pass models, Helios generates video **autoregressively in 33-frame chunks** (9 latent frames each). This enables minute-scale video with temporal coherence:

```
Chunk 1: [frames 1-33]   → denoise from noise
Chunk 2: [frames 34-66]  → denoise with history from chunk 1
Chunk 3: [frames 67-99]  → denoise with history from chunks 1-2
...
```

For a 99-frame video at 24 fps, this produces ~4 seconds of video across 3 chunks.

### Multi-Scale History Memory

Each chunk beyond the first receives context from prior chunks via three Conv3d patch embeddings at different temporal/spatial scales:

| Scale | Kernel | Latent Frames | Purpose |
|---|---|---|---|
| **Long** | 4×8×8 | 16 | Coarse global context |
| **Mid** | 2×4×4 | 2 | Medium-term motion |
| **Short** | 1×2×2 | 1 | Fine local detail |

Total history: 19 latent tokens prepended to the current chunk's 9 tokens, giving the model a 28-token sequence that sees both broad context and recent detail.

### Pipeline Steps

1. **Text encoding** — UMT5-XXL encodes the prompt (shared with Wan)
2. **Per-chunk 3-stage pyramid denoising**:
   - Sample Gaussian noise for 9 latent frames
   - **Downsample** noise to 1/4 spatial resolution
   - **Stage 0** (quarter res): Denoise 2 steps — very fast (16× fewer tokens)
   - **Upsample** 2×, mix in structured block noise (alpha/beta correction)
   - **Stage 1** (half res): Denoise 2 steps — fast (4× fewer tokens)
   - **Upsample** 2×, mix block noise
   - **Stage 2** (full res): Denoise 2 steps — full quality
   - Prepend multi-scale history tokens (if not first chunk)
   - Extract current-chunk latents; update history buffer
3. **VAE decoding** — AutoencoderKLWan decodes latents to RGB (shared with Wan, supports tiled decoding)
4. **Video output** — Frames saved as MP4 via OpenCV

### Pyramid Denoising

The 3-stage pyramid dramatically speeds up generation by performing most denoising at reduced spatial resolution:

```
Stage 0:  ████░░░░░░░░░░░░  (1/4 res, 2 steps) — 16× fewer tokens
Stage 1:  ████████░░░░░░░░  (1/2 res, 2 steps) — 4× fewer tokens
Stage 2:  ████████████████  (full res, 2 steps) — final refinement
```

Customize with `--pyramid-steps`:
- `--pyramid-steps 2 2 2` — default, 6 total forward passes (fastest)
- `--pyramid-steps 4 4 4` — 12 passes (higher quality)
- `--pyramid-steps 2 2 4` — more refinement at full resolution

Use `--amplify-first-chunk` to double the steps for the first chunk, which typically has the biggest impact on overall quality.

## Architecture

Helios shares 95% of its architecture with Wan 2.1:

| Component | Details |
|---|---|
| Transformer | 40 layers, dim=5120, 40 heads, head_dim=128 |
| FFN | SiLU-gated, dim=13824 |
| Patch embedding | (1, 2, 2) — 1 temporal, 2×2 spatial |
| RoPE | 3-way factorized (44, 42, 42), θ=10000 |
| Modulation | 6-vector AdaLN-Zero (shift/scale/gate × 2) |
| VAE | AutoencoderKLWan, stride (4, 8, 8), z_dim=16 |
| Text encoder | UMT5-XXL, dim=4096, 512 token context |

**Helios-specific additions:**
- Restricted self-attention (history tokens attend only among themselves)
- Zero-timestep embedding for history tokens
- Multi-scale history Conv3d patching (short/mid/long)

## Frame Count Constraints

- Output frames are auto-rounded to multiples of **33** (the chunk size)
- Each chunk produces 33 pixel frames = 9 latent frames
- The VAE temporal stride is 4, with formula: `latent_frames = (pixel_frames - 1) / 4 + 1`

Examples:
- `--num-frames 33` → 1 chunk, ~1.4s at 24fps
- `--num-frames 99` → 3 chunks, ~4.1s at 24fps
- `--num-frames 231` → 7 chunks, ~9.6s at 24fps

## Resolution Guide

Height and width must be divisible by 64. Recommended resolutions:

| Resolution | Aspect Ratio | VRAM (bf16) | VRAM (4-bit) |
|---|---|---|---|
| 384 × 640 | 3:5 | ~28 GB | ~7 GB |
| 384 × 384 | 1:1 | ~24 GB | ~6 GB |
| 256 × 448 | 9:16 | ~20 GB | ~5 GB |

## Memory Tips

- Use `--tiling aggressive` for lower VRAM usage during VAE decoding
- Use 4-bit quantization (`--quantize --bits 4` during conversion) to reduce model size from ~28 GB to ~7 GB
- Shorter videos (fewer chunks) require less peak memory for history
- Smaller resolutions significantly reduce memory (quadratic in spatial dimensions)

## File Structure

```
mlx_video/models/helios/
├── __init__.py
├── README.md          ← you are here
├── config.py          ← HeliosModelConfig dataclass
├── rope.py            ← 3-way factorized RoPE (44,42,42)
├── attention.py       ← Self-attention (with history restriction) + cross-attention
├── scheduler.py       ← DMD flow-matching scheduler with 3-stage pyramid support
├── transformer.py     ← HeliosTransformerBlock + HeliosModel backbone
└── loading.py         ← Weight loading (reuses Wan's T5/VAE loaders)

mlx_video/
├── convert_helios.py  ← HF diffusers → MLX weight conversion
└── generate_helios.py ← CLI generation pipeline
```
