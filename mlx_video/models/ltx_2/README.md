# LTX-2 for MLX

MLX port of [LTX-2](https://huggingface.co/Lightricks/LTX-2), a 19B parameter video generation model from Lightricks with synchronized audio-video support.

LTX-2.5 distilled text-to-video and image-to-video are also supported using the
official split checkpoints. Accept the model terms on Hugging Face before the
first download. The lighter convolutional video VAE is the default, and the
experimental diffusion video VAE is available with a fused MLX Metal
neighborhood-attention kernel. Audio-to-video and
synchronized audio-video generation use the split LTX-2.5 audio VAE and vocoder.

## Pipelines

Four pipeline types are available via the `--pipeline` flag:

| Pipeline | Description | CFG | Stages | Speed |
|----------|-------------|-----|--------|-------|
| `distilled` (default) | Fixed sigma schedule, no CFG | No | 2 (8+3 steps) | Fastest |
| `dfr` | Generated keyframes + IC-LoRA spatial detailing | No | 2 (8+3 steps) | Production quality |
| `dev` | Dynamic sigmas, constant CFG | Yes | 1 (30 steps) | Medium |
| `dev-two-stage` | Dev + LoRA refinement | Yes (stage 1) | 2 (30+3 steps) | Slow |
| `dev-two-stage-hq` | res_2s sampler + LoRA both stages | Yes (stage 1) | 2 (15+3 steps) | Slow, highest quality |

## Usage

### Text-to-Video (T2V)

```bash
# Distilled (default) - fast, two-stage
uv run mlx_video.generate --prompt "Two dogs wearing sunglasses, cinematic, sunset" -n 97 --width 768

# LTX-2.5 distilled (downloads only the required split components)
uv run mlx_video.generate --model-repo Lightricks/LTX-2.5 \
    --prompt "Ocean waves at golden hour" -n 121 --width 768 --height 512

# Omit --num-frames to predict duration from the LTX-2.5 caption
uv run mlx_video.generate --model-repo Lightricks/LTX-2.5 \
    --prompt "A traveler crosses a valley and pauses to watch the sunrise" \
    --auto-duration 2 10 --width 768 --height 512

# Enhance a short prompt with a separate generative Gemma 4 model
uv run mlx_video.generate --model-repo Lightricks/LTX-2.5 \
    --prompt "A cat walking through grass" --enhance-prompt \
    --width 768 --height 512

# Dev - single-stage with CFG
uv run mlx_video.generate --pipeline dev --prompt "A cinematic scene" --cfg-scale 3.0

# Dev two-stage - dev + LoRA refinement
uv run mlx_video.generate --pipeline dev-two-stage \
    --prompt "Two dogs of the poodle breed wearing sunglasses, close up, cinematic, sunset" \
    -n 145 --width 1024 --height 768 \
    --model-repo prince-canuma/LTX-2-dev \
    --cfg-scale 3.0 --lora-strength 0.8 \
    --enhance-prompt

# Dev two-stage HQ - res_2s sampler, LoRA both stages (highest quality)
uv run mlx_video.generate --pipeline dev-two-stage-hq \
    --prompt "A cinematic scene of ocean waves at golden hour" \
    --model-repo prince-canuma/LTX-2-dev

# HQ with custom LoRA strengths
uv run mlx_video.generate --pipeline dev-two-stage-hq \
    --prompt "A sunset over mountains" \
    --model-repo prince-canuma/LTX-2-dev \
    --lora-strength-stage-1 0.3 --lora-strength-stage-2 0.6
```

### Multishot generation

LTX-2.5 can generate multiple shots through the regular T2V path. Describe the
shots in order, repeat important subject details in each shot, and allocate
enough frames for the requested sequence:

```bash
uv run mlx_video.generate --model-repo Lightricks/LTX-2.5 \
    --prompt "A cinematic three-shot sequence featuring the same woman with short silver hair, a yellow raincoat, and a red scarf throughout. Shot 1: a medium tracking shot as she walks through a rainy neon-lit alley. Cut to Shot 2: a close-up of the same woman under her umbrella, with the same clothes and alley behind her. Cut to Shot 3: a wide shot of the same woman entering a warm-lit shop, still wearing the same yellow raincoat and red scarf. Maintain the same face, hairstyle, wardrobe, environment, and color grade across all three shots." \
    --num-frames 241 --width 768 --height 512
```

Multishot is prompt-driven rather than a frame-accurate editing interface. Shot
lengths and cut positions are not guaranteed, and an explicit hard cut may be
rendered as a smooth transition. For stronger continuity, keep the character,
wardrobe, environment, lighting, and visual style descriptions consistent
across shots. When using `--enhance-prompt`, verify that the expanded prompt
preserves the requested shot structure.

### Diffusion Fidelity Rendering (DFR)

LTX-2.5 DFR adds generated keyframe slots to the base pass, then uses the
half-resolution result as an in-context reference for a detailing IC-LoRA at
full resolution. The initial MLX path uses the convolutional video VAE and
supports T2V with optional generated audio. Accept the terms for both gated
repositories before the first run:

```bash
uv run mlx_video.generate --model-repo Lightricks/LTX-2.5 \
    --pipeline dfr \
    --prompt "A red fox walking through sunlit grass, detailed fur, cinematic" \
    --num-frames 121 --width 768 --height 512
```

The official detailing adapter is downloaded from
`Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler` by default. Override it
with `--detailing-lora PATH_OR_REPO`; its default merge strength is `0.5` and
can be changed with `--detailing-lora-strength`.

DFR may pad its internal canvas to a 24- or 32-frame segment boundary and trims
the decoded result back to the requested frame count. Image conditioning,
A2V, temporal upscaling, streaming, and diffusion-VAE decoding are not yet
supported by this initial path.

### Diffusion video VAE

Use the LTX-2.5 diffusion decoder instead of the default convolutional decoder:

```bash
uv run mlx_video.generate --model-repo Lightricks/LTX-2.5 \
    --prompt "A red fox walking through sunlit grass, detailed fur" \
    --num-frames 121 --width 768 --height 512 \
    --video-decoder diffusion
```

The decoder loads `vae/ltx-2.5-video-vae-bf16.safetensors` and uses a fused
inference-only Metal kernel for 3D neighborhood attention. The kernel performs
online softmax without materializing the neighborhood score tensor and has a
SIMD-group specialization for the model's head dimension of 64.

DFR-generated keyframes are passed through all five decoder stages when DFR and
the diffusion decoder are selected together. The joint Metal attention lets
each video query attend to its two nearest keyframe planes and lets each
keyframe query attend to its two nearest video frames.

Stage 4 and stage 5 can be spatially tiled with a receptive-field halo. For
example, `--diffusion-vae-spatial-tiles 2` processes a 2x2 tile grid to reduce
peak memory at the cost of repeated halo computation. The same spatial tiling
works for DFR keyframe-aware decoding by cropping every keyframe plane to the
video tile's halo. `--diffusion-vae-temporal-tiles 2` similarly splits the
stage-4/5 time axis with its causal receptive-field halo and supports both
plain and keyframe-aware decoding. Spatial and temporal tiling cannot yet be
combined in one decode. The convolutional decoder remains the default
lower-memory path.

### Image-to-Video (I2V)

```bash
# Distilled I2V
uv run mlx_video.generate --prompt "A person dancing" --image photo.jpg

# Dev I2V
uv run mlx_video.generate --pipeline dev --prompt "Waves crashing" --image beach.png --cfg-scale 3.5
```

### Audio-to-Video (A2V)

Generate video conditioned on an input audio file. Works with all four pipelines. The audio is encoded to latent space and frozen during denoising -- the transformer's cross-attention reads the audio signal to guide video generation.

```bash
# A2V - distilled (default, fastest)
uv run mlx_video.generate --audio-file music.wav --prompt "A band playing music"

# A2V - dev (single-stage with CFG)
uv run mlx_video.generate --pipeline dev --audio-file ocean.wav --prompt "Ocean waves"

# A2V - dev-two-stage (dev + LoRA refinement)
uv run mlx_video.generate --pipeline dev-two-stage --audio-file music.wav \
    --prompt "A band playing music" --model-repo prince-canuma/LTX-2-dev

# A2V - dev-two-stage-hq (highest quality)
uv run mlx_video.generate --pipeline dev-two-stage-hq --audio-file music.wav \
    --prompt "A band playing music" --model-repo prince-canuma/LTX-2-dev

# A2V + I2V (audio + image conditioning)
uv run mlx_video.generate --audio-file rain.wav --image forest.jpg --prompt "Rain in forest"

# A2V with custom start time
uv run mlx_video.generate --audio-file song.mp3 --audio-start-time 30.0 --prompt "Concert"
```

> **Note:** `--audio-file` (A2V) and `--audio` (generate audio) are mutually exclusive. Supported formats: WAV, FLAC, MP3, OGG, and video files with audio tracks.

### Audio-Video Generation (experimental)

Generate synchronized audio alongside video from scratch:

```bash
uv run mlx_video.generate --prompt "Ocean waves crashing" --audio
uv run mlx_video.generate --pipeline dev --prompt "A jazz band playing" --audio --enhance-prompt

# With full guidance (STG + modality_scale, matches PyTorch defaults)
uv run mlx_video.generate --pipeline dev --prompt "Ocean waves crashing" --audio \
    --stg-scale 1.0 --stg-blocks 29 --modality-scale 3.0
```

### LoRA

LoRA weights can be loaded from a file, directory, or HuggingFace repo:

```bash
# From HuggingFace repo
uv run mlx_video.generate --pipeline dev-two-stage \
    --prompt "Camera dolly out of a forest" \
    --lora-path Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Out \
    --lora-strength 1.0

# From local file
uv run mlx_video.generate --pipeline dev-two-stage \
    --prompt "A scene" \
    --lora-path ./my-lora/weights.safetensors

# From local directory (auto-detects .safetensors file)
uv run mlx_video.generate --pipeline dev-two-stage \
    --prompt "A scene" \
    --lora-path ./LTX-2-distilled/lora
```

### Upscaling

```bash
# Upscale an image 2x
uv run mlx_video.upscale --input photo.png --output upscaled.png

# Upscale a video 2x
uv run mlx_video.upscale --input video.mp4 --output upscaled.mp4

# Upscale with refinement (higher quality, requires text prompt)
uv run mlx_video.upscale --input video.mp4 --output upscaled.mp4 --refine --prompt "A cinematic scene"
```

## CLI Options

### General

| Option | Default | Description |
|--------|---------|-------------|
| `--prompt`, `-p` | (required) | Text description of the video |
| `--pipeline` | `distilled` | Pipeline type: `distilled`, `dfr`, `dev`, `dev-two-stage`, or `dev-two-stage-hq` |
| `--height`, `-H` | 512 | Output height (divisible by 64 for two-stage, 32 for dev) |
| `--width`, `-W` | 512 | Output width (divisible by 64 for two-stage, 32 for dev) |
| `--num-frames`, `-n` | 33 for LTX-2/2.3; predicted for LTX-2.5 | Number of frames (must be 1 + 8*k) |
| `--auto-duration` | `1 20` | Min/max seconds for LTX-2.5 duration prediction when `--num-frames` is omitted |
| `--prompt-enhancer-repo` | `mlx-community/gemma-4-e2b-it-bf16` | Separate generative Gemma 4 checkpoint used by `--enhance-prompt` with LTX-2.5 |
| `--seed`, `-s` | 42 | Random seed for reproducibility |
| `--fps` | 24 | Frames per second |
| `--output-path`, `-o` | output.mp4 | Output video path |
| `--model-repo` | Lightricks/LTX-2 | HuggingFace model repository |
| `--text-encoder-repo` | None | Separate text encoder repo (if not in model repo) |
| `--save-frames` | false | Save individual frames as images |
| `--enhance-prompt` | false | Enhance prompt using Gemma |
| `--image`, `-i` | None | Conditioning image for I2V |
| `--image-strength` | 1.0 | Conditioning strength for I2V |
| `--audio`, `-a` | false | Enable synchronized audio generation |
| `--audio-file` | None | Path to audio file for A2V conditioning |
| `--audio-start-time` | 0.0 | Start time in seconds for audio file |
| `--tiling` | `auto` | VAE tiling mode: `auto`, `none`, `aggressive`, `conservative` |
| `--stream` | false | Stream frames as they decode |
| `--spatial-upscaler` | auto (x2) | Spatial upscaler file for two-stage pipelines (see below) |
| `--detailing-lora` | official LTX-2.5 adapter | DFR detailing IC-LoRA path or repository |
| `--detailing-lora-strength` | `0.5` | DFR detailing IC-LoRA merge strength |
| `--video-decoder` | `conv` | LTX-2.5 decoder: `conv` or experimental `diffusion` |
| `--diffusion-vae-spatial-tiles` | `1` | DiffVAE stage-4/5 tiles per spatial axis |
| `--diffusion-vae-temporal-tiles` | `1` | Number of DiffVAE stage-4/5 temporal tiles |

### Spatial Upscalers (LTX-2.3)

LTX-2.3 ships with multiple spatial upscaler variants. Use `--spatial-upscaler` to select one:

| Variant | Scale | Output (from 256x256) | Architecture |
|---------|-------|-----------------------|--------------|
| `ltx-2.3-spatial-upscaler-x2-1.0.safetensors` (default) | 2.0x | 512x512 | Conv2d + PixelShuffle(2) |
| `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | 2.0x | 512x512 | Same arch, newer weights |
| `ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors` | 1.5x | 384x384 | Conv2d + PixelShuffle(3) + BlurDownsample |

```bash
# Default (x2-1.0, auto-detected)
uv run mlx_video.generate --prompt "A sunset" --model-repo ./LTX-2.3-distilled

# x2-1.1 (newer weights)
uv run mlx_video.generate --prompt "A sunset" --model-repo ./LTX-2.3-distilled \
    --spatial-upscaler ltx-2.3-spatial-upscaler-x2-1.1.safetensors

# x1.5 (smaller output, faster)
uv run mlx_video.generate --prompt "A sunset" --model-repo ./LTX-2.3-distilled \
    --spatial-upscaler ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors
```

> **Note:** Stage 1 always runs at half the target resolution. With x1.5, the final output is 75% of `--width`/`--height` (e.g., 512 target -> 256 stage 1 -> 384 output). With x2, the output matches the target exactly.

### Dev / Dev-Two-Stage

| Option | Default | Description |
|--------|---------|-------------|
| `--steps` | 30 | Number of denoising steps |
| `--cfg-scale` | 3.0 | CFG guidance scale |
| `--cfg-rescale` | 0.7 | CFG rescale factor (reduces over-saturation) |
| `--negative-prompt` | (default) | Negative prompt for CFG |
| `--apg` | false | Use Adaptive Projected Guidance (more stable for I2V) |
| `--stg-scale` | 0.0 | STG scale (PyTorch default: 1.0, requires `--audio`) |
| `--stg-blocks` | None | Transformer blocks for STG ([29] for LTX-2, [28] for LTX-2.3) |
| `--modality-scale` | 1.0 | Cross-modal guidance scale (PyTorch default: 3.0, requires `--audio`) |

### Dev-Two-Stage LoRA

| Option | Default | Description |
|--------|---------|-------------|
| `--lora-path` | auto-detect | Path to LoRA file, directory, or HuggingFace repo |
| `--lora-strength` | 1.0 | LoRA merge strength |

### Dev-Two-Stage HQ

| Option | Default | Description |
|--------|---------|-------------|
| `--lora-strength-stage-1` | 0.25 | LoRA strength for stage 1 |
| `--lora-strength-stage-2` | 0.5 | LoRA strength for stage 2 |

HQ defaults: 15 steps (vs 30), `cfg-rescale` 0.45 (vs 0.7), STG disabled. Uses the res_2s second-order sampler (2 model evals per step) for better quality at the same compute budget.

## How It Works

### Distilled Pipeline (default)
1. **Stage 1**: Generate at half resolution with 8 denoising steps (fixed sigmas)
2. **Upsample**: Spatial upsampling via LatentUpsampler (x2 or x1.5, selectable via `--spatial-upscaler`)
3. **Stage 2**: Refine at upsampled resolution with 3 denoising steps
4. **Decode**: VAE decoder converts latents to RGB video

### Dev Pipeline
1. **Generate**: Full resolution with configurable steps and constant CFG
2. **Decode**: VAE decoder converts latents to RGB video

### Dev Two-Stage Pipeline
1. **Stage 1**: Dev denoising at half resolution with CFG
2. **Upsample**: Spatial upsampling via LatentUpsampler (x2 or x1.5)
3. **Stage 2**: Distilled refinement at upsampled resolution with LoRA weights (3 steps, no CFG)
4. **Decode**: VAE decoder converts latents to RGB video

### Dev Two-Stage HQ Pipeline
1. **Stage 1**: res_2s denoising at half resolution with CFG + LoRA@0.25 (15 steps, 2 evals/step)
2. **Upsample**: Spatial upsampling via LatentUpsampler (x2 or x1.5)
3. **Stage 2**: res_2s refinement at upsampled resolution with LoRA@0.5 (3 steps, no CFG)
4. **Decode**: VAE decoder converts latents to RGB video

The res_2s sampler uses an exponential Rosenbrock-type Runge-Kutta integrator with SDE noise injection, producing higher quality results than Euler at the same compute budget (~30 total model evaluations).

### Audio-to-Video (A2V) Conditioning

A2V works by encoding input audio into the same latent space as generated audio, then **freezing** those latents during denoising:

1. Load audio file, resample to 16kHz, compute mel-spectrogram
2. `AudioEncoder(mel_spec)` produces audio latents `(B, 8, T, 16)`
3. Normalize via `PerChannelStatistics`
4. Freeze during denoising: `timesteps=0`, `sigma=0`, skip Euler/RK updates
5. Transformer's A2V cross-attention reads frozen audio to guide video generation
6. Output: denoised video + original input audio waveform (skip audio VAE decode)

## Converting Models

Convert original Lightricks/LTX-2 weights to the modular mlx-video format:

```bash
# Convert distilled model
uv run python -m mlx_video.models.ltx_2.convert \
    --source Lightricks/LTX-2 --output ./LTX-2-distilled --variant distilled

# Convert dev model
uv run python -m mlx_video.models.ltx_2.convert \
    --source Lightricks/LTX-2 --output ./LTX-2-dev --variant dev
```

This extracts 7 components from the monolithic checkpoint:

```
LTX-2-distilled/
├── transformer/          # DiT transformer (19B params)
├── vae/
│   ├── decoder/          # Video VAE decoder
│   └── encoder/          # Video VAE encoder
├── audio_vae/
│   ├── decoder/          # Audio VAE decoder
│   └── encoder/          # Audio VAE encoder
├── vocoder/              # Mel-spectrogram to waveform
└── text_projections/     # Text embedding projections
```

Pre-converted weights are available on HuggingFace:
- [prince-canuma/LTX-2-distilled](https://huggingface.co/prince-canuma/LTX-2-distilled)
- [prince-canuma/LTX-2-dev](https://huggingface.co/prince-canuma/LTX-2-dev)
- [prince-canuma/LTX-2.3-distilled](https://huggingface.co/prince-canuma/LTX-2.3-distilled)
- [prince-canuma/LTX-2.3-dev](https://huggingface.co/prince-canuma/LTX-2.3-dev)

## Model Specifications

- **Transformer**: 48 layers, 32 attention heads, 128 dim per head (19B parameters)
- **Latent channels**: 128
- **Patch size**: 4 (for VAE patchify/unpatchify)
- **Text encoder**: Gemma 3 with 3840-dim output
- **RoPE**: Split mode with double precision (LTX-2.3) or standard (LTX-2)
- **Audio VAE**: Encoder (~35M), Decoder (~50M), Vocoder (~13M)

### Audio VAE Architecture

```
Audio Encoder: mel-spectrogram -> latents (B, 8, T, 16)
  - Channel multipliers: (1, 2, 4)
  - ResNet blocks with optional attention
  - GroupNorm or PixelNorm normalization
  - Optional causal convolutions

Audio Decoder: latents -> mel-spectrogram
  - Mirrors encoder with upsampling path
  - Per-channel statistics for latent normalization

Vocoder: mel-spectrogram -> waveform (~13M params)
  - HiFi-GAN style architecture
  - Upsample rates: [6, 5, 2, 2, 2]
  - ResBlock1 with dilations [1, 3, 5]
```

## Project Structure

```
mlx_video/models/ltx_2/
├── __init__.py
├── config.py             # LTXModelConfig, AudioEncoderModelConfig, AudioDecoderModelConfig
├── convert.py            # Weight conversion from Lightricks/LTX-2
├── generate.py           # Unified generation pipeline (T2V, I2V, A2V, +Audio)
├── postprocess.py        # Video post-processing
├── samplers.py           # Euler and res_2s samplers
├── utils.py              # Shared utilities (get_model_path, load_safetensors, etc.)
├── ltx.py                # Main LTXModel (DiT transformer with AV support)
├── transformer.py        # Transformer blocks, Modality dataclass
├── attention.py          # Multi-head attention with RoPE
├── feed_forward.py       # Feed-forward layers
├── adaln.py              # Adaptive Layer Normalization
├── rope.py               # Rotary Position Embeddings (split/combined)
├── text_projection.py    # Text embedding projection
├── text_encoder.py       # Text encoder with AV embeddings support
├── upsampler.py          # LatentUpsampler for 2-stage generation
├── conditioning/
│   ├── keyframe.py       # Image-to-video keyframe conditioning
│   └── latent.py         # Video-to-video latent conditioning
├── video_vae/
│   ├── decoder.py        # VAE decoder with timestep conditioning
│   ├── encoder.py        # VAE encoder for image/video encoding
│   ├── convolution.py    # CausalConv3d, CausalConv2d
│   ├── ops.py            # patchify, unpatchify, PerChannelStatistics
│   ├── resnet.py         # ResBlock3D, ResBlockGroup
│   ├── sampling.py       # DepthToSpaceUpsample, SpaceToDepthDownsample
│   └── video_vae.py      # Full VAE (encoder + decoder)
└── audio_vae/
    ├── audio_vae.py      # Audio encoder and decoder
    ├── audio_processor.py # Mel-spectrogram computation (librosa)
    ├── vocoder.py        # Mel-spectrogram to waveform synthesis
    ├── ops.py            # AudioPatchifier, PerChannelStatistics
    ├── resnet.py         # ResNet blocks for audio
    ├── attention.py      # Attention blocks for audio VAE
    ├── normalization.py  # Normalization layers
    ├── causal_conv_2d.py # Causal 2D convolutions
    ├── downsample.py     # Downsampling layers
    └── upsample.py       # Upsampling layers
```

## LTX-2 vs LTX-2.3

LTX-2.3 introduces prompt-conditioned adaptive layer normalization (adaln):

| Feature | LTX-2 | LTX-2.3 |
|---------|--------|---------|
| AdaLN | Standard | Prompt-conditioned (`has_prompt_adaln=True`) |
| Attention gate | None | `2.0 * sigmoid(gate_logits)` |
| Scale-shift table | 6 params | 9 params (+ cross-attn Q) |
| Text encoder connectors | 2 blocks | 8 blocks with gate_logits |
| Feature extractor | V1 (batch-level) | V2 (per-token RMSNorm) |
| RoPE | Standard | Double precision |
| STG blocks | [29] | [28] |
| Text encoder repo | Included | Separate (`--text-encoder-repo`) |
