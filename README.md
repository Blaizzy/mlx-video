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

## Supported Models

### LTX-2
[LTX-2](https://huggingface.co/Lightricks/LTX-Video) is a 19B parameter video generation model from Lightricks.

## Features

- Text-to-video generation with the LTX-2 19B DiT model
- Image-to-video (I2V) conditioning
- Synchronized audio-video generation
- Two-stage generation pipeline (distilled model)
- Single-stage generation with CFG (dev model)
- Prompt enhancement using Gemma
- Memory-efficient tiled decoding
- Streaming frame output
- Optimized for Apple Silicon using MLX

---

## Quick Start

### Text-to-Video Generation (Distilled Model)

```bash
python -m mlx_video.generate \
    --prompt "Two dogs of the poodle breed wearing sunglasses, close up, cinematic, sunset" \
    --num-frames 100 \
    --width 768
```

<img src="https://github.com/Blaizzy/mlx-video/raw/main/examples/poodles.gif" width="512" alt="Poodles demo">

---

## CLI Reference

### 1. Distilled Model - Two-Stage Generation

**Command**: `python -m mlx_video.generate`

Uses a two-stage pipeline: generate at half resolution, upsample, then refine.

```bash
python -m mlx_video.generate \
    --prompt "Ocean waves crashing on a beach at sunset" \
    --height 768 \
    --width 768 \
    --num-frames 65 \
    --seed 123 \
    --output my_video.mp4
```

#### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--prompt`, `-p` | (required) | Text description of the video |
| `--height`, `-H` | 512 | Output height (divisible by 64) |
| `--width`, `-W` | 512 | Output width (divisible by 64) |
| `--num-frames`, `-n` | 100 | Number of frames (1 + 8*k format) |
| `--seed`, `-s` | 42 | Random seed for reproducibility |
| `--fps` | 24 | Frames per second |
| `--output-path`, `-o` | output.mp4 | Output video path |
| `--save-frames` | false | Save individual frames as images |
| `--model-repo` | Lightricks/LTX-2 | HuggingFace model repository |
| `--text-encoder-repo` | - | Custom text encoder repository |
| `--verbose` | false | Enable verbose output |
| `--enhance-prompt` | false | Enhance prompt using Gemma |
| `--max-tokens` | 512 | Max tokens for prompt enhancement |
| `--temperature` | 0.7 | Temperature for prompt enhancement |
| `--image`, `-i` | - | Path to image for I2V conditioning |
| `--image-strength` | 1.0 | I2V conditioning strength (0.0-1.0) |
| `--image-frame-idx` | 0 | Frame index to condition on |
| `--tiling` | auto | Tiling mode (see below) |
| `--stream` | false | Stream frames as they're decoded |
| `--audio` | false | Generate synchronized audio |

---

### 2. Dev Model - Single-Stage with CFG

**Command**: `python -m mlx_video.generate_dev`

Uses classifier-free guidance (CFG) for higher quality generation.

```bash
python -m mlx_video.generate_dev \
    --prompt "A cinematic shot of a mountain landscape at golden hour" \
    --negative-prompt "blurry, low quality" \
    --height 512 \
    --width 768 \
    --num-frames 33 \
    --steps 40 \
    --cfg-scale 4.0
```

#### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--prompt`, `-p` | (required) | Text description of the video |
| `--negative-prompt` | (comprehensive default) | CFG negative prompt |
| `--height`, `-H` | 512 | Video height (divisible by 64) |
| `--width`, `-W` | 768 | Video width (divisible by 64) |
| `--num-frames`, `-n` | 33 | Number of frames |
| `--steps` | 40 | Number of inference steps |
| `--cfg-scale` | 4.0 | CFG guidance scale |
| `--seed`, `-s` | 42 | Random seed |
| `--fps` | 24 | Frames per second |
| `--output-path` | output_dev.mp4 | Output video path |
| `--output-audio` | - | Output audio path (if `--audio` enabled) |
| `--save-frames` | false | Save individual frames |
| `--audio` | false | Generate synchronized audio |
| `--verbose` | false | Enable verbose output |
| `--enhance-prompt` | false | Enhance prompt using Gemma |
| `--image`, `-i` | - | Path to image for I2V |
| `--image-strength` | 1.0 | I2V conditioning strength |
| `--image-frame-idx` | 0 | Frame index to condition |
| `--tiling` | none | Tiling mode |

---

### 3. Audio-Video Generation (Distilled)

**Command**: `python -m mlx_video.generate_av`

Generates synchronized video and audio.

```bash
python -m mlx_video.generate_av \
    --prompt "Ocean waves crashing on rocks, seagulls calling" \
    --height 512 \
    --width 512 \
    --num-frames 65 \
    --output-path output_av.mp4 \
    --output-audio output.wav
```

#### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--prompt`, `-p` | (required) | Text description |
| `--height`, `-H` | 512 | Video height |
| `--width`, `-W` | 512 | Video width |
| `--num-frames`, `-n` | 65 | Number of frames |
| `--seed`, `-s` | 42 | Random seed |
| `--fps` | 24 | Frames per second |
| `--output-path` | output_av.mp4 | Output video path |
| `--output-audio` | - | Output audio path (.wav) |
| `--verbose` | false | Enable verbose output |
| `--enhance-prompt` | false | Enhance prompt |
| `--image`, `-i` | - | Path to image for I2V |
| `--image-strength` | 1.0 | I2V conditioning strength |
| `--image-frame-idx` | 0 | Frame index to condition |
| `--tiling` | auto | Tiling mode |

---

## Tiling Modes

Tiling enables memory-efficient decoding for large videos:

| Mode | Description |
|------|-------------|
| `auto` | Automatically determines based on video size |
| `none` | Disable tiling (requires more memory) |
| `default` | 512px spatial, 64 frame temporal |
| `aggressive` | 256px spatial, 32 frame temporal (lowest memory) |
| `conservative` | 768px spatial, 96 frame temporal (faster) |
| `spatial` | Spatial tiling only |
| `temporal` | Temporal tiling only |

---

## Image-to-Video (I2V) Conditioning

Condition video generation on an input image:

```bash
# First frame conditioning
python -m mlx_video.generate \
    --prompt "A cat walking across a sunny garden" \
    --image cat.jpg \
    --image-strength 1.0 \
    --image-frame-idx 0

# Middle frame conditioning
python -m mlx_video.generate_dev \
    --prompt "A person turning around" \
    --image person.jpg \
    --image-frame-idx 16 \
    --num-frames 33
```

---

## Python API

### Basic Video Generation

```python
from mlx_video.generate import generate_video

# Generate a video
generate_video(
    model_repo="Lightricks/LTX-2",
    prompt="A beautiful sunset over the ocean",
    height=512,
    width=768,
    num_frames=65,
    seed=42,
    fps=24,
    output_path="output.mp4",
)
```

### Dev Model with CFG

```python
from mlx_video.generate_dev import generate_video_dev

generate_video_dev(
    model_repo="Lightricks/LTX-2",
    prompt="Cinematic shot of a forest",
    negative_prompt="blurry, low quality",
    height=512,
    width=768,
    num_frames=33,
    num_inference_steps=40,
    cfg_scale=4.0,
    output_path="output_dev.mp4",
)
```

### Audio-Video Generation

```python
from mlx_video.generate_av import generate_video_with_audio

generate_video_with_audio(
    model_repo="Lightricks/LTX-2",
    prompt="Thunder and lightning in a storm",
    height=512,
    width=512,
    num_frames=65,
    output_path="output_av.mp4",
    output_audio_path="output.wav",
)
```

### Image-to-Video Conditioning

```python
from mlx_video.generate import generate_video

# Condition on first frame
generate_video(
    model_repo="Lightricks/LTX-2",
    prompt="A cat walking",
    image="cat.jpg",
    image_strength=1.0,
    image_frame_idx=0,
    output_path="output.mp4",
)
```

### Model Loading

```python
from mlx_video.convert import (
    get_model_path,
    load_transformer_weights,
    load_vae_weights,
)
from mlx_video.models.ltx import LTXModel, LTXModelConfig

# Get model path (downloads if needed)
model_path = get_model_path("Lightricks/LTX-2")

# Load transformer
config = LTXModelConfig()
model = LTXModel(config)
weights = load_transformer_weights(model_path)
model.load_weights(list(weights.items()))

# Load VAE weights
vae_weights = load_vae_weights(model_path)
```

### VAE Encoder/Decoder

```python
from mlx_video.models.ltx.video_vae.encoder import load_vae_encoder
from mlx_video.models.ltx.video_vae.decoder import load_vae_decoder
from mlx_video.models.ltx.video_vae.tiling import TilingConfig

# Load encoder
encoder = load_vae_encoder(model_path)
latents = encoder(image)  # (B, 3, H, W) -> (B, 128, H/32, W/32)

# Load decoder
decoder = load_vae_decoder(model_path)

# Standard decode
video = decoder.decode(latents)

# Memory-efficient tiled decode
video = decoder.decode_tiled(
    latents,
    tiling_config=TilingConfig.auto(height, width, num_frames),
    tiling_mode="auto",
)

# Streaming decode (callback per batch of frames)
def on_frames_ready(frames):
    # Process frames as they're decoded
    pass

decoder.decode_tiled(
    latents,
    tiling_config=TilingConfig.auto(height, width, num_frames),
    on_frames_ready=on_frames_ready,
)
```

### Audio VAE

```python
from mlx_video.generate_av import load_audio_decoder, load_vocoder, save_audio

# Load audio components
audio_decoder = load_audio_decoder(model_path)
vocoder = load_vocoder(model_path)

# Decode audio latents to mel-spectrogram
mel_spectrogram = audio_decoder(audio_latents)

# Convert mel to waveform
audio_waveform = vocoder(mel_spectrogram)

# Save audio
save_audio(audio_waveform, "output.wav", sample_rate=24000)
```

### Text Encoder with Prompt Enhancement

```python
from mlx_video.models.ltx.text_encoder import LTX2TextEncoder

# Load text encoder
text_encoder = LTX2TextEncoder.load(model_path, text_encoder_path)

# Get embeddings
video_embeddings, audio_embeddings = text_encoder(
    prompt="A cat walking",
    return_audio_embeddings=True,
)

# Enhance prompt with Gemma
enhanced_prompt = text_encoder.enhance_t2v(
    prompt="cat walking",
    max_tokens=512,
    temperature=0.7,
    verbose=True,
)
```

### Configuration

```python
from mlx_video.models.ltx.config import (
    LTXModelConfig,
    LTXModelType,
    LTXRopeType,
)

# Create config for different model types
config = LTXModelConfig(
    model_type=LTXModelType.AudioVideo,  # or VideoOnly, AudioOnly
    num_attention_heads=32,
    attention_head_dim=128,
    num_layers=48,
    rope_type=LTXRopeType.SPLIT,
    double_precision_rope=True,
)

# Access derived properties
print(config.inner_dim)        # Video inner dimension
print(config.audio_inner_dim)  # Audio inner dimension
```

---

## How It Works

### Two-Stage Pipeline (Distilled Model)

1. **Stage 1**: Generate at half resolution (e.g., 384x384) with 8 denoising steps
2. **Upsample**: 2x spatial upsampling via LatentUpsampler
3. **Stage 2**: Refine at full resolution (e.g., 768x768) with 3 denoising steps
4. **Decode**: VAE decoder converts latents to RGB video

### Single-Stage Pipeline (Dev Model)

1. **Denoise**: Full resolution with CFG guidance over N steps
2. **Decode**: VAE decoder converts latents to RGB video

### Audio Generation

Audio is generated in parallel with video using:
- Shared transformer backbone with modality-specific attention
- Audio VAE for latent encoding/decoding
- HiFi-GAN vocoder for mel-to-waveform conversion

---

## Audio Configuration

| Constant | Value | Description |
|----------|-------|-------------|
| `AUDIO_SAMPLE_RATE` | 24000 | Output audio sample rate |
| `AUDIO_LATENT_SAMPLE_RATE` | 16000 | VAE internal sample rate |
| `AUDIO_HOP_LENGTH` | 160 | Mel-spectrogram hop length |
| `AUDIO_LATENT_CHANNELS` | 8 | Audio latent channels |
| `AUDIO_MEL_BINS` | 16 | Mel frequency bins |

---

## Requirements

- macOS with Apple Silicon
- Python >= 3.11
- MLX >= 0.22.0

## Model Specifications

- **Transformer**: 48 layers, 32 attention heads, 128 dim per head
- **Latent channels**: 128 (video), 8 (audio)
- **Text encoder**: Gemma 3 with 3840-dim output (video), 2048-dim (audio)
- **RoPE**: Split mode with double precision

## Project Structure

```
mlx_video/
├── __init__.py
├── generate.py             # Two-stage video generation (distilled)
├── generate_av.py          # Audio-video generation (distilled)
├── generate_dev.py         # Single-stage generation with CFG (dev)
├── convert.py              # Weight conversion (PyTorch -> MLX)
├── postprocess.py          # Video post-processing utilities
├── utils.py                # Helper functions
├── components/
│   └── patchifiers.py      # Video/Audio patchifiers
├── conditioning/
│   ├── keyframe.py         # Keyframe conditioning
│   └── latent.py           # Latent state & I2V conditioning
└── models/
    └── ltx/
        ├── ltx.py          # Main LTXModel (DiT transformer)
        ├── config.py       # Model configuration
        ├── transformer.py  # Transformer blocks
        ├── attention.py    # Multi-head attention with RoPE
        ├── text_encoder.py # Text encoder with AV support
        ├── upsampler.py    # 2x spatial upsampler
        ├── video_vae/      # Video VAE encoder/decoder
        └── audio_vae/      # Audio VAE encoder/decoder/vocoder
```

## License

MIT
