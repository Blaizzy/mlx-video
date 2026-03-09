# Wan2.2 Model - Quick Reference Guide

## 1. CORE ARCHITECTURE AT A GLANCE

```
INPUT
├── Text: prompt → T5 Encoder → [512, 4096] → MLP → [512, 5120]
├── Time: scalar t ∈ [0,1000] → sinusoid + MLP → [5120]
└── Video: noise → Patchify (patch_size=1,2,2) → [L, 5120]

DUAL MODELS (Wan2.2 only)
├── High-Noise Model: timestep >= 875 (early denoising)
└── Low-Noise Model: timestep < 875 (fine details)

EACH MODEL (40 blocks × 40 layers total)
├── 40× WanAttentionBlock:
│   ├── Self-Attention (with 3-way RoPE + QK norm)
│   ├── Cross-Attention (text conditioning, pre-cached K/V)
│   ├── FFN (gated GELU)
│   └── ALL modulated by 6 learnable time-conditioned vectors
└── Output Head: Project to patch_size × out_dim

DIFFUSION LOOP (40 steps, e.g.)
├── For each timestep:
│   ├── Forward pass (with/without CFG batch)
│   ├── Optional: TeaCache block skipping (2x speedup)
│   ├── Optional: Spectrum feature prediction (3.5x speedup)
│   ├── Classifier-free guidance: noise = uncond + 4.0 × (cond - uncond)
│   └── Scheduler step (Euler/DPM++/UniPC)
└── Decode with VAE → video pixels
```

## 2. KEY FILES TO UNDERSTAND

| File | Lines | Purpose |
|------|-------|---------|
| `model.py` | 518 | Main WanModel, patchify, unpatchify, forward pass, TeaCache/Spectrum logic |
| `transformer.py` | 96 | WanAttentionBlock (self-attn + cross-attn + FFN + modulation) |
| `attention.py` | 207 | WanSelfAttention, WanCrossAttention (with pre-cached K/V) |
| `rope.py` | 178 | 3-way factorized RoPE (temporal, height, width separate) |
| `config.py` | 157 | Configuration presets (Wan2.2 T2V/I2V/TI2V, Wan2.1) |
| `generate_wan.py` | 38KB | Main generation pipeline (setup → precompute → denoise → decode) |
| `spectrum.py` | 288 | Chebyshev + Taylor feature prediction for acceleration |
| `scheduler.py` | 428 | Flow matching: Euler, DPM++, UniPC solvers |
| `loading.py` | 183 | Model loading, weight conversion, T5 encoder |
| `convert_wan.py` | 27KB | PyTorch→MLX conversion, quantization, LoRA merging |

## 3. DENOISING LOOP SKELETON

```python
# From generate_wan.py:589-693
for i, t in enumerate(tqdm(range(steps), desc="Diffusion")):
    timestep_val = timestep_list[i]
    
    # 1. SELECT MODEL (dual model only)
    if is_dual:
        if timestep_val >= boundary:
            model = high_noise_model
            kv = cross_kv_high
        else:
            model = low_noise_model
            kv = cross_kv_low
    
    # 2. FORWARD PASS
    if cfg_disabled:
        # B=1 (faster)
        preds = model([latents], t=t_batch, context=context_cond, ...)
        noise_pred = preds[0]
    else:
        # B=2 (cfg)
        preds = model(
            [latents, latents],
            t=[t, t],
            context=[context_cond, context_uncond]
        )
        noise_cond, noise_uncond = preds[0], preds[1]
        noise_pred = noise_uncond + guide_scale * (noise_cond - noise_uncond)
    
    # 3. SCHEDULER STEP
    latents = scheduler.step(noise_pred, timestep_val, latents)
    
    # 4. (I2V) REAPPLY MASK
    if is_i2v_mask_blend:
        latents = (1 - mask) * z_img + mask * latents
    
    # 5. EVAL & CLEANUP
    mx.eval(latents)
    del noise_pred
```

## 4. TRANSFORMER BLOCK FORWARD PASS

```python
# From transformer.py
class WanAttentionBlock:
    def __call__(self, x, e, context, cross_kv_cache=None, ...):
        # Modulation: 6 time-conditioned vectors
        mod = (self.modulation + e)  # Add time embedding
        e0, e1, e2, e3, e4, e5 = mod.split()
        
        # Self-Attention with modulation
        x_norm = self.norm1(x) * (1 + e1) + e0  # Affine transform
        x = x + e2 * self.self_attn(x_norm, ...)  # Residual + gate
        
        # Cross-Attention (text conditioning)
        x_norm = self.norm3(x) if self.norm3 else x
        x = x + self.cross_attn(x_norm, context, kv_cache=cross_kv_cache)
        
        # FFN with modulation
        x_norm = self.norm2(x) * (1 + e4) + e3
        x = x + e5 * self.ffn(x_norm)
        
        return x
```

## 5. 3-WAY FACTORIZED ROPE

```python
# From rope.py: Temporal, height, width have separate frequency components
# This prevents temporal motion from being confused with spatial position

d_t = head_dim - 2*(head_dim//3)  # ~1/3 of dimensions
d_h = head_dim // 3                # ~1/3 of dimensions  
d_w = head_dim // 3                # ~1/3 of dimensions

# For each position (f, h, w):
freqs_combined = [
    freqs_t[f],     # Temporal component
    freqs_h[h],     # Height component
    freqs_w[w],     # Width component
]
# Apply rotation separately, then concatenate
```

## 6. TEACACHE ACCELERATION

```python
# Skips transformer blocks when time embedding change is small

if t_embedding_distance < THRESHOLD:
    # Reuse cached residual from previous step
    x = x + cached_residual  # Instant!
else:
    # Full forward pass
    x = run_all_blocks(x)
    cached_residual = x - x_input

# Speedup: ~2x at threshold=0.1, ~3x at threshold=0.2
# Polynomial coefficients pre-profiled per model variant
```

## 7. SPECTRUM ACCELERATION

```python
# Predicts transformer features using Chebyshev polynomials

# Warmup: 5 steps always compute full forward
# Then: Fit Chebyshev polynomials to cached features
# Finally: Predict features on non-compute steps

# Blended prediction:
h_predicted = (1 - w) * h_taylor + w * h_chebyshev

# Window grows adaptively:
if num_steps_without_compute % window_size == 0:
    compute_full_forward()
    window_size *= flex_window  # Grow confidence

# Speedup: ~3.5x with flex_window=0.75, up to ~5x with 3.0
```

## 8. PRECOMPUTATION (Per Generation)

```python
# ONCE before denoising loop:

# 1. Text embeddings (MLP applied to T5 output)
context_emb = model.embed_text([context, context_null])
# Result: [2, 512, 5120] (cond + uncond)

# 2. Cross-attention K/V caches (40 blocks)
cross_kv_caches = [
    block.cross_attn.prepare_kv(context_emb)  # for each block
]
# Result: list of (k, v) [2, 40, 512, 128] per block

# 3. RoPE frequencies (grid sizes constant)
rope_cos_sin = model.prepare_rope(grid_sizes)
# Result: (cos, sin) [seq_len, 1, head_dim//2]

# These are REUSED across all 40 denoising steps!
```

## 9. CONFIGURATION QUICK LOOKUP

```python
# Wan2.2 T2V 14B (default dual-model)
config = WanModelConfig.wan22_t2v_14b()
# dim=5120, heads=40, layers=40, patch_size=(1,2,2)
# dual_model=True, boundary=0.875
# vae_z_dim=48, vae_stride=(4,16,16)

# Wan2.2 I2V 14B (image-to-video, dual-model)
config = WanModelConfig.wan22_i2v_14b()
# Same as above but: model_type="i2v", in_dim=36
# boundary=0.900, sample_guide_scale=(3.5, 3.5)

# Wan2.2 TI2V 5B (text+image, single-model)
config = WanModelConfig.wan22_ti2v_5b()
# dim=3072, heads=24, layers=30
# dual_model=False, model_type="ti2v"
# in_dim=48, out_dim=48, vae_z_dim=48

# Wan2.1 T2V 14B (single-model, backward compatible)
config = WanModelConfig.wan21_t2v_14b()
# dim=5120, heads=40, layers=40
# dual_model=False, boundary=0.0

# Wan2.1 T2V 1.3B (smaller single-model)
config = WanModelConfig.wan21_t2v_1_3b()
# dim=1536, heads=12, layers=30
# dual_model=False, boundary=0.0
```

## 10. GENERATION COMMAND CHEAT SHEET

```bash
# Basic text-to-video (Wan2.2 auto-detected)
python -m mlx_video.generate_wan \
  --model-dir /path/to/wan22 \
  --prompt "A cat running through a forest" \
  --output output.mp4

# Image-to-video
python -m mlx_video.generate_wan \
  --model-dir /path/to/wan22_i2v \
  --prompt "The cat jumps over a fence" \
  --image reference.jpg \
  --output output_i2v.mp4

# Fast with TeaCache (2x speedup, lower quality)
python -m mlx_video.generate_wan \
  --model-dir /path/to/wan22 \
  --prompt "..." \
  --teacache-thresh 0.1 \
  --output fast.mp4

# High quality with Spectrum (3.5x speedup, better than TeaCache)
python -m mlx_video.generate_wan \
  --model-dir /path/to/wan22 \
  --prompt "..." \
  --spectrum \
  --spectrum-flex-window 0.75 \
  --spectrum-warmup 5 \
  --output spectrum.mp4

# With quantization (memory efficient)
python -m mlx_video.generate_wan \
  --model-dir /path/to/wan22_4bit \
  --prompt "..." \
  --output quant.mp4

# Custom guidance & solver
python -m mlx_video.generate_wan \
  --model-dir /path/to/wan22 \
  --prompt "..." \
  --guide-scale 5.0 \
  --scheduler dpm++ \
  --steps 50 \
  --output custom.mp4

# With LoRA fine-tuning
python -m mlx_video.generate_wan \
  --model-dir /path/to/wan22 \
  --prompt "..." \
  --loras "./lora_style.safetensors,0.5" "./lora_motion.safetensors,0.3" \
  --output lora.mp4
```

## 11. WEIGHT CONVERSION

```bash
# Convert Wan2.2 checkpoint to MLX format
python -c "
from mlx_video.convert_wan import convert_wan_checkpoint

convert_wan_checkpoint(
    checkpoint_dir='/path/to/wan22_official',
    output_dir='/path/to/wan22_mlx',
    dtype='bfloat16',
    model_version='auto',
    quantize=True,
    bits=4,
    group_size=64
)
"

# Structure expected:
# wan22_official/
#   ├── low_noise_model/     (safetensors)
#   ├── high_noise_model/    (safetensors)
#   ├── models_t5_umt5-xxl-enc-bf16.pth
#   └── Wan2.1_VAE.pth

# Output:
# wan22_mlx/
#   ├── low_noise_model.safetensors
#   ├── high_noise_model.safetensors
#   ├── t5_encoder.safetensors
#   ├── vae.safetensors
#   └── config.json
```

## 12. NO MOE ARCHITECTURE

⚠️ **Important**: Wan2.2 does **NOT** use Mixture of Experts. The FFN is standard:

```python
class WanFFN(nn.Module):
    def __init__(self, dim: int, ffn_dim: int):
        self.fc1 = nn.Linear(dim, ffn_dim)        # 5120 → 13824
        self.act = nn.GELU(approx="tanh")
        self.fc2 = nn.Linear(ffn_dim, dim)        # 13824 → 5120
    
    def __call__(self, x):
        return self.fc2(self.act(self.fc1(x)))    # Dense → GELU → Dense
```

No routing, no gating, no expert selection.

---

## REFERENCE LINKS

- Main generation: `/Users/daniel/Projects/mlx-video/mlx_video/generate_wan.py`
- Model forward: `/Users/daniel/Projects/mlx-video/mlx_video/models/wan/model.py`
- Transformer blocks: `/Users/daniel/Projects/mlx-video/mlx_video/models/wan/transformer.py`
- Acceleration: `/Users/daniel/Projects/mlx-video/mlx_video/models/wan/spectrum.py`
- Config presets: `/Users/daniel/Projects/mlx-video/mlx_video/models/wan/config.py`
- Full documentation: `/Users/daniel/Projects/mlx-video/WAN22_EXPLORATION_SUMMARY.md`
