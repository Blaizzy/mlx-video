# Wan2.2 Model Architecture & Implementation - Comprehensive Exploration

## Executive Summary
The mlx-video project contains a complete implementation of Wan2.2 (Text-to-Video & Image-to-Video generation) diffusion models on Apple Silicon using MLX. The architecture includes dual-model configurations, advanced caching mechanisms (TeaCache and Spectrum), and supports both quantization and LoRA fine-tuning.

---

## 1. HOW WAN2.2 MODEL WORKS

### 1.1 High-Level Architecture
Wan2.2 is a **flow-matching diffusion model** with:
- **Dual-model configuration**: Separate high-noise and low-noise transformer models for different stages of denoising
- **Single-model fallback**: Wan2.1 uses single transformer for compatibility
- **Condition types**: Text embeddings (from UMT5-XXL T5 encoder) + time embeddings + optional image conditioning (I2V)

### 1.2 Input Flow (Text-to-Video Pipeline)
```
User Prompt → T5 Encoder → Text Embeddings [B, text_len, 4096]
                                      ↓
                              Embed Layer (linear projection)
                                      ↓
                         Transformer-friendly [B, text_len, dim]
                         
Noise [C, T, H, W] → VAE Encode → Latents [z_dim, T_latent, H_latent, W_latent]
                                      ↓
                                  Patchify
                                      ↓
                           Patch Embeddings [1, L, dim]
                           
Time Step (continuous 0-1000) → Sinusoidal Embedding [B, freq_dim]
                                      ↓
                         Time embedding MLP → [B, dim]
                                      ↓
                         Time projection (6x scale) → [B, 1, 6, dim]
```

### 1.3 Core Configuration
**File**: `/Users/daniel/Projects/mlx-video/mlx_video/models/wan/config.py`

Key Wan2.2 T2V 14B configuration:
- **Model Dimension**: 5120
- **Number of Transformer Blocks**: 40
- **Attention Heads**: 40 (head_dim = 128)
- **FFN Dimension**: 13,824
- **Patch Size**: (1, 2, 2) - temporal, height, width
- **Input Channels**: 16 (VAE latent depth)
- **Output Channels**: 16
- **Text Length**: 512 tokens
- **Text Embedding Dimension**: 4096 (from T5)
- **Frequency Dimension**: 256 (for time embedding sinusoids)
- **Number of Training Timesteps**: 1000

**Configuration Variants**:
```python
WanModelConfig.wan22_t2v_14b()       # Text-to-Video (default, dual-model)
WanModelConfig.wan22_i2v_14b()       # Image-to-Video (dual-model, in_dim=36)
WanModelConfig.wan22_ti2v_5b()       # Text+Image-to-Video (single-model, dim=3072)
WanModelConfig.wan21_t2v_14b()       # Wan2.1 14B (single-model)
WanModelConfig.wan21_t2v_1_3b()      # Wan2.1 1.3B (single-model, dim=1536)
```

### 1.4 Dual Model Architecture (Wan2.2 Specific)
**File**: `/Users/daniel/Projects/mlx-video/mlx_video/generate_wan.py` lines 407-419

Wan2.2 uses **adaptive model switching** based on noise levels:
- **High-Noise Model** (early denoising steps): Handles coarse structure generation from pure noise
  - Used when: `timestep_val >= boundary` (default: 87.5% of training timesteps)
  - More sensitive to global structure
  
- **Low-Noise Model** (late denoising steps): Handles fine details and refinement
  - Used when: `timestep_val < boundary`
  - More sensitive to local details

**Boundary Calculation**: `boundary = config.boundary * config.num_train_timesteps = 0.875 * 1000 = 875`

**Guidance Scales**: Can differ per model
- Example: `guide_scale=(3.0, 4.0)` means (low_noise_scale, high_noise_scale)

---

## 2. DENOISING LOOP STRUCTURE

### 2.1 Main Denoising Loop
**File**: `/Users/daniel/Projects/mlx-video/mlx_video/generate_wan.py` lines 589-693

```python
for i, t in enumerate(tqdm(range(steps), desc="Diffusion")):
    timestep_val = timestep_list[i]
    
    # Select model based on noise level
    if is_dual:
        if timestep_val >= boundary:
            model = high_noise_model
            kv = cross_kv_high
            rcs = rope_cos_sin_high
        else:
            model = low_noise_model
            kv = cross_kv_low
            rcs = rope_cos_sin_low
    
    # CFG or direct inference
    if cfg_disabled:
        # B=1 forward (faster)
        noise_pred = forward_pass([latents], timestep_val, context_cond)
    else:
        # B=2 forward: conditional + unconditional
        noise_pred = forward_pass(
            [latents, latents],
            [timestep_val, timestep_val],
            [context_cond, context_uncond]
        )
        # Classifier-free guidance
        noise_pred = noise_uncond + guide_scale * (noise_cond - noise_uncond)
    
    # Scheduler step (Euler, DPM++, or UniPC)
    latents = scheduler.step(noise_pred, timestep_val, latents)
    
    # (I2V only) Freeze first frame
    if is_i2v_mask_blend:
        latents = (1 - mask) * z_img + mask * latents
```

### 2.2 Timestep Handling
- **Timesteps are contiguous values** (not discrete indices) from 0-1000
- **Reverse scheduling**: Start at high noise (≈1000), end at low noise (≈0)
- **Scheduler converts to sigma** (noise level): Used for denoising step size
- **Step index tracking**: Pre-converted to Python list to avoid GPU sync per step
  ```python
  timestep_list = sched.timesteps.tolist()  # Pre-convert to avoid .item() sync
  ```

### 2.3 Batch Processing Optimization (CFG)
**No-CFG Mode** (`guide_scale ≤ 1.0`):
- Batch size = 1 (only conditional)
- **2x faster** than CFG mode
- Skips unconditional forward pass

**CFG Mode** (standard):
- Batch size = 2 (conditional + unconditional in same forward)
- Single forward pass processes both, then blends

### 2.4 Pre-computation to Avoid Per-Step Overhead
**Text Embeddings**:
```python
context_emb = model.embed_text([context, context_null])  # Once, reused all steps
```

**Cross-Attention K/V Cache**:
```python
cross_kv = model.prepare_cross_kv(context_emb)  # Once, indices passed to each block
```

**RoPE Frequencies**:
```python
rope_cos_sin = model.prepare_rope(grid_sizes)  # Once (grid sizes constant)
```

---

## 3. TRANSFORMER FORWARD PASS FOR WAN2.2

### 3.1 Main Forward Pass
**File**: `/Users/daniel/Projects/mlx-video/mlx_video/models/wan/model.py` lines 295-518

```python
def __call__(
    self,
    x_list: list,              # Video latents [C, F, H, W]
    t: mx.array,               # Timesteps [B] or [B, L]
    context: list | mx.array,  # Text embeddings [B, text_len, dim]
    seq_len: int,              # Max sequence length
    cross_kv_caches: list | None = None,  # Pre-computed K/V
    y: list | None = None,     # I2V conditioning
    rope_cos_sin: tuple | None = None,    # Pre-computed RoPE
) → list:
```

### 3.2 Input Processing

**Step 1: Detect CFG Batching**
- Check if all batch elements are identical (common for CFG batch of 2)
- Broadcast optimization if `all_same`

**Step 2: Channel Concatenation (I2V)**
```python
if y is not None:  # Image-to-Video conditioning
    x_list = [mx.concatenate([u, v], axis=0) for u, v in zip(x_list, y)]
    # Now x has extra channels from conditioning image
```

**Step 3: Patchification**
```python
# Input: [C, F, H, W]
# Output: [1, L, dim] where L = (F/patch_t) * (H/patch_h) * (W/patch_w)
patches, grid_sizes = model._patchify(x_list[0])
```

### 3.3 Transformer Block Structure
**File**: `/Users/daniel/Projects/mlx-video/mlx_video/models/wan/transformer.py` lines 7-96

Each `WanAttentionBlock` contains:

```python
class WanAttentionBlock(nn.Module):
    def __init__(self, dim, ffn_dim, num_heads, ...):
        # Self-attention path
        self.norm1 = WanLayerNorm(dim)
        self.self_attn = WanSelfAttention(dim, num_heads)
        
        # Cross-attention path
        self.norm3 = WanLayerNorm(dim) if cross_attn_norm else None
        self.cross_attn = WanCrossAttention(dim, num_heads)
        
        # FFN path
        self.norm2 = WanLayerNorm(dim)
        self.ffn = WanFFN(dim, ffn_dim)
        
        # Learned modulation (6 vectors × dim)
        self.modulation = mx.random.normal((1, 6, dim)) * (dim**-0.5)

    def __call__(self, x, e, context, cross_kv_cache=None, ...):
        # Modulation (compute in float32 for precision)
        mod = (self.modulation + e)  # Add time embedding
        e0, e1, e2, e3, e4, e5 = mod.split()  # 6 modulation vectors
        
        # Self-attention with modulation
        x_norm = self.norm1(x) * (1 + e1) + e0  # Affine transform
        x = x + e2 * self.self_attn(x_norm)
        
        # Cross-attention (conditioned on text)
        x_norm = self.norm3(x) if self.norm3 else x
        x = x + self.cross_attn(x_norm, context, kv_cache=cross_kv_cache)
        
        # FFN with modulation
        x_norm = self.norm2(x) * (1 + e4) + e3
        x = x + e5 * self.ffn(x_norm)
        
        return x
```

### 3.4 Block Processing Loop
**File**: `/Users/daniel/Projects/mlx-video/mlx_video/models/wan/model.py` lines 508-511

Standard path (no caching):
```python
for i, block in enumerate(self.blocks):
    kv = cross_kv_caches[i] if cross_kv_caches is not None else None
    x = block(x, cross_kv_cache=kv, **kwargs)
```

With **TeaCache** enabled (lines 450-507):
- Skip blocks if time embedding change is small
- Reuse residual from previous step

With **Spectrum** enabled (lines 430-449):
- Predict features with Chebyshev polynomials
- Full compute only at strategic intervals

### 3.5 Learned Modulation (DiT-style)
The 6 modulation vectors control:
1. **e0**: Self-attention shift (residual connection bias)
2. **e1**: Self-attention scale (learnable gain)
3. **e2**: Self-attention gate (residual weight)
4. **e3**: FFN shift
5. **e4**: FFN scale
6. **e5**: FFN gate

This is a **Diffusion Transformer (DiT)** style architecture where modulation is **per-block, not per-layer-pair**.

### 3.6 Self-Attention Details
**File**: `/Users/daniel/Projects/mlx-video/mlx_video/models/wan/attention.py` lines 46-121

```python
class WanSelfAttention(nn.Module):
    def __call__(self, x, seq_lens, grid_sizes, freqs, rope_cos_sin=None, attn_mask=None):
        b, s, _ = x.shape
        
        # Project to Q, K, V
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        
        # QK normalization (optional)
        if self.norm_q:
            q = self.norm_q(q)  # RMS norm
        if self.norm_k:
            k = self.norm_k(k)
        
        # Reshape: [B, L, dim] → [B, L, num_heads, head_dim]
        q, k, v = q.reshape(...), k.reshape(...), v.reshape(...)
        
        # Apply 3-way factorized RoPE (temporal, height, width)
        q = rope_apply(q.astype(mx.float32), grid_sizes, freqs, rope_cos_sin)
        k = rope_apply(k.astype(mx.float32), grid_sizes, freqs, rope_cos_sin)
        
        # Memory-efficient scaled-dot-product attention
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=attn_mask)
        
        # Output projection
        return self.o(out)
```

**Key Features**:
- QK normalization prevents attention collapse
- RoPE applied in float32 then cast back
- Attention mask handles variable sequence lengths

### 3.7 Cross-Attention Details
**File**: `/Users/daniel/Projects/mlx-video/mlx_video/models/wan/attention.py` lines 124-207

```python
class WanCrossAttention(nn.Module):
    def prepare_kv(self, context):
        """Pre-compute K, V once before denoising loop"""
        k = self.k(context)  # [B, L_ctx, dim]
        v = self.v(context)
        k = k.reshape(B, -1, num_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, -1, num_heads, head_dim).transpose(0, 2, 1, 3)
        return k, v
    
    def __call__(self, x, context, context_lens=None, kv_cache=None):
        # Q from hidden state
        q = self.q(x).reshape(...)
        
        # Use pre-computed K/V if available
        if kv_cache is not None:
            k, v = kv_cache
        else:
            k = self.k(context).reshape(...)
            v = self.v(context).reshape(...)
        
        # Attention with optional context masking
        out = mx.fast.scaled_dot_product_attention(q, k, v)
        return self.o(out)
```

---

## 4. EXISTING CACHING & ACCELERATION MECHANISMS

### 4.1 TeaCache (Timestep Embedding Aware Cache)
**File**: `/Users/daniel/Projects/mlx-video/mlx_video/models/wan/model.py` lines 15-59

**Purpose**: Skip redundant transformer blocks when time embedding change is small

**How it Works**:
1. Monitor relative L1 distance between consecutive projected time embeddings (e0)
2. Use polynomial regression to map distance → output distance
3. Accumulate rescaled distances
4. Skip blocks if accumulated distance < threshold
5. Reuse cached residual instead

**Configuration**:
```python
teacache_thresh: float = 0.0  # Threshold (0=disabled, 0.1≈2x speedup, 0.2≈3x)
teacache_verbose: bool = False  # Debug per-step decisions
```

**State Tracking**:
```python
@dataclass
class TeaCacheState:
    enabled: bool = False
    threshold: float = 0.0
    coefficients: tuple = ()  # Polynomial coefficients (ret-mode)
    verbose: bool = False
    
    # Per-batch state
    previous_e0: object = None  # Cached time embedding
    accumulated_distance: float = 0.0
    previous_residual: object = None  # Cached residual to reuse
    
    # Bookkeeping
    cnt: int = 0
    num_steps: int = 0
    ret_steps: int = 2  # Always compute first N steps
    cutoff_steps: int = 0  # Always compute last N steps
    
    # Stats
    steps_skipped: int = 0
    steps_computed: int = 0
```

**Polynomial Coefficients** (pre-profiled for different models):
- Wan2.2 T2V 14B: `(-3.03e5, 4.91e4, -2.66e3, 5.87e1, -0.316)`
- Wan2.2 I2V 14B: `(2.57e5, -3.54e4, 1.40e3, -13.6, 0.133)`
- Wan2.1 T2V 1.3B: `(-5.22e4, 9.23e3, -5.28e2, 13.7, -0.050)`

**Speedup Results**:
- threshold=0.1 → ~2x speedup
- threshold=0.2 → ~3x speedup

### 4.2 Spectrum (Adaptive Spectral Feature Forecasting)
**File**: `/Users/daniel/Projects/mlx-video/mlx_video/models/wan/spectrum.py` (entire file)

**Purpose**: Predict transformer features using Chebyshev polynomials; skip expensive forward passes

**Algorithm**:
1. **Warmup Phase** (first 5 steps): Always compute full transformer
2. **Caching Phase**: Store (timestep, features) pairs
3. **Fitting Phase**: Fit Chebyshev polynomials to cached features via ridge regression
4. **Prediction Phase**: Use fitted polynomials + Taylor extrapolation to predict next features
5. **Adaptive Scheduling**: Grow compute interval as confidence increases

**Chebyshev Forecaster**:
```python
class ChebyshevForecaster:
    def __init__(self, M=4, K=100, lam=0.1, num_steps=50):
        self.M = M  # Polynomial degree (4)
        self.K = K  # Cache window size (100)
        self.lam = lam  # Ridge regularization (0.1)
        self.t_buf = None  # (<=K,) step indices
        self.H_buf = None  # (<=K, F) flattened features
    
    def update(self, step, h_flat):
        """Append new feature to cache"""
        # Maintains sliding window of K most recent features
    
    def predict(self, step):
        """Predict features using fitted Chebyshev polynomials"""
        # Fit: (P, P) ridge system solved via Cholesky (tiny, CPU-fast)
        # Predict: Evaluate Chebyshev basis at target step
```

**SpectrumForecaster** (blends approaches):
```python
class SpectrumForecaster:
    def __init__(self, M=4, K=100, lam=0.1, w=0.5, num_steps=50):
        self.cheb = ChebyshevForecaster(...)
        self.w = w  # Blend weight: 0=Taylor, 1=Chebyshev (default 0.5)
    
    def predict(self, step):
        h_cheb = self.cheb.predict(step)
        h_taylor = taylor_forward_differences(step)
        return (1 - w) * h_taylor + w * h_chebyshev
```

**Configuration**:
```python
spectrum: bool = False  # Enable/disable
spectrum_w: float = 0.5  # Blend weight (0=Taylor-only, 1=Chebyshev-only)
spectrum_flex_window: float = 0.75  # Window growth rate (default ~3.5x speedup)
spectrum_warmup: int = 5  # Always compute first N steps
```

**State Machine**:
```python
@dataclass
class SpectrumState:
    enabled: bool = False
    m: int = 4  # Chebyshev degree
    lam: float = 0.1  # Ridge regularization
    w: float = 0.5  # Chebyshev/Taylor blend
    k_max: int = 100  # Cache size
    warmup_steps: int = 5
    window_size: int = 2  # Initial compute interval
    flex_window: float = 0.75  # Growth rate (α in paper)
    
    # Runtime state
    num_steps: int = 0
    cnt: int = 0
    curr_ws: float = 2.0  # Current window (grows during run)
    num_consecutive_cached: int = 0
    
    def should_compute(self) -> bool:
        """Adaptive scheduling: compute on warmup or window boundaries"""
        if cnt < warmup_steps:
            return True
        should = (num_consecutive_cached + 1) % floor(curr_ws) == 0
        return should
    
    def step(self, computed: bool):
        """Update counters after each step"""
        if computed:
            curr_ws += flex_window  # Grow window
        else:
            num_consecutive_cached += 1
        cnt += 1
```

**Speedup Results** (from paper):
- flex_window=0.75 → ~3.5x speedup
- flex_window=3.0 → ~5x speedup

### 4.3 Compiled Forward Pass
**File**: `/Users/daniel/Projects/mlx-video/mlx_video/generate_wan.py` lines 578-584

When TeaCache and Spectrum are disabled, compile the full model for speed:
```python
if not use_caching:
    models_to_compile = [high_noise_model, low_noise_model] if is_dual else [single_model]
    for m in models_to_compile:
        m._compiled = mx.compile(m)

# In loop:
_call = getattr(model, "_compiled", model)
preds = _call([latents], t=t_batch, ...)  # JIT-compiled forward
```

### 4.4 Memory Optimizations

**Cross-attention K/V Caching**:
- Project context embeddings once before denoising loop
- Reuse same K/V across all 40 diffusion steps
- **Eliminates 40 redundant linear projections** per generation

**RoPE Precomputation**:
- Grid sizes are constant across steps
- Precompute all cos/sin values once
- **Eliminates per-step broadcast/concat overhead**

**Text Embedding Caching**:
- Embed text once with T5 encoder
- Reuse across all denoising steps
- **Free T5 from memory immediately** after embedding

---

## 5. MoE (MIXTURE OF EXPERTS) ARCHITECTURE

### 5.1 Finding: NO MoE in Wan2.2

**Search Results**:
- No `MoE`, `moe`, `mixture`, or `expert` references in any Wan model files
- FFN is standard **gated feed-forward** (no MoE)

**Feed-Forward Network** (file: `/Users/daniel/Projects/mlx-video/mlx_video/models/wan/transformer.py` lines 84-96):
```python
class WanFFN(nn.Module):
    """Gated feed-forward network with GELU(tanh) activation."""
    
    def __init__(self, dim: int, ffn_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, ffn_dim)  # 5120 → 13824
        self.act = nn.GELU(approx="tanh")
        self.fc2 = nn.Linear(ffn_dim, dim)  # 13824 → 5120
    
    def __call__(self, x: mx.array) -> mx.array:
        # Standard gated FFN: dense → GELU → dense
        x_w = x.astype(_linear_dtype(self.fc1))
        return self.fc2(self.act(self.fc1(x_w)))
```

**Architecture**: Vanilla transformer FFN, not sparse/routed.

---

## 6. MODEL LOADING AND CONFIGURATION

### 6.1 Model Loading Pipeline
**File**: `/Users/daniel/Projects/mlx-video/mlx_video/models/wan/loading.py`

```python
def load_wan_model(model_path, config, quantization=None, loras=None):
    """Load WanModel with optional quantization and LoRA."""
    
    # 1. Create model with config
    model = WanModel(config)
    
    # 2. Apply quantization stubs if specified
    if quantization:
        nn.quantize(
            model,
            group_size=quantization["group_size"],
            bits=quantization["bits"],
            class_predicate=lambda path, m: _quantize_predicate(path, m)
        )
    
    # 3. Load weights
    weights = mx.load(str(model_path))
    
    # 4. Apply LoRA (dequantize + merge if quantized, weight merge if bf16)
    if loras:
        # ... handle quantized vs non-quantized paths ...
    
    # 5. Load and eval
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())
    return model
```

### 6.2 Configuration Loading
**File**: `/Users/daniel/Projects/mlx-video/mlx_video/generate_wan.py` lines 93-173

Auto-detection logic:
```python
if (model_dir / "config.json").exists():
    # Use explicit config
    config = WanModelConfig(**config_dict)
else:
    # Auto-detect from weights
    if (model_dir / "low_noise_model.safetensors").exists():
        config = WanModelConfig.wan22_t2v_14b()  # Dual model
    else:
        # Read weight shapes to determine size
        probe = mx.load(str(model_path))
        for k, v in probe.items():
            if "patch_embedding_proj.weight" in k:
                dim = v.shape[0]
                config = (
                    WanModelConfig.wan21_t2v_1_3b() if dim <= 2048
                    else WanModelConfig.wan21_t2v_14b()
                )
```

### 6.3 Weight Conversion (PyTorch → MLX)
**File**: `/Users/daniel/Projects/mlx-video/mlx_video/convert_wan.py` lines 60-161

Key transformations:
```python
def sanitize_wan_transformer_weights(weights):
    """Convert Wan PyTorch keys to MLX structure."""
    
    # Conv3d → Linear reshape
    if key == "patch_embedding.weight":
        # [dim, in_dim, 1, 2, 2] → [dim, in_dim*1*2*2]
        value = value.reshape(value.shape[0], -1)
        new_key = "patch_embedding_proj.weight"
    
    # Sequential layers → individual modules
    new_key = key.replace("text_embedding.0.", "text_embedding_0.")  # Seq[0]
    new_key = key.replace("text_embedding.2.", "text_embedding_1.")  # Seq[2]
    new_key = key.replace("time_embedding.0.", "time_embedding_0.")
    new_key = key.replace("time_embedding.2.", "time_embedding_1.")
    new_key = key.replace("time_projection.1.", "time_projection.")
    
    # FFN renaming
    new_key = key.replace(".ffn.0.", ".ffn.fc1.")
    new_key = key.replace(".ffn.2.", ".ffn.fc2.")
    
    return {new_key: value, ...}
```

### 6.4 Quantization Strategy
**File**: `/Users/daniel/Projects/mlx-video/mlx_video/convert_wan.py` (full file)

```python
def _quantize_predicate(path, module):
    """Determine which layers to quantize."""
    # Typically: Linear layers in transformer blocks (high memory)
    # Skip: Layer norms, embeddings, output head
    return isinstance(module, nn.Linear) and "blocks" in path
```

Quantization options:
- 4-bit with group_size=64 (default)
- 8-bit with group_size=128
- Mixed: Quantize transformer blocks, keep other layers in bf16

---

## 7. GENERATE SCRIPTS FOR WAN2.2

### 7.1 Main Generation Script
**File**: `/Users/daniel/Projects/mlx-video/mlx_video/generate_wan.py` (38,528 bytes)

Main entry point:
```python
def generate_video(
    model_dir: str,
    prompt: str,
    negative_prompt: str | None = None,
    image: str | None = None,
    width: int = 1280,
    height: int = 720,
    num_frames: int = 81,
    steps: int = None,
    guide_scale: str | float | tuple = None,
    shift: float = None,
    seed: int = -1,
    output_path: str = "output.mp4",
    scheduler: str = "unipc",
    teacache_thresh: float = 0.0,
    teacache_verbose: bool = False,
    spectrum: bool = False,
    spectrum_w: float = 0.5,
    spectrum_flex_window: float = 0.75,
    spectrum_warmup: int = 5,
    loras: list | None = None,
    loras_high: list | None = None,
    loras_low: list | None = None,
    tiling: str = "auto",
):
```

### 7.2 Generation Pipeline

**Phase 1: Setup** (lines 91-306)
- Load and parse config (auto-detect or explicit)
- Validate dimensions (align to patch_size × vae_stride)
- Load T5 encoder and tokenizer
- Encode prompts (text → embeddings)
- Free T5 from memory

**Phase 2: I2V Preparation** (lines 309-392)
- Load VAE encoder
- Encode reference image to latent
- Build masking for blend (TI2V-5B) or channel concat (I2V-14B)

**Phase 3: Model Loading** (lines 394-420)
- Load high-noise and low-noise transformers (or single for Wan2.1)
- Apply LoRA if specified
- Load and apply quantization if specified

**Phase 4: Precomputation** (lines 422-484)
- Embed text once (text_embedding MLP)
- Prepare cross-attention K/V caches (40 blocks × 2 models)
- Precompute RoPE cos/sin for grid sizes
- Setup schedulers (Euler, DPM++, or UniPC)

**Phase 5: Denoising Loop** (lines 508-693)
- For each timestep:
  - Select model based on noise level (dual model)
  - Forward pass (with or without CFG)
  - Apply classifier-free guidance
  - Scheduler step
  - (Optional) Reapply I2V mask
  - Evaluate and free memory

**Phase 6: VAE Decoding** (lines 743-800+)
- Load VAE decoder
- Optional tiling for memory efficiency
- Decode latents to video pixels
- Save MP4

### 7.3 Supported Schedulers
**File**: `/Users/daniel/Projects/mlx-video/mlx_video/models/wan/scheduler.py`

1. **Euler** (1st-order): `--scheduler euler`
   - Simplest: `x_next = x + (σ_next - σ_cur) * v`
   
2. **DPM++2M** (2nd-order): `--scheduler dpm++`
   - Convergence faster than Euler
   - Uses previous step for 2nd-order correction
   
3. **UniPC** (higher-order): `--scheduler unipc` (default)
   - Faster convergence
   - Recommended for quality

### 7.4 Command-Line Usage

```bash
# Basic T2V (Wan2.2 dual-model auto-detected)
python -m mlx_video.generate_wan \
  --model-dir ./wan22_converted \
  --prompt "A cat running through a forest" \
  --output output.mp4

# I2V with reference image
python -m mlx_video.generate_wan \
  --model-dir ./wan22_i2v_converted \
  --prompt "The cat leaps over a fence" \
  --image reference.jpg \
  --output output_i2v.mp4

# With TeaCache acceleration (2x speedup)
python -m mlx_video.generate_wan \
  --model-dir ./wan22_converted \
  --prompt "..." \
  --teacache-thresh 0.1 \
  --output output_fast.mp4

# With Spectrum acceleration (3.5x speedup, better quality)
python -m mlx_video.generate_wan \
  --model-dir ./wan22_converted \
  --prompt "..." \
  --spectrum \
  --spectrum-flex-window 0.75 \
  --output output_spectrum.mp4

# With quantization (memory efficient)
python -m mlx_video.generate_wan \
  --model-dir ./wan22_quantized_4bit \
  --prompt "..." \
  --output output_quant.mp4
```

---

## 8. KEY FILE STRUCTURE & SIZES

```
mlx_video/models/wan/
├── __init__.py                    (2 lines)
├── model.py                       (518 lines)  ← Main WanModel class
├── config.py                      (157 lines)  ← Configuration variants
├── transformer.py                 (96 lines)   ← Transformer blocks & FFN
├── attention.py                   (207 lines)  ← Self/Cross-attention
├── rope.py                        (178 lines)  ← 3-way factorized RoPE
├── loading.py                     (183 lines)  ← Model loading & T5 encoder
├── text_encoder.py                (240 lines)  ← T5 text encoder
├── scheduler.py                   (428 lines)  ← Flow matching schedulers
├── spectrum.py                    (288 lines)  ← Spectrum acceleration
├── i2v_utils.py                   (58 lines)   ← Image-to-Video utilities
├── vae.py                         (589 lines)  ← Wan2.1 VAE
└── vae22.py                       (908 lines)  ← Wan2.2 VAE (48-dim latent)

mlx_video/
├── generate_wan.py                (38 KB)      ← Main generation script
├── convert_wan.py                 (27 KB)      ← Weight conversion & quantization
├── train_wan.py                   (9 KB)       ← Training script
└── utils.py                       (9 KB)       ← Utilities
```

---

## 9. SUMMARY TABLE: WAN2.2 ARCHITECTURE

| Component | Details | File |
|-----------|---------|------|
| **Dual Model** | High-noise (0-875 steps) + Low-noise (875-1000) | model.py:406-419 |
| **Transformer Blocks** | 40 blocks, DiT-style learned modulation | transformer.py:7-96 |
| **Self-Attention** | QK norm + 3-way factorized RoPE + windowing | attention.py:46-121 |
| **Cross-Attention** | Pre-computed K/V for 40 steps reuse | attention.py:124-207 |
| **Time Embedding** | Sinusoidal positional encoding + MLP projection | model.py:138-144 |
| **Text Embedding** | T5 UMT5-XXL (24 layers, 64 heads) → MLP | text_encoder.py + loading.py |
| **Patch Size** | (1, 2, 2) for temporal stability | config.py:13 |
| **VAE Latent Depth** | Wan2.2: 48-dim (stride 4×16×16), Wan2.1: 16-dim (stride 4×8×8) | config.py:44-45 |
| **TeaCache** | Polynomial-based block skipping (~2-3x speedup) | model.py:15-59 |
| **Spectrum** | Chebyshev + Taylor prediction (~3.5x speedup) | spectrum.py (full) |
| **Schedulers** | Euler, DPM++, UniPC (flow matching) | scheduler.py (full) |
| **Quantization** | 4-bit or 8-bit on transformer blocks | convert_wan.py:305+ |
| **LoRA Support** | Runtime or weight-merge application | lora/ + convert_wan.py |

---

## 10. DIAGRAM: DATA FLOW

```
┌─ TEXT PROMPT ──────────────────────────────────────────────────────────┐
│                                                                        │
│  1. Tokenize & T5 Encode                                              │
│     "A cat running"                                                    │
│     ↓                                                                   │
│     [tokens] → T5 Encoder (24 layers) → [512, 4096]                   │
│     ↓                                                                   │
│     T5 kept in float32 for precision (only called once)               │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
                              ↓
                    MODEL LOADS
                    ↓
        ┌─────────────────────────────────┐
        │  High-Noise Model  Low-Noise    │
        │  (40 blocks)       Model        │
        │  dim=5120          (40 blocks)  │
        │                    dim=5120     │
        └─────────────────────────────────┘
                      ↓
        ┌─────────────────────────────────┐
        │  Precompute Once (per gen):     │
        │  - Text embed: [512, 5120]      │
        │  - Cross-KV cache: 40 × 2       │
        │  - RoPE cos/sin: [seq_len, d/2] │
        └─────────────────────────────────┘

                ┌─ NOISE ───────────────────────────────────┐
                │                                           │
                │  Random [48, T_lat, H_lat, W_lat]         │
                │  ↓                                         │
                │  Patchify: [L, 5120]                      │
                │  ↓                                         │
                │  Time Embed: [B, 5120]                    │
                │  ↓                                         │
                │  CFG Batch: [B=2]                         │
                │              (cond + uncond)              │
                └───────────────────────────────────────────┘
                              ↓
        ┌──────────────────────────────────────────────────────────┐
        │  DENOISING LOOP (40 steps)                               │
        │                                                          │
        │  for t in timesteps:                                     │
        │    if t >= 875:  use HIGH_NOISE_MODEL                    │
        │    else:         use LOW_NOISE_MODEL                     │
        │                                                          │
        │    ┌────────────────────────────────────────────────┐   │
        │    │  Forward Pass: 40 Transformer Blocks           │   │
        │    │                                                 │   │
        │    │  for block in blocks:                           │   │
        │    │    ├─ Self-Attn (with 3-way RoPE)             │   │
        │    │    ├─ Cross-Attn (text conditioning)           │   │
        │    │    └─ FFN (gated)                              │   │
        │    │    All with learned modulation (6 vectors)     │   │
        │    │                                                 │   │
        │    │  Optional: TeaCache block skipping             │   │
        │    │  Optional: Spectrum feature prediction         │   │
        │    │                                                 │   │
        │    └────────────────────────────────────────────────┘   │
        │                    ↓                                     │
        │    Classifier-Free Guidance:                            │
        │    noise_pred = uncond + 4.0 * (cond - uncond)         │
        │                    ↓                                     │
        │    Scheduler Step (Euler/DPM++/UniPC)                   │
        │    latents = f(noise_pred, timestep, latents)          │
        │                                                          │
        └──────────────────────────────────────────────────────────┘
                              ↓
        ┌──────────────────────────────────────────────────────────┐
        │  VAE DECODE                                              │
        │                                                          │
        │  Latents [48, T_lat, H_lat, W_lat]                       │
        │  ↓                                                        │
        │  Wan2.2 VAE Decoder (CausalConv3d)                       │
        │  Denormalize + Upscale                                   │
        │  ↓                                                        │
        │  Video Frames [3, T_pix, H_pix, W_pix]                   │
        │  (Channels-last: [T, H, W, 3])                           │
        │                                                          │
        └──────────────────────────────────────────────────────────┘
                              ↓
                    ┌─────────────────────┐
                    │  Save MP4 Video     │
                    │  output.mp4         │
                    └─────────────────────┘
```

---

## CONCLUSION

Wan2.2 is a sophisticated dual-model video diffusion architecture implemented efficiently on Apple Silicon using MLX. Key innovations:

1. **Dual-model design**: Separate high-noise and low-noise transformers for coarse-to-fine generation
2. **Learned modulation**: DiT-style timestep conditioning with 6 per-block modulation vectors
3. **Aggressive precomputation**: Text embeddings, cross-attention K/V, RoPE frequencies computed once
4. **Advanced caching**: TeaCache (polynomial skipping) + Spectrum (Chebyshev prediction) for 2-3.5x speedup
5. **3-way factorized RoPE**: Separate frequency components for temporal, height, and width dimensions
6. **Flexible I2V**: Both channel-concatenation (I2V-14B) and masking-based (TI2V-5B) approaches

