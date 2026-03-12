# Porting Diffusion Video Models to MLX: Lessons Learned

A practical guide distilled from porting Wan2.1/2.2 (1.3B–14B) and Helios 14B DiT
video generation models from PyTorch to MLX on Apple Silicon. These lessons apply
broadly to any diffusion-based video (or image) model port.

---

## Table of Contents

1. [Debugging Methodology](#1-debugging-methodology)
2. [Precision & Dtype Pitfalls](#2-precision--dtype-pitfalls)
3. [MLX-Specific Gotchas](#3-mlx-specific-gotchas)
4. [Autoregressive Chunk Boundaries](#4-autoregressive-chunk-boundaries)
5. [VAE Decoder Artifacts](#5-vae-decoder-artifacts)
6. [Scheduler & Timestep Issues](#6-scheduler--timestep-issues)
7. [Weight Conversion](#7-weight-conversion)
8. [Text Conditioning Failures](#8-text-conditioning-failures)
9. [Position Encodings (RoPE)](#9-position-encodings-rope)
10. [Multi-Stage / Pyramid Pipelines](#10-multi-stage--pyramid-pipelines)
11. [Common Symptoms → Root Causes](#11-common-symptoms--root-causes)
12. [Verification Checklist](#12-verification-checklist)
13. [Diagnostic Tools](#13-diagnostic-tools)

---

## 1. Debugging Methodology

### Component isolation first

Never debug the full pipeline. Test each component in isolation:

1. **Text encoder** — Does it produce embeddings with reasonable statistics? (std > 0.01)
2. **Scheduler** — Do sigma/timestep values match the reference exactly?
3. **Transformer** — Does a single forward pass match the reference? (cosine similarity > 0.999)
4. **VAE decoder** — Feed reference latents into your VAE. Does the output look correct?

If every component matches individually but the pipeline fails, the bug is in
**orchestration** — how components are wired together.

### Statistical fingerprinting

Track per-step statistics through the diffusion loop:

```python
# After each denoising step
print(f"step {i}: mean={latent.mean():.6f} std={latent.std():.6f} "
      f"min={latent.min():.4f} max={latent.max():.4f}")
```

**What to look for:**
- **Progressive mean drift** (e.g., -0.002 → -0.040 → -0.123) signals accumulating errors
- **Collapsing std** (std dropping toward 0) signals broken conditioning or wrong noise schedule
- **Exploding values** signal wrong sigma scaling or scheduler formula

### Cross-framework numerical comparison

The most powerful debugging tool: save intermediate tensors from your MLX pipeline,
feed them to the PyTorch reference, compare outputs.

```python
# In MLX pipeline, save inputs before transformer call
mx.save("debug_inputs.npz", {"latent": latent, "timestep": t, "text_emb": text_emb})

# In PyTorch script, load and compare
inputs = np.load("debug_inputs.npz")
mlx_out = np.load("debug_output.npz")["flow"]
pt_out = reference_model(torch.from_numpy(inputs["latent"]), ...)
cos_sim = F.cosine_similarity(pt_out.flatten(), torch.from_numpy(mlx_out).flatten(), dim=0)
# cos_sim > 0.999 = model is correct; bug is elsewhere
# cos_sim < 0.99  = model has a bug; compare per-layer
```

### Ablation testing

When a pipeline has multiple "fixes" or features, disable them one at a time:

- **Frozen history**: Fix history to the same value for all chunks → proves whether
  history propagation is the source of drift/zoom
- **Single chunk**: Generate only 1 chunk → isolates per-chunk quality from
  multi-chunk interaction bugs
- **Disable post-processing**: Remove cross-fade, blending, corrections → reveals
  what the raw model output looks like

### Use reference on same hardware

Run the PyTorch reference on the same device (MPS for Apple Silicon). CUDA and MPS
produce different numerical results due to different float handling. Comparing your
MLX output against a CUDA reference adds noise to the comparison.

```python
# MPS may not support float64 — patch the reference:
original_linspace = torch.linspace
def patched_linspace(*args, **kwargs):
    kwargs.pop("dtype", None)
    return original_linspace(*args, dtype=torch.float32, **kwargs)
torch.linspace = patched_linspace
```

---

## 2. Precision & Dtype Pitfalls

### The #1 source of subtle bugs

Precision issues caused the most insidious bugs in our port. They don't cause
crashes — they cause progressive quality degradation that's hard to attribute.

### Residual connections MUST be float32

**Bug**: Progressive zoom/shrinking across autoregressive chunks.

**Root cause**: Residual additions (`x = x + attn_out`) in bfloat16. With 7-bit
mantissa, high-frequency spatial detail is systematically truncated. Over 144
residual ops × 6+ model calls per chunk, detail is progressively smoothed away.

**Fix**: Promote to float32 for the addition:
```python
# BAD — bfloat16 accumulation
x = x + attn_out

# GOOD — match reference's .float() pattern
x = (x.astype(mx.float32) + attn_out).astype(weight_dtype)
```

**Rule**: If the reference uses `.float()` anywhere, copy that pattern exactly. It's
there for a reason, even if a quick test seems to work without it.

### Scheduler computations need high precision

Diffusion schedulers involve:
- `x0 = xt - sigma * flow` — catastrophic cancellation near sigma ≈ 1
- `log(sigma)` and `exp()` — sensitive to small precision differences

Some references use float64 for these computations. MLX GPU doesn't support float64,
so use float32 and accept small numerical differences, but **never** use bfloat16
for scheduler math.

### Dtype propagation is invisible

Track dtype through your pipeline. A single bfloat16 intermediate can silently
downcast everything downstream:

```python
# This looks harmless but if model output is bfloat16:
result = noise - sigma * model_output  # result is bfloat16!

# Fix: explicit cast
result = (noise.astype(mx.float32) - sigma * model_output.astype(mx.float32))
```

### Type promotion rules differ across frameworks

- PyTorch: bfloat16 + float32 → float32
- MLX: bfloat16 + float32 → float32 (same, but verify)
- NumPy: no bfloat16 support

Always check what your framework does and match the reference's implicit promotations.

### Float32 for VAE decoding

**Bug** (Wan2.2): VAE decode in bfloat16 produced visibly worse quality than reference.

Official Wan2.2 runs VAE decode in `torch.float` (float32), but our converted weights
were bfloat16. The VAE has many sequential layers where precision loss compounds.

**Fix**: Upcast VAE weights to float32 at load time. The VAE runs once per generation,
so the performance impact is negligible compared to the transformer.

### Modulation/gate vectors need float32

**Bug** (Wan2.2): Quality degradation from bfloat16 modulation across 30 blocks × 50 steps.

The official Wan2.2 explicitly uses `torch.amp.autocast('cuda', dtype=torch.float32)`
for time embeddings, modulation parameters, norm outputs before modulation, and gate ops.

**Fix**: Keep modulation in float32, cast to working dtype only when applying to the
hidden state:
```python
# Modulation computed in float32
e0 = self.modulation(time_emb)  # float32
scale, shift, gate = e0.split(3, axis=-1)

# Cast to bfloat16 only for the matmul with hidden state
x = (x * (1 + scale.astype(x.dtype)) + shift.astype(x.dtype))
```

### Map PyTorch autocast zones precisely

PyTorch models use nested `torch.amp.autocast` scopes to switch precision. Map these
exactly:
- **Outer scope** (`bfloat16`): attention QKV projections, FFN matmuls
- **Inner scope** (`float32`): modulation, gates, norms, RoPE
- **Residual stream**: float32 (the "backbone" between blocks)

```python
# Wan2.2 dtype flow (matches official):
# Modulation/gates: float32 (explicit)
# QKV/FFN linear projections: bfloat16 (weight dtype)
# RoPE: float32 (official uses float64, MLX lacks float64)
# Attention Q/K: cast back to bfloat16 after RoPE
# Residual stream: float32
```

### Float32 promotion cascades kill performance

**Bug** (Wan2.2): ~2x slowdown from accidental float32 promotion.

A single float32 tensor (e.g., time embedding) flowing into bfloat16 operations
promotes the entire computation graph to float32. In Wan2.2:
- Time embedding MLP output (float32) fed into transformer → all layers float32
- RoPE frequencies (float32) applied to Q/K → all attention float32

**Fix**: Cast intermediate results to model dtype at promotion boundaries:
```python
# After time embedding MLP (float32), cast before feeding to transformer
time_emb = time_mlp(t).astype(model_dtype)

# After RoPE (float32), cast Q/K back to attention dtype
q = rope_apply(q, freqs).astype(v.dtype)
```

---

## 3. MLX-Specific Gotchas

### Underscore-prefixed attributes are invisible

**Bug** (Wan2.2): 87 of 110 VAE weights silently dropped during loading.

MLX's `nn.Module.parameters()` and `nn.Module.load_weights()` **skip** attributes
whose names start with underscore. If you name a layer `self._layer_0`, its weights
will never be loaded or saved.

```python
# BAD — weights silently ignored
self._layer_0 = nn.Linear(...)  # nn.Module skips _prefixed attrs

# GOOD
self.layer_0 = nn.Linear(...)
```

This is especially insidious because there's no error — the model loads, runs, and
produces output. The output is just garbage because most weights are random.

### nn.Sequential indexing vs named children

PyTorch's `nn.Sequential` uses integer indices (`sequential.0.weight`), while MLX's
module hierarchy uses named attributes. When mirroring a PyTorch module structure,
you need explicit key sanitization:

```python
def sanitize_key(key):
    # PyTorch: "decoder.middle.0.residual.1.weight"
    # MLX:    "decoder.middle.layer_0.residual.layer_1.weight"
    key = re.sub(r'\.(\d+)', lambda m: f'.layer_{m.group(1)}', key)
    return key
```

### Reshape axis ordering differs from PyTorch

**Bug** (Wan2.2): Green checkerboard pattern from VAE attention.

`[B,C,T,H,W]` cannot be directly reshaped to `[BT,C,H,W]` because in memory C
comes before T. PyTorch's `reshape` works because it handles non-contiguous tensors.
MLX requires explicit transpose first:

```python
# BAD — mixes channels with time
x = x.reshape(B*T, C, H, W)  # Corrupts spatial layout

# GOOD — make B,T adjacent first
x = x.transpose(0, 2, 1, 3, 4)  # [B,T,C,H,W]
x = x.reshape(B*T, C, H, W)     # Now correct
```

### Patchify channel ordering

**Bug** (Wan2.2): Solid green video output from wrong patchify order.

When converting a Conv3d patchify to a manual reshape+linear, the dimension ordering
in the reshape must match the Conv3d weight layout. Conv3d expects `[C, pt, ph, pw]`
(channels slowest), but a naive reshape produces `[pt, ph, pw, C]` (channels fastest):

```python
# BAD — channel scrambling
patches = x.reshape(B, F', H', W', pt, ph, pw, C)

# GOOD — match Conv3d weight layout
patches = x.reshape(B, F', pt, H', ph, W', pw, C)
patches = patches.transpose(0, 1, 3, 5, 7, 2, 4, 6)  # [B, F', H', W', C, pt, ph, pw]
```

Verify numerically: the fixed version should match Conv3d output to ~1e-6.

### mx.zeros / padding inherits dtype

Use dtype-aware `mx.zeros` for padding and concatenation to avoid promotion:

```python
# BAD — default float32 padding promotes bfloat16 input
pad = mx.zeros((B, pad_len, C))  # float32!
x = mx.concatenate([pad, x], axis=1)  # x promoted to float32

# GOOD — match input dtype
pad = mx.zeros((B, pad_len, C), dtype=x.dtype)
x = mx.concatenate([pad, x], axis=1)  # stays bfloat16
```

### Use mx.fast kernels

Replace manual implementations with fused MLX kernels where possible:

```python
# Manual RMS norm → mx.fast.rms_norm
# Manual LayerNorm → mx.fast.layer_norm
# Manual attention → mx.fast.scaled_dot_product_attention
```

These are faster and handle precision internally.

---

## 4. Autoregressive Chunk Boundaries

For models that generate long videos by autoregressively extending chunks (Helios,
CogVideoX, etc.), chunk boundaries are the primary source of visual artifacts.

### Don't add post-processing the reference doesn't have

**Bug**: Added pixel cross-fade to smooth boundaries → caused 40% sharpness drop.

The reference pipeline used **no cross-fade at all**. The first frame of each new
chunk is intentionally a sharp reconstruction conditioned on history. Blending it with
the previous chunk's tail (which has different content) creates blur.

**Rule**: Before adding smoothing/blending, verify the reference doesn't do it.
Reference simplicity is usually correct.

### First-frame artifacts are common

The first pixel frame of each non-first chunk is typically a distorted reconstruction
of the conditioning frame. In many models, this is expected behavior:

- **Fix**: Drop the first frame from each chunk
- **Verify frame math**: If 33 raw frames at 16fps → drop 1 → 32 frames = exactly 2 seconds

### History conditioning errors compound

Small errors in how history is prepared, sliced, patchified, or position-encoded
will compound across chunks. The error is invisible in chunk 1, small in chunk 2,
and catastrophic by chunk 5.

**Debug strategy**: Generate with frozen history (same history for every chunk).
If the artifact disappears, the bug is in history handling.

---

## 5. VAE Decoder Artifacts

### Causal temporal convolutions cause boundary warmup

Video VAEs (WanVAE, CogVideoX-VAE) use causal temporal convolutions. When decoding
each chunk independently, the first few frames lack temporal context (only zero
padding), causing:

- **~7% contrast drop** in first frames of each chunk
- **Spatial brightness redistribution** (face darkens, background brightens)

This is inherent to the architecture. The reference has the same effect but at
lower magnitude.

### Post-processing to fix VAE warmup

Two-stage correction applied to first N frames of each non-first chunk:

```python
# Stage 1: Spatially-varying brightness correction
# Downsample reference (previous chunk's last frame) and current frame
ref_small = cv2.resize(ref_frame, (w//16, h//16), interpolation=cv2.INTER_AREA)
cur_small = cv2.resize(cur_frame, (w//16, h//16), interpolation=cv2.INTER_AREA)
diff_small = ref_small - cur_small
diff_full = cv2.resize(diff_small, (w, h), interpolation=cv2.INTER_LINEAR)
corrected = cur_frame + ramp * diff_full  # ramp: 1.0 → 0.0 over N frames

# Stage 2: Per-channel contrast matching
for c in range(3):
    ref_std = np.std(ref_frame[:,:,c])
    cur_std = np.std(corrected[:,:,c])
    scale = 1.0 + ramp * (ref_std / (cur_std + 1e-6) - 1.0)
    corrected[:,:,c] = (corrected[:,:,c] - mean) * scale + mean
```

### VAE overlap decode does NOT work

**Attempted**: Prepend previous chunk's last latent frames to give the decoder
temporal context.

**Result**: Made things **worse** (22% contrast drop vs 7%). The causal convolutions
see conflicting content from different chunks and create larger artifacts than
zero-padding.

**Lesson**: Overlap only works when tiles contain the same content from the same
denoising process (e.g., spatial tiling). It fails for temporal chunks with
different content.

### Per-chunk VAE decoding is correct

Decode each chunk's latents independently, not concatenated. Concatenating all chunks
and decoding together lets boundary discontinuities propagate through temporal
convolutions, creating worse artifacts.

### First-frame quality: causal padding strategies

Multiple approaches were tried for the first-frame quality issue in Wan VAE:

| Approach | Result |
|----------|--------|
| Zero padding (default) | First ~4 frames degraded, but matches training |
| Replicate padding | Fixes artifacts but causes color intensity bias (conv applies all kernel weights to same value) |
| Warmup frame prepend | Helps motion but warmup frame itself has artifacts |
| Mirror-reflect warmup | Best compromise — varied context without zeros, no intensity bias |

**Lesson**: Don't assume "replicate padding is better than zero padding." The model
was trained with zero padding; changing it shifts the gain. Instead, prepend warmup
frames and trim them after decoding.

### RMS_norm vs L2 normalization

**Bug** (Wan2.2): Garbled output from incorrect normalization.

A PyTorch class named `RMS_norm` actually uses `F.normalize` (L2 norm: `x / ||x||_2`),
not RMS normalization (`x / sqrt(mean(x²))`). The difference is a factor of `sqrt(C)`,
causing values to explode through the decoder.

**Lesson**: Don't trust class names — read the actual implementation.

### Temporal frame count: causal boundary effects

**Bug** (Wan2.2): VAE produced 12 frames instead of 9 for a 9-frame input.

PyTorch reference processes frames one-by-one with caching, skipping temporal conv for
the first chunk. All-at-once decoding produces extra frames from zero-padded causal
context.

**Fix**: Use `first_chunk=True` flag to trim causal boundary frames, matching the
reference's chunked behavior.

### Chunked VAE encoding for I2V

**Bug** (Wan2.2 I2V-14B): Incorrect latents from non-chunked encoding.

Non-chunked encoding with causal zero-padding produces incorrect latents because
temporal features don't propagate correctly without caching. The reference uses chunked
encoding (1+4+4+... frames) with persistent temporal cache.

**Fix**: Implement chunked encoding with `feat_cache` propagation through CausalConv3d,
ResidualBlock, and Resample layers.

---

## 6. Scheduler & Timestep Issues

### Copy formulas exactly

Even small differences in scheduler formulas compound over many steps:

```python
# Dynamic time shifting — reference uses specific formula
mu = 0.5 + shift * 0.5  # NOT shift * 0.6 or any other constant

# Euler step
x_next = x + (sigma_next - sigma) * flow  # order matters: next - current
```

### Verify sigma schedules numerically

Print and compare sigma values at each step:

```python
# Reference
sigmas_ref = [1.0, 0.99375, 0.9875, ...]

# Your implementation
sigmas = scheduler.get_sigmas(steps)
for i, (r, m) in enumerate(zip(sigmas_ref, sigmas)):
    assert abs(r - m) < 1e-6, f"Step {i}: ref={r}, mlx={m}"
```

### Timestep embedding precision

Integer vs float timesteps matter. Some models expect `timestep=999` (int), others
expect `timestep=0.999` (float). Wrong type can silently produce wrong embeddings
with reasonable-looking but incorrect statistics.

### Boundary conditions: ±inf at sigma endpoints

**Bug** (Wan2.2): Greenish/yellow constant output from DPM++/UniPC schedulers.

The `lambda(sigma)` function must return `-inf` at `sigma=1.0` (pure noise) and `+inf`
at `sigma=0.0` (clean signal). Our implementation returned `0.0`, causing massive x0
overscaling on the first denoising step.

PyTorch naturally computes `torch.log(0) = -inf`, and `math.expm1(-inf) = -1.0`
handles the formulas correctly. Reproduce this behavior explicitly:

```python
def _lambda(self, sigma):
    if sigma >= 1.0:
        return float('-inf')
    if sigma <= 0.0:
        return float('inf')
    return -math.log(sigma / (1 - sigma))
```

### UniPC corrector coefficients

**Bug** (Wan2.2): Accumulated artifacts across 47+ steps from wrong polynomial weights.

The UniPC corrector must compute `rhos_c` via `linalg.solve` for order ≥ 2. Hardcoded
`0.5` was 7× too large for the history weight (actual: ~0.08), causing massive
overweighting of history corrections.

---

## 7. Weight Conversion

### Always verify statistically

After converting weights from PyTorch to MLX format:

```python
for name in mlx_weights:
    pt = pytorch_weights[map_name(name)]
    mx_val = np.array(mlx_weights[name])
    pt_val = pt.numpy()
    cos_sim = np.dot(mx_val.flat, pt_val.flat) / (
        np.linalg.norm(mx_val) * np.linalg.norm(pt_val) + 1e-10
    )
    if cos_sim < 0.9999:
        print(f"MISMATCH: {name} cos_sim={cos_sim:.6f}")
```

### Conv3d → Linear reshaping

When converting 3D convolutions to linear layers (common for MLX which prefers
linear ops), the flattening order must match:

```python
# PyTorch Conv3d weight: (out_ch, in_ch, kT, kH, kW)
# Flatten to Linear: (out_ch, in_ch * kT * kH * kW)
# The reshape order MUST match how the input is patchified
```

### Sanitization functions

Write explicit weight sanitization that maps reference key names to your key names.
Don't rely on automatic matching — key naming conventions differ between frameworks.

### Module structure must mirror reference for direct loading

**Bug** (Wan2.2): Rewrote entire VAE module hierarchy to match PyTorch `nn.Sequential`
structure. ResidualBlock needed `None` gaps at specific indices to match the original
`nn.Sequential(RMSNorm, SiLU, Conv3d, ...)` indexing.

When possible, structure your modules to accept reference weights directly without
sanitization. This eliminates an entire class of bugs.

### Save VAE weights in float32

Even if the model uses bfloat16 for the transformer, save VAE weights in float32.
bfloat16 → float32 roundtrip loses precision that cannot be recovered by load-time
upcast.

### Temporal downsample/upsample order

**Bug** (Wan2.2): `temporal_downsample=[True, True, False]` but reference uses
`[False, True, True]`. Stage 0 created a `time_conv` with random weights (no matching
file key), and Stage 2 missed its `time_conv` (weights silently dropped).

Always verify boolean flags for each stage by inspecting the actual weight file keys.

### Silent weight drops are the worst bugs

When `load_weights()` with `strict=False` silently skips keys that don't match, you
get a model with random weights for those layers. This produces output that looks
"almost right" but is subtly wrong. Always log which keys were loaded vs skipped:

```python
loaded_keys = set()
for key, value in weights:
    if key in model_params:
        loaded_keys.add(key)
# Check for missing
expected = set(model_params.keys())
missing = expected - loaded_keys
if missing:
    print(f"WARNING: {len(missing)} weights not loaded: {list(missing)[:5]}...")
```

---

## 8. Text Conditioning Failures

### Symptom: model predicts noise back to itself

If the model output correlates > 0.8 with its input noise, text conditioning is
likely broken. The model has learned nothing from the prompt and is just returning
its input.

### Check embedding statistics

```python
text_emb = text_encoder(prompt)
print(f"text_emb: mean={text_emb.mean():.4f} std={text_emb.std():.4f}")
# std < 0.01 → embeddings are collapsed → broken encoder or wrong weights
# std > 10.0 → embeddings are exploding → wrong normalization
```

### Verify with ablation

```python
# Generate with real text
output_text = denoise(latent, text_emb=real_embeddings)
# Generate with zeros
output_zero = denoise(latent, text_emb=mx.zeros_like(real_embeddings))
# Compare
text_influence = np.mean(np.abs(output_text - output_zero))
print(f"Text influence: {text_influence:.4f}")  # Should be > 0 (typically 30-60% of output)
```

### Text preprocessing must match exactly

**Bug** (Wan2.2): Patchy-blurry output from wrong negative prompt tokenization.

The official Wan2.2 tokenizer applies `ftfy.fix_text` + `html.unescape` + whitespace
normalization before tokenization. Without this, fullwidth Chinese commas (U+FF0C)
tokenize differently from ASCII commas (U+002C), causing **27 different token IDs**
in the negative prompt. This made CFG's unconditional prediction wrong.

**Fix**: Apply the same text cleaning pipeline as the reference:
```python
import ftfy
import html
import re

def clean_text(text):
    text = ftfy.fix_text(text)
    text = html.unescape(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
```

### T5 encoder precision

**Bug** (Wan2.2): Quality degradation from bfloat16 T5 attention.

T5 uses **no scaling** in attention (no `1/sqrt(d)` factor), so attention logits can
be very large. bfloat16 softmax loses significant precision across 24 encoder layers.

**Fix**: Compute T5 QK^T and softmax in float32. The T5 encoder only runs once per
generation, so the performance impact is negligible.

### Dual-model text embeddings

**Bug** (Wan2.2 I2V-14B): Low/high noise models have different `text_embedding` weights
(~42% relative difference). Using one model's embeddings for both caused incorrect
text conditioning for the high-noise model that handles critical early denoising steps.

**Fix**: Compute separate text embeddings for each model in dual-model setups.

---

## 9. Position Encodings (RoPE)

### Multi-scale consistency

In pyramid/multi-resolution models, RoPE must be computed consistently across scales.
If the model operates at 1/4 resolution in an early stage, the position grid must
reflect the actual spatial dimensions, not the final target dimensions.

### History vs current chunk

When conditioning on history from a previous chunk, the position encoding for
history frames must match what the model saw during training. Mismatches between
history and current-chunk position encodings can cause subtle spatial distortions
that compound across chunks.

### Factorized RoPE

3D video models often use factorized RoPE (separate temporal, height, width
frequencies). Verify each axis independently:

```python
# Compare temporal frequencies
assert np.allclose(mlx_rope_t, ref_rope_t, atol=1e-5)
# Compare spatial frequencies
assert np.allclose(mlx_rope_h, ref_rope_h, atol=1e-5)
assert np.allclose(mlx_rope_w, ref_rope_w, atol=1e-5)
```

### Per-axis frequency construction

**Bug** (Wan2.2): Grey/artifact-filled output from wrong frequency distribution.

The reference uses three separate `rope_params()` calls with different dimension
normalizations (e.g., 44, 42, 42 for Wan) so each axis gets its own full frequency
range. Consolidating into a single `rope_params(head_dim)` call and splitting gave
height frequencies starting at 0.042 and width at 0.002 (should be 1.0 for both).

**Fix** (and subsequent revert): This bug was introduced as a "fix" for a previous
RoPE issue, then had to be reverted. The lesson: RoPE changes have far-reaching effects.
Always verify with actual generation, not just numerical comparison of frequencies.

**Lesson**: Read the reference's frequency construction very carefully. Don't
"simplify" three separate calls into one unless you verify the frequency distribution
matches exactly.

---

## 10. Multi-Stage / Pyramid Pipelines

### Each stage is a potential failure point

Pyramid pipelines (generate at low res, upsample, refine at high res) multiply the
number of things that can go wrong:

- Downsampling method (bilinear vs area) must match reference
- Energy compensation factors (e.g., ×2 after bilinear downsample) must be present
- Alpha/beta noise mixing coefficients are stage-dependent
- Frame indices and history resolution change per stage

### Test single-stage first

If the model works at full resolution for a single stage but fails in the pyramid,
the bug is in stage orchestration — typically in how latents are passed between
stages or how position encodings adapt to different resolutions.

### Integration bugs are the hardest

We verified every Helios component matched the reference individually, but the
pyramid still produced uniform color. The bug was in dtype handling during stage
transitions. Integration bugs only appear when components interact.

---

## 11. Common Symptoms → Root Causes

| Symptom | Likely Root Causes |
|---------|-------------------|
| **Pure noise output** | Wrong sigma schedule, broken text conditioning, incorrect weight mapping |
| **Uniform color** | Model predicting noise back; text embeddings collapsed; wrong timestep format |
| **Progressive zoom/shrink** | bfloat16 residuals truncating high-freq detail; RoPE mismatch across chunks |
| **Brightness jumps at boundaries** | VAE causal warmup; cross-fade blending misaligned content |
| **Color drift across chunks** | Dtype in scheduler step; history normalization missing |
| **Blur at boundaries** | Cross-fade enabled; latent blending; wrong VAE decode order |
| **Grid/checker patterns** | Patchify channel ordering bug; latent blend artifacts; reshape axis error |
| **Green/magenta tint** | VAE weight key mismatch; wrong denormalization constants; cv2 YUV color matrix |
| **Mean drift across steps** | bfloat16 accumulation; wrong scheduler formula; missing energy compensation |
| **Garbled/scrambled output** | Silent weight drops (underscore prefix, wrong key mapping); RMS vs L2 norm |
| **Greenish-yellow constant** | Scheduler boundary condition (log(0) not returning -inf); x0 overscaling |
| **~2x slower than expected** | Float32 promotion cascade from single mistyped intermediate |
| **Extra output frames** | Causal padding producing extra temporal frames; missing `first_chunk` trim |
| **Grey/artifact output** | RoPE frequency construction wrong (per-axis vs single-call) |
| **Patchy-blurry with CFG** | Text preprocessing mismatch (fullwidth vs ASCII chars → wrong tokenization) |
| **I2V temporal mismatch** | Non-chunked VAE encoding vs reference's chunked encoding with temporal cache |

---

## 12. Verification Checklist

Use this checklist when porting a new diffusion video model:

### Model
- [ ] Weight conversion: all keys mapped, cosine similarity > 0.9999
- [ ] No silent weight drops (log loaded vs expected keys)
- [ ] Single forward pass matches reference (cos_sim > 0.999)
- [ ] Residual connections use float32 accumulation
- [ ] Attention computation matches reference precision
- [ ] Modulation/gate vectors in float32 (if reference uses autocast)
- [ ] No underscore-prefixed module attributes (MLX ignores them)

### Scheduler
- [ ] Sigma values match reference at every step (diff < 1e-6)
- [ ] Timestep format correct (int vs float, scale factor)
- [ ] Dynamic shifting formula copied exactly
- [ ] Step function returns correct dtype (float32)
- [ ] Boundary conditions: lambda(-inf) at sigma=1, lambda(+inf) at sigma=0
- [ ] Higher-order coefficients computed (not hardcoded) for UniPC/DPM++

### Text Encoder
- [ ] Embedding statistics reasonable (0.01 < std < 10)
- [ ] Text influence > 0 (ablation test)
- [ ] Tokenization matches (special tokens, padding, max length)
- [ ] Text preprocessing matches (ftfy, html unescape, whitespace normalization)
- [ ] T5/CLIP attention precision (float32 softmax if no 1/sqrt(d) scaling)
- [ ] Separate embeddings for dual-model setups (if applicable)

### VAE
- [ ] Denormalization constants match training pipeline
- [ ] Per-chunk decoding (not concatenated)
- [ ] Temporal frame count correct (account for causal padding)
- [ ] Weight keys mapped correctly (encoder vs decoder)
- [ ] Weights stored/loaded in float32 (not bfloat16)
- [ ] Temporal downsample/upsample order matches reference
- [ ] RMS_norm vs L2_norm: check actual implementation, not class name
- [ ] Chunked encoding for I2V (if applicable)
- [ ] Reshape axis ordering correct ([B,C,T,H,W] → transpose before reshape)

### Pipeline Orchestration
- [ ] Position encodings consistent across stages/chunks
- [ ] History slicing and conditioning correct
- [ ] Noise generation matches (distribution, correlation structure)
- [ ] Multi-chunk output visually consistent (no progressive degradation)
- [ ] Dimension auto-alignment (divisible by patch_size × vae_stride)
- [ ] Dtype-aware padding (mx.zeros with explicit dtype)

### Output
- [ ] Frame count matches expected (account for warmup/trim)
- [ ] FPS correct
- [ ] Color range [0, 255] uint8 for video
- [ ] No first-frame duplication artifacts
- [ ] Video codec correct (imageio/libx264 preferred over cv2/mp4v on macOS)

### Performance
- [ ] No float32 promotion cascades (check with profiler)
- [ ] Using mx.fast kernels (rms_norm, layer_norm, sdpa)
- [ ] Time embedding computed once per sample (not per position)
- [ ] Memory cleanup (delete temporaries before mx.eval)

---

## 13. Diagnostic Tools

### General video diagnostics (`scripts/video/`)

| Script | Purpose |
|--------|---------|
| `compare_videos.py` | PSNR, SSIM, temporal coherence, color fidelity between two videos |
| `video_quality.py` | Sharpness, stability, defect detection, chunk boundary analysis |

```bash
# Quick quality check
python scripts/video/video_quality.py output.mp4 --chunk-size 32

# Compare against reference
python scripts/video/compare_videos.py reference.mp4 output.mp4 --diff-video diff.mp4
```

### Model-specific diagnostics (`mlx_video/models/helios/scripts/`)

| Script | Purpose |
|--------|---------|
| `analyze_boundaries.py` | Detailed boundary quality metrics for Helios |
| `run_reference.py` | Run PyTorch reference on MPS |
| `compare_pipelines.py` | Compare scheduler/pipeline mechanics |
| `compare_models.py` | Cross-framework model output comparison |

### Inline debugging pattern

Add temporary debug output to the diffusion loop:

```python
for i, sigma in enumerate(sigmas):
    flow = model(latent, sigma, text_emb)
    latent = scheduler.step(latent, flow, sigma, sigma_next)

    # Debug: track statistics
    print(f"[step {i}] sigma={sigma:.4f} "
          f"latent: mean={latent.mean():.6f} std={latent.std():.6f} "
          f"flow: mean={flow.mean():.6f} std={flow.std():.6f}")

    # Debug: save for cross-framework comparison
    if os.environ.get("DEBUG"):
        mx.save(f"/tmp/debug_step_{i}.npz", {
            "latent": latent, "flow": flow, "sigma": mx.array(sigma)
        })
```

---

## Key Takeaways

1. **Precision is the #1 bug source** — bfloat16 residuals, scheduler math, type
   promotion, modulation vectors. Copy the reference's `.float()` and `autocast` zones.

2. **Don't add what the reference doesn't have** — cross-fade, overlap decode,
   temporal blending. If the reference works without it, you probably have a bug
   elsewhere.

3. **Silent failures are the hardest bugs** — underscore-prefixed weights, `strict=False`
   weight loading, wrong normalization class names. Always verify weight load counts
   and output statistics.

4. **Component isolation → integration testing** — verify each part matches, then
   debug their interaction.

5. **Statistical comparison beats visual inspection** — mean drift, contrast ratios,
   and cosine similarity catch bugs before they're visible.

6. **Autoregressive errors compound** — a 1% error per chunk becomes 10% by chunk 10.
   Fix precision first, add corrections second.

7. **MLX has unique pitfalls** — underscore attribute names, reshape axis ordering,
   dtype-unaware padding, and float32 promotion cascades. Know your framework.

8. **Text preprocessing matters** — Unicode normalization, fullwidth chars, HTML entities.
   A single mismatched comma can break CFG guidance.

9. **VAE is deceptively complex** — causal padding, temporal frame counts, chunked vs
   batch processing, norm implementations. Budget significant debugging time for VAE.
