# Helios Diagnostics & Engineering Notes

Technical reference for the Helios (distilled) video generation pipeline in mlx-video.
Covers all findings from the bring-up, verified behaviors, resolved bugs, open problems,
and things to watch out for during future development.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Verified Components](#verified-components)
- [Bug History & Resolutions](#bug-history--resolutions)
- [Open Problems](#open-problems)
- [Things to Watch Out For](#things-to-watch-out-for)
- [Key Constants & Formulas](#key-constants--formulas)
- [Diagnostic Recipes](#diagnostic-recipes)

---

## Architecture Overview

Helios is a 14B-parameter DiT for autoregressive video generation. It shares ~95% of its
architecture with Wan (same VAE, same T5 encoder, same dim/heads/layers). Key Helios-specific
additions:

| Component | Description |
|-----------|-------------|
| **Autoregressive chunking** | 33-frame chunks (9 latent frames), each chunk conditioned on history from prior chunks |
| **Multi-scale history** | Short (1×), Mid (2× downsampled), Long (4× downsampled) history via Conv3d patchifiers |
| **3-stage pyramid denoising** | Denoise at 1/4 → 1/2 → full resolution for efficiency |
| **DMD scheduler** | x0-prediction with re-noising (distilled model uses 2+2+2 steps) |
| **Block noise** | Structured per-patch noise via correlated multivariate normal |

### Pipeline flow (distilled, 3-stage pyramid)

```
Full-res noise → bilinear↓2 * 2 → bilinear↓2 * 2 → [1/4 res latents]

Stage 0 (1/4 res):  2 DMD steps → denoised₀
Stage 1 (1/2 res):  nearest↑2(denoised₀) → α·up + β·block_noise → 2 DMD steps → denoised₁
Stage 2 (full res): nearest↑2(denoised₁) → α·up + β·block_noise → 2 DMD steps → final

VAE decode → video frames
```

### File layout (~2500 lines total)

```
mlx_video/generate_helios.py          # Pipeline orchestration (554 lines)
mlx_video/models/helios/
  config.py        # HeliosModelConfig dataclass (69 lines)
  transformer.py   # 14B DiT backbone (511 lines)
  attention.py     # Self/cross attention with history (270 lines)
  rope.py          # 3-way factorized RoPE (215 lines)
  scheduler.py     # DMD + Euler schedulers (264 lines)
  loading.py       # Weight loading wrappers (51 lines)
mlx_video/convert_helios.py           # HF→MLX weight conversion
tests/test_helios.py                  # 46 tests (554 lines)
```

---

## Verified Components

These components have been numerically verified against the reference PyTorch implementation
and can be considered correct. If output quality issues arise, look elsewhere first.

### 1. Transformer model ✅

**Verification**: Fed identical random inputs (latents, encoder_hidden_states, timestep=500)
to both MLX and reference PyTorch implementations.

| Metric | Value |
|--------|-------|
| Mean abs diff | 0.004190 |
| Correlation | **0.999773** |
| Per-channel means | Match to 3 decimal places |

The model produces correct flow predictions. Color issues are in the pipeline, not the model.

### 2. VAE decoder ✅

- All weight key mappings verified exact (0.000000 max diff per key)
- Decoder output correlation 0.999+ with reference
- **Temporal offset**: First `stride_t - 1 = 3` frames are warmup garbage from causal
  padding. The pipeline trims these before saving.

### 3. Scheduler (DMD) ✅

Verified against reference PyTorch scheduler with identical inputs. Both produce:

| Parameter | Stage 0 (1/4 res) | Stage 1 (1/2 res) | Stage 2 (full res) |
|-----------|-------------------|-------------------|-------------------|
| seq_len | 540 | 540 | 2160+ |
| mu (shift) | 0.5481 | 0.5481 | 0.8223 |
| sigmas | [0.998, 0.354, 0.0] | [0.998, 0.354, 0.0] | [0.999, 0.451, 0.0] |
| timesteps | [998.5, 834.0] | [742.6, 512.5] | [385.2, 174.8] |

Alpha/beta blending coefficients:
- Stage 1: α=0.6001, β=0.6926 (α²+β²=0.84)
- Stage 2: α=0.7498, β=0.4333 (α²+β²=0.75)

### 4. Other verified components

- **T5 text encoder**: Reused from Wan, works with sanitized HF UMT5 keys
- **RoPE**: 3-way factorized (44,42,42) split, pad+center_downsample for history
- **Bilinear downsample × 2**: Matches reference `F.interpolate(bilinear) * 2`
- **Nearest upsample**: Matches reference `F.interpolate(nearest)`
- **Block noise**: Mathematically equivalent to reference
- **History ordering**: [long | mid | short | current] matches reference
- **Video encoding**: imageio with libx264 (no color space issues)

---

## Bug History & Resolutions

### Bug 1: Timestep projection permutation

**Symptom**: Garbage output, model crash.
**Root cause**: Reference permutes `timestep_proj` from `(B,6,L,dim)` → `(B,L,6,dim)` before
passing to blocks. Our code had the wrong axis order.
**Fix**: Added `.transpose(0, 2, 1, 3)` in `HeliosModel.__call__`.

### Bug 2: T5 encoder key mismatch

**Symptom**: `ValueError: Received 242 parameters not in model`
**Root cause**: HuggingFace UMT5 weight keys don't match MLX T5Encoder keys
(e.g., `encoder.block.0.layer.0.SelfAttention.q.weight` vs `encoder.layers.0.self_attn.q_proj.weight`).
**Fix**: Added `_sanitize_helios_t5_weights()` with complete HF→MLX key mapping.

### Bug 3: RoPE reshape crash

**Symptom**: `ValueError: Cannot reshape array of size 88 into shape (1,1,1,22,2)`
**Root cause**: RoPE frequency computation assumed fixed spatial dimensions. With pyramid
denoising, dimensions change per stage.
**Fix**: Rewrote `rope.py` with 5D compute + downsample approach (`_rope_compute_5d`,
`_rope_pad_and_downsample`).

### Bug 4: Grey/uniform output

**Symptom**: All pixels ~128 (mid-grey), no content visible.
**Root cause**: Two bugs:
1. Wrong `_time_shift` formula: Was `mu*t/(mu+(1-mu)*t)`, correct is `mu*t/(1+(mu-1)*t)`
2. VAE weight keys not mapped: decoder was using random weights.
**Fix**: Corrected formula + added `sanitize_helios_vae_weights()`.

### Bug 5: Multi-chunk noise (chunks 2+ were random noise)

**Symptom**: First chunk had content, subsequent chunks were pure noise.
**Root cause**: Code was downsampling history to match each pyramid stage's resolution.
The reference passes **full-resolution** history at **all** pyramid stages — the model's
Conv3d patchifiers handle the spatial mismatch.
**Fix**: Removed `_downsample_history` calls.

### Bug 6: Video color space (macOS)

**Symptom**: Colors appeared wrong in some players.
**Root cause**: OpenCV's `mp4v` codec on macOS uses a YUV color matrix that some players
interpret differently.
**Fix**: Switched to imageio + libx264 for video encoding.

### Bug 7: Color distortion — solid red/yellow (pyramid-specific)

**Symptom**: Output heavily biased toward a single color. R≈224, G≈100-197, B≈28-48.
Single-stage denoising produced correct colors; pyramid denoising did not.

**Investigation** (9 controlled experiments):

| Experiment | Result |
|-----------|--------|
| 1-stage, 8 steps, full res | ✅ Balanced colors |
| 3-stage pyramid, 2+2+2 | ❌ Red-biased |
| 3-stage pyramid, 8+8+8 | ❌ Still red-biased (more steps ≠ better) |
| Pyramid with zero block noise | ❌ Still red-biased (noise not the cause) |
| 2 steps at full res (stage 2 sigmas) | ✅ Balanced |
| 2 steps at full res (stage 0 sigmas) | ✅ Balanced |
| 3-stage at full res (no spatial scaling, with blend) | ❌ Wrong colors |
| 3-stage at full res, pure noise start_point | ✅ Balanced |

**Root cause**: DMD re-noising cascades per-channel mean bias across pyramid stages.

The formula `prev = (1-σ_next)·x0 + σ_next·start_point` re-injects the blended signal's
mean at each step. Channel means grow monotonically through stages:

```
Stage 0 → ch0: -0.25, ch2: +0.36
Stage 1 → ch0: -0.49, ch2: +0.82
Stage 2 → ch0: -0.83, ch2: +1.41  ← ~4× amplification
```

More steps per stage make this WORSE (4+4+4 gives ch2=+2.09).

**Fix applied** (commit `c5acde72`): Normalize the start_point per-channel to zero mean
and unit std for stages > 0. This preserves spatial structure (which patches are high/low)
while breaking the mean cascade:

```python
sp_mean = mx.mean(sp, axis=keepdim_axes, keepdims=True)
sp_std = mx.clip(sp.std(axis=keepdim_axes, keepdims=True), a_min=1e-6, a_max=None)
start_point_list.append((sp - sp_mean) / sp_std)
```

Result: R=206,G=107,B=43 → **R=152,G=111,B=75** (balanced warm tones for beach prompt).

**REVERTED**: See Bug 9 below — this normalization was found to be the cause of the
pure noise output. The reference implementation does NOT normalize start_point.
Mild per-channel mean growth across stages is the expected behavior.

### Bug 8: Precision mismatch (MLX vs PyTorch)

**Symptom**: Subtle color shifts compared to reference.
**Root cause**: MLX promotes `bfloat16 × float32 → float32`, so the model was computing
in float32 instead of bfloat16 (which PyTorch uses on CUDA tensor cores). Also, the
scheduler's `step_dmd` never cast back to the original dtype.
**Fix** (commit `c5acde72`):
1. Cast latents + history to `bfloat16` before model calls
2. Return `prev_sample.astype(orig_dtype)` from `step_dmd`

Note: This alone had minimal impact on color — the normalized start_point was the primary fix.

### Bug 9: Pure noise output — start-point normalization breaks DMD trajectory

**Symptom**: Output was pure noise even for the first chunk. No recognizable content.

**Root cause**: The start_point normalization added in Bug 7's fix (commit `c5acde72`)
changed the scale of the noise tensor used in DMD re-noising. The DMD formula:
```
prev = (1 - sigma_next) * x0_pred + sigma_next * start_point
```
relies on `start_point` having the correct magnitude — it's the original noisy latent
at each pyramid stage, scaled by the alpha/beta blending coefficients. Normalizing to
unit std destroys this scale relationship, causing the denoising trajectory to diverge.

The reference implementation (`pipeline_helios_diffusers.py` line 703) simply appends
the blended latent without any normalization:
```python
start_point_list.append(latents)
```

**Investigation** (systematic comparison against reference):
1. Line-by-line comparison of `generate_helios.py` vs `pipeline_helios_diffusers.py`
2. Verified scheduler sigmas, timesteps, DMD expansion/trim all match reference
3. Verified VAE denormalization is correct (WanVAE handles internally)
4. Verified block noise Cholesky approach matches reference MultivariateNormal
5. Identified start_point normalization as the only functional deviation from reference

**Fix**: Removed the normalization, restoring `start_point_list.append(latents)` to
match the reference.

**Debug output** (seed=42, "A calm ocean at sunset", 384×640, 33 frames):
```
Stage 0 (1/4 res, 12×20): sigmas=[0.998, 0.354, 0.0], ts=[998.5, 834.0]
  Step 0: model_out std=0.505 → latent std=0.719
  Step 1: model_out std=0.515 → latent std=0.603
Stage 1 (1/2 res, 24×40): alpha=0.60, beta=0.69
  Step 0: model_out std=0.622 → latent std=0.552
  Step 1: model_out std=0.581 → latent std=0.548
Stage 2 (full res, 48×80): alpha=0.75, beta=0.43
  Step 0: model_out std=0.668 → latent std=0.603
  Step 1: model_out std=0.594 → latent std=0.762

Output frame analysis:
  R=114, G=59, B=17 (warm sunset tones ✓)
  Gradient: dx=0.2, dy=0.4 (smooth, structured)
  Entropy: 5.54 bits (normal range)
  Frame-to-frame diff: 3.46 avg (temporally coherent ✓)
```

**Status**: Mean cascade still exists (mean grows -0.07 → -0.15 → -0.23 across stages)
but is mild. This appears to be inherent model behavior, not a bug. The per-channel
growth is within the VAE's normalization range and decodes to warm, plausible colors.

### Bug 10: Uniform color output — wrong zero-history timestep embedding

**Symptom**: Output video showed near-uniform red/orange color (R=114, G=59, B=17 with
very low per-channel variance). No recognizable content despite plausible color range.

**Root cause**: The zero-history timestep embedding was computed incorrectly. The reference
passes `timestep=0` through the sinusoidal `Timesteps()` encoder which produces
`[cos(0), sin(0)] = [1,1,...,1, 0,0,...,0]` (128 ones followed by 128 zeros). Our code
used `mx.zeros_like(t_emb)` — all zeros — which produces a completely different MLP output.

Since history tokens make up ~81.6% of all tokens (2400 out of 2940 at 1/4 resolution),
the vast majority of tokens received wrong scale/shift/gate modulation vectors from the
`scale_shift_table`. This corrupted self-attention (history and current tokens interact),
making the transformer output effectively random.

**Diagnosis** (block-by-block comparison against reference PyTorch):
1. Verified all inputs to the transformer match: patches, RoPE, text embeddings, time
   embeddings, history patches — all cosine_sim ≈ 1.0
2. Block 0 output diverged catastrophically: cosine_sim = -0.30 (essentially uncorrelated)
3. Traced the bug to `HeliosModel.__call__` line 459 where `t0_emb = mx.zeros_like(t_emb)`
   should have been the sinusoidal encoding of timestep=0

**Fix** (commit `061f191b`):
```python
# Before (wrong):
t0_emb = mx.zeros_like(t_emb)

# After (correct):
t0_emb = mx.array([0.0]) * self._inv_freq
t0_emb = mx.concatenate([mx.cos(t0_emb), mx.sin(t0_emb)], axis=-1)
```

**Result**: Block 0 cosine similarity: -0.30 → 0.999982. Full pipeline output now shows
recognizable structured content with high per-channel variance (R=100±100, G=81±81, B=33±37).

### Bug 11: Scheduler step_dmd returning bfloat16

**Symptom**: Minor precision loss across denoising steps (contributed to warm color bias
but not the primary cause of bad output).

**Root cause**: `step_dmd()` cast the result back to `orig_dtype` (bfloat16) at the end.
The reference keeps latents in float32 between steps. Since the DMD formula
`prev = (1-σ)·x0 + σ·start_point` involves near-cancellation when σ≈1, float32 precision
is important.

**Fix** (commit `061f191b`): Return float32 from `step_dmd()`, use `float()` for sigma
values to avoid array overhead.

---

## Open Problems

### 1. Chunk boundary blur artifacts

**Status**: Improved via per-chunk VAE decoding. First few latent frames of each new chunk
have ~45% less spatial detail than peak frames due to lack of temporal context during
denoising. This is inherent to autoregressive chunking — the reference has the same
limitation and does **no post-processing** at chunk boundaries.

**Key finding**: The reference decodes each chunk independently (9 latent frames at a time),
then concatenates pixel frames. Our initial approach of decoding the full concatenated
latent sequence caused the VAE's causal temporal convolutions to propagate quality
discontinuities across chunk boundaries, adding secondary artifacts (grid, brightness).

**Current approach**: Per-chunk VAE decoding (matching reference). Each chunk is decoded
independently with fresh causal padding, producing cleaner boundary transitions.

**Optional**: Latent-space blend (`--chunk-blend N`, default 0 = off). Blends the first N
latent frames of each new chunk toward the previous chunk's last frame. Generally not
recommended as it introduces its own artifacts (grid patterns, brightness shift).

### 2. Color warmth / saturation

**Status**: RESOLVED by Bug 10 fix. The uniform warm color was caused by the wrong
zero-history timestep embedding, not inherent model behavior. Output now shows proper
color variation matching the prompt.

### 3. Generation speed

**Status**: ~14s/step at 384×640 resolution. This is limited by the full-resolution stages
(stages 0-1 at reduced resolution are fast: ~5s/step).

**Not yet explored**:
- `mx.compile()` for the model forward pass
- Quantization (model supports 4/8-bit via convert_helios.py)
- Memory-efficient attention

### 4. `amplify_first_chunk` not tested

The reference recommends `--is-amplify-first-chunk` for distilled models. This doubles the
DMD timestep expansion for the first chunk (2n+1 steps instead of n+1). Not yet tested in
our pipeline.

### 5. Non-distilled model not supported

Only the distilled model (DMD scheduler, 2+2+2 steps, no CFG) is implemented. The
non-distilled model uses Euler/UniPC schedulers with 20+20+20 steps and requires CFG.
The scheduler infrastructure exists (`step()` method) but the pipeline hasn't been tested.

---

## Things to Watch Out For

### Precision: bfloat16 vs float32 in MLX

MLX type promotion rules:
```
bfloat16 × float32 → float32   (NOT bfloat16!)
bfloat16 × float16 → float32   (promoted to higher common type)
bfloat16 × bfloat16 → bfloat16 (stays in bf16)
```

The reference runs the model in bfloat16 throughout (CUDA tensor cores). To match, we
**must** cast latents and history to bfloat16 before model calls. The model weights are
stored in bfloat16 (`model.safetensors`), so if inputs are also bfloat16, all computations
stay in bfloat16.

However: Empirically, the precision difference has minimal impact on output quality. The
normalized start_point fix was far more impactful.

### VAE temporal offset

The WanVAE's causal Conv3d layers produce `stride_t - 1 = 3` warmup frames at the start.
These frames are garbage and must be trimmed:

```python
video = video[:, :, 3:, :, :]  # trim causal padding warmup
```

The pipeline handles this automatically but be careful if decoding latents manually.

### History is always full resolution

When passing history latents to the transformer, they must be at **full resolution** regardless
of which pyramid stage is active. The model's Conv3d patchifiers (with stride 2×2×2 and 4×4×4)
handle the downsampling internally. Passing pre-downsampled history causes noisy output.

### Frame count requirements

- Frames per chunk: **33** (hardcoded: `(9 - 1) * 4 + 1`)
- Total frames must be `1 + 32*k` (e.g., 33, 65, 97, 129)
- The pipeline automatically rounds up to the nearest valid count

### Dimension alignment

- Height and width must be divisible by **16** (VAE spatial compression × patch size)
- Latent dimensions: `h_lat = h // 8`, `w_lat = w // 8`
- For pyramid, each dimension halves twice (so full-res latent dims must be divisible by 4)

### DMD re-noising formula

```
x0_pred = sample - sigma_t * flow_pred        (float32, upcasted)
prev = (1 - sigma_next) * x0_pred + sigma_next * noisy_start   (if not last step)
prev = x0_pred                                                  (if last step)
```

The `noisy_start` is stored per-stage. For stage 0, it's the initial downsampled noise.
For stages > 0, it's the **normalized** blended signal (zero-mean, unit-std per channel).

**Critical**: Passing the raw blended signal as `noisy_start` causes mean cascading
(see Bug 7 above). Always normalize.

### Block noise structure

`sample_block_noise()` generates per-patch correlated noise using:
```
noise = N(0, gamma*I) + mean(patch_noise) * (1 - gamma)
```
where `gamma = 1/3`. This is NOT standard Gaussian noise. It reduces visible block
artifacts at patch boundaries.

### Dynamic shifting

The sigma schedule is shifted based on spatial resolution via `calculate_shift()`:
```
mu = base_shift + (max_shift - base_shift) * (seq_len - base_seq) / (max_seq - base_seq)
```

This `mu` is used in `_time_shift`: `shifted_t = mu * t / (1 + (mu - 1) * t)`.

**Important**: `image_seq_len` must be computed at **pre-upsample** resolution for each
pyramid stage (matching reference). Using post-upsample seq_len gives wrong sigmas.

### Restrict self-attention

Set to `False` (full attention) to match reference behavior. Setting `True` restricts
self-attention to only current-chunk tokens (excluding history), which significantly
degrades quality.

---

## Key Constants & Formulas

### Scheduler defaults

```python
num_train_timesteps = 1000
stages = 3
stage_range = [0, 1/3, 2/3, 1]
gamma = 1/3
base_shift = 0.5
max_shift = 1.15
base_image_seq_len = 256
max_image_seq_len = 4096
```

### `ori_start_sigmas` (per-stage starting signal coefficient)

```python
ori_start_sigmas = {0: 0.999, 1: 0.666, 2: 0.334}
# ori_sigma (signal coeff) = 1 - ori_start_sigmas[i_s]
# Stage 0: ori_sigma = 0.001 (almost pure noise)
# Stage 1: ori_sigma = 0.334
# Stage 2: ori_sigma = 0.666
```

### Alpha/beta blending (stage transitions)

```python
ori_sigma = 1 - scheduler.ori_start_sigmas[i_s]
gamma = 1/3
alpha = 1 / (sqrt(1 + 1/gamma) * (1 - ori_sigma) + ori_sigma)
beta = alpha * (1 - ori_sigma) / sqrt(gamma)
# Note: alpha + beta > 1 (NOT a convex combination) — this is intentional
```

### Time shift formula

```python
def _time_shift(mu, t):
    return mu * t / (1 + (mu - 1) * t)
# WARNING: a common bug is mu*t/(mu+(1-mu)*t) — this is WRONG
```

---

## Diagnostic Recipes

### Check latent channel statistics per step

Add this inside the denoising loop (after `scheduler.step_dmd`):

```python
mx.eval(latents)
ch_means = [latents[c].mean().item() for c in range(min(4, latents.shape[0]))]
ch_stds = [latents[c].std().item() for c in range(min(4, latents.shape[0]))]
print(f"S{i_s} step{idx}: mean={ch_means} std={ch_stds}")
```

**What to look for**:
- Means should not grow unboundedly across stages (if they do, start_point normalization may be off)
- Stds should stay in 0.3–1.0 range (collapsing to <0.1 indicates degenerate output)

### Check pixel-level output quality

```python
import numpy as np, imageio.v3 as iio
vid = iio.imread('/tmp/output.mp4')
for fi in [0, vid.shape[0]//2, vid.shape[0]-1]:
    f = vid[fi]
    print(f'Frame {fi}: R={f[:,:,0].mean():.1f} G={f[:,:,1].mean():.1f} B={f[:,:,2].mean():.1f} '
          f'std=({f[:,:,0].std():.1f},{f[:,:,1].std():.1f},{f[:,:,2].std():.1f})')
```

**Healthy output indicators**:
- RGB means between 60–200 (not pegged to extremes)
- Per-channel std > 15 (indicates spatial diversity, not flat color)
- No single channel dominating (R≈G≈B for neutral scenes)

### Check motion between frames

```python
for a, b in [(0, 16), (16, 32), (32, 33)]:
    diff = np.abs(vid[b].astype(float) - vid[a].astype(float)).mean()
    print(f'Motion {a}→{b}: {diff:.1f}')
```

**Healthy values**: 10–30 for natural motion. >50 suggests major artifacts or scene breaks.
32→33 (chunk boundary) should be <30 for smooth transitions.

### Compare scheduler values with reference

```python
from mlx_video.models.helios.scheduler import HeliosScheduler
s = HeliosScheduler()
for stage in range(3):
    s.set_timesteps(2, stage_index=stage, image_seq_len=540)
    print(f"Stage {stage}: sigmas={s.sigmas.tolist()}, timesteps={s.timesteps.tolist()}")
```

Expected output should match the values in the [Verified Components](#3-scheduler-dmd-) table.

### Run all tests

```bash
.venv2/bin/python3 -m pytest tests/test_helios.py -v
# Expected: 46 passed
```

### Quick generation test

```bash
.venv2/bin/python3 -m mlx_video.generate_helios \
  --model-dir /path/to/Helios-Distilled-MLX \
  --prompt "A golden retriever running on a sunny beach" \
  --num-frames 33 --height 384 --width 640 \
  --output-path /tmp/test.mp4 \
  --pyramid-steps 2 2 2 --seed 42
```

---

## Appendix: Commit History

| Commit | Description |
|--------|-------------|
| `45c20851` | Initial Helios model with 3-stage pyramid denoising |
| `70214cea` | Fix grey output (time_shift formula + VAE key mapping) |
| `fcefee27` | Add CFG support and VAE frame trimming |
| `e61eb33b` | Fix pyramid color distortion (restrict_self_attn, float32 precision, int timestep) |
| `c5acde72` | Fix color bias (normalized start_point + bfloat16 inputs) |
