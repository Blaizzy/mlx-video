# Wan2.2 Architecture Diagrams & Visualizations

## 1. HIGH-LEVEL DATA FLOW

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         GENERATION PIPELINE                                 │
└─────────────────────────────────────────────────────────────────────────────┘

STAGE 1: ENCODING
═══════════════════════════════════════════════════════════════════════════════

  TEXT PROMPT                    NOISE + I2V CONDITION
  "A cat running"    ──────┐    Random [48, T, H, W]  ──────┐
       ↓                    │          ↓                     │
   T5 Encoder            [2]    VAE Encode             [4]
   (24 layers)      ┌──────────→ "Patchify"                │
   float32           │      │     Reshape→Linear Proj      │
       ↓             │      │          ↓                     │
  [512, 4096]       │      └─────[1, L, 5120]              │
       ↓             │                                       │
  Text MLP          │                                       │
  (layer norm)      │                                       │
       ↓             │                                       │
  [2, 512, 5120]    │      TIME EMBEDDING                   │
  (cond+uncond)     │      ════════════════════════════     │
       │             │      Timestep t ∈ [0, 1000]         │
       │             │           ↓                          │
       │             │      Sinusoidal Positional Encoding │
       │             │      + MLP                           │
       │             │           ↓                          │
       │             │      [B, 5120]  (scalar for step)   │
       │             │           ↓                          │
       │             │      Time Projection (×6 scale)     │
       │             └─────→[B, 1, 6, 5120]                │
       │                    (6 modulation vectors per block)
       │
       └─────→ PREPARE K/V CACHES (once, pre-loop)
              For each of 40 blocks, cross-attn:
              k_cache, v_cache = block.cross_attn.prepare_kv(context_emb)


STAGE 2: DENOISING LOOP
═══════════════════════════════════════════════════════════════════════════════

  FOR STEP t IN 40 TIMESTEPS:
  
    SELECT MODEL (dual-model only)
    ═══════════════════════════════
    if t >= 875:
        ├─ USE: HIGH_NOISE_MODEL       (high-noise, timesteps 875-1000)
        ├─ PURPOSE: Coarse structure generation
        └─ GUIDANCE: guide_scale[1]
    else:
        ├─ USE: LOW_NOISE_MODEL        (low-noise, timesteps 0-875)
        ├─ PURPOSE: Fine detail refinement
        └─ GUIDANCE: guide_scale[0]
    
    
    FORWARD PASS
    ═════════════════════════════════
    ┌──────────────────────────────────────────────┐
    │ INPUT: latents [B, L, 5120]                  │
    │        time_emb [B, 1, 6, 5120]              │
    │        context [B, 512, 5120]                │
    └──────────────────────────────────────────────┘
                      ↓
    ┌────────────────────────────────────────────────────┐
    │ FOR i IN 0..39 BLOCKS:                             │
    │                                                    │
    │  ┌────────────────────────────────────────────┐   │
    │  │ MODULATION (6 vectors)                     │   │
    │  │ mod = self.modulation[i] + time_emb        │   │
    │  │ e0,e1,e2,e3,e4,e5 = mod.split()            │   │
    │  └────────────────────────────────────────────┘   │
    │                 ↓                                   │
    │  ┌────────────────────────────────────────────┐   │
    │  │ SELF-ATTENTION (with modulation)           │   │
    │  │                                            │   │
    │  │ x_norm = norm1(x) * (1 + e1) + e0          │   │
    │  │ attn_out = self_attn(x_norm)               │   │
    │  │ x = x + e2 * attn_out     (gated residual) │   │
    │  │                                            │   │
    │  │ Inside self_attn:                          │   │
    │  │  - Q, K, V projections                     │   │
    │  │  - QK normalization (RMS norm)             │   │
    │  │  - Apply 3-way factorized RoPE             │   │
    │  │  - Scaled-dot-product attention            │   │
    │  │  - Output projection                       │   │
    │  └────────────────────────────────────────────┘   │
    │                 ↓                                   │
    │  ┌────────────────────────────────────────────┐   │
    │  │ CROSS-ATTENTION (text conditioning)        │   │
    │  │                                            │   │
    │  │ x_cross = norm3(x) if norm3 else x         │   │
    │  │ cross_out = cross_attn(x_cross, context)   │   │
    │  │ x = x + cross_out                          │   │
    │  │                                            │   │
    │  │ Inside cross_attn:                         │   │
    │  │  - Q from x_cross (use pre-cached K,V)     │   │
    │  │  - K,V from context (from K/V cache!)      │   │
    │  │  - Scaled-dot-product attention            │   │
    │  │  - Output projection                       │   │
    │  └────────────────────────────────────────────┘   │
    │                 ↓                                   │
    │  ┌────────────────────────────────────────────┐   │
    │  │ FFN (Gated Feed-Forward with modulation)   │   │
    │  │                                            │   │
    │  │ x_norm = norm2(x) * (1 + e4) + e3          │   │
    │  │ ffn_out = fc2(GELU(fc1(x_norm)))           │   │
    │  │ x = x + e5 * ffn_out      (gated residual) │   │
    │  │                                            │   │
    │  │ fc1: [5120] → [13824]                      │   │
    │  │ fc2: [13824] → [5120]                      │   │
    │  └────────────────────────────────────────────┘   │
    │                 ↓                                   │
    │     [OPTIONAL] TeaCache/Spectrum decision     │   │
    │     (skip block or use prediction)            │   │
    │                                                    │
    └────────────────────────────────────────────────────┘
                      ↓
    ┌──────────────────────────────────────────────┐
    │ OUTPUT HEAD                                  │
    │ ═════════════════════════════════════════   │
    │ x_out = head(x, time_emb)                    │
    │                                              │
    │ Inside head:                                 │
    │  - x_norm = norm(x)                          │
    │  - Modulation with time_emb                  │
    │  - Linear proj: [5120] → [out_tokens]        │
    │                          ↓                    │
    │                    [B, L, 64]                │
    │                    (16 channels × patch_size)
    └──────────────────────────────────────────────┘
    
    
    CLASSIFIER-FREE GUIDANCE
    ═════════════════════════════════
    if CFG_DISABLED (guide_scale ≤ 1.0):
        noise_pred = output_uncond  # Just uncond, B=1 forward
    else:
        noise_pred_cond = output[0]
        noise_pred_uncond = output[1]
        noise_pred = noise_uncond + guide_scale × (cond - uncond)
    
    
    SCHEDULER STEP
    ═════════════════════════════════
    latents = scheduler.step(noise_pred, timestep, latents)
    
    # Euler: x_next = x + (σ_next - σ_cur) * v
    # DPM++: x_next with 2nd-order correction
    # UniPC: Higher-order solver
    
    
    [OPTIONAL] I2V MASK REAPPLY (TI2V-5B only)
    ═════════════════════════════════════════════
    if is_i2v_mask_blend:
        latents = (1 - mask) * z_img + mask * latents


STAGE 3: DECODING
═══════════════════════════════════════════════════════════════════════════════

  Latents [48, T_lat, H_lat, W_lat]      (output of denoising loop)
       ↓
  Denormalize                             (per-channel mean/std)
       ↓
  Wan2.2 VAE Decoder:
    - CausalConv3d blocks                (per-frame + temporal)
    - Upsampling × 4 (time), × 16 (spatial)
       ↓
  Video Frames [3, T_pix, H_pix, W_pix]
       ↓
  Save MP4
```

---

## 2. SINGLE TRANSFORMER BLOCK INTERNALS

```
┌────────────────────────────────────────────────────────────────────────┐
│                    WANATTENTIONBLOCK FORWARD PASS                      │
└────────────────────────────────────────────────────────────────────────┘

INPUT
══════════════════════════════════════════════════════════════════════════
x           [B, L, 5120]       ← hidden states (latent patch embeddings)
e           [B, 1, 6, 5120]    ← modulation vectors (time-conditioned)
context     [B, 512, 5120]     ← text embeddings (from T5)
cross_kv_cache: (k, v) tuples  ← pre-computed cross-attention caches


STEP 1: COMPUTE MODULATION VECTORS
════════════════════════════════════════════════════════════════════════

  mod = self.modulation[1, 6, 5120] + e[B, 1, 6, 5120]
        ↓
  Broadcast + element-wise add: [1, 6, 5120] → [B, 1, 6, 5120]
        ↓
  Extract 6 vectors:
  ┌─────────────────────────────────────┐
  │ e0 = mod[:, 0, :]  [B, 5120] shift  │
  │ e1 = mod[:, 1, :]  [B, 5120] scale  │
  │ e2 = mod[:, 2, :]  [B, 5120] gate   │
  ├─────────────────────────────────────┤
  │ e3 = mod[:, 3, :]  [B, 5120] shift  │
  │ e4 = mod[:, 4, :]  [B, 5120] scale  │
  │ e5 = mod[:, 5, :]  [B, 5120] gate   │
  └─────────────────────────────────────┘
         ↓
  These control self-attn, cross-attn, and FFN paths


STEP 2: SELF-ATTENTION WITH MODULATION
════════════════════════════════════════════════════════════════════════

  Input residual saved:
  residual1 = x

  Apply affine transformation:
  x_norm = norm1(x)            [B, L, 5120] (layer norm)
  x_norm = x_norm * (1 + e1)   (scale by learned gain)
  x_norm = x_norm + e0         (shift by learned bias)

  Self-attention:
  x_attn = self_attn(x_norm, grid_sizes, freqs, rope_cos_sin, attn_mask)
           ├─ Q = linear(x_norm)
           ├─ K = linear(x_norm)
           ├─ V = linear(x_norm)
           ├─ QK norm (RMS)
           ├─ Apply 3-way RoPE (temporal, height, width)
           ├─ Scaled-dot-product attention
           └─ Output projection
  
  Apply gated residual:
  x = residual1 + e2 * x_attn  (gate controls contribution)


STEP 3: CROSS-ATTENTION (TEXT CONDITIONING)
════════════════════════════════════════════════════════════════════════

  (Optional norm, depends on cross_attn_norm flag)
  x_cross = norm3(x) if norm3 else x

  Cross-attention:
  x_cross_out = cross_attn(x_cross, context, kv_cache=cross_kv_cache)
                ├─ Q = linear(x_cross)  [B, L, 5120]
                ├─ K, V from cache:
                │  (if kv_cache)  use pre-computed [B, 40, 512, 128]
                │  (else)         compute from context
                ├─ Scaled-dot-product attention
                └─ Output projection

  Residual:
  x = x + x_cross_out  (no gating on cross-attn)


STEP 4: FFN WITH MODULATION
════════════════════════════════════════════════════════════════════════

  Input residual saved:
  residual2 = x

  Apply affine transformation:
  x_norm = norm2(x)
  x_norm = x_norm * (1 + e4)   (scale)
  x_norm = x_norm + e3         (shift)

  FFN:
  x_ffn = fc2(GELU(fc1(x_norm)))
          ├─ fc1: [5120] → [13824]  (4× expansion)
          ├─ GELU activation (approx='tanh')
          └─ fc2: [13824] → [5120]  (project back)

  Apply gated residual:
  x = residual2 + e5 * x_ffn  (gate controls contribution)


OUTPUT
══════════════════════════════════════════════════════════════════════════
x  [B, L, 5120]  ← updated hidden states (one block's output = next block's input)
```

---

## 3. SELF-ATTENTION WITH 3-WAY FACTORIZED ROPE

```
┌────────────────────────────────────────────────────────────────────────┐
│              WAN SELF-ATTENTION WITH 3-WAY ROPE                        │
└────────────────────────────────────────────────────────────────────────┘

ROPE PRECOMPUTATION (once per generation)
═════════════════════════════════════════════════════════════════════════

Grid sizes: [F_grid, H_grid, W_grid]  e.g., [3, 10, 20] for (4D, 2D, 2D)
Seq length: L = F_grid * H_grid * W_grid = 3 × 10 × 20 = 600

Freqs split into 3 parts:
  head_dim_half = 64
  d_t = 64 - 2×(64//3) ≈ 38    (temporal gets more capacity)
  d_h = 64 // 3 ≈ 21           (height)
  d_w = 64 // 3 ≈ 21           (width)

For each position (f, h, w) ∈ grid:
  ┌───────────────────────────────────────────┐
  │ Temporal component:   freqs_t[f]          │
  │ Height component:     freqs_h[h]          │
  │ Width component:      freqs_w[w]          │
  └───────────────────────────────────────────┘
  Concatenate: [38 + 21 + 21 = 64] dimensions


SELF-ATTENTION COMPUTATION
═════════════════════════════════════════════════════════════════════════

Inputs:
  x [B, L, 5120]
  grid_sizes = [F_grid, H_grid, W_grid]
  freqs [1024, 64, 2]  (precomputed cos/sin pairs)
  rope_cos_sin (precomputed, optional)
  attn_mask [B, 1, 1, L] (optional, for variable seq lengths)

Step 1: Project to Q, K, V
  Q = linear(x)         [B, L, 5120]
  K = linear(x)         [B, L, 5120]
  V = linear(x)         [B, L, 5120]

Step 2: Apply QK normalization (RMS norm)
  Q = norm_q(Q)         (if qk_norm=True)
  K = norm_k(K)         (if qk_norm=True)

Step 3: Reshape to heads
  Q = Q.reshape(B, L, 40, 128)    [B, 40, L, 128]
  K = K.reshape(B, L, 40, 128)
  V = V.reshape(B, L, 40, 128)

Step 4: Apply 3-way RoPE
  ┌─ Compute cos/sin for each (f, h, w) position
  ├─ Three separate frequency components
  ├─ Interleaved as: [temp freq] + [height freq] + [width freq]
  └─ Rotate Q and K by these frequencies (complex multiplication)
     
     For each head and position:
       cos_t, sin_t = freqs_t[f]    (temporal rotation)
       cos_h, sin_h = freqs_h[h]    (height rotation)
       cos_w, sin_w = freqs_w[w]    (width rotation)
       
       Apply rotation: z' = z * e^(iθ) = cos(θ)z + sin(θ)z_rotated

Step 5: Scaled-dot-product attention
  scale = 1/√(head_dim) = 1/√128 ≈ 0.088
  
  scores = Q @ K^T × scale     [B, 40, L, L]
  
  (Optional) Add attention mask:
  scores[i, :, :, j:] += mask[i, :, :, j:]  (−∞ for masked positions)
  
  attn = softmax(scores, dim=-1)
  out = attn @ V

Step 6: Output projection
  out = out.reshape(B, L, 5120)
  out = linear(out)             [B, L, 5120]

Output: [B, L, 5120] ← gated and summed into residual
```

---

## 4. CROSS-ATTENTION K/V CACHING

```
┌────────────────────────────────────────────────────────────────────────┐
│              CROSS-ATTENTION K/V PRE-COMPUTATION                       │
└────────────────────────────────────────────────────────────────────────┘

NORMAL (UNCACHED) CROSS-ATTENTION
═════════════════════════════════════════════════════════════════════════

For each denoising step:
  Q = linear(x)                 [B, L, 5120]
  K = linear(context)           [B, 512, 5120]  ← EXPENSIVE! (40 times)
  V = linear(context)           [B, 512, 5120]  ← EXPENSIVE! (40 times)
  attn = Q @ K^T
  out = softmax(attn) @ V

Cost: 2 × 40 = 80 linear projections across all blocks and steps


WAN2.2 OPTIMIZATION: PRE-COMPUTE K/V
═════════════════════════════════════════════════════════════════════════

Step 1: Before denoising loop (once per generation)

  context_emb = embed_text([text, neg_text])    [2, 512, 5120]
  
  For each block (40 blocks):
    K = block.cross_attn.linear_k(context_emb)   [2, 512, 5120]
    V = block.cross_attn.linear_v(context_emb)   [2, 512, 5120]
    
    Reshape to attention format:
    K = K.reshape(2, 512, 40_heads, 128).transpose(0, 2, 1, 3)
      = [2, 40, 512, 128]  ← ready for attention
    V = V.reshape(2, 512, 40_heads, 128).transpose(0, 2, 1, 3)
      = [2, 40, 512, 128]
    
    cross_kv_caches[block_idx] = (K, V)

  Total storage: 40 blocks × 2 tensors × [2, 40, 512, 128]
                = 40 × 2 × 2 × 40 × 512 × 128 ≈ 13 million params (cache)


Step 2: During denoising loop (for each step, 40 times)

  For each block:
    K, V = cross_kv_caches[block_idx]  ← retrieve cached
    
    Q = block.cross_attn.linear_q(x)   [B, L, 5120]
    Q = Q.reshape(B, L, 40_heads, 128).transpose(0, 2, 1, 3)
      = [B, 40, L, 128]
    
    attn = Q @ K^T / √128              [B, 40, L, 512]
    out = softmax(attn) @ V            [B, 40, L, 128]
    out = out.transpose(...).reshape(B, L, 5120)
    out = block.cross_attn.linear_out(out)  ← only this projection per step


Benefit:
  OLD: 80 linear projections (K and V, all 40 steps, both models)
  NEW: 2 linear projections (K and V, once before loop)
  
  Speedup factor: 40× on context projection overhead!
  (Text encoder inference is free after T5 embed)
```

---

## 5. TEACACHE vs SPECTRUM vs NO CACHING

```
┌────────────────────────────────────────────────────────────────────────┐
│                   CACHING COMPARISON                                   │
└────────────────────────────────────────────────────────────────────────┘

NO CACHING (Default, if no acceleration)
═══════════════════════════════════════════════════════════════════════════

  FOR EACH STEP (40 iterations):
    RUN ALL 40 TRANSFORMER BLOCKS
    ├─ 40 self-attn forward passes
    ├─ 40 cross-attn forward passes
    ├─ 40 FFN forward passes
    └─ Total: 120 forward passes per step
    
  Total forwards: 40 steps × 120 = 4,800 block forwards


TEACACHE ACCELERATION (2-3x speedup)
═══════════════════════════════════════════════════════════════════════════

  Pre-profile: Train polynomial to map
    rel_l1_distance(time_embedding) → output_distance

  FOR EACH STEP:
    ├─ Compute e0 (projected time embedding)
    ├─ Compare with previous e0:
    │   rel_l1 = ||e0 - prev_e0|| / ||prev_e0||
    ├─ Map through polynomial:
    │   rescaled_distance = poly(rel_l1)
    ├─ Accumulate:
    │   total_distance += rescaled_distance
    │
    ├─ DECISION:
    │   if total_distance < threshold:
    │     SKIP ALL BLOCKS
    │     x = x + cached_residual  ← instant (no transformer)
    │   else:
    │     RUN ALL 40 BLOCKS (as normal)
    │     cached_residual = x_output - x_input
    │     total_distance = 0 (reset)
    │
    └─ Statistics:
       ├─ threshold=0.1 → ~40% of steps skipped → 2x speedup
       ├─ threshold=0.2 → ~60% of steps skipped → 3x speedup
       └─ Higher threshold = more skips = faster but lower quality

  Why it works:
    Time embeddings change smoothly across steps.
    When change is small, relative contribution to output is small.
    Cached residual from previous step is a good approximation!


SPECTRUM ACCELERATION (3.5-5x speedup)
═══════════════════════════════════════════════════════════════════════════

  Warmup (steps 0-4):
    └─ RUN ALL BLOCKS, cache features and timesteps

  Fitting (online, after each compute):
    ├─ Collect: (t_i, h_i) pairs
    ├─ Fit Chebyshev polynomial: h = Σ c_k T_k(τ)
    │  where τ ∈ [-1, 1] normalized timestep
    │  Solve: (X^T X + λI) c = X^T h  (Cholesky-based ridge)
    └─ Storage: coefficients (P=M+1 params per feature)

  Prediction (steps 5+):
    ├─ Use adaptive windowing:
    │   window_size starts at 2
    │   grows by flex_window (0.75) after each compute
    │   Example: 2, 2.75, 3.5, 4.25, 5.0, ...
    │
    ├─ Prediction happens every window_size steps
    │ PREDICT:
    │   h_taylor = h_recent + (step - t_recent) * dh/dt
    │   h_cheb = sum(c_k * T_k(τ_step))
    │   h_pred = (1-w) * h_taylor + w * h_cheb
    │   x = x + h_pred  ← no transformer needed!
    │
    └─ Compute only on boundaries (every ~2-5 steps)

  Results:
    ├─ flex_window=0.75 → ~3.5x speedup (high quality)
    ├─ flex_window=3.0  → ~5x speedup (experimental, lower quality)
    └─ Warmup prevents cold-start artifacts


COMPARISON TABLE
═══════════════════════════════════════════════════════════════════════════

Method        │ Speedup  │ Quality │ Implementation Complexity
──────────────┼──────────┼─────────┼─────────────────────────────
No caching    │ 1.0x    │ Reference
TeaCache      │ 2-3x    │ Good    │ Polynomial threshold (pre-profiled)
Spectrum      │ 3.5-5x  │ Great   │ Chebyshev+Taylor blending
──────────────┴──────────┴─────────┴─────────────────────────────
```

---

## 6. LATENT SPACE PATCHIFICATION

```
┌────────────────────────────────────────────────────────────────────────┐
│                       PATCHIFY OPERATION                               │
└────────────────────────────────────────────────────────────────────────┘

INPUT: Video latent [C, F, H, W]
═════════════════════════════════════════════════════════════════════════

Example: [48, 21, 90, 160]
  C = 48       (latent channel depth, from VAE)
  F = 21       (temporal frames in latent space)
  H = 90       (height in latent space)
  W = 160      (width in latent space)

Patch size = (1, 2, 2)
  pt = 1       (temporal patch size)
  ph = 2       (height patch size)
  pw = 2       (width patch size)


RESHAPE OPERATION
═════════════════════════════════════════════════════════════════════════

Step 1: Compute output grid dimensions
  f_out = F / pt = 21 / 1 = 21
  h_out = H / ph = 90 / 2 = 45
  w_out = W / pw = 160 / 2 = 80

  grid_size = (f_out, h_out, w_out) = (21, 45, 80)
  total_patches = 21 × 45 × 80 = 75,600 patches


Step 2: Reshape to separate patch dimensions
  [48, 21, 90, 160]
  ↓
  Reshape: [48, 21÷1, 1, 90÷2, 2, 160÷2, 2]
         = [48, 21, 1, 45, 2, 80, 2]
  ↓
  Transpose to group: [F', H', W', C, pt, ph, pw]
  [21, 45, 80, 48, 1, 2, 2]


Step 3: Flatten patch dimension
  [21, 45, 80, 48, 1, 2, 2]
  ↓
  Reshape to: [21×45×80, 48×1×2×2]
            = [75600, 192]  (L, C×patch_product)


Step 4: Project through linear layer
  weights: [5120, 192]  (output_dim, input_dim)
  [75600, 192] @ [192, 5120]
  ↓
  [75600, 5120]  (L, dim)


Step 5: Add batch dimension
  [1, 75600, 5120]  (batch_size=1, seq_len=L, dim)

OUTPUT
═════════════════════════════════════════════════════════════════════════

patches: [1, 75600, 5120]
grid_sizes: (21, 45, 80)

Used for:
  ├─ Input to transformer blocks (L = 75,600 tokens)
  ├─ RoPE grid construction
  ├─ Attention masking for variable lengths
  └─ Unpatchify to reconstruct video


UNPATCHIFY (REVERSE OPERATION)
═════════════════════════════════════════════════════════════════════════

Input: [B, L, out_dim * patch_product]
       [1, 75600, 64]  (where 64 = 16 channels × patch_product)

Step 1: Extract sequence length for this batch
  seq_len = grid_size[0] * grid_size[1] * grid_size[2]
          = 21 * 45 * 80 = 75600
  
Step 2: Reshape
  [75600, 64] → [21, 45, 80, 1, 2, 2, 16]
                  (f, h, w, pt, ph, pw, C)

Step 3: Transpose to group channels and patch dims
  [16, 21, 1, 45, 2, 80, 2]  (C, F', pt, H', ph, W', pw)

Step 4: Flatten to reconstruct spatial dimensions
  [16, 21×1, 45×2, 80×2]
  [16, 21, 90, 160]  (C, F, H, W)

Output: [16, 21, 90, 160]  ← back to latent space

This is then decoded through VAE decoder to get pixel-space video.
```

---

## 7. FRAME ALIGNMENT & EXTRA FRAMES

```
┌────────────────────────────────────────────────────────────────────────┐
│                    FRAME HANDLING IN WAN2.2                            │
└────────────────────────────────────────────────────────────────────────┘

USER REQUEST: num_frames=81

VAE STRIDE: (4, 16, 16) for Wan2.2

COMPUTE LATENT FRAMES
═════════════════════════════════════════════════════════════════════════

User requested 81 pixel frames with VAE stride 4 in temporal:
  
  gen_frames = 81
  extra_frames = 4  (VAE stride[0])  ← T2V: add extra for boundary artifacts
  
  total_latent_frames = gen_frames + extra_frames = 85 pixel frames
  t_latent = (total_latent_frames - 1) // 4 + 1
           = (85 - 1) // 4 + 1
           = 84 // 4 + 1
           = 21 + 1 = 22 latent frames

WHY EXTRA FRAMES?
─────────────────
The VAE uses causal convolution (temporal causality).
At boundaries (first/last frame), padding creates artifacts.
Extra frames are generated and later discarded:
  - First 4 extra frames absorbed by VAE padding
  - Original 81 frames remain clean
  - Final 4 frames discarded

For I2V (image-to-video):
  - extra_frames = 0 (reference image provides real first frame)
  - No need for padding


ALIGNMENT REQUIREMENT
═════════════════════════════════════════════════════════════════════════

Dimensions must align to patch_size × vae_stride:
  
  Spatial:
    align_h = patch_size[1] * vae_stride[1] = 2 * 16 = 32
    align_w = patch_size[2] * vae_stride[2] = 2 * 16 = 32
    
    User requests: 720 × 1280
    
    Check:
      720 % 32 = 0 ✓
      1280 % 32 = 0 ✓
    
    If not aligned, auto-round down:
      height_aligned = (720 // 32) * 32 = 22 * 32 = 704
      width_aligned = (1280 // 32) * 32 = 40 * 32 = 1280

  Temporal: (not usually constrained for T2V, constrained for I2V)


EXAMPLE GENERATION
═════════════════════════════════════════════════════════════════════════

User requests: 1280×720, 81 frames, T2V

Alignment check:
  ✓ 1280 % 32 = 0
  ✓ 720 % 32 = 0

Dimension computation:
  vae_stride = (4, 16, 16)
  gen_frames = 81 + 4 = 85
  
  h_latent = 720 / 16 = 45
  w_latent = 1280 / 16 = 80
  t_latent = (85 - 1) / 4 + 1 = 22
  z_dim = 48
  
  target_shape = (48, 22, 45, 80)

Patchification:
  patch_size = (1, 2, 2)
  f_grid = 22 / 1 = 22
  h_grid = 45 / 2 = 22.5 ← ERROR! Not divisible
  
  Actually: h_grid = 45 / 2 = 22 (integer division)
  Remainder frames are padding (handled by attn_mask)

Generation:
  Denoising loop: 40 steps
  → Output shape: [48, 22, 45, 80] latent frames

VAE Decode:
  [48, 22, 45, 80]
  ↓ Upscale spatial × 16: [48, 22, 720, 1280]
  ↓ Upscale temporal × 4: [48, 88, 720, 1280]
  
  (Extra frames included; trim after decode)
  ↓ Final: [3, 81, 720, 1280]  (RGB, 81 frames)
```

---

## 8. MEMORY HIERARCHY

```
┌────────────────────────────────────────────────────────────────────────┐
│                    MEMORY USAGE BREAKDOWN                              │
└────────────────────────────────────────────────────────────────────────┘

Model weights (Wan2.2 T2V 14B × 2 models):
  High-noise model:
    Transformer (40 blocks): ~26 GB (bf16)
    Text embeddings: ~1 MB
  Low-noise model:
    Transformer (40 blocks): ~26 GB (bf16)
    Text embeddings: ~1 MB
  
  With 4-bit quantization: ~6.5 GB per model (80% reduction)


Temporary buffers (per denoising step):
  Input latents: [2, 48, 22, 45, 80] = ~7.5 MB (CFG batch)
  Patches: [2, 75600, 5120] = ~1.5 GB
  Transformer hidden states: [2, 75600, 5120] = ~1.5 GB
  Attention maps: [2, 40 heads, 75600, 75600] = ~21 GB ← peak!
  
  Intermediate buffers are discarded after each step.


Pre-computed caches (reused across all steps):
  Text embeddings (embedded T5 output):
    [2, 512, 5120] per model = ~10 MB
  
  Cross-attn K/V caches (all 40 blocks):
    Per model: 40 blocks × 2 tensors × [2, 40, 512, 128] = ~20 MB
    Total (2 models): ~40 MB
  
  RoPE frequencies:
    [seq_len, 1, head_dim/2] ≈ 3 MB
  
  Total cache: ~50 MB (negligible compared to model weights)


PEAK MEMORY ESTIMATE
═════════════════════════════════════════════════════════════════════════

Scenario: Wan2.2 T2V 14B, dual-model, no quantization

  Model weights (in memory): 2 × 26 GB = 52 GB
  Transformer hidden states: ~1.5 GB
  Attention maps (peak):     ~21 GB
  ─────────────────────────────────────
  Total peak:                ~74.5 GB ← too much for most GPUs!

  Solutions:
    ├─ Quantization (4-bit): 2 × 6.5 GB = 13 GB + temp = ~35 GB ✓
    ├─ Single model (no dual): 26 GB + temp = ~48 GB ✓
    ├─ Spectrum (skip blocks): reduces temp buffers
    └─ Tiling VAE (spatial/temporal): processes in chunks

With 4-bit quantization + Apple Silicon optimizations:
  → Fits in ~8-16 GB systems


MEMORY TIMELINE
═════════════════════════════════════════════════════════════════════════

Phase 1: T5 Encoding
  ├─ Load T5: ~15 GB
  ├─ Encode prompts: ~1 GB temp
  └─ Free T5: Release 15 GB ✓

Phase 2: Model Loading
  ├─ Load transformers: ~13 GB (quantized)
  ├─ Precompute caches: ~50 MB
  └─ Total: ~13 GB

Phase 3: Denoising Loop (per step)
  ├─ Latents: ~7 MB
  ├─ Transformer forward: ~1.5 GB peak
  ├─ Clean temp: Release between steps
  └─ Steady state: ~13 GB

Phase 4: VAE Decode
  ├─ Load VAE: ~2 GB
  ├─ Decode latents: ~2-3 GB
  ├─ Free transformers: Release ~13 GB
  └─ Total: ~4 GB

Phase 5: Video Saving
  ├─ Write MP4: CPU-based, minimal GPU memory
  └─ ~500 MB for frame buffer
```

---

Created: March 6, 2025
Project: mlx-video (Wan2.2 Implementation on MLX)
Filename: WAN22_ARCHITECTURE_DIAGRAMS.md
