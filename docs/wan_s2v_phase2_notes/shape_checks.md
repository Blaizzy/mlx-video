# Wan 2.2 S2V — Phase 2 shape reasoning

Worked example. All shapes verified by hand against the released S2V-14B
configuration (dim=5120, heads=40, layers=40, num_audio_token=4). This document
substitutes for the runtime shape test that cannot be executed in the
sandbox (no MLX / Metal).

## Setup — released S2V-14B

- `dim` = 5120, `head_dim` = 128, `num_heads` = 40
- `patch_size` = (1, 2, 2) → post-patch grid downsamples H,W by 2
- `vae_stride` = (4, 8, 8) → pixel→latent shrinks T by 4, HW by 8
- `audio_dim` = 1024, `num_audio_token` = 4
- `audio_inject_layers` = 12 layers: {0,4,8,12,16,20,24,27,30,33,36,39}
- Ref frame: 1 latent frame
- Framepack `zip_frame_buckets` = (1, 2, 16) → 19 latent motion frames

## 1. Video latent → noise tokens

- Pixel clip: 81 frames at 512×512.
- VAE-encoded latent: `(C=16, F_lat=21, H_lat=64, W_lat=64)`.
- `patchify(1,2,2)` splits the latent into `(F', H', W') = (21, 32, 32)` grid
  → **F_video = 21**, **N_per_frame = 1024**.
- Noise token count `seq_orig = F_video * N_per_frame = 21 × 1024 = 21504`.

## 2. Reference image tokens

- Ref image at 512×512 → VAE latent `(16, 1, 64, 64)`.
- Patchify → `(1, 32, 32)` = **1024 ref tokens**.
- `trainable_cond_mask.segment_embedding(1, 1024)` adds a per-token seg embed.

## 3. Framepack motion tokens (only when previous clip exists)

- Motion latent input: `(C=16, F_motion=19, 64, 64)`.
- Split into 3 buckets in chronological order:
  - Coarsest 16 frames → `proj_4x` kernel `(4, 8, 8)` → grid `(4, 8, 8)` = **256 tokens**
  - Next 2 frames → `proj_2x` kernel `(2, 4, 4)` → grid `(1, 16, 16)` = **256 tokens**
  - Last 1 frame → `proj` kernel `(1, 2, 2)` → grid `(1, 32, 32)` = **1024 tokens**
- Total motion tokens = **1536**.
- `trainable_cond_mask.segment_embedding(2, ...)` per bucket.

## 4. Combined sequence length

- Without motion: `seq_total = seq_orig + 1024 (ref) = 22528`.
- With motion: `22528 + 1536 = 24064`.

## 5. Audio encoder

- Input wav2vec features (interpolated to `T_audio = 4 × F_video = 4 × 21 = 84`):
  shape `(1, 25, 1024, 84)`.
- Weighted layer-sum → `(1, 1024, 84)`.
- Transpose for Conv1d → `(1, 84, 1024)`.

### Local path
- `conv1_local(1024 → 4·1280 = 5120, k=3, s=1)` → `(1, 84, 5120)`.
- Reshape `(1, 84, 4, 1280)` → transpose → `(1, 4, 84, 1280)` → merge batch
  = `(4, 84, 1280)`.
- `conv2(1280 → 2560, s=2)` → `(4, 42, 2560)`.
- `conv3(2560 → 5120, s=2)` → `(4, 21, 5120)`.
- Rearrange → `(1, 21, 4, 5120)`.
- Since `F_video = 21` matches → `local: (B=1, F=21, 4, 5120)` ✓

### Global path
- `conv1_global(1024 → 1280)` → `(1, 84, 1280)`.
- `conv2 → (1, 42, 2560)`.
- `conv3 → (1, 21, 5120)`.
- `final_linear(5120 → 5120)` → `(1, 21, 5120)`.
- Reshape → `global: (1, 21, 1, 5120)` ✓

## 6. Audio injection (per injected block)

Given `hidden_states: (B=1, seq_total=22528, 5120)`:

1. Extract noise slice: `(1, 21504, 5120)`.
2. Reshape → `(1, 21, 1024, 5120)` → `(21, 1024, 5120)` (merge B×F).
3. AdaLN with `audio_emb_global[:, :, 0]`:
   - `temb: (21, 1, 5120)` (broadcast over 1024 tokens)
   - `linear(SiLU(temb)) : (21, 1, 10240)`
   - `(scale, shift) = chunk(2, -1) → each (21, 1, 5120)`
   - `x_out = x * (1 + scale) + shift` → `(21, 1024, 5120)`
4. Cross-attention:
   - Q from `(21, 1024, 5120)` → `(21, heads=40, 1024, 128)`
   - K/V from `audio_emb: (21, 4, 5120)` → `(21, 40, 4, 128)`
   - Attention output → `(21, 1024, 5120)`
5. Reshape → `(1, 21504, 5120)` residual-add → back to `(1, 22528, 5120)`.

## 7. Head + unpatchify

- Head input (noise slice only): `(1, 21504, 5120)`.
- Head Linear `(5120 → 4·16 = 64)` (out_dim × prod(patch_size)):
  → `(1, 21504, 64)`.
- Unpatchify with grid `(21, 32, 32)` → `(16, 21, 64, 64)`.
- Batched: list of 1 tensor of shape `(16, 21, 64, 64)`.

## 8. Sanity check: Tiny test config

Test parameters (`_make_tiny_s2v_config`):
- dim=64, layers=4, heads=4, audio_dim=32
- num_audio_token=4, audio_inject_layers=(0, 3)
- cond_dim=4, in_dim=out_dim=vae_z_dim=4
- Clip: `(C=4, F_vid=5, H=8, W=8)` → patch grid `(5, 4, 4)` → 80 noise tokens.
- T_audio = 4 × 5 = 20 → after stride-4 → 5 frames. ✓
- Local: `(1, 20, 64)` → local conv1 → `(1, 20, 4·16 = 64)` → `(4, 20, 16)`
  → conv2 → `(4, 10, 32)` → conv3 → `(4, 5, 64)` → `(1, 5, 4, 64)`. ✓
- Global: `(1, 20, 16)` → `(1, 10, 32)` → `(1, 5, 64)` → linear → `(1, 5, 1, 64)`. ✓
- Injection at block 0: noise slice `(1, 80, 64)`; audio `(1, 5, 4, 64)`;
  reshape to `(5, 16, 64)`; adaln with `(5, 1, 64)`; cross-attn output
  `(5, 16, 64)`; reshape back to `(1, 80, 64)`. ✓
- Head → `(1, 80, 4·4 = 16)` → unpatchify to `(4, 5, 8, 8)`. ✓

## 9. Known bugs / caveats

- **RoPE for ref/motion tokens**: current implementation does *not* apply
  temporal RoPE for the appended ref/motion tokens (they still get RoPE for
  positions past `seq_orig` since we pass `seq_lens=[seq_total]*B`, but the
  effective grid_sizes still only cover the noise slice). The reference
  places ref at temporal index 30 and motion at negative t. This is
  documented as `TODO(verify)` at the block-loop callsite.
- **Framepack drop_mode="padd"**: zero-pads prefix when motion history is
  shorter than 19 latents. Untested against the reference (see TODO).
- **CondEncoder**: pose-overlay conditioning path is *not* wired into
  forward (only the module exists). Talking-head inference typically
  doesn't need it; enable when adding pose-driven conditioning.
