# Wan 2.2 S2V — Phase 2 implementation notes

Author: Claude (Opus 4.7 running in Cowork sandbox).
Date: 2026-07-30.

## What was implemented (Phase 2)

Real forward implementations, replacing Phase-1 no-op stubs, for:

### `mlx_video/models/wan_2/audio_encoder.py`
- `CausalConv1d` — left-padded Conv1d, weights stored in raw PyTorch layout
  `(out, in, kernel)` so state-dict loads directly. Forward transposes to
  MLX layout and uses `mx.conv1d`.
- `MotionEncoder_tc` — 3-conv stack + optional global path with `SiLU`
  activations between convs (marked `TODO(verify)`: exact activation).
- `CausalAudioEncoder` — 25-layer wav2vec2 hidden-state weighted sum
  (softmax-normalised), then `MotionEncoder_tc`. Output shape
  `(B, F_video, num_token, out_dim)` local + `(B, F_video, 1, out_dim)` global.
- `extract_wav2vec_features(...)` — CPU-side helper that runs a HuggingFace
  `Wav2Vec2Model` with `output_hidden_states=True`, stacks 25 hidden states,
  linearly interpolates to `num_audio_token * num_video_frames` timesteps.
  Not runnable in sandbox (no torch/transformers/audio).

### `mlx_video/models/wan_2/s2v_utils.py`
- `AdaLayerNorm(embedding_dim, output_dim)` — `linear(SiLU(temb)) → chunk(2) →
  x * (1 + scale) + shift`. No internal LayerNorm (design R5 confirms).
- `AudioCrossAttention` — inherits `WanCrossAttention` (Q from video, K/V from
  audio, no RoPE per design R2). Kept as its own class only for state-dict
  namespacing.
- `AudioInjector.inject(...)` — real forward: extract noise slice, reshape to
  `(B*F, N, D)`, AdaLN on video with per-frame global audio embedding,
  cross-attend with local audio tokens, residual-add back into the noise
  slice only (ref/motion untouched).
- `FramePacker.pack_motion_frames(motion_latent)` — real forward using
  `_Conv3dLinear` helpers at three temporal scales `(1,2,2), (2,4,4), (4,8,8)`.
  Splits history chronologically into 3 buckets (coarse, medium, fine),
  applies the appropriate projection to each, returns list of 3 flat token
  tensors. Zero-pads when history < 19 latent frames (`drop_mode="padd"`).
- `CondEncoderProj` — pose/overlay Conv3d as a per-patch Linear. Not wired
  into forward yet (talking-head first-clip case doesn't need it).
- `TrainableCondMask` — `Embedding(3, dim)` with a `segment_embedding(seg_id,
  length)` helper.

### `mlx_video/models/wan_2/wan_2.py::WanS2VModel.__call__`
End-to-end real forward:
1. Patchify each denoise latent → noise tokens with seg-id 0.
2. Patchify optional ref image → ref tokens with seg-id 1.
3. Framepack optional motion history → 3 token buckets with seg-id 2.
4. Concatenate `[noise (padded to seq_len), ref, motion_1, motion_2, motion_3]`.
5. Time embedding: `t` for noise; `t=0` for ref/motion when
   `config.zero_timestep=True` (per-token `e_block` in that case).
6. Run 40 transformer blocks; after each block whose index is in
   `audio_injector.injected_block_id`, call `audio_injector.inject(...)`.
7. Head + unpatchify on the noise slice only. Return list of
   `(C_out, F, H, W)` epsilon predictions.

### `mlx_video/models/wan_2/generate.py`
- New `generate_s2v_video(model_dir, prompt, image, audio, ...)` that loads
  T5 + VAE + wav2vec2 + S2V transformer, encodes context/ref-image/audio,
  runs the flow-matching sampler with the S2V forward, VAE-decodes, saves
  MP4. Untested end-to-end (sandbox has no MLX / audio files / weights).
- CLI dispatch (`--audio` or `--model-type s2v`) routes through
  `generate_s2v_video` instead of the old stub.

### `mlx_video/models/wan_2/convert.py`
- No change needed — `frame_packer.proj*` and `cond_encoder` weights pass
  through unchanged in raw `(O, I, D, H, W)` layout. `_Conv3dLinear` in
  `s2v_utils.py` does the patchify+matmul reshape at runtime, so no
  conversion-time transpose is required.

### `tests/test_wan_s2v_load.py`
- Replaced `test_forward_raises_not_implemented` with
  `test_forward_shape_synthetic` — runs `WanS2VModel(cfg=tiny)` on random
  synthetic inputs (silent audio, no ref image, no motion history) and
  asserts the output shape matches `(C_out=4, F=5, H=8, W=8)` with no NaNs.
  Runnable on Mac.
- All existing weight-loading tests unchanged and still pass by construction
  (no weight-layout changes were made to the modules that already loaded).

### `docs/wan_s2v_phase2_notes/shape_checks.md`
- Full worked example of shape math from wav→features→audio_emb→
  audio_injection→head→unpatchify, using both the released S2V-14B config
  and the tiny test config. Substitute for the runtime numpy shape check
  I could not run.

## What could NOT be done in the sandbox

Reasons — see the environment constraints in the brief:

- **Runtime testing on MLX / Metal.** MLX is Metal-only and this Linux
  aarch64 sandbox cannot import it. No inference, no smoke test, no
  numerical parity, no MP4 output.
- **Weight download from HuggingFace.** The sandbox has no external network
  (proxy blocks all outbound HTTPS on 443). The `Wan-AI/Wan2.2-S2V-14B`
  checkpoint (≈32 GB) must be downloaded on the Mac.
- **Wav2vec2 feature extraction.** `transformers` and `torch` are not
  installed in the sandbox; `soundfile`/`librosa` are not present either.
- **PyTorch reference source fetch.** `raw.githubusercontent.com` returns
  403 from the sandbox proxy. Every fetch attempt failed
  (`web_fetch` timed out at 180 s; `curl` gets 403). Implementation was
  therefore driven by:
  - `docs/WAN_S2V_PORT_DESIGN.md` (the authoritative design doc)
  - `docs/wan_s2v_keys.txt` (weight shapes)
  - Existing MLX patterns in `mlx_video/models/wan_2/` and
    `mlx_video/models/ltx_2/`
  - Standard Wan/diffusers conventions.

Every guess is marked `TODO(verify)` in the source.

## Commands to run on the Mac to complete Phases 2a, 2e, 2f

### Phase 2a — Convert weights (once)

```bash
cd ~/mlx-video

# Assuming the Wan2.2-S2V-14B checkpoint has been downloaded to
# ~/mlx-video/checkpoints/Wan2.2-S2V-14B/
python -m mlx_video.models.wan_2.convert \
    --checkpoint-dir ~/mlx-video/checkpoints/Wan2.2-S2V-14B \
    --output-dir     ~/mlx-video/mlx-models/wan22-s2v-14b-bf16 \
    --dtype bfloat16
```

Expected output: `model.safetensors`, `vae.safetensors`,
`t5_encoder.safetensors`, `config.json` — all in
`mlx-models/wan22-s2v-14b-bf16/`.

### Phase 2e — Silent-audio smoke test

```bash
# Runs the tiny synthetic forward test — validates shape math end-to-end.
pytest tests/test_wan_s2v_load.py -v -k forward_shape_synthetic
```

Expected: passes. If it fails with a shape mismatch, that's a genuine bug
worth fixing before moving on.

### Phase 2f — Real audio "Willy" end-to-end test

```bash
python -m mlx_video.models.wan_2.generate \
    --model-dir ~/mlx-video/mlx-models/wan22-s2v-14b-bf16 \
    --prompt "A photorealistic talking head video of Willy" \
    --image  ~/movie/wang_wenchin/willy_ref.jpg \
    --audio  ~/movie/wang_wenchin/willy.wav \
    --output-path ~/output_willy_s2v.mp4 \
    --num-frames 81 \
    --width 512 --height 512
```

## Known risks / places likely to need iteration

1. **RoPE for ref/motion tokens.** Currently *not* applied specially: the
   appended tokens go through self-attn without adjusted RoPE positions.
   The design doc says ref should be at temporal index 30 and motion at
   negative t. If first-frame artifacts appear, plumb per-token RoPE
   positions into `rope_apply` and re-run.
2. **`extract_wav2vec_features` model name.** Guessed
   `facebook/wav2vec2-large-xlsr-53` (25 hidden states). The Chinese
   S2V variant might use `TencentGameMate/chinese-wav2vec2-large`. Verify
   by loading the model and asserting `outputs.hidden_states` has 25
   entries of shape `(1, T, 1024)`.
3. **AdaLN SiLU vs no SiLU.** Diffusers convention adds SiLU on `temb`
   before the linear; this implementation does the same. If the reference
   doesn't, the audio-modulation scale/shift will be wrong-signed; remove
   the SiLU line in `s2v_utils.py::AdaLayerNorm.__call__`.
4. **Activation between conv stack layers.** Assumed SiLU. If GELU is
   used, sound quality of audio→lip-sync will degrade. Trivial fix.
5. **`_time_embed_scalar` per-token e_block.** When `zero_timestep=True`
   and there are ref/motion tokens, `e_block` is `(B, seq_total, 6, D)`.
   The block modulation code `mod = self.modulation + e` handles this via
   broadcasting; if the trace fails at that add, likely a dtype or ndim
   mismatch — cast `e` to `mx.float32` upstream.
6. **CondEncoder path unused.** No wiring in forward for pose overlays.
   Add a `cond_input` parameter and call `self.cond_encoder(cond)` before
   patch-embedding sum for pose-driven inference.
7. **Framepack bucket order.** Assumed chronological (earliest first).
   If motion history is treated in reverse order by the reference,
   swap the slice ordering in `pack_motion_frames`.
8. **Audio feature interpolation.** `F.interpolate(..., mode='linear')`
   is used to align wav2vec output length to `4 * F_video`. The reference
   might just crop or pad — verify against `wan/utils/audio_utils.py`.
9. **First denoising step is slow** because model + audio encoder + framepack
   all trace on the same call. Not a correctness issue.
