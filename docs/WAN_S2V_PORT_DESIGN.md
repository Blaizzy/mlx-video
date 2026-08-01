# Wan 2.2 S2V-14B — MLX Port Design

Status: **Phase 0 + Phase 1** (design + weight-loading skeleton). Phase 2+ (audio
encoder forward, cross-attn forward, motion frames, framepack) is NOT in this
document as executable code — only as file-level plans.

Authoritative PyTorch sources (fetched from `Wan-Video/Wan2.2` @ `main`):

- `wan/modules/s2v/model_s2v.py`   — `WanModel_S2V`, `WanS2VAttentionBlock`, `Head_S2V`
- `wan/modules/s2v/audio_utils.py` — `CausalAudioEncoder`, `AudioInjector_WAN`, `AudioCrossAttention`
- `wan/modules/s2v/auxi_blocks.py` — `MotionEncoder_tc`, `CausalConv1d`
- `wan/modules/s2v/motioner.py`    — `MotionerTransformers`, `FramePackMotioner`
- `wan/modules/s2v/s2v_utils.py`   — `rope_precompute`
- HF config: `Wan-AI/Wan2.2-S2V-14B/config.json`
- HF weight map: `diffusion_pytorch_model.safetensors.index.json` (1260 keys)

The **released S2V-14B checkpoint** uses:

```json
{ "dim": 5120, "num_heads": 40, "num_layers": 40,
  "audio_dim": 1024, "num_audio_token": 4,
  "audio_inject_layers": [0,4,8,12,16,20,24,27,30,33,36,39],
  "cond_dim": 16, "enable_adain": true, "adain_mode": "attn_norm",
  "enable_framepack": true, "enable_motioner": false,
  "framepack_drop_mode": "padd",
  "zero_init": true, "zero_timestep": true,
  "add_last_motion": true, "trainable_token_pos_emb": false }
```

Empirically confirmed from the state dict:
`motioner.*` and `zip_motion_out.*` are **absent** (enable_motioner=false).
`injector_pre_norm_feat`, `injector_pre_norm_vec`, `injector_adain_output_layers`
are **absent** (adain_mode="attn_norm" + need_adain_ont=False).

## 1. Architecture summary

S2V adds **audio conditioning + reference image + motion-frame history** on top
of the standard Wan 2.2 T2V-14B DiT. Audio (a wav2vec2-XLS-R feature stack,
25 layers × 1024 dim) is fused via a lightweight `CausalAudioEncoder` into per-
video-frame tokens; those tokens are then cross-attended into the video latent
at 12 selected transformer blocks (layers 0, 4, 8, 12, 16, 20, 24, 27, 30, 33,
36, 39 — every ~3 layers). The reference image is patch-embedded and appended
to the video-token sequence (RoPE grid shifted to `[30..31]`) so self-attention
sees it. A "framepack" module compresses the previous-clip motion history at
three temporal scales (1, 2, 16 frames) and prepends them to the sequence.

```
        wav2vec2 features                                     text (T5)
        [B, 25, 1024, T_audio]                                [L, 4096]
              │                                                   │
              ▼                                                   │
    ┌───────────────────────┐                                     │
    │ CausalAudioEncoder    │  weighted layer sum → 3 CausalConv1d│
    │  (out = num_frames    │  → adain global vec (b,f,1,dim)     │
    │   × num_token × dim)  │  → local tokens (b,f,4,dim)         │
    └────────────┬──────────┘                                     │
                 │                                                │
                 │             video latent (VAE-encoded)         │
                 │             [C=16, F, H, W]                    │
                 │                     │                          │
                 │                     ▼                          │
                 │       ┌─────────────────────────────┐          │
                 │       │ patch_embed 3D (1×2×2)      │          │
                 │       │ + trainable_cond_mask       │          │
                 │       │ + cond_encoder (pose,       │          │
                 │       │   optional 16-ch overlay)   │          │
                 │       │ + ref image patches (append)│          │
                 │       │ + framepack motion patches  │          │
                 │       │   (append at negative time) │          │
                 │       └────────────────┬────────────┘          │
                 │                        │                       │
                 │                        ▼                       │
                 │                ┌──────────────┐                │
                 │                │ block[0]     │◄───────────────┤ cross_attn (text)
                 │                └──────┬───────┘                │
                 │  ┌──────────────►     ▼        after_transformer_block:
                 │  │      audio_injector.injector[k]
                 │  │       AudioCrossAttention(video, audio_emb)
                 │  │       + AdaLN(temb=audio_emb_global)   ← for injected layers only
                 │  │       inject on layers {0,4,8,12,16,20,24,27,30,33,36,39}
                 │  ▼
                 │        block[1..3]    (no audio inject)
                 │        block[4]       (audio inject k=1)
                 │        …
                 └──►     block[39]      (audio inject k=11)
                                │
                                ▼
                          Head_S2V → unpatchify → epsilon prediction
```

Key differences vs T2V-14B (which mlx-video already supports):
- Extra 165 tensors: `audio_injector.*`, `casual_audio_encoder.*`,
  `frame_packer.*`, `cond_encoder.*`, `trainable_cond_mask.*`
- `WanS2VAttentionBlock` splits modulation into **two segments** by `seg_idx`
  (index separating denoise tokens from ref/motion tokens; ref/motion get the
  zero-timestep modulation vector when `zero_timestep=True`)
- `Head_S2V` accepts per-token `e` (float32) — MLX head already supports this
- Sequence length is variable per clip because of ref+motion append

## 2. Weight mapping table (PyTorch → MLX)

Bucket counts and complete mapping. Sources: `wan_s2v_keys.txt`.

### 2.1 Already handled by `sanitize_wan_transformer_weights`
| PyTorch prefix (T2V-compatible) | MLX prefix | Count | Status |
|---|---|---|---|
| `patch_embedding.{weight,bias}` | `patch_embedding_proj.{weight,bias}` | 2 | [already-mapped] |
| `text_embedding.{0,2}.{weight,bias}` | `text_embedding_{0,1}.{weight,bias}` | 4 | [already-mapped] |
| `time_embedding.{0,2}.{weight,bias}` | `time_embedding_{0,1}.{weight,bias}` | 4 | [already-mapped] |
| `time_projection.1.{weight,bias}` | `time_projection.{weight,bias}` | 2 | [already-mapped] |
| `blocks.<N>.{norm1,norm2,norm3,modulation}` | same | 40×5 | [already-mapped] |
| `blocks.<N>.{self_attn,cross_attn}.{q,k,v,o}.{weight,bias}` | same | 40×16 | [already-mapped] |
| `blocks.<N>.{self_attn,cross_attn}.norm_{q,k}.weight` | same | 40×4 | [already-mapped] |
| `blocks.<N>.ffn.{0,2}.{weight,bias}` | `blocks.<N>.ffn.{fc1,fc2}.{weight,bias}` | 40×4 | [already-mapped] |
| `head.head.{weight,bias}`, `head.modulation` | same | 3 | [already-mapped] |

**Total already handled: 1095 keys**

### 2.2 New S2V mappings (`sanitize_wan_s2v_extra_weights`)

All new modules attach at the **top level** of the MLX `WanS2VModel`, mirroring
the PyTorch names to keep the mapping trivial. Bias/weight names are already
compatible with `nn.Linear` / `nn.Conv1d` in MLX.

| PyTorch key pattern | MLX key pattern | Count | Notes |
|---|---|---|---|
| `audio_injector.injector.<N>.{q,k,v,o}.{weight,bias}` | `audio_injector.injector.<N>.{q,k,v,o}.{weight,bias}` | 12×8 | Cross-attn Q/K/V/O — 4 heads per layer? No — same `dim=5120, num_heads=40` as backbone. Passthrough. |
| `audio_injector.injector.<N>.norm_{q,k}.weight` | same | 12×2 | RMSNorm — [needs-new-mapping] but shape-compat, direct copy. |
| `audio_injector.injector_adain_layers.<N>.linear.{weight,bias}` | `audio_injector.injector_adain_layers.<N>.linear.{weight,bias}` | 12×2 | `diffusers.AdaLayerNorm` — `linear(temb)→(scale,shift)`. Output dim = `2*5120`. |
| `casual_audio_encoder.weights` | `casual_audio_encoder.weights` | 1 | 25-layer wav2vec weighting `[1,25,1,1]`. Verbatim upstream typo `casual` (not `causal`) — keep. |
| `casual_audio_encoder.encoder.conv1_local.conv.{weight,bias}` | same | 2 | Conv1d `1024→(5120/4)*4=5120` ker=3. |
| `casual_audio_encoder.encoder.conv1_global.conv.{weight,bias}` | same | 2 | Conv1d `1024→5120/4=1280` ker=3. |
| `casual_audio_encoder.encoder.conv2.conv.{weight,bias}` | same | 2 | Conv1d `1280→2560` stride=2. |
| `casual_audio_encoder.encoder.conv3.conv.{weight,bias}` | same | 2 | Conv1d `2560→5120` stride=2. |
| `casual_audio_encoder.encoder.final_linear.{weight,bias}` | same | 2 | Linear `5120→5120` (global path only). |
| `casual_audio_encoder.encoder.padding_tokens` | same | 1 | `[1,1,1,5120]` learnable padding. |
| `frame_packer.proj.{weight,bias}` | same | 2 | Conv3d `16→5120` kernel `(1,2,2)`. **Needs weight reshape (Conv3d→Linear)** like `patch_embedding`. |
| `frame_packer.proj_2x.{weight,bias}` | same | 2 | Conv3d `16→5120` kernel `(2,4,4)`. **Needs weight reshape.** |
| `frame_packer.proj_4x.{weight,bias}` | same | 2 | Conv3d `16→5120` kernel `(4,8,8)`. **Needs weight reshape.** |
| `cond_encoder.{weight,bias}` | `cond_encoder_proj.{weight,bias}` | 2 | Conv3d `16→5120` kernel `(1,2,2)`. **Needs weight reshape** like patch_embedding. Rename to match style. |
| `trainable_cond_mask.weight` | `trainable_cond_mask.weight` | 1 | `nn.Embedding(3, 5120)` — 3 mask ids (noise, ref, motion). |

**Total new: 165 keys.** Sum: **1260 keys**, matches the checkpoint index exactly.

### 2.3 Absent-in-released-checkpoint (mark, don't map)
- `motioner.*` (would appear if enable_motioner=true) — [absent] — skip for now.
- `zip_motion_out.*` — [absent] — skip.
- `injector_pre_norm_feat.*`, `injector_pre_norm_vec.*` — [absent] because adain_mode="attn_norm" replaces them.
- `injector_adain_output_layers.*` — [absent] because need_adain_ont=False.
- `token_freqs` — [absent] because trainable_token_pos_emb=False.

If a future S2V variant enables any of these, extend the sanitizer.

## 3. Cross-attention insertion points

12 injection layers: **{0, 4, 8, 12, 16, 20, 24, 27, 30, 33, 36, 39}**.

Insertion happens **after** the standard block (in `after_transformer_block`),
so the block-level API is unchanged. Shape signatures (batch B, video frames F,
video tokens per frame N, dim D=5120):

```
hidden_states_in  : [B, seq_total, 5120]         # seq_total = orig + ref + motion
video_slice       : [B, seq_orig, 5120]          # only first seq_orig tokens
video_frame_view  : [B*F, N, 5120]               # rearranged for per-frame cross-attn
audio_emb         : [B, F, num_token=4, 5120]    # from CausalAudioEncoder
audio_emb_global  : [B, F, 1, 5120]              # for AdaLN
--- inside AudioInjector[k] ---
adain_out = AdaLN(video_frame_view, temb=audio_emb_global[:,:,0])   # [B*F, N, 5120]
attn_out  = AudioCrossAttention(adain_out, context=audio_emb, ctx_lens=[4,...])  # [B*F, N, 5120]
residual  = rearrange(attn_out, '(b f) n c -> b (f n) c')            # [B, seq_orig, 5120]
hidden_states[:, :seq_orig] += residual
```

For Phase 1 (this PR), these ops are **no-op stubs** — the parameters exist and
load, but forward returns input unchanged. Phase 2 implements the actual ops.

## 4. File-by-file changes

### 4.1 `mlx_video/models/wan_2/config.py` — ADD
```python
@classmethod
def wan22_s2v_14b(cls) -> "WanModelConfig":
    return cls(
        model_type="s2v",
        model_version="2.2",
        dim=5120, ffn_dim=13824, num_heads=40, num_layers=40,
        in_dim=16, out_dim=16,
        # S2V-specific
        audio_dim=1024, num_audio_token=4,
        audio_inject_layers=(0,4,8,12,16,20,24,27,30,33,36,39),
        cond_dim=16, enable_adain=True,
        enable_framepack=True, enable_motioner=False,
        zero_timestep=True,
        dual_model=False, boundary=0.0,
        sample_shift=5.0, sample_steps=40,
        sample_guide_scale=(4.0, 4.0),  # placeholder
        max_area=704 * 1280,
    )
```
Add the 8 new fields as `dataclasses.field` defaults so T2V/I2V/TI2V configs
remain unchanged.

### 4.2 `mlx_video/models/wan_2/convert.py` — ADD
- New function `sanitize_wan_s2v_weights(weights)` that first runs the existing
  T2V sanitizer, then handles the 165 new keys.
- New Conv3d→Linear reshapes for `frame_packer.proj*.weight` and
  `cond_encoder.weight`.
- Renames `cond_encoder.{weight,bias}` → `cond_encoder_proj.{weight,bias}`.
- Extend `convert_wan_checkpoint()` to detect `model_type=="s2v"` from source
  `config.json` and skip the dual-model branch.

### 4.3 `mlx_video/models/wan_2/audio_encoder.py` — NEW
- `class CausalConv1d(nn.Module)` — Conv1d with left-pad kernel-1.
- `class MotionEncoder_tc(nn.Module)` — 3 CausalConv1d + norms + optional global path.
- `class CausalAudioEncoder(nn.Module)` — weighted layer sum + MotionEncoder_tc.
- **Phase 1**: `__call__` returns a zero tensor of the expected output shape.
  Parameters are real and shape-compatible with the state dict so
  `model.load_weights` passes strict validation for the S2V-owned keys.

### 4.4 `mlx_video/models/wan_2/s2v_utils.py` — NEW
- `AudioCrossAttention(nn.Module)` — subclass of `WanCrossAttention` for
  audio Q/K/V/O + norm. **Phase 1**: forward returns input unchanged.
- `AdaLayerNorm(nn.Module)` — matches `diffusers.AdaLayerNorm` output shape
  `[..., dim]` from `linear(temb) → (scale, shift)`. **Phase 1**: no-op.
- `AudioInjector(nn.Module)` — holds `injector: list[AudioCrossAttention]`,
  `injector_adain_layers: list[AdaLayerNorm]`, and the `injected_block_id`
  dict `{block_idx: injector_idx}`.
- `FramePackerStub(nn.Module)` — three Conv3d-as-Linear projections
  `proj`, `proj_2x`, `proj_4x` with the right weight shapes. Forward is
  a no-op returning empty lists.

### 4.5 `mlx_video/models/wan_2/wan_2.py` — EXTEND
- New `class WanS2VModel(WanModel)` (or a sibling if inheritance is awkward)
  that in `__init__` additionally instantiates:
  - `self.casual_audio_encoder = CausalAudioEncoder(dim=audio_dim, out_dim=dim, num_token=num_audio_token, need_global=enable_adain)`
  - `self.audio_injector = AudioInjector(dim, num_heads, inject_layer=audio_inject_layers, enable_adain=enable_adain)`
  - `self.trainable_cond_mask = nn.Embedding(3, dim)`
  - `self.cond_encoder_proj = nn.Linear(cond_dim * prod(patch_size), dim)`
  - `self.frame_packer = FramePackerStub(dim=dim, num_heads=num_heads)` (if enable_framepack)
- `__call__` **Phase 1** just raises `NotImplementedError("S2V inference not yet implemented — Phase 2")`.

### 4.6 `mlx_video/models/wan_2/utils.py` — EXTEND
- `load_wan_model()` gains a `model_type` branch: if `config.model_type == "s2v"`, instantiate `WanS2VModel(config)` instead of `WanModel(config)`.

### 4.7 `mlx_video/models/wan_2/generate.py` — EXTEND
- Add `--model-type` CLI flag (default: auto-detect from config).
- If `config.model_type == "s2v"`, call new `generate_s2v_video()` that
  loads the model (must succeed) and then raises
  `NotImplementedError("S2V inference not yet implemented")` — matches spec.

### 4.8 `tests/test_wan_s2v_load.py` — NEW
- Sample a random small subset of S2V weights (synthetic), then instantiate
  `WanS2VModel` and call `load_weights(..., strict=True)`. Assert:
  - Every provided key is consumed.
  - No MLX parameter is left un-loaded (`model.parameters()` matches keys).

Fallback if the real checkpoint is not available: **synthesize** a minimal state
dict with all 1260 key names and correct-per-config shapes from a small config
(`dim=64`, `num_layers=4`, `num_heads=4`, `num_audio_token=4`,
`audio_inject_layers=(0,3)`), and use that. This exercises the mapping code
even without downloading 32 GB.

## 5. Test plan (Phase 2+)

1. **Load smoke test (Phase 1, this PR)**
   - `tests/test_wan_s2v_load.py::test_load_synthetic_s2v_weights` — synth
     state dict → strict `load_weights` → assert no missing/unused keys.
   - `tests/test_wan_s2v_load.py::test_load_real_s2v_weights` (skipped if
     `~/mlx-video/checkpoints/Wan2.2-S2V-14B/` absent).

2. **Silent-audio smoke test (Phase 2)**
   - Feed a zero-valued wav2vec feature tensor `[1, 25, 1024, T]` and a
     single-frame reference image. Expect a valid output tensor of the right
     shape (contents can be garbage — just no NaNs, no shape errors).

3. **Real-audio parity test (Phase 3)**
   - Encode a real 3-second clip with `wav2vec2-large-xlsr-53-english` (the
     checkpoint ships this encoder in the same HF repo), compare a small
     number of denoising steps to the PyTorch reference on identical seeds.
     Allowable delta: `atol=1e-2, rtol=5e-2` for bfloat16.

4. **End-to-end talking-head test (Phase 4)**
   - `python -m mlx_video.generate --pipeline wan_s2v --model-dir … --image portrait.jpg --audio talk.wav --prompt "…"`
   - Manual quality check on M-series hardware.

## 6. Risks / unknowns

| # | Risk | Mitigation |
|---|---|---|
| R1 | Wav2vec2 encoder is a HF transformers checkpoint — we'd need to port it or run it on CPU with `transformers` and pass in features. | Phase 2: use `transformers` on CPU to extract features, keep DiT on MLX. Not blocking Phase 1. |
| R2 | `AudioCrossAttention` may need custom RoPE or none at all. | Confirmed from `audio_utils.py`: it's a plain `WanCrossAttention` — no RoPE on cross-attn K/V (text cross-attn behavior). |
| R3 | Reference-image insertion changes sequence length per step, breaking the current shape-caching in `wan_2.py::__call__`. | Phase 2: write a separate `__call__` in `WanS2VModel` — do NOT reuse T2V fast paths. |
| R4 | `zero_timestep=True` means the ref/motion tokens use a **different** modulation vector than the noise tokens; requires per-token segmented modulation. | Encode as an extra `[e0, seg_idx]` tuple as the reference does; supported by MLX broadcast. |
| R5 | AdaLayerNorm (from diffusers) — need to reimplement (~10 LOC). | Trivial: `x * (1 + scale) + shift` with `scale, shift = linear(temb).chunk(2, dim=-1)`. |
| R6 | Framepack Conv3d has non-`(1,2,2)` kernels (also `(2,4,4)`, `(4,8,8)`) — the patchify-as-linear trick needs generalization. | Straightforward: same reshape recipe, different strides. |
| R7 | The `casual` typo — if upstream fixes it, we break. | Detect both `casual_audio_encoder` and `causal_audio_encoder` in the sanitizer. |
| R8 | `enable_motioner=False` in released checkpoint; but design must not preclude re-enabling for future checkpoints. | `WanS2VModel.__init__` gates `motioner` on `config.enable_motioner`. |
| R9 | GGUF Q5_K_M path is unimplemented for S2V; only fp16 safetensors supported. | Ship fp16 first, GGUF is Phase 5. |
| R10 | `sample_guide_scale` for S2V is unknown; official code shows `guidance_scale=1.0`? | Read from `wan_s2v_pipeline.py` in Phase 2. |

## 7. Estimated wall-clock

| Phase | Work | Est. |
|---|---|---|
| **Phase 1** (this PR) | Design + skeleton + load test | done |
| **Phase 2a** | Real `CausalAudioEncoder.__call__` (MLX Conv1d) | 3–4 h |
| **Phase 2b** | Real `AudioInjector.__call__` (AdaLN + cross-attn + rearrange) | 4–6 h |
| **Phase 2c** | Wire ref-image + framepack tokens into sequence, rope adjustment | 6–8 h |
| **Phase 2d** | Wav2vec2 feature extraction via `transformers` (CPU) | 2 h |
| **Phase 3** | Numerical parity vs PyTorch reference (~5 min per test run) | 4–6 h |
| **Phase 4** | End-to-end MP4 output + CLI polish | 3 h |
| **Total Phase 2** | | **~20–25 h** |

## 8. Acceptance criteria for Phase 1 (this PR)

- [x] Design doc merged
- [ ] `WanS2VModel` instantiates from `WanModelConfig.wan22_s2v_14b()` without error
- [ ] `sanitize_wan_s2v_weights` consumes every key in a synthetic 1260-key state dict
- [ ] `test_wan_s2v_load.py` passes
- [ ] Existing T2V generation regression test passes (no path in Wan T2V changed)
