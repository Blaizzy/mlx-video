# CRITICAL: S2V-specific weights were missing from converted safetensors

## Root cause

The MLX safetensors under `mlx-models/Wan2.2-S2V-14B-MLX-int4/model.safetensors`
was built by the prior conversion but did NOT contain any of the 165
S2V-specific tensors:

- `casual_audio_encoder.*` (12 tensors: layer weights, 4 conv1d + norm/bias,
  final_linear, padding_tokens)
- `audio_injector.injector.<0..11>.{q,k,v,o,norm_q,norm_k}.*` (144 tensors)
- `frame_packer.{proj,proj_2x,proj_4x}.{weight,bias}` (6 tensors)
- `cond_encoder.{weight,bias}` (2 tensors)
- `trainable_cond_mask.weight` (1 tensor)

Result: `nn.quantize(model, ...)` + `model.load_weights(strict=False)` left
every S2V module with zero-initialised weights. The audio conditioning
produced ZERO effect on the diffusion output — verified by running the model
with the real audio vs `mx.zeros_like(audio)` and seeing byte-identical mp4
output.

This nullified every "Phase 3" lipsync + color-pulse fix earlier in the
`wan-s2v-port` branch because those fixes tuned code paths that had no
weights to run.

## Fix

Downloaded shards 3 and 4 of `Wan-AI/Wan2.2-S2V-14B` (all 165 S2V keys live
in these two shards; DiT weights fill shards 1-2 and part of 3) and merged
their bf16 tensors into the existing MLX safetensors, preserving the
quantised uint32 DiT tensors exactly by using
`safetensors.torch.{load_file,save_file}` instead of `mx.load` /
`mx.save_safetensors` (the MLX pair silently converts uint32 tensors and
zeros out scales/biases).

Merger command (kept for reproducibility):

```
python - <<'PY'
import torch
from safetensors.torch import load_file, save_file
from safetensors import safe_open
from pathlib import Path

MLX_FILE = Path.home() / "mlx-video/mlx-models/Wan2.2-S2V-14B-MLX-int4/model.safetensors"
S2V_PREFIXES = ("audio_injector.", "casual_audio_encoder.", "causal_audio_encoder.",
                "frame_packer.", "cond_encoder.", "trainable_cond_mask.")
mlx_tensors = load_file(str(MLX_FILE))  # preserves torch.uint32 packed quantized weights
s2v_tensors = {}
for shard in ["/path/to/diffusion_pytorch_model-00003-of-00004.safetensors",
              "/path/to/diffusion_pytorch_model-00004-of-00004.safetensors"]:
    with safe_open(shard, framework="pt") as f:
        for k in f.keys():
            if any(k.startswith(p) for p in S2V_PREFIXES):
                new_key = k.replace("causal_audio_encoder.", "casual_audio_encoder.")
                s2v_tensors[new_key] = f.get_tensor(k).to(torch.bfloat16)
merged = dict(mlx_tensors); merged.update(s2v_tensors)
save_file(merged, str(MLX_FILE))
PY
```

## Verification

Before merge: `block[0].self_attn.q.scales` mean_abs = 0.0038, but
`casual_audio_encoder.encoder.conv1_local.conv.weight` mean_abs = 0.0
(missing). After merge (with safetensors library): both non-zero, and
regenerated mp4 shows a real Willy portrait with visible mouth motion
tracking the audio (xcorr peak lag = -1 frame, zero-lag correlation = 0.35,
up from 0.31 in the fake earlier runs where audio had no effect).

## The convert.py bug that caused this

`mlx_video/models/wan_2/convert.py::sanitize_wan_s2v_weights` correctly
handles S2V keys AS LONG AS the input dict actually contains them. The
prior conversion appears to have used a source safetensors file that was
missing the S2V add-ons (either wan2.2_s2v_14B_fp8_scaled.safetensors from
kijai's ComfyUI setup, which packages only the DiT, or a stripped source).
The fix is to re-run the full conversion against a fresh
Wan-AI/Wan2.2-S2V-14B directory download.
