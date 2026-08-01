# Escha-W2 tensor enumeration (per-parent group)
Source: `/Users/kaede/models/Qwen3.6-35B-A3B-Escha-W2`  
Total tensors: 1362  
Distinct suffix patterns: 4

## Pattern 1: n=318 groups, suffixes=['weight']

Example: `model.language_model.layers.0.input_layernorm`

| suffix | dtype | example shape |
|---|---|---|
| `weight` | `float16` | `(2048,)` |

## Pattern 2: n=252 groups, suffixes=['weight_int8', 'weight_scale']

Example: `lm_head`

| suffix | dtype | example shape |
|---|---|---|
| `weight_int8` | `int8` | `(248320, 2048)` |
| `weight_scale` | `float16` | `(248320,)` |

## Pattern 3: n=80 groups, suffixes=['escha_code', 'escha_config', 'escha_rin', 'escha_rout', 'escha_s_in', 'escha_s_out']

Example: `model.language_model.layers.0.mlp.experts.down_proj`

| suffix | dtype | example shape |
|---|---|---|
| `escha_code` | `int16` | `(256, 32, 128, 48)` |
| `escha_config` | `int32` | `(9,)` |
| `escha_rin` | `float16` | `(256, 512)` |
| `escha_rout` | `float16` | `(256, 2048)` |
| `escha_s_in` | `float32` | `(256, 512)` |
| `escha_s_out` | `float32` | `(256, 2048)` |

## Pattern 4: n=30 groups, suffixes=['A_log', 'dt_bias']

Example: `model.language_model.layers.0.linear_attn`

| suffix | dtype | example shape |
|---|---|---|
| `A_log` | `float16` | `(32,)` |
| `dt_bias` | `float16` | `(32,)` |

## Suffix counts (all layers)

- `weight` × 318
- `weight_int8` × 252
- `weight_scale` × 252
- `escha_s_in` × 80
- `escha_s_out` × 80
- `escha_config` × 80
- `escha_rin` × 80
- `escha_rout` × 80
- `escha_code` × 80
- `A_log` × 30
- `dt_bias` × 30

## Findings

The Escha-W2 safetensors contain EXACTLY these per-projection tensors for the 80 quantized MoE projections (40 layers × 2 projections: `gate_up_proj` + `down_proj`):

- `escha_code` — **int16** packed codes
- `escha_rin` — **float16** pre-transform (T128) scale on input dim
- `escha_rout` — **float16** post-transform (T128) scale on output dim
- `escha_s_in`, `escha_s_out` — **float32** all-ones outer scales (dropped at load)
- `escha_config` — **int32[9]** metadata `[block=16, K, V=2, cbA_id=1, E=256, in_f, out_f, in_p, out_p]`

**No** `packed_codes` / `scale` / `transform_left` / `transform_right` / `a1` / `a2` tensors. The task description was speculative — the actual escha wheel API (see `escha/gptoss_experts.py`) uses (`code`, `rin`, `rout`, `s_in`, `s_out`) as documented in `docs/ESCHA_PORT_FEASIBILITY.md §1a`.

The C op `torch.ops.escha.escham_reconstruct(code, in_features, out_features, K, cbA, mul1) -> Tensor` decodes `code` alone — the codebook lattice (`cbA_id=1`, K=2/3) is baked into `escha/_C.…so` as compile-time `.nv.constant0` data. **The codebook is NOT a per-weight safetensors tensor.** Route I's premise was wrong; the codebook lives in the .so and must be extracted (or its generator formula reverse-engineered) once.
