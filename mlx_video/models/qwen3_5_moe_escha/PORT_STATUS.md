# Escha-W2 → MLX port — Phase 1 status

Read `docs/ESCHA_PORT_FEASIBILITY.md` first for the design audit that motivates
this port. This file tracks Phase 1 implementation progress against the plan in
§6 of that doc.

## Phase 1 checklist

| Step | File(s) | Status | Notes |
|---|---|---|---|
| 1a  Branch + module scaffold                 | this directory                                                 | ✅ done  | branch `escha-mlx-port` |
| 1b  Weight loader                            | `weight_loader.py`, `quantization.py`                          | ✅ done  | drops all-ones s_in/s_out, unknown suffixes fail loud |
| 1c  Hybrid attention (linear-attn + full)    | *(planned)* `attention.py`                                     | ⬜ todo  | reuse mlx-lm Qwen3Next as base |
| 1d  MoE routing + shared expert              | *(planned)* `moe.py`                                           | ⬜ todo  | pattern from mlx-lm mixtral, wrap `PackedScaledExpertLinear` |
| 1e  Codebook extraction (Linux, out-of-band) | `codebooks/extract_codebooks.py` + `.dockerfile`               | ✅ script ready · ⬜ run  | needs a Linux+GPU box, ~1 hr wall |
| 1f  escham_reconstruct + T128 + PackedScaledExpertLinear | `eschamoe.py`, `transform.py`                    | ✅ done (pure MLX)  | correct, slow; refactor for Metal in Phase 2 |
| 1g  End-to-end forward + logits verify       | *(planned)* `tests/models/test_eschamoe.py`                    | ⬜ todo  | dump reference activations from Linux, np.allclose |

## Phase 2 (post-correctness) — see feasibility §6 tier 2

- Metal `escham_reconstruct` kernel (fused decode-GEMV)
- Cached bf16 Hadamard as constant Metal buffer
- Profile + tune to ≥75 tok/s on M4 Max

## Immediate blocking dependency (UPDATED Aug 2026 Route J)

**Codebook extraction succeeded on Modal — but the dequant math is more
complex than classical AQLM.** See `docs/ESCHA_PORT_FEASIBILITY.md §11`.

Escha-W2's `escham_reconstruct` spreads each code across 5 rows × 2 cols of
the output tile (positions vary structurally by K-slice and within-slice
offset), not the 1 row × 16 cols the current `eschamoe.py` assumes. Extracted
`(65536, 32)` fp16 codebooks per K exist at `codebooks/cb_K2.npy` /
`codebooks/cb_K3.npy`, but plugging into current MLX escham_reconstruct
mis-reconstructs (verified: L2 diff 1.16 on first row vs Modal reference).

**Fastest path to a working Mac Escha:** pre-dequant all 80 MoE projections
on Modal (produces ~9 GB bf16), ship a MLX-native fp16 checkpoint that skips
`escham_reconstruct` entirely. See feasibility doc §11d for details.

## What compiles today

```python
from mlx_video.models.qwen3_5_moe_escha import (
    escham_reconstruct, PackedScaledExpertLinear, t128, hadamard_128,
)
# hadamard_128() and t128(x) work standalone right now (no codebook needed).
# escham_reconstruct(...) raises FileNotFoundError until the codebook is present.
```

The full model forward pass is not wired up yet — Phase 1c/d/g finish that.
```
