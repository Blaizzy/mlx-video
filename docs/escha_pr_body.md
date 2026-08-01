World-first MLX port of EschaLabs' Qwen3.6-35B-A3B-Escha-W2 — a 35B-total / 3B-active
MoE using a novel 2-bit AQLM residual codebook that today only decodes on Linux+CUDA
via a closed-source `escha` wheel.

## What this PR ships

**Dense dequant path** ("Option B"): pre-compute the effective per-expert matrix
`M` on Modal A10G by pushing the identity through the real Escha op, then load
those bf16 matrices from MLX. All four expert ops (block-128 Hadamard, per-channel
diagonal scales, matmul) are linear, so they compose to a single M we can serve
without `escham_reconstruct`.

- **Modal driver**: `mlx_video/models/qwen3_5_moe_escha/codebooks/modal_dequant_all.py` — 10 min A10G sweep, ~$0.30
- **MLX modules**: `model.py`, `moe.py`, `gated_deltanet.py`, `quantization.py`, `transform.py`, `eschamoe.py` (with new `DequantExpertLinear`), `weight_loader_dequant.py`
- **Smoke script**: `scripts/escha_smoke.py`
- **Published weights (experimental)**: [KaedeTai/Qwen3.6-35B-A3B-Escha-W2-MLX](https://huggingface.co/KaedeTai/Qwen3.6-35B-A3B-Escha-W2-MLX)

## Measured (128 GB M-series unified memory)

| Metric | Value |
|---|---|
| Load | 9–20 s |
| Prefill (1 token) | 1.22 s |
| Greedy decode | 17.79 tok/s |
| Peak Metal memory | 69.87 GB |

## Status: EXPERIMENTAL / DRAFT

Generation currently degenerates into repetitive tokens after ~5 steps —
router, shared expert, and per-expert dequant pass individual verification
against the reference, but the attention / GatedDeltaNet forward has no
Modal cross-check yet and is the likely culprit. Opening as draft so:

- reviewers can see the port shape early
- the community can help track down the semantic bug
- the packed variant (Option A) can land on top when the layout RE succeeds

## Option A (packed codebook) — codebook shipped, one probe short of full inference

The prior "91 h serial A10G" cost estimate for the codebook sweep was wrong.
An op audit revealed:

1. `escham_reconstruct` is EXACTLY linear in codes (superposition |diff|=0 for
   up to 100 slot activations).
2. The codebook is (bi, bj)-invariant — same code at any tile position produces
   the same 16×16 delta.
3. Op accepts leading batch dims on `packed`.

Combined, these let us extract the FULL codebook in **1024 op calls total (~2
minutes A10G, $0.04)** by placing one distinct (bi, bj, k, v) probe per tile-
block (8192 blocks per op).

- **Audit**: `mlx_video/models/qwen3_5_moe_escha/codebooks/modal_op_audit.py`
- **Extractor**: `mlx_video/models/qwen3_5_moe_escha/codebooks/modal_smart_probe.py`
- **Compact codebook artifact**: `mlx_video/models/qwen3_5_moe_escha/codebooks/layout_v2/compact.pkl` (120 MB)
- **MLX decoder**: `mlx_video/models/qwen3_5_moe_escha/eschamoe.py::escham_reconstruct` (rewritten to use `layout_v2/compact.pkl`)
- **Published codebook reference**: [KaedeTai/Qwen3.6-35B-A3B-Escha-W2-Codebook-Ref](https://huggingface.co/KaedeTai/Qwen3.6-35B-A3B-Escha-W2-Codebook-Ref)
- **Full op audit report**: `docs/escha_op_signature.md`

**Verified**: for any isolated `code[bi, bj, k]=v` probe, MLX decoder output
matches the CUDA op up to bf16 rounding (~1e-2 abs).

**Not yet verified**: for a real expert (all 262 K slots active), the current
decoder output plus a naive baseline mis-composes M by ~3 kOhm norm out of
~10 kOhm reference. Extracting the missing baseline term (one more `op(all-
zeros)` dump — 30 seconds A10G) was blocked by the Modal workspace hitting
its spend limit after the codebook extraction completed. See
`docs/ESCHA_LAYOUT_NOTES.md` § "What would unblock full packed inference".

## Files

- `mlx_video/models/qwen3_5_moe_escha/` — new module (~1600 lines MLX code)
- `mlx_video/models/qwen3_5_moe_escha/codebooks/` — Modal drivers + probes + `layout_v2/`
- `scripts/escha_smoke.py`, `scripts/escha_forward_smoke.py`
- `tests/test_escha_*.py` — unit tests (router, weight loader, GDN, reconstruct)
- `docs/ESCHA_LAYOUT_NOTES.md`, `docs/escha_op_signature.md`, `docs/twitter_escha_ab_launch.md`
