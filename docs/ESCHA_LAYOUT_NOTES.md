# Escha packed layout — reverse-engineering notes (Phase 2 blocker)

## Summary

The packed `escha_code` → dense weight layout used by
`torch.ops.escha.escham_reconstruct` cannot be modelled by a compact bit-permutation
or simple lattice, based on probing on Modal A10G. A full lookup table
extraction is not feasible in a normal iteration window (~15 days of
serial op calls at 1 ms each). Option A ("packed 12 GB MLX port") therefore
remains **blocked** pending either:

- (a) upstream (EschaLabs) exposing a portable decoder, OR
- (b) a bulk sweep in a properly parallelised setup, OR
- (c) linking against the CUDA op via a MLX metal-kernel FFI (out of scope).

The layout probe results are pickled at
`mlx_video/models/qwen3_5_moe_escha/codebooks/layout_map.pkl` (0.87 MB).

## What we probed

For K in {2, 3}, canonical tile `(in_f=128, out_f=128)`, we set exactly one
code slot `code[bi, bj, k_slot] = 1` and recorded the delta pattern in the
`(128, 128)` output vs the all-zero baseline.

- 2048 unique (bi, bj, k_slot) probes for K=2  (8*8*32)
- 3072 unique (bi, bj, k_slot) probes for K=3  (8*8*48)
- A separate value sweep at `(bi=0, bj=0, k_slot=0)` for code values
  {1, 2, 3, 4, 5, 7, 10, ..., 32767, -1, ..., -32768} showing 22 distinct
  outputs.

Runtime: ~10 min total on A10G.

## Key findings

1. **Every probe has a unique row/col pattern**. All 2048 K=2 slots produce
   different `(nz_rows, nz_cols)` tuples — the layout is genuinely dense,
   not derivable from a small permutation table.

2. **Codebook values differ per code integer.** For a fixed slot
   `(bi=0, bj=0, k_slot=0)`, changing the code value 1 → 2 → 3 → ... → -32768
   produces DIFFERENT row patterns and magnitudes. E.g.:
   - `v=1`: 5 rows, 2 cols
   - `v=64`: 8 rows, 2 cols
   - `v=16384`: 4 rows, 2 cols
   - `v=-32768`: 4 rows, 2 cols
   No obvious linear or bit-plane structure.

3. **Delta norm is roughly slot-invariant (~7.5 for K=2, ~3.1–8.1 for K=3)**
   at code=1, suggesting the codebook is roughly normalised per residual.
   K=3 slot 0 has a larger norm than slots 1–2, consistent with the
   residual-quantization scheme (first residual carries the coarsest signal).

4. **Row/col support is sparse per slot.** For a single code perturbation,
   ~2–3 output columns and ~4–8 output rows change out of 128. Aggregated
   across all 2048 slots, coverage of the (128, 128) grid should be dense —
   we did not verify.

## Why a compact rewrite is hard

Escha's `escham_reconstruct` looks like it internally does:

    out[row_pattern(bi, bj, k_slot, v), col_pattern(bi, bj, k_slot, v)]
        += mag(v, k_slot) * some_hadamard_local_pattern

where `row_pattern`, `col_pattern`, and `mag` are all baked into a
compiled `.rodata` table (or the CUDA kernel's constant memory) that is
NOT exposed to Python.

To decode this in MLX we'd need one of:

- The (65536 × K × 16) full codebook table → doable if we can extract it
  (see routes G/H attempts in prior tracks — they returned zero-length or
  wrong-endianness data).
- A closed-form derivation from the observed patterns → seems unlikely
  given each slot has its own pattern.
- A blazingly-parallel Modal sweep: 65k values × 2048 K=2 slots × 3072 K=3
  slots ≈ 328M op calls. At 1 ms/op that's 91 h of A10G time (~$100).
  Parallelised across 20 nodes it becomes ~5 h — still expensive but
  potentially feasible for a follow-up push.

## Recommended follow-ups (in decreasing ROI)

1. Ask EschaLabs directly (HF Discussion) for a portable codebook dump.
2. Contact them re: an ONNX or GGUF export path.
3. Bulk value-sweep at higher parallelism (per-slot, per-value) with all
   65k values → build a 65536 × 3072 delta table per K and lookup.
4. Attach a Metal kernel via `mx.fast.metal_kernel` that mirrors the CUDA
   sequence, if we can extract the .cubin.

## Data in this repo

- `mlx_video/models/qwen3_5_moe_escha/codebooks/layout_map.pkl` — the
  probe results dictionary. Keys `K2`, `K3`, `value_sweep`.
- `mlx_video/models/qwen3_5_moe_escha/codebooks/modal_layout_probe.py` —
  the Modal driver used to produce the above.
