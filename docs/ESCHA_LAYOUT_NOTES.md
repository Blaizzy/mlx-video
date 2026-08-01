# Escha packed layout — reverse-engineering notes (updated Aug 2026, post-Option-A)

## Summary — where we are now

**Modal audit + smart probe: SUCCESS.** The full (K, k_slot, code_value) →
(row, col, weight_value) codebook has been extracted in a compact form (120 MB,
~1024 A10G op calls total, ~2 min GPU wall time) and verified for isolated
single-code lookups.

**End-to-end packed inference: PARTIALLY WORKING.** MLX `escham_reconstruct`
correctly reproduces the CUDA op's output for any single-slot code (max abs
diff = fp16/bf16 rounding, ~0.01). For a real expert whose codes activate all
262 K slots simultaneously, my reconstruction picks up an unresolved
density-dependent bias equal in norm to the raw delta (~4 kOh). This bias is
NOT expert-independent, contradicting the simple `w_bare = w0 + delta` model
that the linearity audit (`docs/escha_op_signature.md`) predicts.

Root cause candidates (all unverified — Modal workspace hit spend limit before
the follow-up probe could run):
  1. Escha's `escha_t128` implementation differs from the plain-Hadamard
     `t128` in `transform.py` by a scaling / permutation / bias term. If so,
     our compose+invert self-round-trip stays consistent but never reaches
     the reference op's true `w_bare`.
  2. The op has a per-tile bias (function of `(bi, bj)` alone, but non-uniform
     across blocks) that our extraction folded into the "baseline"
     via the `w0 = op(all-zeros)` subtraction.
  3. The op is not truly linear at high code density (superposition test only
     verified for 100 random slots out of 262 K).

Fix path — one more Modal probe would resolve it: dump `op(all-zeros, in_p,
out_p, K)` as an fp16 tensor (in_p × out_p, ~6 MB total for both K values),
plus one `op(real_expert_code)` reference for cross-check. Blocked on Modal
spend limit at 2026-08-01 17:20 CST.

## What we extracted (Modal smart probe — `codebooks/modal_smart_probe.py`)

  Runtime: 46.5 s for K=2 (256 op calls) + ~90 s for K=3 (768 op calls).
  Total ≈ 2 min A10G, ~$0.04.

- `codebooks/layout_v2/compact.pkl` (120 MB): compact sparse codebook.
  Keys `K{2,3}_positions[k]` = (n_nz, 2) int8 row/col positions;
       `K{2,3}_values[k]`    = (65536, n_nz) fp16 codebook values.
- `codebooks/layout_v2/cb_K{2,3}.npy` on the `escha-codebooks` Modal Volume:
  fully dense (k_max, 65536, 16, 16) fp16, ~1 GB + 1.5 GB.

Verified properties (see `docs/escha_op_signature.md`):
  - Op is exactly LINEAR in codes across up to 100 random slot activations
    (superposition |diff|=0).
  - Codebook is (bi, bj)-INVARIANT across all tile positions (up to (bi*16,
    bj*16) offset), including corners like (bi=127, bj=63) vs (bi=0, bj=0).
  - Op supports leading batch dimensions on `packed` — enabling further
    parallelism if needed.

## Structural regularities in the k_slot layout

For K=2 (32 slots per 16×16 block), the (row, col) support of each k_slot is:
  - Row pattern cycles with period 4: {[4,5,11,12,13], [2,3,9,10,11],
    [0,1,8,9,15]+extra, [6,7,13,14,15]}
  - Col pattern: `col_c = k_slot // 4`, cols = `[col_c, col_c + 8]`
    (with an extra col at `k_slot % 4 == 2`)

For K=3 (48 slots), similar cycle with period 6.

These patterns are extractable in ~1 s of GPU time (1 op call per k_slot);
they encode which 8-10 output positions each `cb[k, v]` writes to.

## Bug in the earlier `cb_K2.npy` / `cb_K3.npy` extraction

The prior `modal_extract_v2.py` only captured `d[first_nz_row, :32]` — a single
32-value slice of the first non-zero row of the delta. This missed the other
4-5 rows of each pattern. That's why plugging cb_K2/cb_K3 into `eschamoe.py`
produced L2 diff 1.16 on the first row: the codebook was under-specified by 5x.

The new `compact.pkl` extraction fixes this — it captures every non-zero
position across all 65 K code values.

## Files on disk

- `codebooks/modal_op_audit.py`    — signature + linearity audit (Modal)
- `codebooks/modal_smart_probe.py` — batched 1024-op codebook extraction
- `codebooks/layout_v2/compact.pkl` — 120 MB codebook artifact
- `codebooks/layout_v2/cb_K{2,3}.npy` — dense form (only on Modal Volume)
- `codebooks/extract_baseline.py` — attempt at recovering `w0` from Option B
  M matrices (produces per-expert-varying estimate; see § summary above)
- `eschamoe.py`                    — MLX decoder using the new codebook
- `docs/escha_op_signature.md`     — full op audit report

## What would unblock full packed inference

One more Modal call, ~30 s A10G, cost ~$0.02:

```python
@app.function(gpu="A10G")
def dump_baselines():
    # For each (in_f, out_f, K) shape used by Escha-W2, dump op(all-zeros).
    import torch, escha
    op = torch.ops.escha.escham_reconstruct
    for (in_f, out_f, K, cshape) in [(2048, 1024, 2, (128, 64, 32)),
                                      (512, 2048, 3, (32, 128, 48))]:
        p0 = torch.zeros(cshape, dtype=torch.int16, device="cuda")
        w0 = op(p0, in_f, out_f, K, True, False).cpu().numpy()
        # Also dump op(one_random_expert_code) for cross-check
        code = ...  # load one real expert code
        w_ref = op(code, in_f, out_f, K, True, False).cpu().numpy()
```

Plus: verify the linearity assumption at HIGH code density (dense 262 K slots)
by comparing `op(code)` against `w0 + my_reconstruct(code)`. If they differ
by more than fp16 noise, the op has hidden non-linear structure that would
need one more targeted probe.
