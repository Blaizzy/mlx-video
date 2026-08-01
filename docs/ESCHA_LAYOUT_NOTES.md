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

---

## §3 — SOLVED (2026-08-01): pairwise spatial-overlap interaction

**Findings (from `modal_joint_hypothesis_probes.py` + `modal_pair_grouping_probe.py`):**

The joint decode within one K=1 layer is EXACTLY

```
delta(codes) = sum_k solo_k(codes[k]) + sum_{(i,j) : supp(k=i) ∩ supp(k=j) ≠ ∅} pair_{i,j}(codes[i], codes[j])
```

Rules verified empirically to numerical zero (float rounding):

1. **Cross-K-layer additivity** (control): for k_i ∈ [0..15], k_j ∈ [16..31],
   the pair residual `d_AB - d_A - d_B` is exactly 0 for all tested values.
   → K-layer 1 (slots 0..15) and K-layer 2 (slots 16..31) are independent
   additive residual layers, as originally hypothesized.

2. **Pair non-additivity is fully predicted by spatial support overlap.**
   The 16 slots' spatial-support patterns fall into 4 row-classes (indexed by
   `k % 4`) and 4 col-classes (`k // 4`, plus expansion for `k%4 == 2`). Two
   slots interact iff their supports share at least one (r, c) pixel:

   | k%4 pair | Row overlap? |
   |---|---|
   | (0,1), (0,3), (1,2), (2,3) | Yes (4 rows each) |
   | (0,2), (1,3) | No |

   Combined with the col rule (k_i // 4 == k_j // 4 to share cols, or `k%4=2` cases
   which have extended col support), the exact non-additive pair set within one
   K-layer is 15 pairs (both K-layers share the same structure, translated).

3. **Only pairwise interactions — no 3-way or higher.** Probe P3 confirmed the
   3-way Möbius residual `d_ABC - (d_AB+d_AC+d_BC) + (d_A+d_B+d_C)` is exactly
   zero for four sampled triples including ones with multiple pair overlaps.

4. **Pair-decode structure per interacting pair** (from dense 32×32 sweep of
   pair (0,1)): effective rank ~23 in the unfolded (v0*v1)×spatial matrix;
   per-pixel rank ranges from 1 (non-overlap pixels — depend on only one code)
   to ≤ 14 (overlap pixels — cross term is nontrivial low-rank structure).

**Hypothesis discrimination result:**

- ✅ **H1 (bilinear cross-terms)** — CORRECT for the interaction structure.
  The 3-way residual being zero rules out higher-order terms; the pair-decode
  factors reasonably as solo terms plus low-rank cross terms.
- ❌ H2 (VQ-VI, all 16 codes → one lookup): would produce full-rank joint
  structure, contradicts observed pairwise-only interaction.
- ❌ H3 (16-way tensor decomposition): overspecified; the observed structure is
  strictly pairwise, so a 2-way tensor decomp per overlapping pair suffices.
- ❌ H4 (softmax mixture): codes are int16 discrete indices, not weights, and
  no convex-hull constraints observed.
- ❌ H5 (small dense codebook): incompatible with the per-slot 65536-index
  sweep having distinct outputs at all values.

**Extraction plan (Step 3):**

- For each K-layer, for each slot k ∈ [0..15]: extract solo_k(v) for
  v ∈ [0, 65536) — 16 × 65536 × 16 fp16 sparse tiles. Already done in
  `codebooks/layout_v2/compact.pkl` (120 MB, `smart_probe`).
- For each of 15 interacting pairs (i, j) per K-layer, extract the "cross"
  term as follows: sweep v_i with v_j = fixed non-zero (say v_j = 1) to get
  `cross(v_i, 1)`; sweep v_j with v_i = 1 to get `cross(1, v_j)`; fit low-rank
  factorization per overlap pixel (rank ≤ 4 suffices per per-pixel analysis).
- Cross-term storage: ~4 MB per pair × 15 pairs × 2 K-layers = 120 MB.
- Solo + cross ≈ 240 MB total per (in_f, out_f, K) — well within budget.

**Verify** by running MLX reconstruction and comparing to `op(real_expert_code)`
extracted via Modal on 10 randomly-picked experts across layers 0, 5, 20, 39.
Max abs diff should be < 1e-2 (bf16 rounding tolerance).

Both K variants (K=2 for gate_up, K=3 for down) share the same joint-lookup
structure — only the number of K-layers differs. Extraction script and MLX
implementation cover both K in the same code path.

**Modal cost so far**: ~$0.10 total across the two probes (both finished in
under 1s wall time on A10G).

---

## §3.1 — 2026-08-01 evening: extraction ran, factorisation insufficient

**Full extraction** (`modal_extract_v3.py`, 24 min A10G, ~$0.30):
- Extracted all 32 solo functions for K=2 gate_up + all 48 for K=3 down_proj.
- Extracted cross function samples (rows + cols with 4 ref values each) for
  ALL 15 interacting pairs × K K-layers × 2 variants = 90 cross pairs.
- Saved `full_extract_v4.pkl` (21.7 GB raw, ~1.5 GB compressed npz codebook).

**Factorisation with rank=4 cross-approximation**: `build_codebook_v3.py`
produces `codebook_v3.npz` (1 GB). Reconstruction test
(`test_recon_v3.py`) against a real expert code gives:

- gate_up (K=2): |w_recon|max = 37 vs |w_ref|max = 3.9. Rel err = **180%**.
- down_proj (K=3): |w_recon|max = 67 vs 3.9. Rel err = **301%**.

**Root cause of high error** (`debug_cross_matrix.py`):
- For **most (v0, v1) combinations tested**, cross(v0, v1) ≈ 0. The pair truly
  equals solo_i(v0) + solo_j(v1). Additivity holds.
- Cross is only non-zero at overlap pixels for SPECIFIC combinations of code
  values (e.g. v0=-1 vs any v1 gave cross ∈ {-0.77, 1.58, 0.05, 3.55}).
- Real expert codes contain values across all int16 range (including many
  negatives) — hence non-additivity accumulates to ~5-10 per weight per block.

**Solo structure is much more subtle than "codebook lookup"**
(`debug_recon.py`, `full_extract_v4.pkl` inspection):
- solo_0(v)[pixel (2, 0)] = 0 for v ∈ [1, 10000]; non-zero for v ∈ [32000+]
  and for v with high bit set (negative int16).
- Different v values activate **different sparsity patterns** within a slot's
  union support. This is bit-pattern dependent, not scalar-magnitude dependent.

**Extraction of higher-rank cross factorisation (rank ≥ 16, wider ref set) is
likely required.** Alternatively, the pair op may use a bit-level decomposition
that would be more compactly captured by directly extracting bit-set patterns
per code position.

## §3.2 — Recommended next steps (bounded exploration exhausted)

Given the observed subtlety of the code→pixel mapping (bit-pattern dependent),
and the failure of a naive rank-4 cross-approximation to close the reconstruction
error to acceptable levels, three viable forward paths:

- **Option B (pre-dequantize)** — call the reference op on all 20 480 real
  expert code tensors on Modal, save dense bf16 weights (~35 GB), load
  directly. Guaranteed correctness, ~5-10 min Modal, ~30-60 min download.
- **Option A-2 (bit-decomposition extraction)** — hypothesise each code v =
  b₀·2⁰ + b₁·2¹ + … + b₁₅·2¹⁵ and extract 16 per-bit contribution tiles per
  slot. If the op is linear in v as a bit vector (not as a scalar), this
  captures the full structure with 16 × 16 slot = 256 extractions per K-layer.
- **Option A-3 (denser sampling + rank ≥ 16)** — re-run extraction with 32+
  ref values (mix of powers of 2, sign-bit patterns, and random) and rank
  ≥ 16 factorisation. Storage ≥ 4 GB per variant.

**Decision**: pursue Option B as the highest-probability shipping path.
