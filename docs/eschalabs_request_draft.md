# Draft HF discussion post — Escha AQLM codebook structure clarification

Post this at: <https://huggingface.co/EschaLabs/escha-runtime-qwen3moe/discussions>

Subject and body below; edit before posting.

---

**Subject:** RE writeup + one clarifying question on `escham_reconstruct`'s per-slot LUT

Hi Escha team,

We've been working on a native Apple Silicon (MLX) port of
`Qwen3.6-35B-A3B-Escha-W2`. The model loader, MoE routing, RoPE, attention,
Q3-Next GatedDeltaNet, and the fp8/int8 residual paths all pass numerical
parity against a Modal-hosted reference. The blocker is the `escham_reconstruct`
codebook — we've done enough reverse-engineering to be confident in some
structural properties, but have hit a wall on one specific question. Writing
up what we found so far, in case it's useful and to see if you're willing to
sanity-check one hypothesis.

**What we've verified empirically** (Modal A10G probes, `escha==1.0.2+qwen3moe`,
CUDA 12.8, ~$0.42 total spend):

1. **Op is EXACTLY linear across (bi, bj) tile blocks** — superposition
   |diff|=0 across all tile positions. A single op call on a
   `(bi_max, bj_max, k_max)` code tensor is a free test bed for `bi_max × bj_max`
   distinct codebook queries.
2. **K-layers are additive** — across (k in 0..15) vs (k in 16..31) the
   pair residual is exactly zero.
3. **Within one K-layer, slots interact ONLY pairwise, at overlapping spatial
   support** — 3-way Möbius `R_3 = δ_ABC - Σδ_pairs + Σδ_solos = 0` for all
   sampled triples. 15 interacting slot-pairs per K-layer, structure derivable
   from `(row_class = k%4, col_class = k//4)` with the `k%4==2` col-expansion
   rule (details in our notes doc, linked below).
4. **The per-slot LUT `δ_k(v)` for a single `int16 v` is a full 65536-entry
   function that is nonlinear in v's bit representation at every order.**

Point (4) is what's blocking us. We probed:
- Additive-bit hypothesis (v = Σbit_i · 2^i → δ_k(v) = Σ_{bit_i(v)==1} P[k,i]):
  extracted P[k, i] cleanly via single-bit isolation; but 256 random-v tests
  show max_abs_diff ≈ 34.83, rel_l2 ≈ 460%. Predicted magnitude grows linearly
  in bit-count while actual saturates at 3-5.
- 2-way bit Möbius: 120/120 pairs non-zero (max |cross| = 9.70).
- 3-way bit Möbius: 16/16 triples non-zero (max |R_3| = 7.51).
- Symmetry: δ(-v) is neither δ(v) nor -δ(v).

So the LUT really is 65536 distinct outputs per (k, K-layer). The extracted
compact form (`compact.pkl`, 120 MB fp16) empirically reproduces single-slot
op calls to bf16 rounding. Full-density accuracy needs the per-pair
`(v_i, v_j) → cross` term, which our rank-4 factorization did NOT capture
(rel_err 180-300%).

**One question — no pressure to answer:**

Is the per-slot LUT structurally something like:
(a) A scaled Hadamard / DCT of a small learned inner codebook (dim ≪ 65536)?
(b) A lattice quantizer with per-position sign flips (E8 / Leech / product)?
(c) A pair-decomposed structure `(v_hi, v_lo) → LUT_hi[v_hi] + LUT_lo[v_lo]`
    or similar split?
(d) Something else — the runtime just carries the full 65536-entry table?

If (d), we'll ship a Modal-extracted `.safetensors` bundle of the two LUTs
(~2 GB) alongside the port and call it done. If (a)-(c), a one-line hint
would let us compress properly. Happy to open a PR to your repo with an
accessor function once we know the shape.

**What we're NOT asking for:** any proprietary training or optimization
detail, or the CUDA kernel itself. Just the analytic form of the codebook
so downstream non-CUDA ports can compress it correctly.

Cost log, extraction scripts, and the full RE diagnosis:
- Notes: `docs/ESCHA_LAYOUT_NOTES.md` §3, §3.1, §3.3, §3.4 in
  <https://github.com/Blaizzy/mlx-video/tree/escha-mlx-port>
- Probe scripts: `mlx_video/models/qwen3_5_moe_escha/codebooks/modal_*.py`
- Modal artifacts: on the `escha-codebooks` Modal Volume (dense per-slot LUTs
  cb_K{2,3}.npy, compact.pkl, bit_decomp_probe_v1.pkl, bit_mobius_v1.pkl)

Thanks for the runtime — the linearity + slot-pair-only structure is genuinely
elegant, and MLX Mac users are eager for this model.

— KaedeTai
[link to PR #46]

---

**Notes for the poster (delete before posting):**
- Escha's `README.md` says Apache-2.0. Direct disclosure of the LUT structure
  or shape doesn't touch training / optimization internals.
- The maintainer handle is `yzhwang` on HF; auto-subscribes to repo discussions.
- If no reply in ~5 business days, the fallback is shipping the raw LUT bundle
  (~2 GB extracted `.safetensors`) — which is compressed 6x from the naive
  35 GB dequant, but not the "true packed 12 GB" impl we set out to build.
