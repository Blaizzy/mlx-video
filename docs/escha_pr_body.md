World-first attempt at an MLX port of EschaLabs' Qwen3.6-35B-A3B-Escha-W2 —
a 35B-total / 3B-active MoE using a novel 2-bit AQLM residual codebook that
today only decodes on Linux+CUDA via a closed-source `escha` wheel.

## Status: RESEARCH-COMPLETE, PORT PARKED — awaiting EschaLabs codebook clarification

This branch ships as a **reverse-engineering diagnosis** rather than a working
port. Per project mandate the only acceptable ship state is a true packed
~12 GB Mac impl (Escha's actual compression innovation), OR a detailed
failure report contributing to the community. Since we've hit a well-
characterized wall on the codebook structure, we ship the failure report.

**What was tried, what worked, what did not:**

- Route B (pre-dequant to 35 GB bf16 via identity trick): **technically works
  end-to-end (17.79 tok/s greedy, 69.87 GB Metal peak)** but explicitly NOT
  shipped. A 35 GB decoded model isn't an Escha port — it's an Escha-decoded
  model, and the compression is the whole point.
- Route A / Route J (sparse solo codebook extraction, cross-slot pair
  interactions): 120 MB compact codebook extracted; single-slot lookups match
  the CUDA op to bf16 rounding. Multi-slot reconstruction: rank-4 cross-
  factorisation gave 180-300% rel_err at real-expert code density.
- Route A-2 (bit-decomposition of int16 codes): **falsified**. See §3.3 of
  `docs/ESCHA_LAYOUT_NOTES.md` — per-slot LUT `δ_k(v)` is genuinely nonlinear
  in v's bits at every order (2-way Möbius: 120/120 pairs non-zero; 3-way:
  16/16 triples non-zero; no v/-v symmetry).

## What IS definitive (RE artifacts worth reusing)

- **Op signature**: exactly linear across (bi, bj) tile blocks; K-layers strictly
  additive; slots within one K-layer interact **only pairwise**, at overlapping
  spatial support; 3-way Möbius (slot level) = 0. Full audit in
  `docs/escha_op_signature.md`, further probe results in `docs/ESCHA_LAYOUT_NOTES.md`.
- **Extraction infrastructure**: Modal A10G drivers that extract the full
  per-slot LUT in ~2 min at ~$0.04 (`modal_smart_probe.py`), and the
  bit-decomposition probe / Möbius diagnostics (`modal_bit_decomp_probe.py`,
  `modal_bit_mobius.py`) that rule out any additive-bit hypothesis.
- **Extracted artifacts**: `layout_v2/compact.pkl` (120 MB, single-slot valid),
  dense `cb_K{2,3}.npy` (1.0 GB + 1.5 GB) on the `escha-codebooks` Modal Volume.
- **MLX skeleton**: model loader, MoE routing, RoPE, attention, Qwen3-Next
  GatedDeltaNet, per-row int8 residual paths all pass numerical parity against
  a Modal-hosted reference in isolation.

## What's needed to close the loop (both are cheap)

1. **EschaLabs clarification** of the per-slot LUT structure (see draft in
   `docs/eschalabs_request_draft.md`) — one paragraph would let us compress
   properly and finish the packed port.
2. **A spectrally-informed probe** designed against a specific hypothesized
   LUT family (Hadamard-of-inner-codebook, lattice quantizer, split-index
   pair-decomp). Not attempted here — outside the bounded exploration budget.

## Cost log

Total Modal A10G spend across all Escha RE routes: ~$0.42 of the $10 budget.

## Files

- `mlx_video/models/qwen3_5_moe_escha/` — MLX module (~1600 lines) — kept for
  future port completion; not exercised end-to-end without a working packed
  reconstruct.
- `mlx_video/models/qwen3_5_moe_escha/codebooks/modal_*.py` — 18 Modal drivers
  spanning six RE routes; each is standalone and reproducible.
- `docs/ESCHA_LAYOUT_NOTES.md` — running RE notes, §3.3 has the A-2 failure
  diagnosis and §3.4 the handoff summary.
- `docs/escha_op_signature.md` — op audit (linearity, tile invariance, etc).
- `docs/eschalabs_request_draft.md` — HF discussion post asking upstream for
  the LUT analytic form.
- `docs/twitter_escha_ab_launch.md` — public writeup draft (not sent).

## Marking as DRAFT

Draft rather than ready-for-review because: (a) no working end-to-end packed
inference; (b) Route B dequant port is technically working but explicitly
withheld per project mandate. Ready to land when either EschaLabs replies to
the discussion, or someone contributes a probe that cracks the LUT structure.
