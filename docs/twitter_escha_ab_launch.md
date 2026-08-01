# Twitter draft — Escha-W2 MLX RE (packed impl NOT SHIPPED)

DO NOT POST — user will post manually. Draft only.

**Status update (2026-08-01):** Route B (dequant to 35 GB bf16) is *technically*
working end-to-end but per project mandate is NOT shipped — a 35 GB fp16 model
is not an "Escha port", it's an "Escha-decoded model". The compression is the
whole point.

Route A / A-2 / A-3 (true packed impl at 12 GB) is BLOCKED on the per-slot
LUT nonlinearity finding (see below). Falling back to publishing the full RE
diagnosis rather than shipping half-work.

## Main tweet (RE writeup)

> Reverse-engineering EschaLabs' Escha AQLM MoE runtime for a Mac (MLX) port.
> Bounded exploration exhausted at ~$0.42 total Modal spend. TL;DR: the packed
> codebook op is more structured than any bit-additive scheme captures. Ships
> as an honest RE diagnosis, not a working port.
>
> Full notes: [github.com/Blaizzy/mlx-video PR #46]

## Thread

**2/** What we established (definitive):
- op is EXACTLY linear across (bi, bj) tile blocks (superposition |diff|=0)
- K-layers are strictly additive across layers
- Within a K-layer, slots interact **only pairwise** at overlapping spatial
  supports (3-way Möbius = 0)
- 15 interacting slot-pairs per K-layer, structure derivable from (row%4, col//4)

**3/** What we HOPED (A-2 / bit-decomposition):
- Each per-slot int16 code v would decompose as δ(v) = Σ_{i: bit_i(v)==1} P[k,i]
- If true → 16 patterns × 16 slots × 2 K-layers = ~2 MB per K variant.
  World-first real packed Mac impl at ~12 GB.

**4/** Reality:
- 2-way bit Möbius: 120/120 pairs non-zero (up to |cross|max=9.7)
- 3-way bit Möbius: 16/16 triples non-zero (up to |R3|max=7.5)
- No v/-v symmetry — no hidden odd/even linearity
- Predicted magnitude grows linearly with bit-count; actual saturates at 3-5
- The per-slot function is genuinely nonlinear in v at every order.

**5/** Interpretation: the LUT is likely a scaled lattice / structured
quantizer / Hadamard-of-inner-codebook, not a bit-additive superposition.
The 65536 codes really are 65536 distinct outputs per slot. Compression to
&lt;120 MB would require either upstream disclosure of the analytic form, or
a spectral / lattice-informed probe we don't currently have designed.

**6/** Cost log across all routes: ~$0.42 total Modal A10G. Extraction script
and all probes MIT-licensed at [github.com/Blaizzy/mlx-video branch
`escha-mlx-port`]. Failure report so someone else can build on the diagnosis.

**7/** What'd complete the port: a &lt;1-day EschaLabs disclosure of the
`escham_reconstruct` internal codebook math, or a probe designed against a
specific hypothesized LUT family. Both are natural handoffs.

Credit: @EschaLabs for the runtime + weights; @modal for the throwaway A10G;
@Blaizzy for mlx-video.
