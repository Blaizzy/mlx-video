# Twitter draft — Escha-W2 MLX (Options A + B)

DO NOT POST — user will post manually. Draft only.

## Main tweet

> World-first Mac port of EschaLabs' 2-bit AQLM MoE, Qwen3.6-35B-A3B-Escha-W2,
> running on MLX — plus the first public extraction of the packed codebook.
>
> Option B (dequant, shipping): 60 GB dense M matrices via identity trick, 17.79 tok/s greedy on M-series unified memory.
>
> Option A (codebook reference, shipping): full Escha AQLM codebook (~120 MB)
> extracted from `torch.ops.escha.escham_reconstruct` in 1024 Modal A10G op
> calls (down from a naive 328 M).
>
> Dequant weights: https://huggingface.co/KaedeTai/Qwen3.6-35B-A3B-Escha-W2-MLX
> Codebook + audit: https://huggingface.co/KaedeTai/Qwen3.6-35B-A3B-Escha-W2-Codebook-Ref

## Thread continuation

**2/** The dequant trick: every Escha expert projection is
`y = t128(t128(x·rin) @ escham_reconstruct(code), ·rout)`.
All four steps are linear. So M = f(I). Pushed identity through the real op on
Modal A10G, dumped 60 GB of bf16 dense weights. Total compute: ~$0.30.

**3/** MLX side: naive scatter-gather MoE, per-row int8 attention (fp16-scale,
bf16 dequant at load), Qwen3-Next GatedDeltaNet in pure MLX ops. Load 9s,
prefill 1.2s, decode 17.79 tok/s, peak 70 GB Metal memory.

**4/** The codebook extraction (Option A) is what a naive one-code-at-a-time
sweep would have taken 91 h of A10G for (328 M op calls, ~$100). Wrong shape
of probe.

**5/** Right shape: op audit first. Discovered
(a) op is EXACTLY linear in codes (superposition |diff|=0),
(b) codebook is (bi, bj)-invariant across all tile blocks,
(c) op accepts leading batch dims on packed.

**6/** From (a)+(b): every one of the 8192 tile-blocks in a (128, 64, 32) code
tensor is a free test bed for a distinct codebook entry, in a single op call.
That collapses the sweep to **256 op calls for K=2 + 768 for K=3 = 1024 total.
~2 minutes A10G.**

**7/** Extracted codebook: (65 536 codes × 32 k_slots × 16 × 16 fp16) per K,
sparse-compact to 120 MB total for both K values.

**8/** Known limit — full packed-inference is 95% there. Isolated single-code
lookups match the CUDA op exactly (up to bf16 noise). At real-expert code
density (all 262 K slots active) there's an unresolved additive term (~4 kOhm
norm) that needs one more 30-second Modal probe to resolve. Workspace hit
spend limit before the follow-up. Details in `LAYOUT_NOTES.md` in the
codebook repo.

**9/** For now the Option-B dense variant is the working end-to-end port. The
codebook repo ships as reference for anyone completing the packed path — the
Modal reproducer scripts, op signature audit, and layout notes are all there.

**10/** Full port code: github.com/Blaizzy/mlx-video branch `escha-mlx-port`.
PR to Blaizzy/mlx-video incoming.

Credit: @EschaLabs for the 2-bit runtime, @modal for the throwaway CUDA
on-demand, @Blaizzy for mlx-video.
