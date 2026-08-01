# Twitter draft — Escha-W2 MLX (Option B: dequant)

## Main tweet

> World-first Mac port of EschaLabs' novel 2-bit AQLM MoE, Qwen3.6-35B-A3B-Escha-W2, running on MLX.
>
> Escha's Linux-only codebook decoded by pushing identity through the CUDA op on Modal → dense M matrices → 17.79 tok/s greedy on M-series unified memory.
>
> Weights: https://huggingface.co/KaedeTai/Qwen3.6-35B-A3B-Escha-W2-MLX
>
> Experimental — Phase 2 (packed 12 GB, native layout) coming.

## Thread continuation

**2/** The trick: every Escha expert projection is
`y = t128(t128(x·rin) @ escham_reconstruct(code), ·rout)`.
All four ops are linear. So M = f(I). Pushed identity through the real op on Modal A10G, saved 60 GB of bf16 dense weights. Total compute cost: ~$0.30.

**3/** MLX side: naive scatter-gather MoE, per-row int8 attention (fp16-scale, bf16 dequant at load), Qwen3-Next GatedDeltaNet in pure MLX ops. Load 9s, prefill 1.2s, decode 17.79 tok/s, peak 70 GB Metal memory.

**4/** Known: generation degenerates after ~5 steps — attention/GDN path bugs to shake out. Router + shared expert + per-expert dequant math verified against reference. Shipping the plumbing so the community can help debug.

**5/** Phase 2 (in progress): reverse-engineer the packed tile layout so we can ship the native 12 GB variant. Same math, 5x smaller on disk, MLX-native quantized dispatch.

**6/** Full port code: github.com/Blaizzy/mlx-video branch `escha-mlx-port`. PR incoming once Phase 2 lands.

Credit: @EschaLabs for the 2-bit runtime, @modal for the throwaway CUDA on-demand.
