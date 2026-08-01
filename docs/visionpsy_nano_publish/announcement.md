# VisionPsy-Nano-460M MLX port — launch materials

## Twitter / X post (final)

> First MLX port of Tether QVAC's VisionPsy-Nano-460M on Apple Silicon.
> Standard: 99 tok/s decode / Flash: 152 tok/s decode / 2.64 GB peak GPU (bf16).
> Load in 0.4s.
> Weights:
> https://huggingface.co/KaedeTai/VisionPsy-Nano-460M-MLX
> https://huggingface.co/KaedeTai/VisionPsy-Nano-460M-Flash-MLX
> Thanks @qvac_ai @tether_to + mlx-community + the Apple MLX team.

(Reserve version, if 280-char over)

> MLX port of @qvac_ai's VisionPsy-Nano-460M is up on HF: 99/152 tok/s (Std/Flash), 2.6 GB peak GPU, bf16, 0.4s load.
> https://huggingface.co/KaedeTai/VisionPsy-Nano-460M-MLX
> Runs natively on Apple Silicon via mlx-video (port branch linked in the model card).

---

## Discussion post for QVAC HF discussions

**Subject:** MLX port of VisionPsy-Nano-460M (Standard + Flash) now available

Hi QVAC / Tether AI Research team,

Thank you for releasing VisionPsy-Nano under Apache 2.0 — it's a lovely piece of engineering. I've published an MLX port that runs natively on Apple Silicon, and wanted to let you know in case it's useful to link from the model card or the announcement blog.

**Repos:**
- Standard: https://huggingface.co/KaedeTai/VisionPsy-Nano-460M-MLX
- Flash:    https://huggingface.co/KaedeTai/VisionPsy-Nano-460M-Flash-MLX

**What the port covers:**
- SigLIP2-base-patch16-512 vision encoder (channels-last conv weights for MLX)
- Pixel-shuffle modality projector (unchanged)
- SmolLM2-360M-Instruct decoder with GQA + RoPE
- Tile-based image preprocessing (2-8 x 2-8 grid) and processor
- Greedy generation loop with an in-memory KV cache

**Numbers on M-series (bf16, greedy, 64 max new tokens, 7 images x 5 prompts = 35 runs per variant):**

| Variant | Avg decode tok/s | Median tok/s | Peak GPU |
|---|---|---|---|
| Standard | 99 | 90 | 2.64 GB |
| Flash    | 152 | 157 | 2.64 GB |

Flash peaks at ~295 tok/s on text-heavy inputs where the pruned visual-token count really pays off.

**Notes for potential upstreaming:**
- The weight repack renames tensors to mlx-vlm conventions (`language_model.*` / `vision_tower.*` / `multi_modal_projector.*`) and nests config into `text_config` / `vision_config`. It's set up so an eventual `mlx_vlm.models.visionpsy_nano` handler would be a drop-in.
- Weights are cast to bf16; outputs verified byte-identical vs fp32 for greedy decode on both variants.
- Source: https://github.com/KaedeTai/mlx-video/tree/visionpsy-mlx-port

Happy to hear if there are corrections, or if you'd like a PR against the QVAC repo to add MLX usage docs.

— KaedeTai

---

## Escha / community relay note

**For Escha or the mlx-community discord channel:**

> MLX port of Tether QVAC's VisionPsy-Nano-460M (Standard + Flash) is now on HF under KaedeTai/. bf16, ~1 GB per checkpoint, 99/152 tok/s decode on M-series, 2.6 GB peak GPU, ~0.4 s load. Layout matches mlx-vlm conventions so a future `visionpsy_nano` handler would slot in cleanly. Full 70-run benchmark matrix + repack scripts under github.com/KaedeTai/mlx-video @ visionpsy-mlx-port.

---

## Metadata

- Publish date: 2026-08-01
- Bench hardware: M-series Apple Silicon (kaede's workstation)
- Bench config: bf16, max_new_tokens=64, greedy, 60 s per-run deadline (none hit)
- Full CSV: `~/movie/visionpsy_bench/results.csv`
- Full Markdown report: `~/movie/visionpsy_bench/results.md`
