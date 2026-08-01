"""Benchmark harness: (images) x (prompts) x (variants) for VisionPsy-Nano MLX.

Loads each variant once and reuses it across all image/prompt combos.
Writes results.csv + results.md into ~/movie/visionpsy_bench/.
"""
from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
from PIL import Image

sys.path.insert(0, str(Path(os.path.expanduser("~/mlx-video"))))

from mlx_video.models.visionpsy_nano import load_visionpsy_nano  # noqa: E402
from mlx_video.models.visionpsy_nano.processor import load_processor  # noqa: E402


BENCH_ROOT = Path(os.path.expanduser("~/movie/visionpsy_bench"))
BENCH_ROOT.mkdir(parents=True, exist_ok=True)
SYNTH = BENCH_ROOT / "synth"

IMAGES = [
    ("portrait",   os.path.expanduser("~/movie/wang_wenchin/output/willy_portrait.png")),
    ("anime_grid", os.path.expanduser("~/movie/kana/character_sheet/kana_character_sheet.png")),
    ("neon_text",  os.path.expanduser("~/movie/flux2_mlx_smoke/text_768_helloKaede.png")),
    ("screenshot", str(SYNTH / "screenshot.png")),
    ("chart",      str(SYNTH / "chart_revenue.png")),
    ("receipt",    str(SYNTH / "receipt.png")),
    ("diagram",    str(SYNTH / "diagram_flow.png")),
]

PROMPTS = [
    ("describe_en", "Describe this image in one sentence."),
    ("text_en",     "What text appears in this image?"),
    ("count_en",    "Count the objects/people in this image."),
    ("subject_en",  "What is the main subject?"),
    ("describe_zh", "用一句話描述這張圖片。"),
]

VARIANTS = {
    "standard": os.path.expanduser("~/models/VisionPsy-Nano-460M"),
    "flash":    os.path.expanduser("~/models/VisionPsy-Nano-460M-Flash"),
}


def _peak_gpu_gb() -> float:
    try:
        return mx.get_peak_memory() / (1024 ** 3)
    except AttributeError:
        return 0.0


def _reset_peak() -> None:
    try:
        mx.reset_peak_memory()
    except AttributeError:
        pass


def bench_one(model, cfg, proc, image_path: Path, prompt: str, max_new_tokens: int,
              deadline_s: float) -> dict:
    image = Image.open(image_path).convert("RGB")
    t_pre = time.perf_counter()
    batch = proc(prompt, image=image)
    proc_s = time.perf_counter() - t_pre

    n_prompt_tokens = int(batch["input_ids"].shape[-1])
    n_tiles = int(batch["pixel_values"].shape[0])

    _reset_peak()
    eos_id = getattr(proc.tokenizer, "eos_token_id", None)

    # Prefill
    t0 = time.perf_counter()
    logits = model(
        batch["input_ids"],
        pixel_values=batch["pixel_values"],
        image_token_id=batch["image_token_id"],
    )
    mx.eval(logits)
    prefill_s = time.perf_counter() - t0

    # Decode
    tokens = []
    t_dec = time.perf_counter()
    stopped_early = False
    for tok in model.generate(
        batch["input_ids"],
        pixel_values=batch["pixel_values"],
        image_token_id=batch["image_token_id"],
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_id,
    ):
        tokens.append(tok)
        if (time.perf_counter() - t_dec) > deadline_s:
            stopped_early = True
            break
    decode_s = time.perf_counter() - t_dec

    text = proc.decode(tokens, skip_special_tokens=True)
    peak = _peak_gpu_gb()

    return {
        "n_prompt_tokens": n_prompt_tokens,
        "n_tiles":         n_tiles,
        "proc_s":          proc_s,
        "prefill_s":       prefill_s,
        "decode_s":        decode_s,
        "n_decode_tokens": len(tokens),
        "decode_tok_s":    len(tokens) / max(decode_s, 1e-6),
        "peak_gpu_gb":     peak,
        "output":          text,
        "stopped_early":   stopped_early,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--deadline-s", type=float, default=120.0, help="hard per-run decode cap")
    ap.add_argument("--variants", default="standard,flash")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--csv", default=str(BENCH_ROOT / "results.csv"))
    ap.add_argument("--md",  default=str(BENCH_ROOT / "results.md"))
    args = ap.parse_args()

    dtype = {"bf16": mx.bfloat16, "fp16": mx.float16, "fp32": mx.float32}[args.dtype]
    rows = []

    for vname in args.variants.split(","):
        vname = vname.strip()
        model_dir = Path(VARIANTS[vname])
        if not model_dir.exists():
            print(f"skip {vname}: {model_dir} missing", file=sys.stderr)
            continue
        print(f"\n=== loading variant {vname} from {model_dir} ===", flush=True)
        t0 = time.perf_counter()
        model, cfg = load_visionpsy_nano(model_dir, dtype=dtype)
        proc = load_processor(model_dir, cfg=cfg)
        load_s = time.perf_counter() - t0
        print(f"  loaded in {load_s:.2f}s  variant={cfg.variant}", flush=True)

        for img_name, img_path in IMAGES:
            if not os.path.exists(img_path):
                print(f"  SKIP image {img_name}: not found ({img_path})", flush=True)
                continue
            for prompt_name, prompt in PROMPTS:
                t_run = time.perf_counter()
                try:
                    r = bench_one(model, cfg, proc, Path(img_path), prompt,
                                  args.max_new_tokens, args.deadline_s)
                    row = {
                        "variant":     vname,
                        "image":       img_name,
                        "prompt_id":   prompt_name,
                        "prompt":      prompt,
                        "n_prompt_tokens": r["n_prompt_tokens"],
                        "n_tiles":         r["n_tiles"],
                        "proc_s":          round(r["proc_s"], 3),
                        "prefill_s":       round(r["prefill_s"], 3),
                        "decode_s":        round(r["decode_s"], 3),
                        "n_decode_tokens": r["n_decode_tokens"],
                        "decode_tok_s":    round(r["decode_tok_s"], 2),
                        "peak_gpu_gb":     round(r["peak_gpu_gb"], 3),
                        "stopped_early":   int(r["stopped_early"]),
                        "output":          r["output"].replace("\n", " ").strip()[:400],
                    }
                    rows.append(row)
                    print(
                        f"  [{vname} {img_name:10s} {prompt_name:11s}] "
                        f"{row['n_decode_tokens']:3d} tok  {row['decode_tok_s']:6.1f} tok/s  "
                        f"peak={row['peak_gpu_gb']:.2f} GB  {time.perf_counter() - t_run:.1f}s",
                        flush=True,
                    )
                except Exception as e:
                    print(f"  FAIL [{vname} {img_name} {prompt_name}]: {e!r}", flush=True)
                    rows.append({
                        "variant": vname, "image": img_name, "prompt_id": prompt_name,
                        "prompt": prompt, "n_prompt_tokens": -1, "n_tiles": -1,
                        "proc_s": -1, "prefill_s": -1, "decode_s": -1,
                        "n_decode_tokens": -1, "decode_tok_s": -1, "peak_gpu_gb": -1,
                        "stopped_early": -1, "output": f"ERROR: {e!r}",
                    })
        # Teardown between variants.
        del model, proc, cfg
        gc.collect()

    # ---------- write CSV
    if rows:
        fieldnames = list(rows[0].keys())
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nwrote {len(rows)} rows to {args.csv}", flush=True)

    # ---------- write Markdown, grouped by image
    md_lines = [
        "# VisionPsy-Nano-460M MLX benchmark",
        "",
        f"- Config: dtype={args.dtype}, max_new_tokens={args.max_new_tokens}",
        f"- Variants: {args.variants}",
        f"- Images: {len(IMAGES)}, Prompts: {len(PROMPTS)}",
        "",
    ]

    # Summary table first
    md_lines += [
        "## Summary: avg decode tok/s and peak GPU per variant",
        "",
        "| Variant | Avg decode tok/s | Median decode tok/s | Avg peak GPU (GB) | Runs |",
        "|---|---|---|---|---|",
    ]
    from statistics import mean, median
    for vname in args.variants.split(","):
        vname = vname.strip()
        vr = [r for r in rows if r["variant"] == vname and r["decode_tok_s"] > 0]
        if not vr:
            continue
        md_lines.append(
            f"| {vname} | {mean(r['decode_tok_s'] for r in vr):.1f} | "
            f"{median(r['decode_tok_s'] for r in vr):.1f} | "
            f"{mean(r['peak_gpu_gb'] for r in vr):.2f} | {len(vr)} |"
        )
    md_lines.append("")

    # Per-image tables
    for img_name, _ in IMAGES:
        img_rows = [r for r in rows if r["image"] == img_name]
        if not img_rows:
            continue
        md_lines += [
            f"## Image: {img_name}",
            "",
            "| Variant | Prompt | Prompt tok | Tiles | Prefill s | Decode tok | Decode tok/s | Peak GPU GB | Output |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for r in img_rows:
            out_esc = r["output"].replace("|", "\\|")
            md_lines.append(
                f"| {r['variant']} | {r['prompt_id']} | {r['n_prompt_tokens']} | "
                f"{r['n_tiles']} | {r['prefill_s']} | {r['n_decode_tokens']} | "
                f"{r['decode_tok_s']} | {r['peak_gpu_gb']} | {out_esc} |"
            )
        md_lines.append("")

    with open(args.md, "w") as f:
        f.write("\n".join(md_lines))
    print(f"wrote {args.md}")


if __name__ == "__main__":
    main()
