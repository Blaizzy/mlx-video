"""Smoke test for the VisionPsy-Nano MLX port.

Runs a short generation on both the Standard and Flash variants for a given
image + prompt, reporting tok/s and peak RAM.

Usage:
    python scripts/visionpsy_nano_smoke.py \
        --image docs/wan_s2v_phase3_notes/willy_portrait.png \
        --prompt "Describe this image in one sentence."
"""
from __future__ import annotations

import argparse
import gc
import os
import resource
import time
from pathlib import Path

import mlx.core as mx
from PIL import Image


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3)


def peak_gpu_gb() -> float:
    try:
        return mx.get_peak_memory() / (1024 ** 3)
    except AttributeError:
        return 0.0


def run(model_dir: Path, image_path: Path, prompt: str, max_new_tokens: int, dtype_str: str):
    from mlx_video.models.visionpsy_nano import load_visionpsy_nano
    from mlx_video.models.visionpsy_nano.processor import load_processor

    dtype = {"bf16": mx.bfloat16, "fp16": mx.float16, "fp32": mx.float32}[dtype_str]

    print(f"\n=== {model_dir.name} ({dtype_str}) ===")
    t0 = time.perf_counter()
    model, cfg = load_visionpsy_nano(model_dir, dtype=dtype)
    proc = load_processor(model_dir, cfg=cfg)
    print(f"load: {time.perf_counter() - t0:.2f}s   variant={cfg.variant}")

    image = Image.open(image_path).convert("RGB")
    batch = proc(prompt, image=image)

    n_prompt_tokens = int(batch["input_ids"].shape[-1])
    n_tiles = int(batch["pixel_values"].shape[0])
    print(
        f"tiles={n_tiles} grid={batch['grid']} has_global={batch['has_global']} "
        f"prompt_tokens={n_prompt_tokens}"
    )

    # Prefill timing.
    try:
        mx.reset_peak_memory()
    except AttributeError:
        pass
    t0 = time.perf_counter()
    logits = model(
        batch["input_ids"],
        pixel_values=batch["pixel_values"],
        image_token_id=batch["image_token_id"],
    )
    mx.eval(logits)
    prefill_s = time.perf_counter() - t0
    print(
        f"prefill: {prefill_s:.2f}s  ({n_prompt_tokens / prefill_s:.1f} tok/s incl. vision)"
    )

    # Decode.
    eos_id = getattr(proc.tokenizer, "eos_token_id", None)
    tokens = []
    t0 = time.perf_counter()
    for tok in model.generate(
        batch["input_ids"],
        pixel_values=batch["pixel_values"],
        image_token_id=batch["image_token_id"],
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_id,
    ):
        tokens.append(tok)
    decode_s = time.perf_counter() - t0
    n_gen = len(tokens)
    print(f"decode: {decode_s:.2f}s  gen={n_gen}  {n_gen / max(decode_s, 1e-6):.1f} tok/s")

    text = proc.decode(tokens, skip_special_tokens=True)
    print(f"peak RSS: {peak_rss_gb():.2f} GB   peak GPU: {peak_gpu_gb():.2f} GB")
    print(f"OUTPUT: {text!r}")

    # Aggressive teardown so the next variant starts clean.
    del model, proc, logits
    gc.collect()

    return {
        "variant": cfg.variant,
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "gen_tokens": n_gen,
        "output": text,
        "peak_rss_gb": peak_rss_gb(),
        "peak_gpu_gb": peak_gpu_gb(),
        "n_tiles": n_tiles,
        "n_prompt_tokens": n_prompt_tokens,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompt", default="Describe this image in one sentence.")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--models-root", default=os.path.expanduser("~/models"))
    ap.add_argument("--variants", default="standard,flash")
    args = ap.parse_args()

    root = Path(args.models_root)
    dirs = {
        "standard": root / "VisionPsy-Nano-460M",
        "flash": root / "VisionPsy-Nano-460M-Flash",
    }

    results = {}
    for name in args.variants.split(","):
        name = name.strip()
        model_dir = dirs[name]
        if not model_dir.exists():
            print(f"skip {name}: missing {model_dir}")
            continue
        results[name] = run(
            model_dir, Path(args.image), args.prompt, args.max_new_tokens, args.dtype
        )

    print("\n=== summary ===")
    for name, r in results.items():
        print(
            f"{name:10s} {r['gen_tokens']:3d} tok in {r['decode_s']:.2f}s "
            f"({r['gen_tokens'] / max(r['decode_s'], 1e-6):.1f} tok/s), "
            f"peak_gpu={r['peak_gpu_gb']:.2f}GB, tiles={r['n_tiles']}"
        )


if __name__ == "__main__":
    main()
