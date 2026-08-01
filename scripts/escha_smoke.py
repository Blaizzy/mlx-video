"""Option-B smoke test: load dequant checkpoint, generate ~50 tokens.

Uses the flat-dequant loader path (weight_loader_dequant + DequantExpertLinear)
so the MoE experts are REAL (not zero-fallback). Reports:
    - load time (seconds)
    - first-token latency + tokens/sec on generation
    - peak Metal memory (GB)
    - the generated text
"""

from __future__ import annotations

import time
from pathlib import Path

import mlx.core as mx


def main() -> None:
    from tokenizers import Tokenizer
    from mlx_video.models.qwen3_5_moe_escha.model import load_model_dequant

    orig_dir = Path.home() / "models" / "Qwen3.6-35B-A3B-Escha-W2"
    dq_dir = Path.home() / "models" / "Qwen3.6-35B-A3B-Escha-W2-MLX-dequant" / "dequant_v1"

    t0 = time.time()
    model = load_model_dequant(orig_dir, dq_dir)
    mx.eval(model.parameters())
    load_time = time.time() - t0
    print(f"Load: {load_time:.1f}s")

    tok = Tokenizer.from_file(str(orig_dir / "tokenizer.json"))
    prompt = "Hello"
    ids = tok.encode(prompt).ids
    print(f"Prompt: {prompt!r}   token IDs: {ids}")

    # Prefill
    tokens = mx.array([ids], dtype=mx.int32)
    cache = model.make_cache()
    t1 = time.time()
    logits = model(tokens, cache=cache)
    mx.eval(logits)
    prefill_time = time.time() - t1
    print(f"Prefill: {prefill_time:.2f}s  logits {logits.shape} {logits.dtype}")

    # Greedy generation
    N = 50
    generated: list[int] = []
    last = logits[0, -1]
    next_id = int(mx.argmax(last).item())
    generated.append(next_id)

    t_gen_start = time.time()
    for step in range(N - 1):
        t_step = time.time()
        tokens = mx.array([[next_id]], dtype=mx.int32)
        logits = model(tokens, cache=cache)
        mx.eval(logits)
        last = logits[0, -1]
        next_id = int(mx.argmax(last).item())
        generated.append(next_id)
        if step < 3 or step == N - 2:
            print(f"  step {step+1}: {time.time() - t_step:.2f}s  id={next_id}")

    gen_time = time.time() - t_gen_start
    tps = (N - 1) / gen_time if gen_time > 0 else 0
    print(f"Generation: {N} tokens in {gen_time:.1f}s => {tps:.2f} tok/s")

    try:
        peak_gb = mx.metal.get_peak_memory() / 1e9
        print(f"Peak Metal memory: {peak_gb:.2f} GB")
    except Exception as e:
        print(f"peak memory query unavailable: {e}")

    text = tok.decode(generated)
    print(f"\nGenerated text: {text!r}")


if __name__ == "__main__":
    main()
