"""Escha-W2 end-to-end forward smoke test.

Loads the full 12 GB checkpoint from ~/models/Qwen3.6-35B-A3B-Escha-W2 and
runs a single 'Hello' forward pass with the zero-expert fallback (codebooks
still blocked on the parallel RunPod extraction track).

Emits: load time, forward time, logits shape, finite-fraction, and the
top-10 next-token decodings for the last position.

Run:  python scripts/escha_forward_smoke.py
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import mlx.core as mx


def main() -> None:
    from tokenizers import Tokenizer
    from mlx_video.models.qwen3_5_moe_escha.model import load_model

    mdir = Path.home() / "models" / "Qwen3.6-35B-A3B-Escha-W2"

    t0 = time.time()
    model = load_model(mdir)
    print(f"Load: {time.time() - t0:.1f}s")

    tok = Tokenizer.from_file(str(mdir / "tokenizer.json"))
    ids = tok.encode("Hello").ids
    print("Token IDs for 'Hello':", ids)

    tokens = mx.array([ids], dtype=mx.int32)
    t1 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        logits = model(tokens)
    mx.eval(logits)
    print(f"Forward: {time.time() - t1:.1f}s")
    print(f"Logits shape: {logits.shape}, dtype: {logits.dtype}")

    finite = float(mx.mean(mx.isfinite(logits.astype(mx.float32))).item())
    print(f"Finite fraction: {finite:.6f}")

    last = logits[0, -1].astype(mx.float32)
    top = mx.argsort(-last)[:10]
    top_ids = [int(x) for x in top.tolist()]
    top_scores = [round(float(last[i].item()), 3) for i in top_ids]
    print("Top-10 next-token IDs:", top_ids)
    print("Top-10 logits:", top_scores)
    print("Top-10 decoded:", [repr(tok.decode([i])) for i in top_ids])


if __name__ == "__main__":
    main()
