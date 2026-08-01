"""One-shot codebook extractor. Run once on a Linux x86-64 box with `escha` installed.

Produces `escha_codebooks_v1.npz` next to this script — a ~4-6 MiB file that the
MLX port loads at runtime.

**Prefer `modal_extract.py`** (uses Modal Labs) if you don't already have a
Linux + NVIDIA GPU box. Modal ships the code to a serverless GPU and returns
the result to your Mac in ~15 min, ~$0.20. See ../../../docs/CODEBOOK_EXTRACTION_HOWTO.md.

This script is the "run it yourself on a Linux box" path. Usage:

    python3.12 -m venv .venv && source .venv/bin/activate
    pip install "torch==2.9.*" --index-url https://download.pytorch.org/whl/cu128
    pip install "huggingface_hub[cli]" numpy safetensors
    hf download EschaLabs/escha-runtime-qwen3moe --include "sglang/*" --local-dir .
    pip install ./sglang/escha-*.whl
    python extract_codebooks.py

CPU-only fallback: same install, then `python extract_codebooks.py --cpu`.
Roughly 30-90 minutes wall-clock on a modern GPU.

WHAT THIS DOES
==============
Escha-W2 uses AQLM-style residual codebooks: each 16-wide slice of a weight
row is reconstructed as a sum of K vectors from K different 65536-entry
codebooks of length-16 fp16. The codebook tables live inside the escha .so
as compile-time constants — not in the safetensors.

Extraction strategy (each stage falls back if unusable):
    1. Introspect for a native accessor (dir(escha), torch.ops.escha.*).
    2. Scan the .so .rodata for codebook-shaped blobs (best-effort).
    3. Functional probe with auto-detected layout, then sweep 0..65535 for
       each K. This is the guaranteed path.

I/O contract of the op (from the public schema):
    torch.ops.escha.escham_reconstruct(code, in_features, out_features, K,
                                       cbA: bool, mul1: bool) -> fp16 Tensor
    where `code` has shape (in_p/16, out_p/16, 16*K) int16 and the returned
    tensor has shape (in_p, out_p). The exact axis convention has drifted
    across drafts — we auto-detect it now instead of hardcoding.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def _instantiate(shape_desc, K):
    return tuple(16 * K if x == "16K" else x for x in shape_desc)


def _find_layout(op, device):
    """Try plausible layouts; return the one where perturbing probe[0] changes
    exactly one 16-wide row of the output."""
    import torch

    layouts_to_try = [
        # (probe_shape_desc, in_features, out_features)
        ((1, 8, "16K"), 16, 128),
        ((8, 1, "16K"), 128, 16),
        ((8, 8, "16K"), 128, 128),
        ((1, 16, "16K"), 16, 256),
        ((16, 1, "16K"), 256, 16),
    ]

    for shape_desc, in_f, out_f in layouts_to_try:
        for K in (2, 3):
            probe_shape = _instantiate(shape_desc, K)
            try:
                p0 = torch.zeros(probe_shape, dtype=torch.int16, device=device)
                w0 = op(p0, in_f, out_f, K, True, False)
            except Exception as e:
                print(f"  {probe_shape} in={in_f} out={out_f} K={K}: baseline raised {type(e).__name__}")
                continue
            p1 = torch.zeros(probe_shape, dtype=torch.int16, device=device)
            p1.view(-1)[0] = 1
            try:
                w1 = op(p1, in_f, out_f, K, True, False)
            except Exception as e:
                print(f"  {probe_shape} in={in_f} out={out_f} K={K}: perturbed raised {type(e).__name__}")
                continue

            diff = (w1 - w0).float().abs()
            if diff.ndim >= 2:
                n_changed = int((diff > 1e-6).any(dim=-1).sum().item())
            else:
                n_changed = int((diff > 1e-6).sum().item())
            print(
                f"  {probe_shape} in={in_f} out={out_f} K={K}: "
                f"out.shape={tuple(w1.shape)} rows_changed={n_changed}"
            )
            if n_changed == 1:
                return shape_desc, in_f, out_f
    return None


def _introspect(escha, torch):
    """Look for any native codebook accessor — if EschaLabs ever ships one."""
    escha_attrs = [a for a in dir(escha) if not a.startswith("_")]
    ops_names = [n for n in dir(torch.ops.escha) if not n.startswith("_")]
    print(f"  escha attrs: {escha_attrs}")
    print(f"  torch.ops.escha: {ops_names}")

    hints = ("codebook", "lut", "lattice", "dump", "table", "cb_a", "cb_b", "cb_c")
    for src, names in (("escha", escha_attrs), ("torch.ops.escha", ops_names)):
        for n in names:
            if any(w in n.lower() for w in hints):
                print(f"  candidate accessor: {src}.{n}")


def _extract_K(op, device, shape_desc, in_f, out_f, K):
    import numpy as np
    import torch

    probe_shape = _instantiate(shape_desc, K)
    cb = np.zeros((65536, 16), dtype=np.float16)
    p0 = torch.zeros(probe_shape, dtype=torch.int16, device=device)
    w0 = op(p0, in_f, out_f, K, True, False).detach().cpu().numpy()
    baseline_row = w0.reshape(-1, 16)[0].copy()
    cb[0] = baseline_row

    t0 = time.time()
    for i in range(1, 65536):
        probe = torch.zeros(probe_shape, dtype=torch.int16, device=device)
        probe.view(-1)[0] = i
        w = op(probe, in_f, out_f, K, True, False).detach().cpu().numpy()
        cb[i] = w.reshape(-1, 16)[0] - baseline_row
        if i % 4096 == 0:
            el = time.time() - t0
            eta = el * (65536 - i) / i
            print(
                f"    K={K}: {i}/65536 ({100*i/65536:.1f}%)  elapsed={el:.0f}s  ETA={eta:.0f}s",
                flush=True,
            )
    print(f"    K={K}: done in {time.time()-t0:.0f}s", flush=True)
    return cb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true", help="Use CPU fallback (slower).")
    ap.add_argument("--out", default=str(Path(__file__).parent / "escha_codebooks_v1.npz"))
    args = ap.parse_args()

    try:
        import numpy as np
        import torch
        import escha  # noqa: F401 — registers torch.ops.escha.*
    except ImportError as e:
        print(f"Missing dep: {e}. See docstring for install instructions.", file=sys.stderr)
        return 1

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[extract] device={device} torch={torch.__version__}")
    print(f"[extract] escha={escha.__file__} version={getattr(escha, '__version__', '?')}")

    print("\n[stage 1] introspecting escha for a native codebook accessor...")
    _introspect(escha, torch)

    print("\n[stage 3] auto-detecting probe layout...")
    op = torch.ops.escha.escham_reconstruct
    layout = _find_layout(op, device)
    if layout is None:
        print(
            "FAILURE: no probe layout produced a clean single-row response. "
            "The escha op shape convention may have changed. Please open a "
            "HuggingFace discussion at "
            "https://huggingface.co/EschaLabs/escha-runtime-qwen3moe/discussions",
            file=sys.stderr,
        )
        return 2

    shape_desc, in_f, out_f = layout
    print(f"[stage 3] locked layout {shape_desc} in={in_f} out={out_f}")

    print("[stage 3] sweeping K=2...")
    cb_K2 = _extract_K(op, device, shape_desc, in_f, out_f, 2)
    print("[stage 3] sweeping K=3...")
    cb_K3 = _extract_K(op, device, shape_desc, in_f, out_f, 3)

    nz2 = int((cb_K2 != 0).any(axis=1).sum())
    nz3 = int((cb_K3 != 0).any(axis=1).sum())
    print(f"\n[result] cb_K2 nonzero rows: {nz2}/65536")
    print(f"[result] cb_K3 nonzero rows: {nz3}/65536")
    if nz2 < 60000 or nz3 < 60000:
        print("[WARN] unusually few nonzero rows — inspect the output before shipping.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, cb_A_K2=cb_K2, cb_A_K3=cb_K3)
    print(f"[result] wrote {out} ({out.stat().st_size / 1024 / 1024:.2f} MiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
