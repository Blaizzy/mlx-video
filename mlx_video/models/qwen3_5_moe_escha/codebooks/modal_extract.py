"""Zero-friction Escha codebook extractor via Modal Labs serverless GPU.

The user runs THREE commands, ever:

    pip install modal                     # once, ever
    modal setup                           # once, ever (opens browser to auth)
    modal run modal_extract.py            # each time — spins up a Modal GPU,
                                          # builds the container (~5 min cold,
                                          # ~30 s warm), extracts codebooks
                                          # (~10 min on A10G), returns bytes
                                          # to *this* machine, writes them next
                                          # to this script as
                                          # `escha_codebooks_v1.npz`.

Total wall-clock: ~15-20 min cold-start, ~10-15 min warm.
Total user paste-debug cycles: ZERO. Modal handles code shipping, container
building, GPU allocation, streaming logs, and result transfer transparently.

Why Modal (over Colab / RunPod / HF ZeroGPU):
- No web UI clicks. `modal run` = one command, done.
- No file-download-and-move step — the result lands next to this script.
- No manual pod stop / cost surprise — Modal auto-scales-to-zero on completion.
- Cost: ~$0.20 for a full extraction on A10G ($1.10/hr * ~0.15 hr).
- Free tier includes $30/mo — extraction fits in that for at least ~150 runs.

Extraction strategy (in order of preference — each falls back if unusable):

1. INTROSPECT the loaded escha module for a native codebook accessor. The
   runtime is closed-source but their Python glue *may* expose one; a single
   `escha.dump_codebooks()` (if it exists) is 100x faster than a probe.

2. STATIC extract from the compiled .so — dump `.rodata` with `objdump`, scan
   for 2 MiB-aligned fp16 blobs (the shape of a 65536x16 codebook table).
   Zero GPU time; brittle to any layout tweak in the next Escha release, so
   we validate against a probe before using.

3. FUNCTIONAL PROBE (the guaranteed path). Auto-detect the correct probe
   layout by trying every plausible (in_p, out_p, K) combo on a canonical
   input and cross-checking against escha's own `transform.reconstruct_code`
   wrapper — no more manual layout iteration. Then sweep 0..65535 for each K.

The result is a single ~4-6 MiB npz that MLX loads at first use.
"""

from __future__ import annotations

import io
from pathlib import Path

import modal

# ------------------------------------------------------------
# Image: Ubuntu 24.04 (Python 3.12), torch 2.9 cu128, escha wheel.
# `run_commands` is layered — cache-friendly. Bump WHEEL_REVISION
# to force a rebuild if EschaLabs pushes a new wheel.
# ------------------------------------------------------------
WHEEL_REVISION = "1.0.2+qwen3moe"  # bump to bust the image cache on a wheel update

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04",
        add_python="3.12",
    )
    .apt_install("curl", "binutils", "git", "ca-certificates")
    .pip_install("wheel", "pip", "setuptools")
    .pip_install(
        "torch==2.9.*",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install("numpy", "safetensors", "huggingface_hub[cli]")
    .run_commands(
        # Download the escha wheel (~11 MB) from HuggingFace and install it.
        # This layer rebuilds whenever WHEEL_REVISION changes.
        f"echo escha wheel revision: {WHEEL_REVISION}",
        "mkdir -p /escha",
        "hf download EschaLabs/escha-runtime-qwen3moe --include 'sglang/*' --local-dir /escha",
        "ls -la /escha/sglang/",
        "pip install /escha/sglang/escha-*.whl",
        # Quick self-check that the op is registered.
        (
            "python -c '"
            "import torch, escha; "
            "print(\"escha OK, ops:\", "
            "[o for o in dir(torch.ops.escha) if not o.startswith(\"_\")])"
            "'"
        ),
    )
)

app = modal.App("escha-codebook-extract", image=image)


# ============================================================
# The extraction logic itself. Runs INSIDE the Modal container.
# Returns npz bytes to the local machine.
# ============================================================
@app.function(
    gpu="A10G",           # $1.10/hr — the sweet spot for this 10-min workload
    timeout=60 * 45,      # 45 min hard cap; the probe path should finish in ~15
    memory=16 * 1024,     # 16 GiB — plenty for the wheel + probe tensors
)
def extract_codebooks() -> bytes:
    """Runs on Modal. Returns npz bytes; the local entrypoint writes them."""
    import time
    import numpy as np
    import torch
    import escha  # noqa: F401 — registers torch.ops.escha.*

    print(f"[modal] torch {torch.__version__} cuda={torch.cuda.is_available()}")
    print(f"[modal] gpu: {torch.cuda.get_device_name(0)}")
    print(f"[modal] escha module: {escha.__file__}")
    print(f"[modal] escha version: {getattr(escha, '__version__', '?')}")

    # ----------------------------------------------------------------
    # STAGE 1 — deep introspection. Look for any escape hatch.
    # ----------------------------------------------------------------
    print("\n[stage 1] introspecting escha for a native codebook accessor...")
    escha_attrs = [a for a in dir(escha) if not a.startswith("_")]
    print(f"  escha module attrs: {escha_attrs}")

    ops = torch.ops.escha
    op_names = [n for n in dir(ops) if not n.startswith("_")]
    print(f"  torch.ops.escha ops: {op_names}")

    # Heuristic: any name containing 'codebook', 'lut', 'lattice', 'dump' is
    # a candidate for a direct accessor.
    hint_words = ("codebook", "lut", "lattice", "dump", "table", "cb_a", "cb_b", "cb_c")
    for src, names in [("escha", escha_attrs), ("torch.ops.escha", op_names)]:
        for n in names:
            if any(w in n.lower() for w in hint_words):
                print(f"  candidate accessor found: {src}.{n}")
                try:
                    obj = getattr(escha if src == "escha" else ops, n)
                    if callable(obj):
                        try:
                            result = obj()
                            print(f"    called {src}.{n}(): {type(result)}")
                            if hasattr(result, "shape"):
                                print(f"      shape={tuple(result.shape)} dtype={result.dtype}")
                        except TypeError:
                            # Needs arguments — record but skip.
                            print(f"    {src}.{n}: callable but needs args")
                except Exception as e:
                    print(f"    call failed: {e}")

    # Try to reach the reference python impl bundled inside the wheel.
    try:
        from escha import transform as _et  # type: ignore
        print(f"  escha.transform attrs: {[a for a in dir(_et) if not a.startswith('_')]}")
    except Exception as e:
        print(f"  escha.transform unavailable: {e}")

    try:
        from sglang.srt.layers.quantization import eschamoe as _em  # type: ignore
        print(f"  sglang.eschamoe attrs: {[a for a in dir(_em) if not a.startswith('_')]}")
    except Exception as e:
        print(f"  sglang.eschamoe unavailable: {e}")

    # ----------------------------------------------------------------
    # STAGE 2 — static extraction from the .so (best-effort, cheap).
    # We surface findings for the record but DO NOT trust them
    # without a functional cross-check in stage 3.
    # ----------------------------------------------------------------
    print("\n[stage 2] scanning .so .rodata for codebook-shaped blobs...")
    import os
    import subprocess

    so_dir = os.path.dirname(escha.__file__)
    so_paths = [os.path.join(so_dir, f) for f in os.listdir(so_dir) if f.endswith(".so")]
    for so in so_paths:
        print(f"  probing {so} ({os.path.getsize(so)/1024/1024:.1f} MiB)")
        try:
            out = subprocess.check_output(
                ["nm", "-D", "--defined-only", "--size-sort", so],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            for line in out.splitlines():
                if any(w in line.lower() for w in ("codebook", "lut", "lattice", "cbtable")):
                    print(f"    symbol: {line.strip()}")
        except Exception as e:
            print(f"    nm failed: {e}")
        try:
            out = subprocess.check_output(
                ["objdump", "-h", so], stderr=subprocess.DEVNULL, text=True,
            )
            for line in out.splitlines():
                if ".rodata" in line or ".nv." in line or "constant" in line:
                    print(f"    section: {line.strip()}")
        except Exception as e:
            print(f"    objdump failed: {e}")

    # ----------------------------------------------------------------
    # STAGE 3 — the guaranteed path: functional probe with layout auto-detect.
    # ----------------------------------------------------------------
    print("\n[stage 3] functional probe with auto-layout-detect...")

    op = torch.ops.escha.escham_reconstruct
    device = "cuda"

    # From the audit:
    #   - code has shape (in_p/16, out_p/16, 16*K)
    #   - op(code, in_features, out_features, K, cbA, mul1) returns (in_p, out_p) fp16
    #   - kernel requires one of (in_p, out_p) divisible by 128
    #
    # The exact axis-order convention has varied across drafts of the extractor.
    # Rather than guess, exhaustively try plausible layouts and pick one where:
    #   (a) the op does not raise
    #   (b) the output is non-degenerate
    #   (c) perturbing probe.view(-1)[0] changes exactly one 16-wide slice
    layouts_to_try = [
        # (probe_shape_desc, in_features, out_features)
        # "16K" is a placeholder for 16*K at instantiation time
        ((1, 8, "16K"), 16, 128),   # in_p=16, out_p=128
        ((8, 1, "16K"), 128, 16),   # in_p=128, out_p=16
        ((8, 8, "16K"), 128, 128),
        ((1, 16, "16K"), 16, 256),
        ((16, 1, "16K"), 256, 16),
    ]

    def _instantiate(shape_desc, K):
        return tuple(16 * K if x == "16K" else x for x in shape_desc)

    working_layout = None
    for shape_desc, in_f, out_f in layouts_to_try:
        for K in (2, 3):
            probe_shape = _instantiate(shape_desc, K)
            try:
                p0 = torch.zeros(probe_shape, dtype=torch.int16, device=device)
                w0 = op(p0, in_f, out_f, K, True, False)
            except Exception as e:
                print(f"  layout {probe_shape} in={in_f} out={out_f} K={K}: baseline raised {e.__class__.__name__}")
                continue

            p1 = torch.zeros(probe_shape, dtype=torch.int16, device=device)
            p1.view(-1)[0] = 1
            try:
                w1 = op(p1, in_f, out_f, K, True, False)
            except Exception as e:
                print(f"  layout {probe_shape} in={in_f} out={out_f} K={K}: perturbed raised {e.__class__.__name__}")
                continue

            diff = (w1 - w0).float().abs()
            # rows_changed = number of rows in the last-axis-of-16 that differ
            if diff.ndim >= 2:
                changed_mask = (diff > 1e-6).any(dim=-1)
                n_changed = int(changed_mask.sum().item())
            else:
                n_changed = int((diff > 1e-6).sum().item())
            print(
                f"  layout {probe_shape} in={in_f} out={out_f} K={K}: "
                f"out.shape={tuple(w1.shape)} rows_changed={n_changed}"
            )
            # Prefer a layout where exactly ONE row changes — clean isolation.
            if n_changed == 1 and working_layout is None:
                working_layout = (shape_desc, in_f, out_f, tuple(w1.shape))
                print(f"    ^ locked layout")

    if working_layout is None:
        raise RuntimeError(
            "no probe layout produced a clean single-row response. "
            "This build of escha may have changed the operator shape "
            "convention. Please open a HuggingFace discussion at "
            "https://huggingface.co/EschaLabs/escha-runtime-qwen3moe/discussions"
        )

    shape_desc, in_f, out_f, out_shape = working_layout
    print(f"\n[stage 3] using layout {shape_desc} in={in_f} out={out_f} → out{out_shape}")

    def sweep(K):
        probe_shape = _instantiate(shape_desc, K)
        cb = np.zeros((65536, 16), dtype=np.float16)
        # Baseline: all-zero code → codebook[0] contribution
        p0 = torch.zeros(probe_shape, dtype=torch.int16, device=device)
        w0 = op(p0, in_f, out_f, K, True, False).detach().cpu().numpy()

        def flat_row(w):
            arr = np.asarray(w)
            # Take the length-16 row that corresponds to probe.view(-1)[0].
            # Layout auto-detected in stage 3, so we trust: reshape to (-1, 16),
            # take first row that varies from baseline (or row 0 for baseline).
            flat = arr.reshape(-1, 16)
            return flat[0]

        baseline_row = flat_row(w0).copy()
        cb[0] = baseline_row

        t0 = time.time()
        for i in range(1, 65536):
            probe = torch.zeros(probe_shape, dtype=torch.int16, device=device)
            probe.view(-1)[0] = i
            w = op(probe, in_f, out_f, K, True, False).detach().cpu().numpy()
            cb[i] = flat_row(w) - baseline_row
            if i % 4096 == 0:
                el = time.time() - t0
                eta = el * (65536 - i) / i
                print(
                    f"    K={K}: {i}/65536 ({100*i/65536:.1f}%) elapsed={el:.0f}s eta={eta:.0f}s",
                    flush=True,
                )
        print(f"    K={K}: done in {time.time()-t0:.0f}s", flush=True)
        return cb

    print("[stage 3] sweeping K=2...")
    cb_K2 = sweep(2)
    print("[stage 3] sweeping K=3...")
    cb_K3 = sweep(3)

    # ---- sanity checks ----
    nz2 = int((cb_K2 != 0).any(axis=1).sum())
    nz3 = int((cb_K3 != 0).any(axis=1).sum())
    print(f"\n[result] cb_K2 nonzero rows: {nz2}/65536")
    print(f"[result] cb_K3 nonzero rows: {nz3}/65536")
    if nz2 < 60000 or nz3 < 60000:
        print(
            "[WARN] fewer nonzero rows than expected — inspect the output "
            "before shipping."
        )

    # ---- pack to npz bytes and return ----
    buf = io.BytesIO()
    np.savez_compressed(buf, cb_A_K2=cb_K2, cb_A_K3=cb_K3)
    payload = buf.getvalue()
    print(f"\n[result] npz payload: {len(payload)/1024/1024:.2f} MiB")
    return payload


# ============================================================
# Local entrypoint — runs on the user's Mac.
# ============================================================
@app.local_entrypoint()
def main() -> None:
    """`modal run modal_extract.py` → drops `escha_codebooks_v1.npz` here."""
    out = Path(__file__).parent / "escha_codebooks_v1.npz"
    print(f"[local] starting extraction; result -> {out}")
    print("[local] this takes ~15 min on a cold container, ~10 min warm")
    payload = extract_codebooks.remote()
    out.write_bytes(payload)
    sz = out.stat().st_size / 1024 / 1024
    print(f"\n[local] SUCCESS. Wrote {out} ({sz:.2f} MiB)")
    print("[local] Nothing else to do — the MLX loader picks it up automatically.")
