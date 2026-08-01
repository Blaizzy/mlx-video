"""Route H — Escha codebook introspection via Python attribute-walk on Modal GPU.

Op-based probe extraction failed on Modal, but the kernel signature
`..PK__half..` indicates the codebook is a launch-time pointer that must be
fully materialized in memory after model init. This script imports escha,
walks every module's attributes and every live tensor in `gc.get_objects()`
for large fp16/bf16/fp32 tensors, and returns a pickle of them to the local
machine.

Run:

    modal run modal_introspect.py

Writes `escha_introspect_dump.pkl` next to this script on success.
"""

from __future__ import annotations

from pathlib import Path

import modal

# ------------------------------------------------------------
# Image: identical to modal_extract.py (torch 2.9 cu128, escha wheel).
# Do NOT diverge — reusing the same image hits the same cached layers.
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

app = modal.App("escha-codebook-introspect", image=image)


# ============================================================
# Introspection function — runs inside the Modal container.
# Returns a pickle of {name: numpy_array} for every large tensor found.
# ============================================================
@app.function(
    gpu="A10G",
    timeout=1800,
    memory=16 * 1024,
)
def introspect() -> bytes:
    import gc
    import pickle
    import sys

    import numpy as np
    import torch

    import escha  # noqa: F401 — registers torch.ops.escha.*

    print(f"[modal] torch {torch.__version__} cuda={torch.cuda.is_available()}")
    print(f"[modal] gpu: {torch.cuda.get_device_name(0)}")
    print(f"[modal] escha module: {escha.__file__}")
    print(f"[modal] escha version: {getattr(escha, '__version__', '?')}")

    # ----------------------------------------------------------------
    # (b) Print surface-level introspection results.
    # ----------------------------------------------------------------
    print("\n[b] escha module surface:")
    print(f"  escha.__file__: {escha.__file__}")
    print(f"  dir(escha): {dir(escha)}")

    if hasattr(escha, "_C"):
        try:
            print(f"  dir(escha._C): {dir(escha._C)}")
        except Exception as e:
            print(f"  escha._C dir failed: {e}")
    else:
        print("  escha._C: not present")

    try:
        print(f"  dir(torch.ops.escha): {dir(torch.ops.escha)}")
    except Exception as e:
        print(f"  torch.ops.escha unavailable: {e}")

    try:
        if hasattr(torch.classes, "escha"):
            print(f"  dir(torch.classes.escha): {dir(torch.classes.escha)}")
        else:
            print("  torch.classes.escha: not present")
    except Exception as e:
        print(f"  torch.classes.escha check failed: {e}")

    # Discover any additional escha submodules already loaded.
    escha_mods_before = sorted(m for m in sys.modules if m.startswith("escha"))
    print(f"  escha submodules loaded: {escha_mods_before}")

    # ----------------------------------------------------------------
    # Helpers.
    # ----------------------------------------------------------------
    MIN_NUMEL = 65536 * 8  # 524_288 — corresponds to 65536*8 fp16 elements
    WANTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

    def is_wanted_tensor(obj) -> bool:
        try:
            return (
                isinstance(obj, torch.Tensor)
                and obj.dtype in WANTED_DTYPES
                and obj.numel() >= MIN_NUMEL
            )
        except Exception:
            return False

    def to_numpy(t: "torch.Tensor") -> "np.ndarray":
        # bfloat16 has no direct numpy dtype; upcast for the dump.
        if t.dtype == torch.bfloat16:
            return t.detach().to(torch.float32).cpu().numpy()
        return t.detach().cpu().numpy()

    def sweep_modules(results: dict, tag: str) -> None:
        """(c) Attribute-walk every module whose name starts with escha or sglang.*.eschamoe."""
        target_mods = [
            (name, mod) for name, mod in list(sys.modules.items())
            if mod is not None and (
                name.startswith("escha")
                or name == "sglang.srt.layers.quantization.eschamoe"
                or name.startswith("sglang.srt.layers.quantization.eschamoe.")
            )
        ]
        print(f"  [{tag}] attribute-walk over {len(target_mods)} modules")
        for name, mod in target_mods:
            try:
                attrs = dir(mod)
            except Exception:
                continue
            for a in attrs:
                if a.startswith("__"):
                    continue
                try:
                    obj = getattr(mod, a)
                except Exception:
                    continue
                if is_wanted_tensor(obj):
                    key = f"attr::{name}.{a}"
                    if key not in results:
                        try:
                            arr = to_numpy(obj)
                            results[key] = arr
                            print(
                                f"    FOUND {key}: shape={tuple(obj.shape)} "
                                f"dtype={obj.dtype} numel={obj.numel()} "
                                f"numpy_dtype={arr.dtype}"
                            )
                        except Exception as e:
                            print(f"    skip {key}: to_numpy failed {e!r}")

    def sweep_gc(results: dict, tag: str) -> None:
        """(d) Sweep gc.get_objects() for any large fp16/bf16 tensor."""
        n_seen = 0
        n_found = 0
        # Snapshot the object list to avoid iteration issues while adding entries.
        objs = gc.get_objects()
        print(f"  [{tag}] gc sweep over {len(objs)} objects")
        for obj in objs:
            if not isinstance(obj, torch.Tensor):
                continue
            n_seen += 1
            if not is_wanted_tensor(obj):
                continue
            # Deduplicate by data pointer where possible.
            try:
                dp = obj.data_ptr()
            except Exception:
                dp = id(obj)
            key = f"gc::dp{dp:#x}::shape{tuple(obj.shape)}::{str(obj.dtype).split('.')[-1]}"
            if key in results:
                continue
            try:
                arr = to_numpy(obj)
                results[key] = arr
                n_found += 1
                print(
                    f"    FOUND {key}: shape={tuple(obj.shape)} "
                    f"dtype={obj.dtype} numel={obj.numel()} "
                    f"numpy_dtype={arr.dtype}"
                )
            except Exception as e:
                print(f"    skip {key}: to_numpy failed {e!r}")
        print(f"  [{tag}] gc sweep: {n_seen} tensors seen, {n_found} matched criteria")

    results: dict = {}

    # ----------------------------------------------------------------
    # (c)/(d) — sweeps at initial state (just `import escha`).
    # ----------------------------------------------------------------
    print("\n[c/d] initial sweep after `import escha`:")
    sweep_modules(results, "initial")
    sweep_gc(results, "initial")
    print(f"  results size after initial: {len(results)}")

    # ----------------------------------------------------------------
    # (e) Try to force sglang layer import (may trigger codebook load).
    # ----------------------------------------------------------------
    print("\n[e] attempting to import sglang.srt.layers.quantization.eschamoe...")
    try:
        from sglang.srt.layers.quantization import eschamoe as _em  # type: ignore
        print(f"  imported eschamoe OK: {_em}")
        try:
            attrs = [a for a in dir(_em) if not a.startswith("_")]
            print(f"  eschamoe attrs: {attrs}")
        except Exception as e:
            print(f"  eschamoe dir failed: {e}")

        # Also try grabbing anything that looks like a factory / initializer
        # and invoke it if it takes no args.
        for a in dir(_em):
            if a.startswith("_"):
                continue
            obj = getattr(_em, a, None)
            name_l = a.lower()
            if not any(w in name_l for w in ("codebook", "lut", "lattice", "load", "init", "table")):
                continue
            print(f"  candidate zero-arg trigger: eschamoe.{a} -> {type(obj).__name__}")
            if callable(obj):
                try:
                    r = obj()
                    print(f"    called {a}(): {type(r).__name__}")
                    if isinstance(r, torch.Tensor):
                        print(f"    -> tensor shape={tuple(r.shape)} dtype={r.dtype}")
                        if is_wanted_tensor(r):
                            results[f"trigger::eschamoe.{a}()"] = to_numpy(r)
                except TypeError:
                    pass
                except Exception as e:
                    print(f"    {a}() raised {e.__class__.__name__}: {e}")
    except Exception as e:
        print(f"  eschamoe import failed: {e.__class__.__name__}: {e}")

    print("\n[e] re-sweep after sglang layer import:")
    sweep_modules(results, "post-sglang")
    sweep_gc(results, "post-sglang")
    print(f"  results size after sglang: {len(results)}")

    # Also try loading escha.transform if present.
    print("\n[e2] attempting to import escha.transform...")
    try:
        from escha import transform as _et  # type: ignore
        print(f"  imported escha.transform OK: {_et}")
        try:
            print(f"  escha.transform attrs: {[a for a in dir(_et) if not a.startswith('_')]}")
        except Exception as e:
            print(f"  escha.transform dir failed: {e}")
    except Exception as e:
        print(f"  escha.transform import failed: {e.__class__.__name__}: {e}")

    print("\n[e2] re-sweep after escha.transform import:")
    sweep_modules(results, "post-transform")
    sweep_gc(results, "post-transform")
    print(f"  results size after transform: {len(results)}")

    # ----------------------------------------------------------------
    # (f) If still empty, try to call an op with a trivial input to
    #     force any lazy codebook upload. We do NOT try to load full
    #     model files — that requires network + disk.
    # ----------------------------------------------------------------
    if not results:
        print("\n[f] no tensors found yet; trying tiny op call to force lazy init...")
        try:
            op = torch.ops.escha.escham_reconstruct
            # Minimal probe — the exact layout doesn't matter, we just want
            # any launch-time codebook pointer to be materialized.
            for probe_shape, in_f, out_f, K in [
                ((1, 8, 32), 16, 128, 2),
                ((8, 1, 32), 128, 16, 2),
                ((8, 8, 32), 128, 128, 2),
            ]:
                try:
                    p = torch.zeros(probe_shape, dtype=torch.int16, device="cuda")
                    _ = op(p, in_f, out_f, K, True, False)
                    print(f"  op call succeeded for shape={probe_shape} in={in_f} out={out_f} K={K}")
                    break
                except Exception as e:
                    print(f"  op call failed for shape={probe_shape}: {e.__class__.__name__}: {e}")
        except Exception as e:
            print(f"  op access failed: {e.__class__.__name__}: {e}")

        print("\n[f] re-sweep after op call:")
        sweep_modules(results, "post-op")
        sweep_gc(results, "post-op")
        print(f"  results size after op: {len(results)}")

    # ----------------------------------------------------------------
    # Summary + serialize.
    # ----------------------------------------------------------------
    print(f"\n[summary] total tensors captured: {len(results)}")
    for k, v in sorted(results.items()):
        # Highlight anything that matches the 65536-row codebook shape.
        is_cb = (
            (v.ndim >= 1 and v.shape[0] == 65536)
            or (v.size in (65536 * 16, 65536 * 16 * 2, 65536 * 16 * 3, 65536 * 16 * 4))
        )
        marker = " <-- CODEBOOK CANDIDATE" if is_cb else ""
        print(f"  {k}: shape={v.shape} dtype={v.dtype} nbytes={v.nbytes}{marker}")

    # Pickle to bytes and return.
    payload = pickle.dumps(results, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\n[summary] pickle payload: {len(payload) / 1024 / 1024:.2f} MiB")
    return payload


# ============================================================
# Local entrypoint — runs on the user's Mac.
# ============================================================
@app.local_entrypoint()
def main() -> None:
    out = Path(__file__).parent / "escha_introspect_dump.pkl"
    print(f"[local] starting introspection; result -> {out}")
    payload = introspect.remote()
    out.write_bytes(payload)
    sz = out.stat().st_size / 1024 / 1024
    print(f"\n[local] SUCCESS. Wrote {out} ({sz:.2f} MiB)")

    # Peek at the pickle to summarize.
    import pickle
    with open(out, "rb") as f:
        results = pickle.load(f)
    print(f"[local] results.keys() ({len(results)} entries):")
    for k, v in sorted(results.items()):
        print(f"  {k}: shape={v.shape} dtype={v.dtype} nbytes={v.nbytes}")
