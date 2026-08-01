"""wire_baseline_v2.py — install the Colab-captured `baseline_v2.npz` into the MLX Escha port.

Steps performed (in order):
  1. Locate `baseline_v2.npz` (either passed as `--src`, at the CWD, or
     downloaded/decoded from a base64 blob written to a local text file).
  2. Copy / decode it into `mlx_video/models/qwen3_5_moe_escha/codebooks/baseline_v2.npz`.
  3. Patch `eschamoe.py::escham_reconstruct` to subtract the matching `w0`
     baseline (keyed by `(in_features, out_features, K)`) from the recomposed
     dense weight — that is the fix that isolates the codebook sum from the
     shape-dependent additive bias.
  4. Verify: for the primary (in=2048, out=1024, K=2) and (in=512, out=2048, K=3)
     tiles, load the sample-code slice from the shipped MoE weights, run BOTH
     the MLX `escham_reconstruct` AND the reference `op(codes)` PyTorch value
     (if a reference dump is available), and print max-abs / rel diff. Fail
     if diff > 1e-2 (bf16-ish tolerance).

Usage:
    python wire_baseline_v2.py                       # picks up ~/Downloads/baseline_v2.npz
    python wire_baseline_v2.py --src /path/to.npz
    python wire_baseline_v2.py --b64  /path/to.b64.txt

The verification step is skipped (with a loud warning) if no reference dump
is present — the wire-in itself is idempotent and safe to re-run.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT   = Path(__file__).resolve().parents[4]   # …/mlx-video
MODEL_DIR   = Path(__file__).resolve().parents[1]   # …/qwen3_5_moe_escha
CB_DIR      = Path(__file__).resolve().parent       # …/codebooks
DEST_NPZ    = CB_DIR / "baseline_v2.npz"
ESCHAMOE_PY = MODEL_DIR / "eschamoe.py"

BASELINE_LOADER_MARK = "# --- BASELINE_V2 WIRE-IN (auto-inserted) ---"


# ---------------------------------------------------------------------------
# Step 1/2: locate + copy npz
# ---------------------------------------------------------------------------

def _find_default_npz() -> Path | None:
    candidates = [
        Path.home() / "Downloads" / "baseline_v2.npz",
        Path.cwd() / "baseline_v2.npz",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _decode_b64_txt(path: Path) -> bytes:
    """Extract the base64 payload between BEGIN/END markers or accept a raw blob."""
    text = path.read_text()
    m = re.search(
        r"BASELINE_V2_NPZ_BASE64_BEGIN\s*(.*?)\s*BASELINE_V2_NPZ_BASE64_END",
        text,
        flags=re.DOTALL,
    )
    payload = (m.group(1) if m else text).strip()
    payload = re.sub(r"\s+", "", payload)
    return base64.b64decode(payload)


def install_npz(args: argparse.Namespace) -> Path:
    if args.b64:
        blob = _decode_b64_txt(Path(args.b64))
        DEST_NPZ.write_bytes(blob)
        print(f"[wire] decoded base64 → {DEST_NPZ} ({len(blob)/1024:.1f} KiB)")
    else:
        src = Path(args.src) if args.src else _find_default_npz()
        if src is None or not src.exists():
            raise SystemExit(
                "no baseline_v2.npz found. Pass --src /path/to/baseline_v2.npz "
                "or --b64 /path/to/output.txt (Colab cell output pasted to a file)."
            )
        shutil.copy2(src, DEST_NPZ)
        print(f"[wire] copied {src} → {DEST_NPZ}")

    # Sanity: enumerate what's inside.
    with np.load(DEST_NPZ, allow_pickle=False) as npz:
        keys = list(npz.files)
        meta = None
        if "_meta_json" in keys:
            meta = json.loads(bytes(npz["_meta_json"]).decode("utf-8"))
    print(f"[wire] npz entries: {[k for k in keys if k != '_meta_json']}")
    if meta:
        print(f"[wire] wheel: {meta.get('wheel')}   torch: {meta.get('torch')}")
        for k, v in meta.get("shapes", {}).items():
            print(
                f"       {k}: shape={tuple(v['out_shape'])} "
                f"|w0|_2={v['l2_norm']:.3e} nnz={v['nnz_frac']:.3f}"
            )
    return DEST_NPZ


# ---------------------------------------------------------------------------
# Step 3: patch eschamoe.py to consume the baseline
# ---------------------------------------------------------------------------

PATCH_LOADER_SRC = '''
# --- BASELINE_V2 WIRE-IN (auto-inserted) ---
# See codebooks/wire_baseline_v2.py. `w0 = op(all_zeros_code, in_f, out_f, K)`
# captured on the reference CUDA runtime; subtracting it isolates the
# codebook sum from Escha's shape-dependent additive bias.
_BASELINE_NPZ = Path(__file__).parent / "codebooks" / "baseline_v2.npz"
_BASELINE_CACHE: dict[tuple[int, int, int], "mx.array"] = {}


def _baseline_w0(in_features: int, out_features: int, K: int, dtype) -> "mx.array":
    key = (int(in_features), int(out_features), int(K))
    if key in _BASELINE_CACHE:
        return _BASELINE_CACHE[key].astype(dtype)
    if not _BASELINE_NPZ.exists():
        raise FileNotFoundError(
            f"baseline_v2.npz not found at {_BASELINE_NPZ}. "
            f"Run codebooks/baseline_probe_colab.ipynb on a T4 and wire via "
            f"codebooks/wire_baseline_v2.py."
        )
    import numpy as _np
    with _np.load(_BASELINE_NPZ, allow_pickle=False) as npz:
        want = f"in{in_features}_out{out_features}_K{K}"
        match = [k for k in npz.files if k.endswith(f"__{want}")]
        if not match:
            raise KeyError(
                f"baseline_v2.npz has no entry for {want}; "
                f"present keys: {list(npz.files)}"
            )
        arr = mx.array(npz[match[0]].astype(_np.float32))
    _BASELINE_CACHE[key] = arr
    return arr.astype(dtype)
# --- /BASELINE_V2 WIRE-IN ---
'''.strip("\n") + "\n"


PATCH_SUBTRACT_SRC = "    w = w - _baseline_w0(in_features, out_features, K, w.dtype)\n"


def patch_eschamoe() -> None:
    src = ESCHAMOE_PY.read_text()

    if BASELINE_LOADER_MARK in src:
        print(f"[wire] {ESCHAMOE_PY.name} already patched — leaving alone.")
        return

    # 1) inject loader helper after the existing _CB_CACHE definition.
    anchor = "_CB_CACHE: dict[tuple[int, str], mx.array] = {}\n"
    if anchor not in src:
        raise RuntimeError(
            "expected anchor `_CB_CACHE: dict[...]` in eschamoe.py; "
            "file structure may have changed."
        )
    src = src.replace(anchor, anchor + "\n\n" + PATCH_LOADER_SRC)

    # 2) inject baseline subtraction right before the final `return w.astype(mx.float16)`.
    tail_anchor = (
        "    w = per_block.reshape(in_features, out_features)\n"
        "    return w.astype(mx.float16)\n"
    )
    if tail_anchor not in src:
        raise RuntimeError(
            "expected tail anchor with reshape+return in escham_reconstruct; "
            "file structure may have changed."
        )
    src = src.replace(
        tail_anchor,
        "    w = per_block.reshape(in_features, out_features)\n"
        + PATCH_SUBTRACT_SRC
        + "    return w.astype(mx.float16)\n",
    )

    ESCHAMOE_PY.write_text(src)
    print(f"[wire] patched {ESCHAMOE_PY.name}: added _baseline_w0 + subtract line.")


# ---------------------------------------------------------------------------
# Step 4: verify against a reference dump (if present)
# ---------------------------------------------------------------------------

REFERENCE_CANDIDATES = [
    CB_DIR / "reference_dump.pkl",
    CB_DIR / "op_audit.pkl",
]


def verify() -> None:
    """If a reference dump exists, decode a sample code with the MLX path
    and compare to the reference `op(codes)` output. Otherwise, warn and
    skip.
    """
    ref_path = next((p for p in REFERENCE_CANDIDATES if p.exists()), None)
    if ref_path is None:
        print(
            "[verify] no reference dump found "
            f"(looked at {[p.name for p in REFERENCE_CANDIDATES]}). "
            "SKIPPING numerical verification — re-run once a reference "
            "op(codes)+codes tuple is dumped from CUDA."
        )
        return

    import pickle
    with open(ref_path, "rb") as f:
        ref = pickle.load(f)

    import mlx.core as mx
    from mlx_video.models.qwen3_5_moe_escha.eschamoe import escham_reconstruct

    bad = 0
    for entry in ref.get("samples", []):
        code_np   = entry["code"]        # int16 (bi, bj, 16K)
        w_ref_np  = entry["w"]           # fp16 (in_f, out_f)  = op(code)
        in_f      = int(entry["in_f"])
        out_f     = int(entry["out_f"])
        K         = int(entry["K"])
        code_mx   = mx.array(code_np.astype(np.int32))
        w_mx      = np.asarray(escham_reconstruct(code_mx, in_f, out_f, K, 1, False))
        diff = np.abs(w_mx.astype(np.float32) - w_ref_np.astype(np.float32))
        max_abs = float(diff.max())
        rel = max_abs / (float(np.abs(w_ref_np).max()) + 1e-9)
        ok = max_abs < 1e-2
        marker = "OK " if ok else "BAD"
        print(
            f"[verify] {marker} (in={in_f}, out={out_f}, K={K}): "
            f"max|Δ|={max_abs:.3e}  rel={rel:.3e}"
        )
        if not ok:
            bad += 1
            # short diagnostic on where the miss lives
            argmx = np.unravel_index(diff.argmax(), diff.shape)
            print(
                f"          argmax {argmx}: mlx={w_mx[argmx]:+.4e} "
                f"ref={w_ref_np[argmx]:+.4e}"
            )
    if bad:
        raise SystemExit(f"[verify] {bad} tile(s) failed the 1e-2 tolerance.")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", help="path to baseline_v2.npz (default: ~/Downloads/baseline_v2.npz or CWD)")
    ap.add_argument("--b64", help="path to a text file containing the base64 output (with or without BEGIN/END markers)")
    ap.add_argument("--skip-patch", action="store_true", help="only install the npz, don't touch eschamoe.py")
    ap.add_argument("--skip-verify", action="store_true", help="skip the numerical verification pass")
    args = ap.parse_args()

    install_npz(args)
    if not args.skip_patch:
        patch_eschamoe()
    if not args.skip_verify:
        verify()
    print("[wire] done.")


if __name__ == "__main__":
    main()
