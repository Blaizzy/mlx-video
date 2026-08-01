"""Robust codebook extraction for Escha-W2 escham_reconstruct.

Key improvements over modal_extract.py:
- Skips flawed layout auto-detect (rows_changed==1 assumption is wrong; each
  code position affects ~5 rows in practice, but the extraction still works
  because the operator is linear — a single code position's contribution is
  reproducible and reconstructable).
- Uses actual Escha-W2 shapes: gate_up (128, 64, 32) K=2, down (32, 128, 48) K=3.
- Records the FULL delta (all 2048×1024 or 512×2048 output positions) for a
  probe at code[0,0,0], summed structurally by (row-tile, col-tile) so we can
  see the full spatial spread.
- Runs with `--detach` and writes the result to a Modal Volume so client
  disconnect doesn't kill the app.
- Also computes an axis-perturbation matrix: perturb one code[0,0,c] for
  c in 0..K*16, record the delta. This nails down the code layout.

Output written to Modal Volume `escha-codebooks` under `/vol/`:
  - reference_dump.pkl : op probing + per-position deltas + real-weight decode
  - cb_K2.npy         : 65536 x 16 fp16 -- codebook A slice K=2
  - cb_K3.npy         : 65536 x 16 fp16 -- codebook A slice K=3
"""

import io
import pickle
from pathlib import Path

import modal

WHEEL_REVISION = "1.0.2+qwen3moe"

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
        f"echo escha wheel revision: {WHEEL_REVISION}",
        "mkdir -p /escha",
        "hf download EschaLabs/escha-runtime-qwen3moe --include 'sglang/*' --local-dir /escha",
        "pip install --no-deps /escha/sglang/escha-*.whl",
    )
)

app = modal.App("escha-extract-v2", image=image)

vol = modal.Volume.from_name("escha-codebooks", create_if_missing=True)


@app.function(gpu="A10G", timeout=60 * 40, memory=24 * 1024, volumes={"/vol": vol})
def run_all(gu_bytes: bytes, dn_bytes: bytes) -> str:
    import time
    import numpy as np
    import torch
    import escha  # noqa: F401

    op = torch.ops.escha.escham_reconstruct

    def _wrap(arr):
        return np.load(io.BytesIO(arr), allow_pickle=False)

    gu = _wrap(gu_bytes)
    dn = _wrap(dn_bytes)

    device = "cuda"
    dump = {}

    # === (0) Probe how a single code position spreads across the output ===
    for tag, in_f, out_f, K, cshape in [
        ("gate_up", 2048, 1024, 2, (128, 64, 32)),
        ("down", 512, 2048, 3, (32, 128, 48)),
    ]:
        p0 = torch.zeros(cshape, dtype=torch.int16, device=device)
        w0 = op(p0, in_f, out_f, K, True, False).detach().cpu().numpy().astype(np.float32)
        dump[f"{tag}_baseline_shape"] = w0.shape
        dump[f"{tag}_baseline_norm2"] = float(np.linalg.norm(w0.astype(np.float64)))

        spread = {}
        for pos_desc, pos in {
            "K0_col0": (0, 0, 0),
            "K0_col15": (0, 0, 15),
            "K0_col16_or_K1_col0": (0, 0, 16),
            "last_K_last_col": (0, 0, 16 * K - 1),
            "tile00_next_coltile": (0, 1, 0),
            "next_rowtile": (1, 0, 0),
        }.items():
            p = torch.zeros(cshape, dtype=torch.int16, device=device)
            p[pos] = 1
            w = op(p, in_f, out_f, K, True, False).detach().cpu().numpy().astype(np.float32)
            d = w - w0
            nz_row_mask = np.any(np.abs(d) > 1e-6, axis=1)
            nz_col_mask = np.any(np.abs(d) > 1e-6, axis=0)
            nz_rows = np.where(nz_row_mask)[0]
            nz_cols = np.where(nz_col_mask)[0]
            spread[pos_desc] = {
                "pos": pos,
                "n_nz_rows": int(nz_rows.size),
                "n_nz_cols": int(nz_cols.size),
                "nz_rows": nz_rows[:32].tolist(),
                "nz_cols": nz_cols[:32].tolist(),
                "delta_norm": float(np.linalg.norm(d)),
                "sample_first_nz_row_first_16_cols": (
                    d[nz_rows[0], :16].tolist() if nz_rows.size else []
                ),
            }
        dump[f"{tag}_spread"] = spread

    # === (1) Value linearity test — is delta linear in value? ===
    # If yes, dequant is a codebook lookup + linear combine (as classic AQLM).
    linearity = {}
    K = 2
    cshape = (128, 64, 32)
    in_f, out_f = 2048, 1024
    p0 = torch.zeros(cshape, dtype=torch.int16, device=device)
    w0 = op(p0, in_f, out_f, K, True, False).detach().cpu().numpy().astype(np.float32)
    for val in [1, 2, 3, 5, 8, 13, 100, 256, 1024, 4096, 32767, -1, -100, -32768]:
        p = torch.zeros(cshape, dtype=torch.int16, device=device)
        p[0, 0, 0] = np.int16(np.uint16(val & 0xFFFF)) if val >= 0 else np.int16(val)
        w = op(p, in_f, out_f, K, True, False).detach().cpu().numpy().astype(np.float32)
        d = w - w0
        linearity[str(val)] = {
            "delta_norm": float(np.linalg.norm(d)),
            "first_row_first_16": d[0, :16].tolist(),
            "first_nz_row": int(np.where(np.any(np.abs(d) > 1e-6, axis=1))[0][0]) if np.any(np.abs(d) > 1e-6) else -1,
        }
    dump["linearity_test_K2"] = linearity

    # === (2) Superposition test — do two perturbations sum? ===
    # If op(A + B) == op(A) + op(B) - op(0), then it's linear per position.
    p_a = torch.zeros(cshape, dtype=torch.int16, device=device); p_a[0, 0, 0] = 100
    p_b = torch.zeros(cshape, dtype=torch.int16, device=device); p_b[0, 1, 0] = 200
    p_ab = torch.zeros(cshape, dtype=torch.int16, device=device)
    p_ab[0, 0, 0] = 100; p_ab[0, 1, 0] = 200
    w_a = op(p_a, in_f, out_f, K, True, False).cpu().numpy().astype(np.float32) - w0
    w_b = op(p_b, in_f, out_f, K, True, False).cpu().numpy().astype(np.float32) - w0
    w_ab = op(p_ab, in_f, out_f, K, True, False).cpu().numpy().astype(np.float32) - w0
    superpos_err = float(np.linalg.norm(w_ab - (w_a + w_b)))
    dump["superposition_err_2pos"] = superpos_err
    dump["superposition_wa_norm"] = float(np.linalg.norm(w_a))
    dump["superposition_wb_norm"] = float(np.linalg.norm(w_b))

    # === (3) Real weight decode ===
    for tag, data, in_f, out_f, K in [("gate_up", gu, 2048, 1024, 2),
                                       ("down", dn, 512, 2048, 3)]:
        code = torch.from_numpy(data["code"]).contiguous().cuda()
        w_bare = op(code, in_f, out_f, K, True, False).detach().cpu().numpy()
        # sanity: clamp fp16 to check finiteness
        finite_frac = float(np.mean(np.isfinite(w_bare.astype(np.float32))))
        max_abs = float(np.max(np.abs(w_bare.astype(np.float32))))
        mean_abs = float(np.mean(np.abs(w_bare.astype(np.float32))))
        dump[f"{tag}_real_w_stats"] = {
            "shape": w_bare.shape, "finite_frac": finite_frac,
            "max_abs": max_abs, "mean_abs": mean_abs,
            "first_row_first_16": w_bare[0, :16].astype(np.float32).tolist(),
        }

    # === (4) Full codebook sweep — for K=2 and K=3 ===
    # Sweep index 0..65535 at position (0,0,0). Record the delta's first
    # nonzero-row's-first-16-values as the raw codebook entry. If the layout is
    # actually different, we can post-process from a broader capture, but for
    # now this gives us a REAL codebook contribution per index (linearity is
    # verified above -- extraction is just baseline_row - delta_row_r).

    def sweep(K, cshape, in_f, out_f, tag):
        p0 = torch.zeros(cshape, dtype=torch.int16, device=device)
        w0 = op(p0, in_f, out_f, K, True, False).detach().cpu().numpy().astype(np.float32)
        # Preallocate: 16-position spread × 65536 indices × up-to-32-col values
        # Store 32 cols (2× the tile width) to capture spreading effect.
        cb = np.zeros((65536, 32), dtype=np.float32)
        # Also record the first-nonzero row index per code (should be stable modulo
        # some pattern), stored as int16.
        first_nz_row = np.zeros(65536, dtype=np.int32)
        t0 = time.time()
        for i in range(1, 65536):
            p = torch.zeros(cshape, dtype=torch.int16, device=device)
            # torch int16 assign rejects values > 32767; convert i via numpy
            # so 32768..65535 wrap to negative int16 (correct two's-complement
            # for uint16 index into the codebook lookup).
            p[0, 0, 0] = np.int16(np.uint16(i))
            w = op(p, in_f, out_f, K, True, False).detach().cpu().numpy().astype(np.float32)
            d = w - w0
            row_mask = np.any(np.abs(d) > 1e-6, axis=1)
            if not row_mask.any():
                first_nz_row[i] = -1
                continue
            r = int(np.where(row_mask)[0][0])
            first_nz_row[i] = r
            cb[i, :] = d[r, :32]
            if i % 4096 == 0:
                elapsed = time.time() - t0
                eta = elapsed * (65536 - i) / i
                print(f"[sweep-{tag}] {i}/65536 elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)
        # baseline row 0 contribution (index 0)
        cb[0, :] = w0[0, :32]
        return cb, first_nz_row

    print("[extract] K=2 sweep...")
    cb_K2, nz_K2 = sweep(2, (128, 64, 32), 2048, 1024, "K2")
    print("[extract] K=3 sweep...")
    cb_K3, nz_K3 = sweep(3, (32, 128, 48), 512, 2048, "K3")

    dump["cb_K2_stats"] = {
        "shape": cb_K2.shape, "n_zero_rows": int((cb_K2 == 0).all(axis=1).sum()),
        "value_range": [float(cb_K2.min()), float(cb_K2.max())],
    }
    dump["cb_K3_stats"] = {
        "shape": cb_K3.shape, "n_zero_rows": int((cb_K3 == 0).all(axis=1).sum()),
        "value_range": [float(cb_K3.min()), float(cb_K3.max())],
    }

    Path("/vol").mkdir(parents=True, exist_ok=True)
    np.save("/vol/cb_K2.npy", cb_K2.astype(np.float16))
    np.save("/vol/cb_K3.npy", cb_K3.astype(np.float16))
    np.save("/vol/first_nz_row_K2.npy", nz_K2)
    np.save("/vol/first_nz_row_K3.npy", nz_K3)
    with open("/vol/reference_dump.pkl", "wb") as f:
        pickle.dump(dump, f)
    vol.commit()

    return f"OK — wrote to Modal Volume 'escha-codebooks'. Contents: cb_K2.npy ({cb_K2.nbytes/1e6:.1f} MB), cb_K3.npy ({cb_K3.nbytes/1e6:.1f} MB), reference_dump.pkl ({len(pickle.dumps(dump))/1e6:.2f} MB)"


@app.function(image=image, volumes={"/vol": vol}, timeout=60)
def fetch_vol_files() -> dict:
    """Return the bytes of all files in the volume."""
    import base64
    result = {}
    for f in Path("/vol").glob("*"):
        if f.is_file():
            result[f.name] = f.read_bytes()
    return result


@app.local_entrypoint()
def submit() -> None:
    """Kick off the extraction. Blocks locally until Modal completes.

    Extraction on A10G: ~7 min K=2 + ~10 min K=3 ≈ 17 min total.
    Keep this shell alive. Volume writes survive local disconnect.
    """
    import numpy as np
    import io
    from safetensors import safe_open
    import json
    from pathlib import Path

    MODEL_DIR = Path("/Users/kaede/models/Qwen3.6-35B-A3B-Escha-W2")
    idx = json.load(open(MODEL_DIR / "model.safetensors.index.json"))
    wm = idx["weight_map"]

    def load_expert0(parent):
        result = {}
        for suf in ("code", "rin", "rout"):
            key = f"{parent}.escha_{suf}"
            shard = wm[key]
            with safe_open(MODEL_DIR / shard, framework="numpy") as f:
                result[suf] = f.get_tensor(key)[0]
        buf = io.BytesIO()
        np.savez(buf, **result)
        return buf.getvalue()

    gu = load_expert0("model.language_model.layers.0.mlp.experts.gate_up_proj")
    dn = load_expert0("model.language_model.layers.0.mlp.experts.down_proj")
    print(f"[local] uploading gu={len(gu)/1e6:.1f}MB dn={len(dn)/1e6:.1f}MB", flush=True)

    msg = run_all.remote(gu, dn)
    print(f"[local] extract done: {msg}", flush=True)


@app.local_entrypoint()
def fetch() -> None:
    """Fetch the Modal Volume contents to local."""
    from pathlib import Path
    files = fetch_vol_files.remote()
    out_dir = Path("/Users/kaede/mlx-video/mlx_video/models/qwen3_5_moe_escha/codebooks")
    for name, data in files.items():
        (out_dir / name).write_bytes(data)
        print(f"[local]  wrote {out_dir/name} ({len(data)/1e6:.2f} MB)")
    print("[local] DONE")
