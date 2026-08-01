"""Option A — reverse-engineer the packed layout of escham_reconstruct.

Approach: for a canonical (in_f=128, out_f=128, K=2) tile — the smallest
that the op accepts — perturb code[bi, bj, k, r] with the value 1 (one
codebook step of 1) and record which (row, col) of w_bare deltas.

The result is a dense LUT:
    layout[K][bi, bj, k, r] -> (row_start, col_start, code_pattern)
that tells us how the tile packing maps into the (in, out) output grid.

We compare against baseline (all zeros) and only capture the delta pattern,
which is the additive contribution of that single code slot.

We do this for both K=2 and K=3 configurations that Escha-W2 uses, and for
both a 128x128 and a slightly larger tile so we can extrapolate to full-size.

The extracted layout maps are pickled and downloaded to the Mac for use in
rewriting eschamoe.escham_reconstruct.
"""

from __future__ import annotations

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

app = modal.App("escha-layout-probe", image=image)


@app.function(gpu="A10G", timeout=1800, memory=16 * 1024)
def probe_layout() -> bytes:
    import numpy as np
    import torch
    import escha  # noqa: F401

    op = torch.ops.escha.escham_reconstruct
    device = "cuda"

    out: dict = {}

    for K in (2, 3):
        # For K=2 in=2048/out=1024 tile shape is (128, 64, 32). Choose a small
        # tile that the op accepts. From existing probes we know (in_p, out_p)
        # need one axis divisible by 128, and code.shape = (in_p/16, out_p/16, 16*K).
        # Use 128 x 128 minimum.
        in_f, out_f = 128, 128
        cshape = (in_f // 16, out_f // 16, 16 * K)   # (8, 8, 32 or 48)
        bi_max, bj_max, k_max = cshape

        p0 = torch.zeros(cshape, dtype=torch.int16, device=device)
        w0 = op(p0, in_f, out_f, K, True, False).detach().cpu().numpy().astype(np.float32)
        print(f"K={K}: baseline w0 shape={w0.shape} norm={np.linalg.norm(w0):.3e}")

        # For every (bi, bj, k_slot) — where k_slot in [0..16K) — set code=1
        # and record the delta location + shape.
        layout: dict = {}
        for bi in range(bi_max):
            for bj in range(bj_max):
                for k_slot in range(k_max):
                    p = torch.zeros(cshape, dtype=torch.int16, device=device)
                    p[bi, bj, k_slot] = 1
                    w = op(p, in_f, out_f, K, True, False).detach().cpu().numpy().astype(np.float32)
                    d = w - w0
                    nz_rows = np.where(np.any(np.abs(d) > 1e-6, axis=1))[0]
                    nz_cols = np.where(np.any(np.abs(d) > 1e-6, axis=0))[0]
                    layout[(bi, bj, k_slot)] = {
                        "nz_rows": nz_rows.tolist(),
                        "nz_cols": nz_cols.tolist(),
                        "delta_norm": float(np.linalg.norm(d)),
                        # Store the actual delta at the affected slot only —
                        # this is enough to reproduce the layout.
                        "delta_values": d[nz_rows[:, None], nz_cols[None, :]].tolist()
                            if len(nz_rows) and len(nz_cols) else [],
                    }
                    if bi == 0 and bj == 0 and k_slot < 3:
                        print(
                            f"  K={K} bi={bi} bj={bj} k_slot={k_slot}: "
                            f"nz_rows={nz_rows.tolist()[:8]} "
                            f"nz_cols={nz_cols.tolist()[:8]} "
                            f"delta_norm={np.linalg.norm(d):.3e}"
                        )

        out[f"K{K}"] = {
            "baseline_shape": w0.shape,
            "code_shape": cshape,
            "layout": layout,
            "in_f": in_f,
            "out_f": out_f,
        }

    # Also grab the code=1 vs code=v values, to reconstruct the codebook
    # entries (per-value delta pattern for a fixed slot).
    print("\n=== value sweep at (0,0,0) for K=2 ===")
    cshape = (128 // 16, 128 // 16, 32)
    p0 = torch.zeros(cshape, dtype=torch.int16, device=device)
    w0 = op(p0, 128, 128, 2, True, False).detach().cpu().numpy().astype(np.float32)
    value_map: dict = {}
    # Probe a modest subset of the 65536 possible codebook indices for pattern
    # verification (a full sweep would take ~30 min — do it in a followup).
    for v in [1, 2, 3, 4, 5, 7, 10, 16, 32, 64, 128, 256, 512, 1024,
              4096, 16384, 32767, -1, -100, -1000, -16384, -32768]:
        p = torch.zeros(cshape, dtype=torch.int16, device=device)
        p[0, 0, 0] = v
        w = op(p, 128, 128, 2, True, False).detach().cpu().numpy().astype(np.float32)
        d = w - w0
        nz_rows = np.where(np.any(np.abs(d) > 1e-6, axis=1))[0]
        nz_cols = np.where(np.any(np.abs(d) > 1e-6, axis=0))[0]
        value_map[int(v)] = {
            "nz_rows": nz_rows.tolist(),
            "nz_cols": nz_cols.tolist(),
            "delta_values": d[nz_rows[:, None], nz_cols[None, :]].tolist()
                if len(nz_rows) and len(nz_cols) else [],
        }
    out["value_sweep"] = value_map

    return pickle.dumps(out, protocol=pickle.HIGHEST_PROTOCOL)


@app.local_entrypoint()
def main() -> None:
    payload = probe_layout.remote()
    out = Path("/Users/kaede/mlx-video/mlx_video/models/qwen3_5_moe_escha/codebooks/layout_map.pkl")
    out.write_bytes(payload)
    print(f"[local] wrote {out} ({len(payload)/1e6:.2f} MB)")

    # Peek
    import pickle
    d = pickle.loads(payload)
    for k in ("K2", "K3"):
        info = d[k]
        print(f"{k}: baseline_shape={info['baseline_shape']} code_shape={info['code_shape']} "
              f"n_layout_entries={len(info['layout'])}")
        # Peek at a couple entries
        for i, (key, val) in enumerate(list(info['layout'].items())[:3]):
            print(f"  {key}: nz_rows={val['nz_rows'][:6]} nz_cols={val['nz_cols'][:6]} "
                  f"delta_norm={val['delta_norm']:.3e}")
    print(f"value_sweep: {len(d['value_sweep'])} probes")
