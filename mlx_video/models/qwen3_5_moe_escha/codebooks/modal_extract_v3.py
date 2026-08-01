"""modal_extract_v3.py — vectorised, faster extraction of solo + cross codebooks.

Optimisations vs v1:
  - Vectorised value-setting into torch tensor (avoid 8192-iter Python loops).
  - Smaller cross-function reference set (R=8, revisit if per-pixel rank higher).
  - Progress prints every batch with sys.stdout.flush().
  - Simpler layout: no K-layer 2/3 verification (we'll verify by comparing to
    real op(code) on the final w_bare).

Data captured:
  - w0 baseline (per variant).
  - solo_k(v) for k=0..3 (unique row-pattern classes) at REFERENCE col-shift.
    Also solo_k(v) for k=4, 8, 12 to verify col-shift replication.
  - cross(v0, v1) reference sweep for pair (0,1) and (0,3): R rows + R columns
    for R=8. Enough to fit rank-8 factorisation.
  - K-layer 2/3 identity: at end, quick sample of solo_16(v) for 8 values.

Output: pickled dict on Modal volume + returned bytes.
"""

from __future__ import annotations

import pickle
import sys
import time
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
    .pip_install("numpy", "safetensors", "huggingface_hub[cli]", "hf_transfer")
    .run_commands(
        f"echo escha wheel revision: {WHEEL_REVISION}",
        "mkdir -p /escha",
        "hf download EschaLabs/escha-runtime-qwen3moe --include 'sglang/*' --local-dir /escha",
        "pip install --no-deps /escha/sglang/escha-*.whl",
    )
)

vol = modal.Volume.from_name("escha-codebooks", create_if_missing=True)
app = modal.App("escha-extract-v3", image=image)


def _extract_variant(name, in_f, out_f, K, cshape, op, device):
    import numpy as np
    import torch

    bi_max, bj_max, k_max_all = cshape
    max_tiles = bi_max * bj_max
    print(f"[{name}] K={K}, cshape={cshape}, max_tiles={max_tiles}", flush=True)

    p0 = torch.zeros(cshape, dtype=torch.int16, device=device)
    w0 = op(p0, in_f, out_f, K, True, False).detach().cpu().numpy().astype(np.float32)
    w0_blocks = w0.reshape(bi_max, 16, bj_max, 16).transpose(0, 2, 1, 3)  # (bi, bj, 16, 16)

    # Precompute the (bi, bj) grid for max_tiles positions.
    bi_flat = torch.arange(max_tiles, device=device) // bj_max
    bj_flat = torch.arange(max_tiles, device=device) % bj_max

    def _to_int16_arr(vals: np.ndarray) -> torch.Tensor:
        """Convert np.int32 array to torch.int16, wrapping signed."""
        # Modular signed conversion
        v = (vals & 0xFFFF).astype(np.int32)
        v = np.where(v >= 32768, v - 65536, v)
        return torch.from_numpy(v.astype(np.int16)).to(device)

    def _sweep_two_slots(k_i: int, k_j: int,
                         v_i_arr: np.ndarray, v_j_arr: np.ndarray) -> np.ndarray:
        """For pairs (v_i_arr[t], v_j_arr[t]), t=0..T-1, set slot k_i=v_i, k_j=v_j
        at distinct (bi,bj) tiles and return delta of shape (T, 16, 16).
        If k_j < 0, only set slot k_i."""
        T = len(v_i_arr)
        results_list = []
        for start in range(0, T, max_tiles):
            end = min(start + max_tiles, T)
            n = end - start
            p = torch.zeros(cshape, dtype=torch.int16, device=device)
            bi_use = bi_flat[:n]
            bj_use = bj_flat[:n]
            vi_t = _to_int16_arr(v_i_arr[start:end])
            p[bi_use, bj_use, k_i] = vi_t
            if k_j >= 0:
                vj_t = _to_int16_arr(v_j_arr[start:end])
                p[bi_use, bj_use, k_j] = vj_t
            w = op(p, in_f, out_f, K, True, False).detach().cpu().numpy().astype(np.float32)
            w_blocks = w.reshape(bi_max, 16, bj_max, 16).transpose(0, 2, 1, 3)
            for tt in range(n):
                bi_v = int(bi_use[tt].item()); bj_v = int(bj_use[tt].item())
                results_list.append((w_blocks[bi_v, bj_v] - w0_blocks[bi_v, bj_v]).astype(np.float16))
        return np.stack(results_list)  # (T, 16, 16)

    variant = {
        "in_f": in_f, "out_f": out_f, "K": K,
        "cshape": list(cshape),
        "w0": w0.astype(np.float16),
    }
    t_start = time.time()

    # ---- solo ----
    solo_data = {}
    # Extract ALL slots individually (translation invariance is unreliable).
    solo_slots = list(range(16 * K))
    for k in solo_slots:
        vs = np.arange(65536, dtype=np.int32)
        zeros = np.zeros_like(vs)
        deltas = _sweep_two_slots(k, -1, vs, zeros)
        solo_data[k] = deltas
        print(f"  [{name}] solo_k={k}: {deltas.shape} |d|max={float(np.abs(deltas).max()):.3f} "
              f"[{time.time()-t_start:.1f}s]", flush=True)
        sys.stdout.flush()
    variant["solo_data"] = solo_data

    # ---- cross ----
    # Extract cross with DIVERSE ref values so we can factor the joint decode
    # over a full 65536-code range at each interacting pair.
    ref_vals = np.array([1, 100, 10000, -1], dtype=np.int32)
    R = len(ref_vals)
    cross_data = {}
    # Extract cross for EVERY interacting pair (translation invariance unreliable)
    # 15 pairs per K-layer × K K-layers.
    # Type A: (2m, 2m+1) for m=0..7
    # Type B: (2m, 2m+3) for m=0..6
    cross_pairs = []
    for k_layer in range(K):
        base = 16 * k_layer
        # Type A
        for m in range(8):
            cross_pairs.append((base + 2*m, base + 2*m + 1, f"A_L{k_layer}_m{m}"))
        # Type B
        for m in range(7):
            cross_pairs.append((base + 2*m, base + 2*m + 3, f"B_L{k_layer}_m{m}"))

    for (k_i, k_j, tag) in cross_pairs:
        # R rows: for each v0_ref, sweep v1 = 0..65535
        rows_deltas = np.zeros((R, 65536, 16, 16), dtype=np.float16)
        for r_idx, v0_ref in enumerate(ref_vals):
            vs_j = np.arange(65536, dtype=np.int32)
            vs_i = np.full_like(vs_j, int(v0_ref))
            deltas = _sweep_two_slots(k_i, k_j, vs_i, vs_j)  # (65536, 16, 16)
            rows_deltas[r_idx] = deltas
        # R cols: for each v1_ref, sweep v0 = 0..65535
        cols_deltas = np.zeros((R, 65536, 16, 16), dtype=np.float16)
        for c_idx, v1_ref in enumerate(ref_vals):
            vs_i = np.arange(65536, dtype=np.int32)
            vs_j = np.full_like(vs_i, int(v1_ref))
            deltas = _sweep_two_slots(k_i, k_j, vs_i, vs_j)
            cols_deltas[c_idx] = deltas
        cross_data[(k_i, k_j, tag)] = {
            "ref_values": ref_vals.tolist(),
            "rows_deltas": rows_deltas,
            "cols_deltas": cols_deltas,
        }
        print(f"  [{name}] cross ({k_i},{k_j}) {tag}: rows={rows_deltas.shape} "
              f"cols={cols_deltas.shape}  |max|={float(np.abs(cols_deltas).max()):.3f} "
              f"[{time.time()-t_start:.1f}s]", flush=True)
        sys.stdout.flush()
    variant["cross_data"] = {str(k): v for k, v in cross_data.items()}

    variant["extraction_time_s"] = time.time() - t_start
    print(f"[{name}] variant time: {variant['extraction_time_s']:.1f}s", flush=True)
    return variant


@app.function(gpu="A10G", timeout=3600, memory=16 * 1024, volumes={"/vol": vol})
def probe() -> bytes:
    import numpy as np
    import torch
    import escha  # noqa: F401
    from safetensors import safe_open
    from huggingface_hub import snapshot_download
    import os
    import json

    op = torch.ops.escha.escham_reconstruct
    device = "cuda"

    variants = [
        ("gate_up", 2048, 1024, 2, (128, 64, 32)),
        ("down_proj",  512, 2048, 3, (32, 128, 48)),
    ]
    results = {}
    for (name, in_f, out_f, K, cshape) in variants:
        print(f"\n=== extracting variant {name} (K={K}) ===", flush=True)
        results[name] = _extract_variant(name, in_f, out_f, K, cshape, op, device)

    # ---- reference samples ----
    print("\n=== extracting real-expert reference samples ===", flush=True)
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    model_dir = snapshot_download(
        "EschaLabs/Qwen3.6-35B-A3B-Escha-W2",
        cache_dir="/vol/hf_cache",
        allow_patterns=[
            "config.json",
            "model.safetensors.index.json",
            "model-00001-of-*.safetensors",
        ],
    )
    idx = json.load(open(f"{model_dir}/model.safetensors.index.json"))
    wm = idx["weight_map"]
    ref_samples = {}
    for tensor_name, proj_variant in [
        ("model.language_model.layers.0.mlp.experts.gate_up_proj.escha_code", "gate_up"),
        ("model.language_model.layers.0.mlp.experts.down_proj.escha_code", "down_proj"),
    ]:
        shard = wm[tensor_name]
        with safe_open(f"{model_dir}/{shard}", framework="pt") as f:
            t = f.get_tensor(tensor_name)
        code = t[0].to(device)
        code_np = t[0].cpu().numpy().astype(np.int16)
        in_f = results[proj_variant]["in_f"]
        out_f = results[proj_variant]["out_f"]
        K = results[proj_variant]["K"]
        w_ref = op(code, in_f, out_f, K, True, False).detach().cpu().numpy().astype(np.float32)
        ref_samples[proj_variant] = {
            "code": code_np,
            "w_ref": w_ref.astype(np.float16),
        }
        print(f"  ref {proj_variant}: |w|max={float(np.abs(w_ref).max()):.3f}", flush=True)
    results["ref_samples"] = ref_samples

    os.makedirs("/vol/joint_probes", exist_ok=True)
    out_path = "/vol/joint_probes/full_extract_v4.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    with open(out_path, "rb") as f:
        blob = f.read()
    vol.commit()
    print(f"[done] wrote {out_path} ({len(blob)/(1024*1024):.1f} MiB)", flush=True)
    return blob


@app.local_entrypoint()
def main() -> None:
    print("[local] launching v3 extraction ...", flush=True)
    blob = probe.remote()
    out = Path("/Users/kaede/mlx-video/mlx_video/models/qwen3_5_moe_escha/codebooks/full_extract_v3.pkl")
    out.write_bytes(blob)
    print(f"[local] wrote {out} ({len(blob)/(1024*1024):.1f} MiB)")
