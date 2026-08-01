"""Extract the FULL layout of escham_reconstruct in a handful of op calls.

Findings from `modal_op_audit.py`:
  1. Op is EXACTLY linear in codes: op(A+B) = op(A) + op(B) - op(0)  (|diff|=0)
  2. Codebook is SLOT-INVARIANT across (bi, bj): same delta at any (bi, bj)
     up to a (bi*16, bj*16) offset.
  3. Each (k, v) pair produces a fixed (16, 16) block pattern with ~5-9 nonzero
     positions.

Extraction algorithm:
  For each K ∈ {2, 3}:
    Use cshape (128, 64, 32) for K=2  → 8192 blocks per op
    Use cshape (32, 128, 48) for K=3  → 4096 blocks per op
    For each k_slot ∈ [0, 16*K):
      Loop v ∈ [0, 65536) in batches of `blocks_per_op`:
        Build ONE code tensor where slot[bi_i, bj_i, k_slot] = v_i for a batch
        of `blocks_per_op` distinct (bi, bj) pairs.
        Call op once. Read out each (16, 16) block. Store as codebook entry.

Result:
  cb_K2 : (32, 65536, 16, 16) fp16  — 4.3 GB
  cb_K3 : (48, 65536, 16, 16) fp16  — 6.4 GB
Total ~11 GB.

But we know most codes produce SPARSE patterns (5-9 nonzeros). We can store
compact: for each (K, k_slot) collect ONCE a mask of which (row, col) positions
are nonzero, then store only the sparse values as (K, 65536, n_nz) fp16.
This shrinks to <100 MB total. See `_compact_layout` below.
"""

from __future__ import annotations

import pickle
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
app = modal.App("escha-smart-probe", image=image)


@app.function(gpu="A10G", timeout=3600 * 2, memory=32 * 1024, volumes={"/vol": vol})
def extract_layout() -> dict:
    import numpy as np
    import torch
    import escha  # noqa: F401

    op = torch.ops.escha.escham_reconstruct
    device = "cuda"

    result: dict = {}

    for K, cshape, in_f, out_f in [
        (2, (128, 64, 32), 2048, 1024),
        (3, (32, 128, 48), 512, 2048),
    ]:
        bi_max, bj_max, k_max = cshape
        blocks_per_op = bi_max * bj_max
        print(f"\n=== K={K} cshape={cshape} in_f={in_f} out_f={out_f} blocks_per_op={blocks_per_op} ===", flush=True)

        # Baseline (all-zeros).
        p0 = torch.zeros(cshape, dtype=torch.int16, device=device)
        w0 = op(p0, in_f, out_f, K, True, False).detach().cpu().numpy().astype(np.float32)
        print(f"  baseline w0 shape={w0.shape} norm={np.linalg.norm(w0):.3e}", flush=True)

        # Assign a distinct (bi, bj) to each of blocks_per_op probes.
        block_positions = [(bi, bj) for bi in range(bi_max) for bj in range(bj_max)]

        # For each k_slot, we'll store a full (65536, 16, 16) fp16 tensor of block
        # deltas. This is 32 MB per k_slot. Total: 32 × 32 MB = 1 GB (K=2), 48 × 32 MB = 1.5 GB (K=3).
        # We'll compact to sparse form at the end.
        cb_full = np.zeros((k_max, 65536, 16, 16), dtype=np.float16)
        t_k_start = time.time()
        for k in range(k_max):
            t0 = time.time()
            # Loop v in batches of `blocks_per_op`.
            n_ops = 0
            for v_start in range(0, 65536, blocks_per_op):
                v_end = min(v_start + blocks_per_op, 65536)
                n_probes = v_end - v_start
                # Build code tensor.
                p = torch.zeros(cshape, dtype=torch.int16, device=device)
                for i in range(n_probes):
                    v = v_start + i
                    bi, bj = block_positions[i]
                    # int16 wrap: values > 32767 become negative
                    p[bi, bj, k] = np.int16(np.uint16(v & 0xFFFF)) if v < 32768 else np.int16(v - 65536)
                # Call op ONCE for these n_probes probes.
                w = op(p, in_f, out_f, K, True, False)
                # Extract each (bi, bj) block as the codebook entry for (k, v).
                w_np = w.detach().cpu().numpy().astype(np.float32)
                for i in range(n_probes):
                    v = v_start + i
                    bi, bj = block_positions[i]
                    block = w_np[bi*16:(bi+1)*16, bj*16:(bj+1)*16] - w0[bi*16:(bi+1)*16, bj*16:(bj+1)*16]
                    cb_full[k, v] = block.astype(np.float16)
                n_ops += 1
            dt = time.time() - t0
            total = time.time() - t_k_start
            print(f"  K={K} k_slot={k:2d} done: {n_ops} op calls, {dt:.1f}s (cumulative {total:.1f}s)", flush=True)

        result[f"cb_K{K}"] = cb_full   # (k_max, 65536, 16, 16) fp16
        print(f"  K={K} total: {time.time() - t_k_start:.1f}s")

    # === Sanity check: reproduce a real expert weight from the extracted layout. ===
    print("\n=== sanity check: reproduce w_bare for gate_up L0/E0 ===", flush=True)
    from safetensors import safe_open
    from huggingface_hub import snapshot_download
    import json

    print("  downloading model snapshot (cached in volume)...", flush=True)
    model_dir = snapshot_download(
        "EschaLabs/Qwen3.6-35B-A3B-Escha-W2",
        cache_dir="/vol/hf_cache",
    )
    idx = json.load(open(f"{model_dir}/model.safetensors.index.json"))
    wm = idx["weight_map"]

    def _load(name, expert=0):
        shard = wm[name]
        with safe_open(f"{model_dir}/{shard}", framework="pt") as f:
            return f.get_tensor(name)[expert]

    for tag, pfx, in_f, out_f, K, cshape in [
        ("gate_up", "model.language_model.layers.0.mlp.experts.gate_up_proj", 2048, 1024, 2, (128, 64, 32)),
        ("down",    "model.language_model.layers.0.mlp.experts.down_proj",    512,  2048, 3, (32, 128, 48)),
    ]:
        code = _load(f"{pfx}.escha_code", expert=0).cuda()
        w_ref = op(code, in_f, out_f, K, True, False).detach().cpu().numpy().astype(np.float32)

        # Reconstruct via extracted layout.
        cb = result[f"cb_K{K}"]        # (k_max, 65536, 16, 16) fp16
        # Baseline for this cshape:
        p0 = torch.zeros(cshape, dtype=torch.int16, device=device)
        w0_np = op(p0, in_f, out_f, K, True, False).detach().cpu().numpy().astype(np.float32)

        w_recon = w0_np.copy()   # start from baseline
        code_np = code.cpu().numpy()   # (bi_max, bj_max, k_max) int16
        # Convert int16 → uint16 index
        code_u = code_np.astype(np.int32) & 0xFFFF   # (bi_max, bj_max, k_max)
        bi_max, bj_max, k_max = cshape
        for k in range(k_max):
            # Gather cb[k, code_u[:,:,k]] → (bi_max, bj_max, 16, 16)
            blocks = cb[k, code_u[:, :, k]]   # (bi_max, bj_max, 16, 16) fp16
            # Add to w_recon at the right positions.
            # blocks[bi, bj] → w_recon[bi*16:(bi+1)*16, bj*16:(bj+1)*16]
            # Reshape trick:
            blocks_reshaped = blocks.astype(np.float32).transpose(0, 2, 1, 3).reshape(bi_max * 16, bj_max * 16)
            w_recon += blocks_reshaped

        diff = np.abs(w_recon - w_ref)
        max_diff = float(diff.max())
        mean_diff = float(diff.mean())
        rel = float(np.linalg.norm(diff) / np.linalg.norm(w_ref))
        print(f"  {tag}: |w_ref|={np.linalg.norm(w_ref):.3e} max_diff={max_diff:.3e} mean_diff={mean_diff:.3e} rel={rel:.3e}", flush=True)
        result[f"sanity_{tag}"] = {"max_diff": max_diff, "mean_diff": mean_diff, "rel": rel}

    # === Save extracted layout to volume ===
    print("\n=== saving to /vol ===", flush=True)
    Path("/vol/layout_v2").mkdir(parents=True, exist_ok=True)
    for K in (2, 3):
        cb = result[f"cb_K{K}"]
        np.save(f"/vol/layout_v2/cb_K{K}.npy", cb)
        print(f"  wrote /vol/layout_v2/cb_K{K}.npy ({cb.nbytes/1e9:.2f} GB)", flush=True)

    # === Also compute a compact-sparse form ===
    print("\n=== computing sparse-compact form ===", flush=True)
    compact: dict = {}
    for K in (2, 3):
        cb = result[f"cb_K{K}"]   # (k_max, 65536, 16, 16) fp16
        k_max = cb.shape[0]
        # For each k_slot, find the union of (row, col) positions that are ever
        # nonzero across all 65536 codes. Store cb[k, :, mask] compactly.
        mask_per_k: list = []
        vals_per_k: list = []
        for k in range(k_max):
            # Any code that has this position nonzero?
            any_nz = np.any(cb[k].astype(np.float32) != 0, axis=0)   # (16, 16) bool
            positions = np.argwhere(any_nz)   # (n_nz, 2)
            n_nz = positions.shape[0]
            # Extract the (65536, n_nz) values.
            vals = cb[k, :, positions[:, 0], positions[:, 1]]   # (n_nz, 65536)
            # Transpose to (65536, n_nz)
            vals = vals.T.astype(np.float16)
            mask_per_k.append(positions.astype(np.int8))
            vals_per_k.append(vals)
            if k < 3:
                print(f"  K={K} k={k}: n_nz_positions={n_nz} vals_shape={vals.shape}", flush=True)
        # Since n_nz may differ per k, store as a list.
        compact[f"K{K}_positions"] = mask_per_k
        compact[f"K{K}_values"] = vals_per_k
    # Serialize.
    with open("/vol/layout_v2/compact.pkl", "wb") as f:
        pickle.dump(compact, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  wrote /vol/layout_v2/compact.pkl", flush=True)

    vol.commit()

    # Return sanity metrics + sizes (avoid returning multi-GB tensors).
    return {
        "sanity_gate_up": result["sanity_gate_up"],
        "sanity_down": result["sanity_down"],
        "cb_K2_shape": result["cb_K2"].shape,
        "cb_K3_shape": result["cb_K3"].shape,
        "cb_K2_nbytes": int(result["cb_K2"].nbytes),
        "cb_K3_nbytes": int(result["cb_K3"].nbytes),
    }


@app.function(image=image, volumes={"/vol": vol}, timeout=600)
def fetch() -> dict:
    """Fetch the compact codebook + optionally the full form."""
    import os
    result = {}
    for name in ("compact.pkl",):
        path = f"/vol/layout_v2/{name}"
        if os.path.exists(path):
            with open(path, "rb") as f:
                result[name] = f.read()
    return result


@app.local_entrypoint()
def main() -> None:
    metrics = extract_layout.remote()
    print(f"\n[local] extract metrics: {metrics}")

    files = fetch.remote()
    out_dir = Path("/Users/kaede/mlx-video/mlx_video/models/qwen3_5_moe_escha/codebooks")
    for name, data in files.items():
        target = out_dir / f"layout_v2_{name}"
        target.write_bytes(data)
        print(f"[local] wrote {target} ({len(data)/1e6:.2f} MB)")
