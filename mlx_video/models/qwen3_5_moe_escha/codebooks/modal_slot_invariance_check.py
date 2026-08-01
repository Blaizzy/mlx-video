"""Verify slot-invariance claim + reproduce w_ref path via full-block codebook.

Extracts:
 1. For K=2, at k_slot=0, put v=7 at (bi=0,bj=0). Read block(0,0)-w0.
    Then put v=7 at (bi=5,bj=3). Read block(5,3)-w0. Are they equal?
 2. Also: reconstruct sample gate_up_L0_E0 via THE SAME modal_smart_probe recipe
    but using the FULL block-codebook computed INLINE (no compaction).
    If this matches ref → compaction is at fault.
    If NOT → slot-invariance itself fails.
"""

from __future__ import annotations

import base64
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
    .pip_install("numpy", "safetensors", "huggingface_hub[cli]", "hf_transfer")
    .run_commands(
        f"echo escha wheel revision: {WHEEL_REVISION}",
        "mkdir -p /escha",
        "hf download EschaLabs/escha-runtime-qwen3moe --include 'sglang/*' --local-dir /escha",
        "pip install --no-deps /escha/sglang/escha-*.whl",
    )
)

vol = modal.Volume.from_name("escha-codebooks", create_if_missing=True)
app = modal.App("escha-slot-invariance", image=image)


@app.function(gpu="A10G", timeout=1800, memory=32 * 1024, volumes={"/vol": vol})
def check() -> dict:
    import json
    import os
    import numpy as np
    import torch
    from safetensors import safe_open
    from huggingface_hub import snapshot_download
    import escha  # noqa: F401

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    op = torch.ops.escha.escham_reconstruct
    device = "cuda"

    in_f, out_f, K = 2048, 1024, 2
    cshape = (128, 64, 32)  # (bi, bj, k)
    bi_max, bj_max, k_max = cshape

    # Baseline
    p0 = torch.zeros(cshape, dtype=torch.int16, device=device)
    w0 = op(p0, in_f, out_f, K, True, False).detach().cpu().numpy().astype(np.float32)

    out: dict = {"in_f": in_f, "out_f": out_f, "K": K}

    # --- Test 1: slot-invariance for a single (k=0, v=7) ---
    def _block_of(w, bi, bj):
        return w[bi*16:(bi+1)*16, bj*16:(bj+1)*16]

    v = 7
    positions_to_test = [(0, 0), (5, 3), (10, 20), (100, 50)]
    deltas = []
    for bi, bj in positions_to_test:
        p = torch.zeros(cshape, dtype=torch.int16, device=device)
        p[bi, bj, 0] = v
        w = op(p, in_f, out_f, K, True, False).detach().cpu().numpy().astype(np.float32)
        d = _block_of(w, bi, bj) - _block_of(w0, bi, bj)
        deltas.append(d.copy())
        # Also examine OTHER blocks: is there leakage?
        w_delta_full = w - w0
        # Zero out our block; check leakage.
        w_delta_full[bi*16:(bi+1)*16, bj*16:(bj+1)*16] = 0
        leak = float(np.abs(w_delta_full).max())
        print(f"[slot-inv] (bi={bi:3d}, bj={bj:2d}, k=0, v={v}): "
              f"block_max={float(np.abs(d).max()):.3e} leak_max={leak:.3e}",
              flush=True)

    ref_d = deltas[0]
    for (bi, bj), d in zip(positions_to_test[1:], deltas[1:], strict=True):
        diff = float(np.abs(d - ref_d).max())
        print(f"[slot-inv] delta(0,0) vs delta({bi},{bj}) max|Δ|={diff:.3e}", flush=True)
        out[f"diff_{bi}_{bj}"] = diff

    # --- Test 2: does the modal_smart_probe recipe reproduce a real expert? ---
    print("\n[reproduce] downloading model...", flush=True)
    model_dir = snapshot_download(
        "EschaLabs/Qwen3.6-35B-A3B-Escha-W2",
        cache_dir="/vol/hf_cache",
        allow_patterns=[
            "config.json",
            "model.safetensors.index.json",
            "model-00001-of-*.safetensors",
            "model-00002-of-*.safetensors",
        ],
    )
    idx = json.load(open(f"{model_dir}/model.safetensors.index.json"))
    wm = idx["weight_map"]

    def _load_expert(name, expert=0):
        shard = wm[name]
        with safe_open(f"{model_dir}/{shard}", framework="pt") as f:
            return f.get_tensor(name)[expert]

    code = _load_expert(
        "model.language_model.layers.0.mlp.experts.gate_up_proj.escha_code", expert=0
    )
    code_gpu = code.to(device)
    w_ref = op(code_gpu, in_f, out_f, K, True, False).detach().cpu().numpy().astype(np.float32)
    print(f"[reproduce] w_ref shape={w_ref.shape} |w_ref|_inf={float(np.abs(w_ref).max()):.3e}", flush=True)

    # Now compute per-k full block-codebook INLINE (no compaction). For each k,
    # loop v in batches, extract cb_k[v] as delta at the single-set (bi,bj).
    # (Same recipe as modal_smart_probe.py, kept in-memory.)
    cb_full = np.zeros((k_max, 65536, 16, 16), dtype=np.float16)
    block_positions = [(bi, bj) for bi in range(bi_max) for bj in range(bj_max)]
    blocks_per_op = bi_max * bj_max
    import time
    t_start = time.time()
    for k in range(k_max):
        for v_start in range(0, 65536, blocks_per_op):
            v_end = min(v_start + blocks_per_op, 65536)
            n_probes = v_end - v_start
            p = torch.zeros(cshape, dtype=torch.int16, device=device)
            for i in range(n_probes):
                vi = v_start + i
                bi, bj = block_positions[i]
                p[bi, bj, k] = np.int16(np.uint16(vi & 0xFFFF)) if vi < 32768 else np.int16(vi - 65536)
            w = op(p, in_f, out_f, K, True, False)
            w_np = w.detach().cpu().numpy().astype(np.float32)
            for i in range(n_probes):
                vi = v_start + i
                bi, bj = block_positions[i]
                cb_full[k, vi] = (w_np[bi*16:(bi+1)*16, bj*16:(bj+1)*16] - w0[bi*16:(bi+1)*16, bj*16:(bj+1)*16]).astype(np.float16)
    print(f"[reproduce] full cb extracted in {time.time()-t_start:.1f}s", flush=True)

    # Reconstruct via extracted layout.
    code_np = code.cpu().numpy()
    code_u = code_np.astype(np.int32) & 0xFFFF
    w_recon = w0.copy()
    for k in range(k_max):
        blocks = cb_full[k, code_u[:, :, k]].astype(np.float32)   # (bi, bj, 16, 16)
        blocks_r = blocks.transpose(0, 2, 1, 3).reshape(bi_max * 16, bj_max * 16)
        w_recon += blocks_r

    diff = np.abs(w_recon - w_ref)
    max_d = float(diff.max())
    mean_d = float(diff.mean())
    argmx = np.unravel_index(diff.argmax(), diff.shape)
    print(
        f"[reproduce] gate_up_L0_E0: max_diff={max_d:.3e} mean_diff={mean_d:.3e} "
        f"argmax={argmx} w_recon={w_recon[argmx]:.3f} w_ref={w_ref[argmx]:.3f}",
        flush=True,
    )
    out["reproduce_max_diff"] = max_d
    out["reproduce_mean_diff"] = mean_d
    out["reproduce_argmax"] = list(argmx)
    return out


@app.local_entrypoint()
def main() -> None:
    print("[local] launching slot-invariance check on A10G ...", flush=True)
    result = check.remote()
    print(f"\n[local] result: {result}")
