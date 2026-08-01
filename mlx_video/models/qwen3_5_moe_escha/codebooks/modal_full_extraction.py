"""modal_full_extraction.py — extract solo + cross codebooks for K=2 and K=3.

Structure (verified in §3 of ESCHA_LAYOUT_NOTES.md):
  delta(codes) = sum_k solo_k(codes[k]) + sum_{(i,j) interacting} cross_ij(codes[i], codes[j])

Translation invariance:
  - solo_k depends only on k%4 (row pattern) and k//4 (col shift). Extract solo
    for k=0..3 (4 unique row-pattern functions); replicate for larger k via col-shift.
  - cross_ij depends only on (k_j - k_i) type: type A = diff-1, type B = diff-3.
    Extract cross for pair (0,1) [type A] and pair (0,3) [type B]; replicate.

For each K-layer's 16 slots, apply within one K-layer. Then sum across K-layers.

Extraction (per K variant):
  A. Baseline w0: op(zeros).
  B. Solo functions: for k=0..3, sweep v=0..65535 recording delta at reference tile.
     Also extract solo for k=4, k=8, k=12 to VERIFY col-shift translation invariance.
     Same for k=16..19 to verify K-layer independence.
  C. Cross functions:
     - Type A (pair (0,1)): for v1_ref in {1,...,32} sweep v0 = 0..65535 → 32 columns.
                            for v0_ref in {1,...,32} sweep v1 = 0..65535 → 32 rows.
     - Type B (pair (0,3)): same protocol.
  D. Verify K-layer independence: extract solo for slots 16..19 (K-layer 2), compare.

Total op calls: ~4000-8000. Wall time: ~30 seconds A10G. Cost: <$0.10.

For K=3 (down_proj): same protocol on cshape (32, 128, 48). K-layers = 3.

Output: .pkl with all raw sweep data. Local script fits rank-R factorization.
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
app = modal.App("escha-full-extract", image=image)


def _int16(v):
    import numpy as np
    return np.int16(np.uint16(v & 0xFFFF)) if v < 32768 else np.int16(v - 65536)


def _extract_variant(name, in_f, out_f, K, cshape, op, device, batch_size):
    """Extract solo + cross for one (in_f, out_f, K) variant."""
    import numpy as np
    import torch

    bi_max, bj_max, k_max_all = cshape
    n_k_layers = K
    slots_per_layer = 16

    p0 = torch.zeros(cshape, dtype=torch.int16, device=device)
    w0 = op(p0, in_f, out_f, K, True, False).detach().cpu().numpy().astype(np.float32)

    def block(w, bi, bj):
        return w[bi*16:(bi+1)*16, bj*16:(bj+1)*16]

    def batched(specs):
        """Each spec: dict {k: v}. Sends up to bi_max*bj_max specs per call."""
        results = []
        max_tiles = bi_max * bj_max
        for start in range(0, len(specs), max_tiles):
            batch = specs[start:start + max_tiles]
            p = torch.zeros(cshape, dtype=torch.int16, device=device)
            positions = []
            for i, spec in enumerate(batch):
                bi = (i // bj_max) % bi_max
                bj = i % bj_max
                positions.append((bi, bj))
                for k, v in spec.items():
                    p[bi, bj, k] = _int16(int(v))
            w = op(p, in_f, out_f, K, True, False).detach().cpu().numpy().astype(np.float32)
            for (bi, bj) in positions:
                results.append(block(w, bi, bj) - block(w0, bi, bj))
        return np.stack(results)

    variant_result = {
        "in_f": in_f, "out_f": out_f, "K": K, "cshape": list(cshape),
        "w0": w0.astype(np.float16),
    }
    t_start = time.time()

    # ---- B: solo functions ----
    # Extract solo_k(v) for k ∈ [0..3, 4, 8, 12, 16..19, 32..35 if K=3]
    # This lets us verify translation invariance across col-shift and K-layer.
    solo_slots_to_extract = list(range(4))  # k=0,1,2,3 unique row patterns
    solo_slots_to_extract += [4, 8, 12]  # translation invariance across col-shift
    if n_k_layers >= 2:
        solo_slots_to_extract += list(range(16, 20))  # K-layer 2 first 4 slots
    if n_k_layers >= 3:
        solo_slots_to_extract += list(range(32, 36))  # K-layer 3 first 4 slots

    solo_data = {}  # k -> (65536, 16, 16) fp16
    for k in solo_slots_to_extract:
        specs = [{k: v} for v in range(65536)]
        deltas = batched(specs)  # (65536, 16, 16)
        solo_data[k] = deltas.astype(np.float16)
        print(f"  [{name}] solo_k={k}: extracted 65536 codes, "
              f"|d|max={float(np.abs(deltas).max()):.3f}", flush=True)
    variant_result["solo_data"] = solo_data

    # ---- C: cross functions ----
    # Type A: pair (0, 1)
    # Type B: pair (0, 3)
    # Also extract type A/B for K-layer 2 (pairs (16,17), (16,19)) to verify K-layer invariance.
    n_ref = 32  # 32 reference values for both v0 and v1 axes
    ref_values = list(range(1, n_ref + 1))  # v1_ref = 1..32

    cross_pairs_to_extract = [(0, 1, "A"), (0, 3, "B")]
    if n_k_layers >= 2:
        cross_pairs_to_extract += [(16, 17, "A_L2"), (16, 19, "B_L2")]

    cross_data = {}
    for (k_i, k_j, tag) in cross_pairs_to_extract:
        # Sweep v0 = 0..65535 with v1 = v1_ref (batch across v1_ref)
        # Columns: cross(:, v1_ref) for v1_ref in {1..32}
        specs_cols = []
        meta_cols = []
        for v1_ref in ref_values:
            for v0 in range(65536):
                specs_cols.append({k_i: v0, k_j: v1_ref})
                meta_cols.append((v0, v1_ref))
        cols_deltas = batched(specs_cols)  # (65536*n_ref, 16, 16)
        cols_deltas = cols_deltas.reshape(n_ref, 65536, 16, 16)

        # Rows: cross(v0_ref, :) for v0_ref in {1..32}
        specs_rows = []
        for v0_ref in ref_values:
            for v1 in range(65536):
                specs_rows.append({k_i: v0_ref, k_j: v1})
        rows_deltas = batched(specs_rows)
        rows_deltas = rows_deltas.reshape(n_ref, 65536, 16, 16)

        cross_data[(k_i, k_j, tag)] = {
            "ref_values": ref_values,
            "cols_deltas": cols_deltas.astype(np.float16),
            "rows_deltas": rows_deltas.astype(np.float16),
        }
        print(f"  [{name}] cross({k_i},{k_j}) tag={tag}: "
              f"cols={cols_deltas.shape} rows={rows_deltas.shape} "
              f"|cols|max={float(np.abs(cols_deltas).max()):.3f}", flush=True)
    variant_result["cross_data"] = {str(k): v for k, v in cross_data.items()}

    # ---- E: sample real expert code for verification ----
    # Extract op(real_code) for 3 sample experts to enable local verification.
    variant_result["extraction_time_s"] = time.time() - t_start
    print(f"  [{name}] total time: {variant_result['extraction_time_s']:.1f}s", flush=True)
    return variant_result


@app.function(gpu="A10G", timeout=3600, memory=32 * 1024, volumes={"/vol": vol})
def probe() -> bytes:
    import numpy as np
    import torch
    import escha  # noqa: F401
    from safetensors import safe_open
    from huggingface_hub import snapshot_download
    import os

    op = torch.ops.escha.escham_reconstruct
    device = "cuda"

    variants = [
        ("gate_up", 2048, 1024, 2, (128, 64, 32)),
        ("down_proj",  512, 2048, 3, (32, 128, 48)),
    ]
    results = {}
    for (name, in_f, out_f, K, cshape) in variants:
        print(f"\n=== extracting variant {name} (K={K}) ===", flush=True)
        results[name] = _extract_variant(
            name, in_f, out_f, K, cshape, op, device, batch_size=None
        )

    # ---- E: reference real-expert samples for verification ----
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
    import json
    idx = json.load(open(f"{model_dir}/model.safetensors.index.json"))
    wm = idx["weight_map"]

    ref_samples = {}
    for tensor_name, proj_variant in [
        ("model.language_model.layers.0.mlp.experts.gate_up_proj.escha_code", "gate_up"),
        ("model.language_model.layers.0.mlp.experts.down_proj.escha_code", "down_proj"),
    ]:
        shard = wm[tensor_name]
        with safe_open(f"{model_dir}/{shard}", framework="pt") as f:
            t = f.get_tensor(tensor_name)  # (256 experts, ...)
        # Take expert 0
        code = t[0].to(device)
        code_np = t[0].cpu().numpy().astype(np.int16)
        in_f = results[proj_variant]["in_f"]
        out_f = results[proj_variant]["out_f"]
        K = results[proj_variant]["K"]
        w_ref = op(code, in_f, out_f, K, True, False).detach().cpu().numpy().astype(np.float32)
        ref_samples[proj_variant] = {
            "code": code_np,
            "w_ref": w_ref.astype(np.float16),
            "abs_max": float(np.abs(w_ref).max()),
        }
        print(f"  [ref] {proj_variant} L0E0: code shape {code_np.shape}, "
              f"w_ref shape {w_ref.shape}, |w|max={float(np.abs(w_ref).max()):.3f}",
              flush=True)
    results["ref_samples"] = ref_samples

    # Save
    os.makedirs("/vol/joint_probes", exist_ok=True)
    out_path = "/vol/joint_probes/full_extract_v1.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    with open(out_path, "rb") as f:
        blob = f.read()
    vol.commit()
    print(f"\n[done] wrote {out_path} ({len(blob)/(1024*1024):.1f} MiB)", flush=True)
    return blob


@app.local_entrypoint()
def main() -> None:
    print("[local] launching full extraction on A10G ...", flush=True)
    blob = probe.remote()
    out_path = Path("/Users/kaede/mlx-video/mlx_video/models/qwen3_5_moe_escha/codebooks/full_extract_v1.pkl")
    out_path.write_bytes(blob)
    print(f"[local] wrote {out_path} ({len(blob)/(1024*1024):.1f} MiB)")
