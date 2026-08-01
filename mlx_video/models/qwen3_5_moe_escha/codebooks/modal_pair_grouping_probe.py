"""modal_pair_grouping_probe.py — verify slot grouping structure.

Prior finding: pair (0,1) has residual 6.4; pairs (0,2),(0,7),(0,15),(1,2),(7,8)
have EXACTLY 0 residual. Suggests slots grouped in 2s: {(0,1),(2,3),...,(14,15)}.

This probe:
  A. Test all 8 "expected pairs" within K-layer 1: (0,1),(2,3),...,(14,15).
     Expected: all non-additive.
  B. Test 16 "expected non-pairs": (0,3),(2,5),(4,7),(6,9),(8,11),(10,13),
     (12,15),(0,14),(1,4),(3,6),(5,8),(7,10),(9,12),(11,14),(2,7),(4,11).
     Expected: all additive.
  C. Test K-layer 2 pairs: (16,17),(18,19),...,(30,31).
     Expected: all non-additive (same structure).
  D. Test K-layer 2 non-pairs: (16,19),(18,21),(20,23).
     Expected: additive.
  E. Test 4-way test: (0,1,2,3) — if pairs are strict, then
     delta_1234 = delta_12 + delta_34 (i.e., pair 01 is independent from pair 23).
     If groups are 4-wide, delta_1234 has 3-way and 4-way residual.
  F. Pair (0,1) more values — is the joint decode rank-1 across the 2D (v0,v1) grid?

Cost: ~$0.05 A10G, ~2 min wall.
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
app = modal.App("escha-pair-grouping", image=image)


def _int16(v):
    import numpy as np
    return np.int16(np.uint16(v & 0xFFFF)) if v < 32768 else np.int16(v - 65536)


@app.function(gpu="A10G", timeout=1200, memory=32 * 1024, volumes={"/vol": vol})
def probe() -> bytes:
    import numpy as np
    import torch
    import escha  # noqa: F401

    op = torch.ops.escha.escham_reconstruct
    device = "cuda"
    in_f, out_f, K = 2048, 1024, 2
    cshape = (128, 64, 32)
    bi_max, bj_max, _ = cshape

    p0 = torch.zeros(cshape, dtype=torch.int16, device=device)
    w0 = op(p0, in_f, out_f, K, True, False).detach().cpu().numpy().astype(np.float32)

    def block(w, bi, bj):
        return w[bi*16:(bi+1)*16, bj*16:(bj+1)*16]

    def batched_probe(slot_specs):
        p = torch.zeros(cshape, dtype=torch.int16, device=device)
        pos_list = []
        for i, spec in enumerate(slot_specs):
            bi = (i // bj_max) % bi_max
            bj = i % bj_max
            pos_list.append((bi, bj))
            for k, v in spec.items():
                p[bi, bj, k] = _int16(int(v))
        w = op(p, in_f, out_f, K, True, False).detach().cpu().numpy().astype(np.float32)
        return np.stack([block(w, bi, bj) - block(w0, bi, bj) for bi, bj in pos_list])

    results = {}
    t0 = time.time()

    # -------- A + B + C + D: test many pairs uniformly ----------
    # For each pair (k_i, k_j), test with v_i=v_j=v for a range of v.
    # Compute residual R = d_AB - d_A - d_B.
    pairs = []
    # A: expected pairs in K-layer 1
    pairs += [(0,1),(2,3),(4,5),(6,7),(8,9),(10,11),(12,13),(14,15)]
    # B: expected non-pairs within K-layer 1
    pairs += [(0,3),(2,5),(4,7),(6,9),(8,11),(10,13),(12,15),(0,14),
              (1,4),(3,6),(5,8),(7,10),(9,12),(11,14),(2,7),(4,11)]
    # C: expected pairs in K-layer 2
    pairs += [(16,17),(18,19),(20,21),(22,23),(24,25),(26,27),(28,29),(30,31)]
    # D: expected non-pairs in K-layer 2
    pairs += [(16,19),(18,21),(20,23),(16,30)]

    test_vs = [(7, 13), (100, 500), (-3, 5)]  # 3 (v_i, v_j) pairs per test

    all_specs = []
    all_meta = []
    for (k_i, k_j) in pairs:
        for (v_i, v_j) in test_vs:
            # A only, B only, AB
            all_specs.append({k_i: v_i}); all_meta.append(('A', k_i, k_j, v_i, v_j))
            all_specs.append({k_j: v_j}); all_meta.append(('B', k_i, k_j, v_i, v_j))
            all_specs.append({k_i: v_i, k_j: v_j}); all_meta.append(('AB', k_i, k_j, v_i, v_j))

    deltas = batched_probe(all_specs)
    results['pair_tests'] = {
        'pairs': pairs,
        'test_vs': test_vs,
        'meta': all_meta,
        'deltas': deltas.astype(np.float16),
    }
    print(f"[A-D] {len(pairs)} pairs × {len(test_vs)} v-pairs = {len(all_specs)} probes done",
          flush=True)

    # -------- E: 4-way test — pair-independence ----------
    # If pairs (0,1) and (2,3) are independent, then:
    #   delta_0123 = delta_01 + delta_23
    # where delta_01 = op(both k=0,k=1 set) - w0, etc.
    e_specs = []
    e_meta = []
    v_pool = [7, 100, -3, 500]
    for i, (v0, v1, v2, v3) in enumerate([(7, 13, 5, 11), (100, 200, 300, 400), (-3, 5, -7, 11)]):
        # Need: delta_01, delta_23, delta_0123
        e_specs.append({0: v0, 1: v1}); e_meta.append(('01', v0, v1, v2, v3))
        e_specs.append({2: v2, 3: v3}); e_meta.append(('23', v0, v1, v2, v3))
        e_specs.append({0: v0, 1: v1, 2: v2, 3: v3}); e_meta.append(('0123', v0, v1, v2, v3))
    e_deltas = batched_probe(e_specs)
    results['pair_independence'] = {
        'meta': e_meta,
        'deltas': e_deltas.astype(np.float16),
    }
    print(f"[E] pair-independence: {len(e_specs)} probes done", flush=True)

    # -------- F: pair (0,1) DENSER sweep to characterise joint decode ----------
    # 32 x 32 grid: more values, to enable rank analysis of the full pair codebook.
    f_values = [0, 1, 2, 3, 4, 5, 7, 11, 13, 17, 19, 23, 31, 47, 63, 100,
                127, 199, 255, 500, 1000, 2000, 5000, 10000, 20000, -1, -3, -7,
                -100, -500, -5000, -20000]
    f_specs = []
    f_meta = []
    for v0 in f_values:
        for v1 in f_values:
            f_specs.append({0: v0, 1: v1})
            f_meta.append((v0, v1))
    f_deltas = batched_probe(f_specs)  # (32*32, 16, 16)
    results['pair_dense'] = {
        'values': f_values,
        'meta': f_meta,
        'deltas': f_deltas.astype(np.float16),
    }
    print(f"[F] dense pair (0,1): {len(f_specs)} probes done", flush=True)

    # -------- G: pair with FIRST slot fixed at v=0 vs sweep second slot ----------
    # Should recover single-slot function of k=1.
    # Then compare to pair(v0=1, v1=x) - should also be linear in x's contribution
    # if pair-decode has form g(v0) + h(v1) + separable(...).
    g_values = [0, 1, 2, 4, 8, 16, 32, 100, 500, 1000, 5000, -1, -100]
    g_specs = []
    g_meta = []
    for v0_fix in [0, 1, 100, -1]:
        for v1 in g_values:
            g_specs.append({0: v0_fix, 1: v1})
            g_meta.append((v0_fix, v1))
    g_deltas = batched_probe(g_specs)
    results['pair_fix_first'] = {
        'v0_fixed': [0, 1, 100, -1],
        'values': g_values,
        'meta': g_meta,
        'deltas': g_deltas.astype(np.float16),
    }
    print(f"[G] fix-first sweep: {len(g_specs)} probes done", flush=True)

    total = time.time() - t0
    results['wall_time_s'] = total
    print(f"\n[done] wall={total:.1f}s. Saving...", flush=True)

    import os
    os.makedirs("/vol/joint_probes", exist_ok=True)
    out_path = "/vol/joint_probes/pair_grouping_v1.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    with open(out_path, "rb") as f:
        blob = f.read()
    vol.commit()
    print(f"[done] wrote {out_path} ({len(blob)/1024:.1f} KiB)", flush=True)
    return blob


@app.local_entrypoint()
def main() -> None:
    print("[local] launching pair-grouping probe on A10G ...", flush=True)
    blob = probe.remote()
    out_path = Path("/Users/kaede/mlx-video/mlx_video/models/qwen3_5_moe_escha/codebooks/pair_grouping_v1.pkl")
    out_path.write_bytes(blob)
    print(f"[local] wrote {out_path} ({len(blob)/1024:.1f} KiB)")
