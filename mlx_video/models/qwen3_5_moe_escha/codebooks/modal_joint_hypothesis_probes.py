"""modal_joint_hypothesis_probes.py — Route J-final probes for H1-H5.

Prior evidence:
  - k=0..15 and k=16..31 are TWO separate K=1 residual layers, ADDITIVE across them.
  - WITHIN one 16-slot K=1 layer, slots are NOT additive (diff 0.68).
    (op(single k=0)+op(single k=1) != op(k=0 AND k=1)).
  - So the 16 slots per K-layer form a JOINT lookup, not 16 independent lookups.

Hypotheses:
  H1 Bilinear: delta_AB = delta_A + delta_B + f(v_i,v_j) with f a bilinear or
              tensor-product cross term.
  H2 VQ-VI: 16 codes form a query vector; a fixed matrix computes output.
  H3 Tensor decomposition (CP/TT/Tucker): rank-R decomposition of the 16-D
     joint codebook tensor.
  H4 Softmax mixture: codes are weights, output = weighted mean of 16 basis
     vectors. (Very unlikely for int16 codes — but easy to test.)
  H5 AQLM-structured: small dense codebook (e.g. 256 vectors), each slot
     indexes.

Diagnostic probes we run (single Modal function, batched):
  P0 — baseline: op(zeros).
  P1 — single slot: for k=0..3, sweep v in a small set. Save delta_k(v).
  P2 — pair (k_i,k_j): for a chosen small pair (0,1), (0,2), (0,7), (0,15),
       sweep both v_i, v_j in a small set (e.g. 16 values). Save delta_AB.
       Compute residual R(k_i,k_j; v_i,v_j) = delta_AB - delta_A - delta_B + baseline0.
       Analyze structure.
  P3 — triple (k_i,k_j,k_k): fixed values, verify Mobius inversion
       (does R_3 = 0? i.e. higher-order terms zero => bilinear only.)
  P4 — full-slot random probe: fill all 16 slots at (bi=0,bj=0) with random
       codes; measure difference from sum-of-per-slot-deltas.
  P5 — bit-scaling: fix k=0, sweep v = 2^i (single-bit codes) and check
       output linearity in bit patterns.
  P6 — cross-layer additivity (control): pair (k=0, k=17) across K-layers,
       should be additive. Confirms extraction sanity.

Runtime target: 5-10 min on A10G. Cost: ~$0.20.

Output: pickled dict on Modal volume `escha-codebooks:/joint_probes/probe_v1.pkl`
and bytes returned to local. Local script prints structural analysis.
"""

from __future__ import annotations

import base64
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
app = modal.App("escha-joint-hypothesis", image=image)


def _int16(v: int):
    import numpy as np
    return np.int16(np.uint16(v & 0xFFFF)) if v < 32768 else np.int16(v - 65536)


@app.function(gpu="A10G", timeout=1800, memory=32 * 1024, volumes={"/vol": vol})
def probe() -> bytes:
    import numpy as np
    import torch
    import escha  # noqa: F401

    op = torch.ops.escha.escham_reconstruct
    device = "cuda"
    in_f, out_f, K = 2048, 1024, 2
    cshape = (128, 64, 32)
    bi_max, bj_max, k_max = cshape

    results: dict = {"in_f": in_f, "out_f": out_f, "K": K, "cshape": list(cshape)}
    t0 = time.time()

    # ---- P0 baseline ----
    p0 = torch.zeros(cshape, dtype=torch.int16, device=device)
    w0 = op(p0, in_f, out_f, K, True, False).detach().cpu().numpy().astype(np.float32)
    results["w0_stats"] = {
        "shape": list(w0.shape),
        "abs_max": float(np.abs(w0).max()),
        "l2": float(np.linalg.norm(w0)),
    }
    print(f"[P0] baseline w0 shape={w0.shape} abs_max={np.abs(w0).max():.3e}", flush=True)

    def block(w, bi, bj):
        return w[bi*16:(bi+1)*16, bj*16:(bj+1)*16]

    def op_at_00(**slots) -> np.ndarray:
        """Set slots {k: value} at (bi=0, bj=0), return block(0,0) delta from w0."""
        p = torch.zeros(cshape, dtype=torch.int16, device=device)
        for k, v in slots.items():
            p[0, 0, k] = _int16(int(v))
        w = op(p, in_f, out_f, K, True, False).detach().cpu().numpy().astype(np.float32)
        return block(w, 0, 0) - block(w0, 0, 0)

    def op_batched_at_positions(slot_specs: list[dict]) -> np.ndarray:
        """Batch: each spec fills its own (bi,bj) tile with the given slots.
        Returns (n_specs, 16, 16) delta array."""
        p = torch.zeros(cshape, dtype=torch.int16, device=device)
        pos_list = []
        for i, spec in enumerate(slot_specs):
            bi = (i // bj_max) % bi_max
            bj = i % bj_max
            pos_list.append((bi, bj))
            for k, v in spec.items():
                p[bi, bj, k] = _int16(int(v))
        w = op(p, in_f, out_f, K, True, False).detach().cpu().numpy().astype(np.float32)
        out = np.zeros((len(slot_specs), 16, 16), dtype=np.float32)
        for i, (bi, bj) in enumerate(pos_list):
            out[i] = block(w, bi, bj) - block(w0, bi, bj)
        return out

    # ---- P1: single-slot deltas for k in 0..15 for a set of values ----
    # values to sweep: {1, 2, 4, 8, 16, 32, 100, 500, 1000, 10000, -1, -2, -100}
    p1_values = [1, 2, 4, 8, 16, 32, 100, 500, 1000, 10000, -1, -2, -100]
    p1_specs = []
    p1_meta = []
    for k in range(16):
        for v in p1_values:
            p1_specs.append({k: v})
            p1_meta.append((k, v))
    p1_deltas = op_batched_at_positions(p1_specs)  # shape (16*13, 16, 16)
    results["P1_single_deltas"] = {
        "meta": p1_meta,
        "deltas": p1_deltas.astype(np.float16),
    }
    print(f"[P1] single-slot: {len(p1_specs)} probes done, "
          f"|d|max={np.abs(p1_deltas).max():.3e}", flush=True)

    # ---- P2: pair (k_i, k_j) sweep ----
    # For chosen pairs, sweep both values over a small grid.
    # Grid: {1, 2, 4, 8, 16, 32, 100, 500} (8 values).
    p2_pairs = [(0, 1), (0, 2), (0, 7), (0, 15), (1, 2), (7, 8)]
    p2_values = [1, 2, 4, 8, 16, 32, 100, 500]  # 8 values
    p2_data = {}
    for (k_i, k_j) in p2_pairs:
        specs = []
        meta = []
        for v_i in p2_values:
            for v_j in p2_values:
                specs.append({k_i: v_i, k_j: v_j})
                meta.append((v_i, v_j))
        deltas = op_batched_at_positions(specs)  # (64, 16, 16)
        p2_data[(k_i, k_j)] = {
            "meta": meta,
            "deltas_AB": deltas.astype(np.float16),
        }
        print(f"[P2] pair ({k_i},{k_j}): {len(specs)} probes done", flush=True)
    results["P2_pair_sweep"] = {str(k): v for k, v in p2_data.items()}

    # ---- P3: triple (k_i, k_j, k_k) test — Mobius inversion ----
    # If pairwise interactions are the only cross terms, then for 3 slots:
    #   delta_ABC - delta_AB - delta_AC - delta_BC + delta_A + delta_B + delta_C - baseline0 = 0
    # baseline0 = delta_(no slots) = 0 by definition (relative to w0).
    p3_triples = [(0, 1, 2), (0, 7, 15), (0, 5, 10), (1, 2, 3)]
    p3_test_v = [(3, 5, 7), (100, 200, 300)]  # test values
    p3_specs = []
    p3_meta = []
    for tri in p3_triples:
        for vs in p3_test_v:
            k_i, k_j, k_k = tri
            v_i, v_j, v_k = vs
            # Need 8 subsets of {A, B, C}: (nothing = baseline w0), A, B, C, AB, AC, BC, ABC
            for combo in [(k_i, v_i), (k_j, v_j), (k_k, v_k),
                          (k_i, v_i, k_j, v_j),
                          (k_i, v_i, k_k, v_k),
                          (k_j, v_j, k_k, v_k),
                          (k_i, v_i, k_j, v_j, k_k, v_k)]:
                slot_dict = {combo[2*n]: combo[2*n+1] for n in range(len(combo)//2)}
                p3_specs.append(slot_dict)
                p3_meta.append((tri, vs, tuple(sorted(slot_dict.keys()))))
    p3_deltas = op_batched_at_positions(p3_specs)
    results["P3_triple"] = {
        "meta": p3_meta,
        "deltas": p3_deltas.astype(np.float16),
    }
    print(f"[P3] triple: {len(p3_specs)} probes done", flush=True)

    # ---- P4: full-slot random ----
    # Set all 16 slots to random values, measure difference from sum-of-per-slot.
    rng = np.random.default_rng(42)
    p4_full_codes = []
    p4_full_specs = []
    n_full = 8
    for _ in range(n_full):
        codes = rng.integers(-30000, 30000, size=16, dtype=np.int32).tolist()
        p4_full_codes.append(codes)
        p4_full_specs.append({k: codes[k] for k in range(16)})
    p4_deltas_full = op_batched_at_positions(p4_full_specs)  # (8, 16, 16)
    # Also single-slot for each of these 16*8 = 128 code positions
    p4_single_specs = []
    p4_single_meta = []
    for i, codes in enumerate(p4_full_codes):
        for k in range(16):
            p4_single_specs.append({k: codes[k]})
            p4_single_meta.append((i, k, codes[k]))
    p4_deltas_single = op_batched_at_positions(p4_single_specs)  # (128, 16, 16)
    results["P4_full_random"] = {
        "full_codes": np.array(p4_full_codes, dtype=np.int32),
        "deltas_full": p4_deltas_full.astype(np.float16),
        "deltas_single_meta": p4_single_meta,
        "deltas_single": p4_deltas_single.astype(np.float16),
    }
    print(f"[P4] full-random: {n_full} full + {len(p4_single_specs)} single probes done",
          flush=True)

    # ---- P5: bit-scaling: single k=0 with v = 2^i and v = 3*2^i ----
    # Tests whether decode is scale-preserving (bilinear expects f(v)*g depends
    # on v specifics, not just linear rescale).
    p5_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, -1, -2, -4, 3, 5]
    p5_specs = [{0: v} for v in p5_values]
    p5_deltas = op_batched_at_positions(p5_specs)
    results["P5_bit_scale"] = {
        "values": p5_values,
        "deltas": p5_deltas.astype(np.float16),
    }
    print(f"[P5] bit-scaling: {len(p5_specs)} probes done", flush=True)

    # ---- P6: cross-K-layer control (k in [0..15] vs k in [16..31]) ----
    # Should be additive if K-layers are independent.
    p6_pairs = [(0, 16), (0, 17), (5, 20), (7, 23)]
    p6_values = [7, 100, 1000]
    p6_specs = []
    p6_meta = []
    for (k_i, k_j) in p6_pairs:
        for v in p6_values:
            # 4 configs: A only, B only, both, none (implicit baseline)
            p6_specs.append({k_i: v})
            p6_specs.append({k_j: v})
            p6_specs.append({k_i: v, k_j: v})
            p6_meta.append((k_i, k_j, v))
    p6_deltas = op_batched_at_positions(p6_specs)
    results["P6_cross_klayer"] = {
        "meta": p6_meta,
        "deltas": p6_deltas.astype(np.float16),
    }
    print(f"[P6] cross-K control: {len(p6_specs)} probes done", flush=True)

    total = time.time() - t0
    results["wall_time_s"] = total
    print(f"\n[done] wall={total:.1f}s. Saving...", flush=True)

    import os
    os.makedirs("/vol/joint_probes", exist_ok=True)
    out_path = "/vol/joint_probes/probe_v1.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    with open(out_path, "rb") as f:
        blob = f.read()
    vol.commit()
    print(f"[done] wrote {out_path} ({len(blob)/1024:.1f} KiB)", flush=True)
    return blob


@app.local_entrypoint()
def main() -> None:
    print("[local] launching joint hypothesis probes on A10G ...", flush=True)
    blob = probe.remote()
    out_path = Path("/Users/kaede/mlx-video/mlx_video/models/qwen3_5_moe_escha/codebooks/joint_probe_v1.pkl")
    out_path.write_bytes(blob)
    print(f"[local] wrote {out_path} ({len(blob)/1024:.1f} KiB)")
