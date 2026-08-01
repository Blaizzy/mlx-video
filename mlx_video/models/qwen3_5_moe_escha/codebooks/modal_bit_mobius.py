"""modal_bit_mobius.py — 2-way and 3-way bit Möbius residuals.

Decides between two paths under bounded exploration:
  * If 3-way bit Möbius ≡ 0 (like J-final showed for SLOT-level cross terms):
    bit-decomp holds with pair corrections only → extract 16*120=1920 pair patterns
    per K-layer per slot (~16 MB compact). Continue port.
  * If 3-way bit Möbius ≠ 0: additive bit model needs 3-way+ terms → STOP per
    user's bounded-exploration policy; report and file EschaLabs discussion.

Also runs a symmetry / structure probe on the single-slot code→delta mapping
to characterize what the op REALLY does (in case the answer is "not a bit code
at all — it's a scalar function of v").
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
app = modal.App("escha-bit-mobius", image=image)


def _to_int16(v: int):
    import numpy as np
    v_u = v & 0xFFFF
    return np.int16(v_u if v_u < 32768 else v_u - 65536)


@app.function(gpu="A10G", timeout=1800, memory=32 * 1024, volumes={"/vol": vol})
def probe() -> bytes:
    import numpy as np
    import torch
    import escha  # noqa: F401

    op = torch.ops.escha.escham_reconstruct
    device = "cuda"
    K, cshape, in_f, out_f = 2, (128, 64, 32), 2048, 1024
    bi_max, bj_max, k_max = cshape
    blocks_per_op = bi_max * bj_max
    k_test = 0  # slot 0 in K-layer 1

    print(f"=== bit-Möbius probe: K={K} k_slot={k_test} ===", flush=True)

    # ---- baseline ----
    p0 = torch.zeros(cshape, dtype=torch.int16, device=device)
    w0 = op(p0, in_f, out_f, K, True, False).detach().cpu().numpy().astype(np.float32)

    # Pack ALL relevant probes into one op call.
    # Layout: assign each probe (labelled by v) a distinct (bi, bj).
    probes = []  # list of (label, v_int)

    # Single-bit: 16 probes
    for i in range(16):
        probes.append((f"s_{i}", 1 << i))
    # Pair: all C(16,2)=120 probes
    for i in range(16):
        for j in range(i+1, 16):
            probes.append((f"p_{i}_{j}", (1 << i) | (1 << j)))
    # Triple: sampled — all C(4,3) for {0,1,2,15}, {0,7,15}, {0,14,15}, {1,2,3}, {13,14,15}
    tri_choices = [
        (0,1,2), (0,1,15), (0,2,15), (1,2,15),
        (0,7,15), (0,14,15), (1,2,3), (13,14,15),
        (0,1,7), (0,8,15), (7,8,15), (0,3,12),
        (5,10,15), (2,5,11), (0,6,13), (4,9,14),
    ]
    for t in tri_choices:
        i,j,k = t
        probes.append((f"t_{i}_{j}_{k}", (1<<i)|(1<<j)|(1<<k)))

    # Symmetry probes: v vs -v (int16 negation)
    for v_pos in [1, 2, 100, 1000, 10000, 30000, 0x0F0F, 0x5555]:
        probes.append((f"sym_pos_{v_pos}", v_pos))
        neg = (-v_pos) & 0xFFFF
        probes.append((f"sym_neg_{v_pos}", neg))

    # Sweep of v = 2^b for b in [0..15] to double-check single-bit vs full
    # (redundant with s_*, but sanity)

    n_probes = len(probes)
    assert n_probes <= blocks_per_op
    print(f"  n_probes={n_probes}", flush=True)

    # Assign positions
    positions = [(i // bj_max, i % bj_max) for i in range(n_probes)]
    p = torch.zeros(cshape, dtype=torch.int16, device=device)
    for i, (label, v) in enumerate(probes):
        bi, bj = positions[i]
        p[bi, bj, k_test] = _to_int16(int(v))
    t0 = time.time()
    w = op(p, in_f, out_f, K, True, False).detach().cpu().numpy().astype(np.float32)
    print(f"  op time {time.time()-t0:.2f}s", flush=True)

    deltas = {}
    for i, (label, v) in enumerate(probes):
        bi, bj = positions[i]
        deltas[label] = (w[bi*16:(bi+1)*16, bj*16:(bj+1)*16]
                         - w0[bi*16:(bi+1)*16, bj*16:(bj+1)*16]).astype(np.float32)

    # ---- 2-way Möbius: cross(i,j) = delta(2^i | 2^j) - delta(2^i) - delta(2^j) ----
    pair_cross = {}
    pair_stats = []
    for i in range(16):
        for j in range(i+1, 16):
            solo_i = deltas[f"s_{i}"]
            solo_j = deltas[f"s_{j}"]
            pair = deltas[f"p_{i}_{j}"]
            cross = pair - solo_i - solo_j
            pair_cross[(i, j)] = cross
            pair_stats.append({
                "i": i, "j": j,
                "|cross|max": float(np.abs(cross).max()),
                "|cross|l2": float(np.linalg.norm(cross)),
                "|pair|max": float(np.abs(pair).max()),
            })
    pair_stats.sort(key=lambda r: -r["|cross|max"])
    print(f"  2-way pair cross-terms (top 8 by max):", flush=True)
    for r in pair_stats[:8]:
        print(f"    ({r['i']:2d},{r['j']:2d}): |cross|max={r['|cross|max']:.4f} "
              f"|cross|l2={r['|cross|l2']:.4f} |pair|max={r['|pair|max']:.4f}", flush=True)
    n_zero_pairs = sum(1 for r in pair_stats if r["|cross|max"] < 1e-3)
    n_big_pairs = sum(1 for r in pair_stats if r["|cross|max"] > 0.1)
    print(f"  → n_pair_cross ≈ 0 (<1e-3): {n_zero_pairs}/120; large (>0.1): {n_big_pairs}/120", flush=True)

    # ---- 3-way Möbius:
    # R3(i,j,k) = δ(i,j,k) - δ(i,j) - δ(i,k) - δ(j,k) + δ(i) + δ(j) + δ(k)
    tri_stats = []
    for (i, j, kk) in tri_choices:
        d_ijk = deltas[f"t_{i}_{j}_{kk}"]
        d_ij  = deltas[f"p_{i}_{j}"]
        d_ik  = deltas[f"p_{i}_{kk}"]
        d_jk  = deltas[f"p_{j}_{kk}"]
        d_i   = deltas[f"s_{i}"]
        d_j   = deltas[f"s_{j}"]
        d_k   = deltas[f"s_{kk}"]
        r3 = d_ijk - d_ij - d_ik - d_jk + d_i + d_j + d_k
        tri_stats.append({
            "triple": (i, j, kk),
            "|R3|max": float(np.abs(r3).max()),
            "|R3|l2": float(np.linalg.norm(r3)),
            "|dijk|max": float(np.abs(d_ijk).max()),
        })
    tri_stats.sort(key=lambda r: -r["|R3|max"])
    print(f"  3-way Möbius residual (all {len(tri_stats)} tested triples):", flush=True)
    for r in tri_stats:
        i,j,k = r["triple"]
        print(f"    ({i:2d},{j:2d},{k:2d}): |R3|max={r['|R3|max']:.4f} "
              f"|R3|l2={r['|R3|l2']:.4f} |dijk|max={r['|dijk|max']:.4f}", flush=True)
    n_zero_tri = sum(1 for r in tri_stats if r["|R3|max"] < 1e-3)
    n_big_tri = sum(1 for r in tri_stats if r["|R3|max"] > 0.1)
    print(f"  → n_R3 ≈ 0: {n_zero_tri}/{len(tri_stats)}; large (>0.1): {n_big_tri}/{len(tri_stats)}", flush=True)

    # ---- Symmetry: is delta(-v) == -delta(v)? or == delta(v)?
    sym_stats = []
    for v_pos in [1, 2, 100, 1000, 10000, 30000, 0x0F0F, 0x5555]:
        pos = deltas[f"sym_pos_{v_pos}"]
        neg = deltas[f"sym_neg_{v_pos}"]
        sym_stats.append({
            "v": v_pos,
            "|pos|max": float(np.abs(pos).max()),
            "|neg|max": float(np.abs(neg).max()),
            "|pos-neg|max": float(np.abs(pos - neg).max()),
            "|pos+neg|max": float(np.abs(pos + neg).max()),  # ≈0 means odd-symmetric
        })
    print(f"  symmetry (v vs -v): (|pos+neg|max ≈ 0 → odd; |pos-neg|max ≈ 0 → even)", flush=True)
    for r in sym_stats:
        print(f"    v={r['v']:6d}: |pos|max={r['|pos|max']:.4f} |neg|max={r['|neg|max']:.4f} "
              f"|pos-neg|max={r['|pos-neg|max']:.4f} |pos+neg|max={r['|pos+neg|max']:.4f}", flush=True)

    # Save results
    Path("/vol/bit_decomp").mkdir(parents=True, exist_ok=True)
    result = {
        "K": K, "k_slot": k_test,
        "pair_stats": pair_stats, "tri_stats": tri_stats, "sym_stats": sym_stats,
        "n_zero_pairs": n_zero_pairs, "n_big_pairs": n_big_pairs,
        "n_zero_tri": n_zero_tri, "n_big_tri": n_big_tri,
    }
    with open("/vol/bit_decomp/mobius_v1.pkl", "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    vol.commit()
    return base64.b64encode(pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL))


@app.local_entrypoint()
def main() -> None:
    b64 = probe.remote()
    result = pickle.loads(base64.b64decode(b64))
    print("\n[local] summary:")
    print(f"  2-way: {result['n_zero_pairs']}/120 near-zero, {result['n_big_pairs']}/120 >0.1")
    print(f"  3-way: {result['n_zero_tri']}/{len(result['tri_stats'])} near-zero, "
          f"{result['n_big_tri']}/{len(result['tri_stats'])} >0.1")
    if result["n_big_tri"] > 0:
        print("  DECISION: 3-way bit Möbius NON-ZERO → additive bit model insufficient at any order.")
        print("           Per user's bounded-exploration policy: STOP, file failure report.")
    else:
        print("  DECISION: 3-way bit Möbius zero → 2-way bit interactions may suffice.")
        print("           Bit-pair extraction is bounded (~16 MB per K variant); continue.")
    out_dir = Path("/Users/kaede/mlx-video/mlx_video/models/qwen3_5_moe_escha/codebooks")
    (out_dir / "bit_mobius_v1.pkl").write_bytes(pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL))
