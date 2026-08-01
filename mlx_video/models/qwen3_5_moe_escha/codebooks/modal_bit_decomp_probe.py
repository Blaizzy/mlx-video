"""modal_bit_decomp_probe.py — A-2 bit-decomposition extraction.

Hypothesis (from J-final finding: solo_0(v)[pixel (2,0)] = 0 for v ∈ [1..10000] but
non-zero for high-magnitude/negative v — clear bit-pattern signature):

    delta_slot_k(v) = sum_{i=0..15: bit_i(v)==1} pattern[K, k, i]     (16, 16) fp16

where bit_i(v) = (v >> i) & 1. If additive-bit holds, the whole per-slot codebook
collapses from 65536 entries to just 16 bit-patterns per slot.

Probe design (batched into ~5 A10G op calls, cost <$0.05):

  cshape (128, 64, k_max):  8192 (bi,bj) slots per op call.

  P_BIT: for each K ∈ {2, 3}, one op call:
      - Assign 16 bits × k_max slots probes across (bi, bj) grid.
      - Each probe sets p[bi, bj, k_slot] = 2^bit_i (int16-cast; bit 15 → -32768).
      - Extract pattern[K, k_slot, bit] from that block's delta.

  P_ADD: for each K, one op call:
      - 100 random 16-bit v values × k_max slots = up to 4800 probes
      - Predicted = sum_{bit_i(v)==1} pattern[K, k, i]
      - Compare max abs diff. If < 1e-3, bit-decomp holds.

  P_CROSS_K: sanity control — verify cross-K-layer additivity is preserved
      (already established in J-final; regression test).

If P_ADD passes with tight tolerance → bit-decomp is the correct model. The J-final
pair-overlap cross terms then only need to be extracted at BIT level (16×16 = 256
combinations per overlapping pair per K-layer), not full 65536×65536.

If P_ADD fails at high magnitude → the per-slot function has bit-pair interactions
too. Report residual structure and STOP per user's bounded-exploration policy.
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
app = modal.App("escha-bit-decomp-probe", image=image)


def _to_int16(v: int):
    """Wrap uint16 → int16 for the code tensor."""
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

    rng = np.random.default_rng(0xE5C4A)
    out: dict = {}
    t_all = time.time()

    for K, cshape, in_f, out_f in [
        (2, (128, 64, 32), 2048, 1024),
        (3, (32, 128, 48), 512, 2048),
    ]:
        bi_max, bj_max, k_max = cshape
        blocks_per_op = bi_max * bj_max
        print(f"\n=== K={K} cshape={cshape} blocks_per_op={blocks_per_op} ===", flush=True)

        # ---- baseline ----
        p0 = torch.zeros(cshape, dtype=torch.int16, device=device)
        w0 = op(p0, in_f, out_f, K, True, False).detach().cpu().numpy().astype(np.float32)
        print(f"  baseline w0 shape={w0.shape} |w0|={np.linalg.norm(w0):.3e}", flush=True)

        # ---- P_BIT: 16 bits × k_max slots ----
        # Assign (bit, k) → (bi, bj) using a linear index; must fit in blocks_per_op.
        n_probes_bit = 16 * k_max
        assert n_probes_bit <= blocks_per_op, (n_probes_bit, blocks_per_op)
        positions = [(i // bj_max, i % bj_max) for i in range(blocks_per_op)]

        p = torch.zeros(cshape, dtype=torch.int16, device=device)
        idx = 0
        for k in range(k_max):
            for bit in range(16):
                bi, bj = positions[idx]
                p[bi, bj, k] = _to_int16(1 << bit)
                idx += 1

        t0 = time.time()
        w = op(p, in_f, out_f, K, True, False).detach().cpu().numpy().astype(np.float32)
        print(f"  P_BIT op time: {time.time()-t0:.2f}s", flush=True)

        pattern = np.zeros((k_max, 16, 16, 16), dtype=np.float16)  # (k, bit, r, c)
        idx = 0
        for k in range(k_max):
            for bit in range(16):
                bi, bj = positions[idx]
                blk = w[bi*16:(bi+1)*16, bj*16:(bj+1)*16] - w0[bi*16:(bi+1)*16, bj*16:(bj+1)*16]
                pattern[k, bit] = blk.astype(np.float16)
                idx += 1
        out[f"pattern_K{K}"] = pattern
        out[f"w0_K{K}"] = w0.astype(np.float16)

        n_nz_bits = int(np.sum(np.any(pattern != 0, axis=(2, 3))))
        print(f"  extracted pattern[K={K}] shape={pattern.shape} nz_bits={n_nz_bits}/{k_max*16}", flush=True)

        # ---- P_ADD: 100 random v × k_max slots, additivity check ----
        # Uses 100 * k_max <= blocks_per_op probes in ONE op call.
        n_rand = 100
        rand_vs = rng.integers(0, 65536, size=n_rand, dtype=np.int64)
        # A few adversarial values: hard bit patterns
        forced = np.array([0x8000, 0xFFFF, 0x5555, 0xAAAA, 0x8001, 0x7FFF], dtype=np.int64)
        all_vs = np.concatenate([forced, rand_vs])[:n_rand]

        n_probes_add = n_rand * k_max
        assert n_probes_add <= blocks_per_op, (n_probes_add, blocks_per_op)

        p2 = torch.zeros(cshape, dtype=torch.int16, device=device)
        add_pos = [(i // bj_max, i % bj_max) for i in range(n_probes_add)]
        idx = 0
        for k in range(k_max):
            for vi in range(n_rand):
                bi, bj = add_pos[idx]
                p2[bi, bj, k] = _to_int16(int(all_vs[vi]))
                idx += 1

        t0 = time.time()
        w2 = op(p2, in_f, out_f, K, True, False).detach().cpu().numpy().astype(np.float32)
        print(f"  P_ADD op time: {time.time()-t0:.2f}s", flush=True)

        # For each (k, vi): actual delta = w2[block] - w0[block]
        # Predicted = sum over set bits of pattern[k, bit]
        add_max = 0.0
        add_l2 = 0.0
        add_l2_ref = 0.0
        worst = None
        per_k_max = np.zeros(k_max, dtype=np.float32)
        per_k_ref_l2 = np.zeros(k_max, dtype=np.float32)
        per_k_diff_l2 = np.zeros(k_max, dtype=np.float32)
        idx = 0
        for k in range(k_max):
            for vi in range(n_rand):
                bi, bj = add_pos[idx]
                actual = w2[bi*16:(bi+1)*16, bj*16:(bj+1)*16] - w0[bi*16:(bi+1)*16, bj*16:(bj+1)*16]
                v_u = int(all_vs[vi]) & 0xFFFF
                pred = np.zeros((16, 16), dtype=np.float32)
                for bit in range(16):
                    if (v_u >> bit) & 1:
                        pred += pattern[k, bit].astype(np.float32)
                diff = np.abs(actual - pred)
                d_max = float(diff.max())
                if d_max > add_max:
                    add_max = d_max
                    worst = {"k": int(k), "v": int(all_vs[vi]), "v_u": v_u,
                             "actual_max": float(np.abs(actual).max()),
                             "pred_max": float(np.abs(pred).max()),
                             "diff_max": d_max}
                per_k_max[k] = max(per_k_max[k], d_max)
                per_k_ref_l2[k] += float(np.linalg.norm(actual)**2)
                per_k_diff_l2[k] += float(np.linalg.norm(diff)**2)
                add_l2 += float(np.linalg.norm(diff)**2)
                add_l2_ref += float(np.linalg.norm(actual)**2)
                idx += 1
        rel = float(np.sqrt(add_l2) / (np.sqrt(add_l2_ref) + 1e-12))
        print(f"  P_ADD K={K}: max_abs_diff={add_max:.4e} rel_l2={rel:.4e} worst={worst}", flush=True)

        out[f"add_K{K}"] = {
            "max_abs_diff": add_max, "rel_l2": rel, "worst": worst,
            "per_k_max": per_k_max.tolist(),
            "per_k_rel_l2": (np.sqrt(per_k_diff_l2) / (np.sqrt(per_k_ref_l2) + 1e-12)).tolist(),
        }

        # ---- P_CROSS_K: cross-K-layer additivity control (only meaningful when k_max > 16) ----
        # Pick k_a in K-layer 0, k_b in K-layer 1. If additive, delta_AB = delta_A + delta_B.
        # Use two disjoint (bi, bj) sites per probe.
        if k_max > 16:
            k_a, k_b = 0, 16
            v_a, v_b = 0x1234, 0x5678
            # site A: solo k_a=v_a; site B: solo k_b=v_b; site C: both at same (bi, bj)
            p3 = torch.zeros(cshape, dtype=torch.int16, device=device)
            p3[0, 0, k_a] = _to_int16(v_a)
            p3[0, 1, k_b] = _to_int16(v_b)
            p3[0, 2, k_a] = _to_int16(v_a)
            p3[0, 2, k_b] = _to_int16(v_b)
            w3 = op(p3, in_f, out_f, K, True, False).detach().cpu().numpy().astype(np.float32)
            dA = w3[0:16, 0:16] - w0[0:16, 0:16]
            dB = w3[0:16, 16:32] - w0[0:16, 16:32]
            dAB = w3[0:16, 32:48] - w0[0:16, 32:48]
            cross_add_max = float(np.abs(dAB - dA - dB).max())
            out[f"cross_K{K}_max"] = cross_add_max
            print(f"  P_CROSS_K K={K} (k_a={k_a},k_b={k_b}): |dAB-dA-dB|max={cross_add_max:.4e}", flush=True)

    print(f"\n=== total wall time: {time.time()-t_all:.1f}s ===", flush=True)

    # Save to volume for later use.
    Path("/vol/bit_decomp").mkdir(parents=True, exist_ok=True)
    with open("/vol/bit_decomp/probe_v1.pkl", "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    vol.commit()

    # Serialize summary metrics (small subset) for return; full patterns in volume.
    summary = {
        "K2": {
            "pattern_shape": list(out["pattern_K2"].shape),
            "add": out["add_K2"],
            "cross_max": out.get("cross_K2_max"),
        },
        "K3": {
            "pattern_shape": list(out["pattern_K3"].shape),
            "add": out["add_K3"],
            "cross_max": out.get("cross_K3_max"),
        },
    }
    payload = pickle.dumps({"summary": summary, "full": out}, protocol=pickle.HIGHEST_PROTOCOL)
    return base64.b64encode(payload)


@app.local_entrypoint()
def main() -> None:
    t0 = time.time()
    b64 = probe.remote()
    payload = pickle.loads(base64.b64decode(b64))
    print(f"\n[local] probe wall {time.time()-t0:.1f}s")
    print(f"[local] summary:")
    for K in (2, 3):
        s = payload["summary"][f"K{K}"]
        print(f"  K={K}: pattern={s['pattern_shape']} "
              f"add.max_abs_diff={s['add']['max_abs_diff']:.4e} "
              f"add.rel_l2={s['add']['rel_l2']:.4e} "
              f"cross_max={s['cross_max']}")
        worst = s['add']['worst']
        if worst:
            print(f"    worst: k={worst['k']} v=0x{worst['v_u']:04x} "
                  f"|actual|max={worst['actual_max']:.3e} "
                  f"|pred|max={worst['pred_max']:.3e} "
                  f"diff={worst['diff_max']:.3e}")

    out_dir = Path("/Users/kaede/mlx-video/mlx_video/models/qwen3_5_moe_escha/codebooks")
    (out_dir / "bit_decomp_probe_v1.pkl").write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    print(f"[local] wrote {out_dir/'bit_decomp_probe_v1.pkl'}")
