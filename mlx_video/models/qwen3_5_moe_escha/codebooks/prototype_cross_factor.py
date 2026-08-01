"""Prototype: fit rank-R factorisation to pair (0,1) 32×32 dense sweep, verify.

Uses pair_grouping_v1.pkl F section: 32×32 dense grid of (v0, v1) for pair (0,1).

Model:
  pair(v0, v1)[r, c] = solo_0(v0)[r,c] + solo_1(v1)[r,c] + cross(v0, v1)[r,c]

At non-overlap pixels: cross = 0 (single slot dominates).
At overlap pixels: cross(v0, v1) has structure, fit rank-R via SVD.

Test: reconstruct pair via low-rank; measure max abs error vs true pair.
"""

from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np

D = Path(__file__).parent
JOINT = D / "joint_probe_v1.pkl"
PAIRG = D / "pair_grouping_v1.pkl"


def main():
    with open(JOINT, "rb") as f:
        joint = pickle.load(f)
    with open(PAIRG, "rb") as f:
        pg = pickle.load(f)

    # Single-slot deltas for k=0 and k=1
    p1_meta = joint["P1_single_deltas"]["meta"]
    p1_d = joint["P1_single_deltas"]["deltas"].astype(np.float32)
    solo_lookup = {(k, v): p1_d[i] for i, (k, v) in enumerate(p1_meta)}

    # Dense pair (0,1) sweep from pg
    f_values = pg["pair_dense"]["values"]
    f_meta = pg["pair_dense"]["meta"]
    f_deltas = pg["pair_dense"]["deltas"].astype(np.float32)
    n = len(f_values)
    fD = f_deltas.reshape(n, n, 16, 16)

    # For each (v0, v1) in dense sweep, compute:
    #   cross(v0, v1) = pair(v0, v1) - solo_0(v0) - solo_1(v1)
    # But we don't have solo_0 for all f_values. Take v0=0 slice which is
    #   pair(0, v1) = solo_1(v1)  (since solo_0(0) = 0 relative to w0)
    idx0 = f_values.index(0)
    solo_0_from_grid = fD[:, idx0, :, :]  # (n, 16, 16), (v0, spatial)
    solo_1_from_grid = fD[idx0, :, :, :]  # (n, 16, 16), (v1, spatial)

    # cross(v0, v1) = fD(v0, v1) - solo_0(v0) - solo_1(v1)
    cross = fD - solo_0_from_grid[:, None, :, :] - solo_1_from_grid[None, :, :, :]
    print(f"cross shape: {cross.shape}, |cross|max: {np.abs(cross).max():.4f}")

    # Overlap pixels: where solo_0 and solo_1 are BOTH non-zero somewhere
    active_solo_0 = (np.abs(solo_0_from_grid).max(axis=0) > 1e-3)  # (16, 16)
    active_solo_1 = (np.abs(solo_1_from_grid).max(axis=0) > 1e-3)
    overlap_pixels = active_solo_0 & active_solo_1  # (16, 16) bool
    print(f"active solo_0 pixels: {int(active_solo_0.sum())}")
    print(f"active solo_1 pixels: {int(active_solo_1.sum())}")
    print(f"overlap pixels: {int(overlap_pixels.sum())}")
    # Where cross is significantly non-zero
    active_cross = (np.abs(cross).max(axis=(0, 1)) > 1e-3)
    print(f"active cross pixels: {int(active_cross.sum())}")

    ov_r, ov_c = np.where(overlap_pixels)
    print(f"overlap pixel coords: {list(zip(ov_r.tolist(), ov_c.tolist()))}")
    ac_r, ac_c = np.where(active_cross)
    print(f"active cross pixel coords: {list(zip(ac_r.tolist(), ac_c.tolist()))}")

    # Fit rank-R factorisation per overlap pixel:
    # cross(v0, v1)[r,c] ≈ sum_{k=1..R} phi_k(v0) * psi_k(v1)
    for R in [1, 2, 4, 8, 16]:
        err = 0
        total = 0
        recon_all = np.zeros_like(cross)
        for r, c in zip(ac_r, ac_c):
            M = cross[:, :, r, c]  # (n, n)
            U, S, Vt = np.linalg.svd(M, full_matrices=False)
            R_use = min(R, len(S))
            recon = U[:, :R_use] * S[:R_use] @ Vt[:R_use, :]
            recon_all[:, :, r, c] = recon
            err += float(np.linalg.norm(M - recon) ** 2)
            total += float(np.linalg.norm(M) ** 2)
        # Reconstruct the full pair(v0, v1) via solo + rank-R cross
        pair_recon = solo_0_from_grid[:, None, :, :] + solo_1_from_grid[None, :, :, :] + recon_all
        max_err = float(np.abs(fD - pair_recon).max())
        mean_err = float(np.abs(fD - pair_recon).mean())
        rel_err = np.sqrt(err / (total + 1e-12))
        print(f"  rank R={R:2d}: cross-only rel_err={rel_err:.4f}  "
              f"full-pair max_err={max_err:.4f}  mean_err={mean_err:.6f}")

    # Zoom on ONE overlap pixel to see the (v0, v1) structure
    if len(ac_r) > 0:
        r, c = int(ac_r[0]), int(ac_c[0])
        M = cross[:, :, r, c]
        s = np.linalg.svd(M, compute_uv=False)
        print(f"\nSingular values at overlap pixel ({r},{c}): {s[:10]}")


if __name__ == "__main__":
    main()
