"""Analyze pair_grouping_v1.pkl."""

import pickle
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent / "pair_grouping_v1.pkl"


def main() -> None:
    with open(DATA, "rb") as f:
        d = pickle.load(f)

    # A-D: pair tests
    print("=" * 70)
    print("A-D: PAIR TESTS — is (k_i, k_j) additive?")
    print("=" * 70)
    meta = d["pair_tests"]["meta"]
    deltas = d["pair_tests"]["deltas"].astype(np.float32)
    # meta: list of ('A'|'B'|'AB', k_i, k_j, v_i, v_j)
    # index them
    from collections import defaultdict
    by_key = defaultdict(dict)
    for i, (tag, k_i, k_j, v_i, v_j) in enumerate(meta):
        by_key[(k_i, k_j, v_i, v_j)][tag] = deltas[i]

    for (k_i, k_j, v_i, v_j), entries in by_key.items():
        dA = entries['A']; dB = entries['B']; dAB = entries['AB']
        R = dAB - dA - dB
        r_max = float(np.abs(R).max())
        print(f"  ({k_i:2d},{k_j:2d}) v=({v_i:>5d},{v_j:>5d}): |R|={r_max:8.4f}  "
              f"|dAB|={float(np.abs(dAB).max()):6.3f}")

    # E: 4-way pair-independence
    print()
    print("=" * 70)
    print("E: 4-WAY — is delta_{0,1,2,3} = delta_{0,1} + delta_{2,3} (pairs strict)?")
    print("=" * 70)
    e_meta = d["pair_independence"]["meta"]
    e_deltas = d["pair_independence"]["deltas"].astype(np.float32)
    for idx in range(0, len(e_meta), 3):
        tag01, v0, v1, v2, v3 = e_meta[idx]
        d01 = e_deltas[idx]
        d23 = e_deltas[idx + 1]
        d0123 = e_deltas[idx + 2]
        R = d0123 - d01 - d23
        print(f"  v={v0,v1,v2,v3}: |d0123|={float(np.abs(d0123).max()):.3f} "
              f"|d01|={float(np.abs(d01).max()):.3f} |d23|={float(np.abs(d23).max()):.3f} "
              f"|d0123 - d01 - d23|={float(np.abs(R).max()):.3e}")

    # F: dense pair (0,1) sweep — analyze codebook structure
    print()
    print("=" * 70)
    print("F: DENSE (0,1) SWEEP — pair-decode structure")
    print("=" * 70)
    f_values = d["pair_dense"]["values"]
    f_meta = d["pair_dense"]["meta"]
    f_deltas = d["pair_dense"]["deltas"].astype(np.float32)
    n = len(f_values)
    fD = f_deltas.reshape(n, n, 16, 16)  # (v0, v1, 16, 16)
    print(f"  grid: {n}x{n} = {n*n} probes, delta shape (16, 16)")
    print(f"  |fD|max: {np.abs(fD).max():.3f}")
    # Rank analysis: unfold to (v0, v1) x (spatial) matrix, SVD
    M = fD.reshape(n*n, -1)
    print(f"  full matrix (v0*v1, 256) shape: {M.shape}")
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    print(f"  singular values (top 16): {S[:16]}")
    cumsum = np.cumsum(S**2) / (S**2).sum()
    print(f"  cumulative variance @ rank 1/2/4/8/16/32: "
          f"{cumsum[0]:.3f} {cumsum[1]:.3f} {cumsum[3]:.3f} {cumsum[7]:.3f} {cumsum[15]:.3f} {cumsum[31]:.3f}")
    rank_1e3 = int((S > 1e-3 * S[0]).sum())
    print(f"  effective rank @ 1e-3: {rank_1e3}")

    # Check if fD is separable f(v0)*g(v1)*V:
    # Per pixel, is fD[:,:,i,j] a low-rank matrix in (v0, v1)?
    max_pix_rank = 0
    r1_pix_frac = 0
    n_pix = 0
    ranks = []
    for pi in range(16):
        for pj in range(16):
            Mp = fD[:, :, pi, pj]  # (n, n)
            if np.abs(Mp).max() > 1e-3:
                s = np.linalg.svd(Mp, compute_uv=False)
                rank = int((s > 1e-3 * s[0]).sum())
                ranks.append(rank)
                r1 = s[0] / (s.sum() + 1e-12)
                r1_pix_frac += r1
                max_pix_rank = max(max_pix_rank, rank)
                n_pix += 1
    if n_pix > 0:
        print(f"  per-pixel rank: max={max_pix_rank}, avg r1-frac={r1_pix_frac/n_pix:.3f}, "
              f"n_active={n_pix}/256")
        ranks = np.array(ranks)
        print(f"  per-pixel rank histogram: "
              f"1:{(ranks==1).sum()} 2:{(ranks==2).sum()} "
              f"3:{(ranks==3).sum()} 4:{(ranks==4).sum()} "
              f">4:{(ranks>4).sum()}")

    # Test: is delta(v0=0, v1=x) matching a single-slot lookup for k=1?
    # i.e. fD[0, :, :, :] — v0=0, sweep v1. This should match single-slot k=1 delta.
    print(f"\n  fD[v0=0, v1=x, :, :] ≡ single-slot k=1(v1) delta?")
    # We don't have single k=1 sweep in this file, but v0=0 in this grid IS index 0.
    idx0 = f_values.index(0)
    for i, v1 in enumerate([0, 1, 5, 100, -1]):
        j = f_values.index(v1) if v1 in f_values else None
        if j is None: continue
        norm = float(np.linalg.norm(fD[idx0, j]))
        print(f"    v0=0, v1={v1:>5}: |delta|_2={norm:.3f}  "
              f"nnz-count={int((np.abs(fD[idx0, j]) > 1e-4).sum())} pixels")

    # Support pattern of pair decode
    active_pixels = np.where(np.abs(fD).max(axis=(0,1)) > 1e-3)
    print(f"\n  active (row, col) pixels in pair-decode: "
          f"n={len(active_pixels[0])}  rows={sorted(set(active_pixels[0].tolist()))}, "
          f"cols={sorted(set(active_pixels[1].tolist()))}")

    # G: fix-first, sweep second
    print()
    print("=" * 70)
    print("G: FIX v0, SWEEP v1 — does pair-decode collapse to single-slot when v0=0?")
    print("=" * 70)
    g_meta = d["pair_fix_first"]["meta"]
    g_deltas = d["pair_fix_first"]["deltas"].astype(np.float32)
    for i, (v0_fix, v1) in enumerate(g_meta):
        if v1 in [0, 1, 100]:
            norm = float(np.linalg.norm(g_deltas[i]))
            max_v = float(np.abs(g_deltas[i]).max())
            print(f"  v0={v0_fix:>4}, v1={v1:>5}: |delta|_2={norm:.3f} |delta|_inf={max_v:.3f}")


if __name__ == "__main__":
    main()
