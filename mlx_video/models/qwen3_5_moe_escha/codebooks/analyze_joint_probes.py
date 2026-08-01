"""Analyze joint_probe_v1.pkl to discriminate H1-H5."""

import pickle
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent / "joint_probe_v1.pkl"

def main() -> None:
    with open(DATA, "rb") as f:
        d = pickle.load(f)
    print(f"Loaded {DATA.name}, wall={d['wall_time_s']:.1f}s")
    print(f"w0 stats: {d['w0_stats']}")
    print()

    # --- P1 single-slot deltas: build a lookup delta_k(v) -> (16,16) tile ---
    p1_meta = d["P1_single_deltas"]["meta"]
    p1_deltas = d["P1_single_deltas"]["deltas"].astype(np.float32)  # (208, 16, 16)
    p1_lookup = {(k, v): p1_deltas[i] for i, (k, v) in enumerate(p1_meta)}

    # --- P6 cross-K control: verify additivity ---
    print("=" * 60)
    print("P6 — cross-K-layer additivity (control test)")
    print("=" * 60)
    p6_meta = d["P6_cross_klayer"]["meta"]
    p6_deltas = d["P6_cross_klayer"]["deltas"].astype(np.float32)
    # for each (k_i, k_j, v), we recorded 3 probes: A, B, AB
    max_p6_residual = 0
    for idx, (k_i, k_j, v) in enumerate(p6_meta):
        d_A = p6_deltas[3*idx]
        d_B = p6_deltas[3*idx + 1]
        d_AB = p6_deltas[3*idx + 2]
        residual = d_AB - (d_A + d_B)
        max_r = float(np.abs(residual).max())
        max_p6_residual = max(max_p6_residual, max_r)
        print(f"  k=({k_i},{k_j}) v={v}: |d_A|={np.abs(d_A).max():.3f} "
              f"|d_B|={np.abs(d_B).max():.3f} |d_AB|={np.abs(d_AB).max():.3f} "
              f"|resid|={max_r:.3e}")
    print(f"P6 max residual across all: {max_p6_residual:.3e} (expected ~0 = additive)\n")

    # --- P2 pair sweep: analyze structure of residual R(v_i, v_j) ---
    print("=" * 60)
    print("P2 — pair (k_i,k_j) joint sweep analysis")
    print("=" * 60)
    p2 = d["P2_pair_sweep"]
    p2_values = [1, 2, 4, 8, 16, 32, 100, 500]
    for pair_key, info in p2.items():
        meta = info["meta"]  # list of (v_i, v_j)
        deltas_AB = info["deltas_AB"].astype(np.float32)  # (64, 16, 16)
        # Parse pair from string key like "(0, 1)"
        k_i, k_j = eval(pair_key)  # noqa: S307
        n = len(p2_values)

        # For each (v_i, v_j), compute residual = delta_AB - (delta_A + delta_B)
        R = np.zeros((n, n, 16, 16), dtype=np.float32)
        d_AB_all = deltas_AB.reshape(n, n, 16, 16)
        for a, v_i in enumerate(p2_values):
            for b, v_j in enumerate(p2_values):
                d_A = p1_lookup[(k_i, v_i)]
                d_B = p1_lookup[(k_j, v_j)]
                R[a, b] = d_AB_all[a, b] - d_A - d_B

        max_R = float(np.abs(R).max())
        max_d_AB = float(np.abs(d_AB_all).max())
        print(f"\nPair {pair_key}: |d_AB|max={max_d_AB:.3f} |residual|max={max_R:.3f}")
        # If non-additive, analyze R structure.
        # Flatten R over spatial dims: R_flat[a,b,:] = residual pixel vector
        R_flat = R.reshape(n, n, -1)  # (n, n, 256)
        # Test H1 bilinear: R(v_i, v_j) = v_i * v_j * C_ij (separable up to scalar)
        # Check if R[a,b] is proportional to R[0,0] scaled by (v_i * v_j).
        # Better: SVD across (v_i, v_j) x (spatial) — check rank.
        M = R.reshape(n*n, -1)  # (64, 256)
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        print(f"  singular values: {S[:8]}")
        print(f"  rank@1e-3: {int((S > 1e-3 * S[0]).sum())} / {min(n*n, 256)}")

        # Test H1 bilinear form: R = outer(f(v_i), g(v_j)) x spatial matrix
        # If rank-1 in (v_i, v_j) index: R[a,b,:] = alpha_a * beta_b * V[:] for some V
        # Reshape as (n*n, 256), check via SVD if the (n*n)-side is rank-1 across a,b
        # Consider R_flat[a,b] = r_ab (vector). If bilinear, r_ab = phi(v_i) * psi(v_j) * u
        # Then across (a,b), the outer structure of coefficients is rank 1.
        # Fold: treat as (n, n) matrix per spatial pixel; check rank per pixel.
        rank1_frac = 0
        max_pix = 0
        for pix in range(256):
            Mpix = R[:, :, pix // 16, pix % 16]  # (n, n)
            s = np.linalg.svd(Mpix, compute_uv=False)
            if np.abs(Mpix).max() > 1e-3:
                r1 = float(s[0] / (s.sum() + 1e-12))
                rank1_frac += r1
                max_pix += 1
        if max_pix > 0:
            print(f"  avg rank-1 fraction per pixel: {rank1_frac/max_pix:.3f} (1.0=perfect bilinear)")

        # Try to fit R = C_pair * phi(v_i) * phi(v_j) — hypothesis: bilinear in log(v) or v^alpha
        # Look at scaling: R[a, a] vs v_i^2
        diag = np.array([R[a, a, 0, 0] for a in range(n)])
        print(f"  diagonal R[a,a,0,0] vs v={p2_values}: {diag}")
        v_arr = np.array(p2_values, dtype=np.float32)
        if np.abs(diag).max() > 1e-4:
            # log fit
            valid = np.abs(diag) > 1e-4
            if valid.sum() >= 2:
                logv = np.log(np.abs(v_arr[valid]) + 1e-9)
                logd = np.log(np.abs(diag[valid]) + 1e-9)
                slope = np.polyfit(logv, logd, 1)[0]
                print(f"  diagonal log-log slope (v→R[a,a]): {slope:.3f} "
                      f"(H1 bilinear→2, linear→1, sqrt→0.5)")

    # --- P3 triples: test if 3-way residual is zero (pairs suffice) ---
    print()
    print("=" * 60)
    print("P3 — triple slot: is 3-way interaction zero (pairs suffice)?")
    print("=" * 60)
    p3_meta = d["P3_triple"]["meta"]
    p3_deltas = d["P3_triple"]["deltas"].astype(np.float32)  # (56, 16, 16)
    # Group by (tri, vs): 7 records per group = A,B,C,AB,AC,BC,ABC
    from collections import defaultdict
    groups = defaultdict(list)
    for i, (tri, vs, _keys) in enumerate(p3_meta):
        groups[(tri, vs)].append((i, _keys))
    for (tri, vs), idxlist in groups.items():
        d_by_slots = {}
        for i, keys in idxlist:
            d_by_slots[keys] = p3_deltas[i]
        k_i, k_j, k_k = tri
        v_i, v_j, v_k = vs
        d_A = d_by_slots[(k_i,)]
        d_B = d_by_slots[(k_j,)]
        d_C = d_by_slots[(k_k,)]
        d_AB = d_by_slots[tuple(sorted([k_i, k_j]))]
        d_AC = d_by_slots[tuple(sorted([k_i, k_k]))]
        d_BC = d_by_slots[tuple(sorted([k_j, k_k]))]
        d_ABC = d_by_slots[tuple(sorted([k_i, k_j, k_k]))]
        # 3-way residual via Mobius inversion:
        # R3 = d_ABC - (d_AB + d_AC + d_BC) + (d_A + d_B + d_C)
        R3 = d_ABC - (d_AB + d_AC + d_BC) + (d_A + d_B + d_C)
        r3_max = float(np.abs(R3).max())
        # Also pairwise residual for reference
        R_AB = d_AB - d_A - d_B
        print(f"  tri={tri} vs={vs}: |R3|={r3_max:.3e} |R_AB|={float(np.abs(R_AB).max()):.3f} "
              f"|d_ABC|={float(np.abs(d_ABC).max()):.3f}")

    # --- P4 full random: sum-of-singles vs full ---
    print()
    print("=" * 60)
    print("P4 — full-slot random: additive prediction error")
    print("=" * 60)
    p4 = d["P4_full_random"]
    codes = p4["full_codes"]  # (8, 16)
    deltas_full = p4["deltas_full"].astype(np.float32)  # (8, 16, 16)
    deltas_single = p4["deltas_single"].astype(np.float32)  # (128, 16, 16)
    # deltas_single_meta[i] = (probe_idx, k, code_value)
    for probe_idx in range(len(codes)):
        d_full = deltas_full[probe_idx]
        # Sum single-slot deltas for this probe's 16 codes
        d_sum = np.zeros((16, 16), dtype=np.float32)
        for i, (pi, k, v) in enumerate(p4["deltas_single_meta"]):
            if pi == probe_idx:
                d_sum += deltas_single[i]
        residual = d_full - d_sum
        print(f"  probe {probe_idx}: |d_full|={np.abs(d_full).max():.3f} "
              f"|d_sum_single|={np.abs(d_sum).max():.3f} "
              f"|d_full - d_sum|={np.abs(residual).max():.3f}")

    # --- P5 bit scaling ---
    print()
    print("=" * 60)
    print("P5 — bit scaling: how does delta_k=0(v) scale with v?")
    print("=" * 60)
    p5_vals = d["P5_bit_scale"]["values"]
    p5_deltas = d["P5_bit_scale"]["deltas"].astype(np.float32)
    # Look at l2 norm of each delta vs v
    for v, delta in zip(p5_vals, p5_deltas, strict=True):
        norm = float(np.linalg.norm(delta))
        max_ = float(np.abs(delta).max())
        # Compare to expected linear scaling from delta(v=1)
        if v == 1:
            ref_norm = norm
            ref_delta = delta
        print(f"  v={v:6d}: |d|_2={norm:.3f}  |d|_inf={max_:.3f}  "
              f"scale/v={norm/(abs(v)+1e-9):.3f}  "
              f"corr(d, d(v=1))={float((delta.flatten() @ ref_delta.flatten()) / (norm * ref_norm + 1e-9)):.3f}")

if __name__ == "__main__":
    main()
