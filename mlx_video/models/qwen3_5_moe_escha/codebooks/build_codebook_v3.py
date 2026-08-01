"""build_codebook_v3.py — analyse full_extract_v3.pkl and build compact codebook.

Steps:
  1. Verify translation invariance across col-shift (k=0 vs k=4 vs k=8 vs k=12)
     and across K-layer (k=0 vs k=16).
  2. Fit low-rank factorisation to cross data.
  3. Build compact codebook_v3.npz for the MLX runtime.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

D = Path(__file__).parent
EXTRACT = D / "full_extract_v4.pkl"
OUT = D / "codebook_v3.npz"


def analyse_translation(name: str, variant: dict) -> dict:
    print(f"\n=== {name} — translation invariance ===")
    solo = variant["solo_data"]
    # solo_0: (65536, 16, 16). Cols pattern for k=0 is [0, 8].
    # solo_4 should be solo_0 with cols shifted by +1 (i.e., [1, 9]).
    # solo_8 should be solo_0 with cols shifted by +2.
    # solo_12 should be solo_0 with cols shifted by +3.
    # solo_16 should equal solo_0 (K-layer 2 uses same codebook).
    s0 = solo[0].astype(np.float32)  # (65536, 16, 16)
    for k in [4, 8, 12]:
        s_k = solo[k].astype(np.float32)
        shift = k // 4  # expected col shift
        # Shift s_k back by -shift to compare with s_0
        s_k_reshifted = np.roll(s_k, shift=-shift, axis=2)
        diff = float(np.abs(s0 - s_k_reshifted).max())
        print(f"  solo_k={k:2d} vs solo_k=0 shifted by {shift}: max_abs_diff = {diff:.4f}")
    if 16 in solo:
        s16 = solo[16].astype(np.float32)
        diff = float(np.abs(s0 - s16).max())
        print(f"  solo_k=16 (K-layer 2) vs solo_k=0: max_abs_diff = {diff:.4f}")
    if 32 in solo:
        s32 = solo[32].astype(np.float32)
        diff = float(np.abs(s0 - s32).max())
        print(f"  solo_k=32 (K-layer 3) vs solo_k=0: max_abs_diff = {diff:.4f}")
    return solo


def factorize_cross(cross_entry: dict, tag: str, rank_test: list[int] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit rank-R factorisation to cross function using rows+cols samples.

    Model: cross(v0, v1)[r, c] ≈ sum_{q=0..R-1} phi_q(v0)[r,c] * psi_q(v1)[r,c]
    Only non-zero at overlap pixels (r, c).

    Input: rows_deltas (R_samp, 65536, 16, 16), cols_deltas (R_samp, 65536, 16, 16).
      rows_deltas[i, v1, r, c] = pair(v0=ref_i, v1=v1)[r, c]
      cols_deltas[i, v0, r, c] = pair(v0=v0, v1=ref_i)[r, c]

    Approach:
      pair(v0, v1) = solo_0(v0) + solo_1(v1) + cross(v0, v1)
      cross(v0, v1) = pair(v0, v1) - solo_0(v0) - solo_1(v1)
      cross(v0=0, v1)  = 0 (by definition)
      cross(v0, v1=0)  = 0
      cross(v0=ref_i, v1) = rows_deltas[i, v1] - solo_0(ref_i) - solo_1(v1)
        Where solo_0(ref_i) can be computed from rows_deltas by v1=0 index

    We factor cross ≈ sum_q phi_q(v0) * psi_q(v1) at each overlap pixel.
    """
    rows = cross_entry["rows_deltas"].astype(np.float32)   # (R, 65536, 16, 16)
    cols = cross_entry["cols_deltas"].astype(np.float32)   # (R, 65536, 16, 16)
    ref_values = cross_entry["ref_values"]
    R_samp = len(ref_values)

    # solo_0(v0) = cols(v1=0)[v0]. But cols is at v1=ref_values, not 0.
    # Use rows(v1=0): rows_deltas[i, 0, :, :] = pair(v0=ref_i, v1=0) = solo_0(ref_i).
    # solo_0(ref_i) for i=0..R_samp-1
    solo_0_at_refs = rows[:, 0, :, :]   # (R_samp, 16, 16)
    # solo_1(v1) for all v1: pair(v0=0, v1) = solo_1(v1) — from cols[:, v0=0, :, :]?
    # Wait cols[i, v0, :, :] = pair(v0, v1=ref_i). solo_1(v1) = pair(0, v1).
    # We don't have pair(0, v1) directly. But rows[i, v1, :, :] = pair(ref_i, v1).
    # As R_samp → ∞, we could interpolate, but the simplest is: use cols[i, 0, :, :]
    # = pair(0, ref_i) = solo_1(ref_i).
    solo_1_at_refs = cols[:, 0, :, :]   # (R_samp, 16, 16)

    # Overlap detection: pixels where both solo_0 and solo_1 are non-zero.
    active_0 = np.abs(solo_0_at_refs).max(axis=0) > 1e-3  # (16, 16)
    active_1 = np.abs(solo_1_at_refs).max(axis=0) > 1e-3
    overlap = active_0 & active_1
    ov_r, ov_c = np.where(overlap)
    n_ov = len(ov_r)
    print(f"  [{tag}] overlap pixels: {n_ov} at {list(zip(ov_r.tolist(), ov_c.tolist()))}")

    # We need cross values at (v0=ref_i, v1) and (v0, v1=ref_i).
    # cross(ref_i, v1)[r,c] = rows[i, v1, r, c] - solo_0(ref_i)[r,c] - solo_1(v1)[r,c]
    # where solo_1(v1) is only known at v1=ref values, so we can only compute
    # cross(ref_i, ref_j) exactly from the two arrays.
    #
    # Actually for factorisation, we need cross at MANY (v0, v1) pairs, not just
    # ref×ref. We have:
    #   rows[i, v1, r, c] = pair(ref_i, v1)[r, c]
    #   cols[j, v0, r, c] = pair(v0, ref_j)[r, c]
    #
    # Compute cross_from_rows[i, v1, r, c] = rows[i, v1, r, c] - solo_0(ref_i)[r,c] - solo_1(v1)[r,c]
    # But solo_1(v1) is unknown for arbitrary v1. Instead use:
    #   cross_from_rows_at_pixel[i, v1, r, c] = rows[i, v1] - solo_1(v1) - solo_0(ref_i)
    # We can express: solo_1(v1)[r,c] = rows[i0, v1, r, c] - solo_0(ref_{i0}) - cross(ref_{i0}, v1)
    # This is a chicken-and-egg. But we know solo_1(v1) at v1=ref_j:
    #   solo_1(ref_j) = cols[j, 0, r, c]
    # For OTHER v1, we need a different approach.
    #
    # Better idea: factor pair(v0, v1) directly at overlap pixels via matrix
    # completion of a partial (65536 x 65536) matrix known at R rows + R cols.
    # But we have only ref rows/cols; still we can factor a (R_samp, R_samp)
    # submatrix and then extrapolate — but that gets messy.
    #
    # SIMPLEST APPROACH: use the fact that at overlap pixels, pair(v0, v1) is
    # rank-R across (v0, v1), so factor as phi(v0) * psi(v1) directly on pair.
    # phi(v0) captures BOTH solo_0(v0) AND the v0-dependent part of cross.
    # This subsumes the solo term.
    #
    # Combined factoring:
    #   pair(v0, v1)[r, c] ≈ sum_q phi_q(v0)[r,c] * psi_q(v1)[r,c]
    # phi, psi both (65536, R, 16, 16) but only non-zero at active pixels.

    # We have R_samp rows: rows[i, v1] = pair(ref_i, v1) → indexed by (i, v1)
    # We have R_samp cols: cols[j, v0] = pair(v0, ref_j) → indexed by (v0, j)
    # We want phi(v0), psi(v1) such that:
    #   pair(v0, v1) ≈ phi(v0) . psi(v1)  where . is over rank axis
    #
    # At overlap pixels, this is rank-R across (v0, v1). At non-overlap pixels
    # (only solo_0 active), pair(v0, v1)[r,c] = solo_0(v0)[r,c], independent
    # of v1 → rank 1. At non-overlap-slot-1 pixels: rank 1 in v1 only.
    #
    # So a UNIFIED per-pixel rank-R factorisation of pair(v0, v1) works.
    #
    # To fit rank-R, use the classic outer-product recovery from rows + cols:
    #   pair(v0, v1) ≈ cols_of_v0 . inv(pair_at_refs) . rows_of_v1
    # where pair_at_refs is the (R, R) submatrix at ref_i x ref_j.
    # This is the "Cross-approximation" / "CUR-decomposition" recipe.

    # For each pixel, extract:
    #   A = rows[:, :, r, c]  → shape (R_samp, 65536)  — "rows" of pair
    #   B = cols[:, :, r, c]  → shape (R_samp, 65536)  — "cols" of pair, transposed
    #   Actually B[j, v0] = pair(v0, ref_j), so B.T[v0, j] = pair(v0, ref_j).
    # We want pair(v0, v1) ≈ (something with rank R using A and B).
    #
    # Since rows and cols sample R_samp = 4 points along one axis, we can fit
    # rank up to R_samp = 4 exactly using cross-approximation:
    #   pair(v0, v1) ≈ B.T[v0, :] @ inv(P) @ A[:, v1]
    # where P[i, j] = pair(ref_i, ref_j) = A[i, ref_j] = B[j, ref_i]
    #
    # P is the (R_samp, R_samp) "pivot matrix". Invertible iff pair has rank ≥ R_samp.

    # Build phi, psi from this decomposition.
    phi = np.zeros((65536, R_samp, 16, 16), dtype=np.float32)  # phi[v0, q, r, c]
    psi = np.zeros((R_samp, 65536, 16, 16), dtype=np.float32)  # psi[q, v1, r, c]

    # Overlap pixels: where cross factorisation is needed
    # Non-overlap pixels: rank 1, just use solo
    for r in range(16):
        for c in range(16):
            A = rows[:, :, r, c]  # (R, 65536) — pair(ref_i, v1)
            B = cols[:, :, r, c]  # (R, 65536) — pair(v0, ref_j).T viewed as (v0, j)
            if np.abs(A).max() < 1e-4 and np.abs(B).max() < 1e-4:
                continue  # dead pixel
            # Pivot matrix P[i, j] = pair(ref_i, ref_j). But we can compute it two ways:
            # From A: P[i, j] = A[i, ref_j-1] (if ref values are 1..R, they're indices 1..R in v1 axis)
            # From B: P[j, i] = B[j, ref_i-1]
            # Take average (should agree):
            P = np.zeros((R_samp, R_samp), dtype=np.float32)
            for i in range(R_samp):
                for j in range(R_samp):
                    v_ref_j = int(ref_values[j])
                    v_ref_i = int(ref_values[i])
                    P[i, j] = 0.5 * (A[i, v_ref_j] + B[j, v_ref_i])
            # Regularised inverse (in case of low rank)
            try:
                P_inv = np.linalg.pinv(P, rcond=1e-4)
            except np.linalg.LinAlgError:
                P_inv = np.zeros_like(P)
            # phi[v0, q, r, c] = B[:, v0].T @ P_inv[:, q] → (65536, R)
            # Actually B[j, v0] means B[:, v0] is column v0. We want:
            #   phi[v0, :] = P_inv.T @ B[:, v0]  →  (R,)
            phi_pixel = (P_inv.T @ B).T  # (65536, R)
            # psi[q, v1] = A[q, v1] directly (as basis rows)
            psi_pixel = A  # (R, 65536)
            # Check: pair(v0, v1) ≈ phi[v0] @ psi[:, v1] = B[:, v0].T @ P_inv @ A[:, v1]
            phi[:, :, r, c] = phi_pixel
            psi[:, :, r, c] = psi_pixel

    return phi.astype(np.float16), psi.astype(np.float16), overlap, ref_values


def main():
    print(f"Loading {EXTRACT} ...")
    with open(EXTRACT, "rb") as f:
        d = pickle.load(f)
    print(f"Variants: {list(d.keys())}")

    variants_out = {}
    for name in ("gate_up", "down_proj"):
        variant = d[name]
        print(f"\n{'='*60}\nVARIANT {name}: K={variant['K']}, cshape={variant['cshape']}")
        print(f"{'='*60}")
        solo = analyse_translation(name, variant)

        # Extract solo for ALL slots (we no longer rely on translation invariance)
        k_max = 16 * variant["K"]
        solo_full = np.stack([solo[k] for k in range(k_max)], axis=0)  # (k_max, 65536, 16, 16)
        print(f"  solo_full shape: {solo_full.shape}, dtype: {solo_full.dtype}")

        # Cross factorisation
        cross = variant["cross_data"]
        cross_out = {}
        for pair_key_str, entry in cross.items():
            print(f"\n  factorising cross {pair_key_str}")
            phi, psi, overlap, refs = factorize_cross(entry, pair_key_str)
            cross_out[pair_key_str] = {"phi": phi, "psi": psi, "overlap": overlap.astype(np.uint8), "refs": np.array(refs)}
            print(f"    phi: {phi.shape}, psi: {psi.shape}, "
                  f"overlap active pixels: {int(overlap.sum())}")

        variants_out[name] = {
            "K": variant["K"],
            "in_f": variant["in_f"],
            "out_f": variant["out_f"],
            "cshape": variant["cshape"],
            "w0": variant["w0"],  # fp16 (in_f, out_f)
            "solo_full": solo_full,  # (16*K, 65536, 16, 16) fp16
            "cross_out": cross_out,
        }

    # Save compact codebook
    print(f"\nWriting compact codebook to {OUT} ...")
    save_dict = {}
    for name, v in variants_out.items():
        pre = f"{name}__"
        save_dict[pre + "K"] = np.array([v["K"]])
        save_dict[pre + "in_f"] = np.array([v["in_f"]])
        save_dict[pre + "out_f"] = np.array([v["out_f"]])
        save_dict[pre + "cshape"] = np.array(v["cshape"])
        save_dict[pre + "w0"] = v["w0"]
        save_dict[pre + "solo_full"] = v["solo_full"]  # (16*K, 65536, 16, 16) fp16
        for pair_key, entry in v["cross_out"].items():
            # pair_key is e.g. "(0, 1, 'A_L0_m0')"
            tag = eval(pair_key)[2]  # noqa: S307 — extract tag string like "A_L0_m0"
            save_dict[pre + f"cross_{tag}_phi"] = entry["phi"]
            save_dict[pre + f"cross_{tag}_psi"] = entry["psi"]
            save_dict[pre + f"cross_{tag}_overlap"] = entry["overlap"]
            save_dict[pre + f"cross_{tag}_refs"] = entry["refs"]

    np.savez_compressed(OUT, **save_dict)
    print(f"Saved {OUT}, size = {OUT.stat().st_size/(1024*1024):.1f} MiB")

    # Also save reference samples for verification
    ref = d["ref_samples"]
    ref_out = D / "ref_samples_v3.npz"
    np.savez_compressed(
        ref_out,
        gate_up__code=ref["gate_up"]["code"],
        gate_up__w_ref=ref["gate_up"]["w_ref"],
        down_proj__code=ref["down_proj"]["code"],
        down_proj__w_ref=ref["down_proj"]["w_ref"],
    )
    print(f"Saved {ref_out}, size = {ref_out.stat().st_size/(1024*1024):.1f} MiB")


if __name__ == "__main__":
    main()
