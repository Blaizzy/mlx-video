"""Debug cross factorisation: does phi @ psi at overlap pixel match extracted rows/cols?"""
import pickle
from pathlib import Path
import numpy as np

D = Path(__file__).parent
with open(D / "full_extract_v4.pkl", "rb") as f:
    d = pickle.load(f)

CB = np.load(D / "codebook_v3.npz", allow_pickle=False)

name = "gate_up"
v = d[name]
cross_data = v["cross_data"]

# Pick pair (0, 1) A_L0_m0
pair_key = "(0, 1, 'A_L0_m0')"
entry = cross_data[pair_key]
rows = entry["rows_deltas"].astype(np.float32)  # (R=4, 65536, 16, 16)
cols = entry["cols_deltas"].astype(np.float32)  # (R=4, 65536, 16, 16)
refs = entry["ref_values"]
print(f"pair: {pair_key}")
print(f"  refs: {refs}")
print(f"  rows: {rows.shape}, cols: {cols.shape}")

# Compute cross at overlap pixels
solo_0 = v["solo_data"][0].astype(np.float32)  # (65536, 16, 16)
solo_1 = v["solo_data"][1].astype(np.float32)

active_0 = (np.abs(solo_0).max(axis=0) > 1e-3)
active_1 = (np.abs(solo_1).max(axis=0) > 1e-3)
overlap = active_0 & active_1
ov_r, ov_c = np.where(overlap)
print(f"  overlap pixels (from full solo): {list(zip(ov_r.tolist(), ov_c.tolist()))}")

# For each overlap pixel, extract phi, psi from codebook, verify at ref values
pre = f"{name}__"
phi = CB[pre + "cross_A_L0_m0_phi"].astype(np.float32)  # (65536, R, 16, 16)
psi = CB[pre + "cross_A_L0_m0_psi"].astype(np.float32)  # (R, 65536, 16, 16)
overlap_cb = CB[pre + "cross_A_L0_m0_overlap"].astype(bool)
print(f"  overlap (from codebook build): {int(overlap_cb.sum())} pixels at "
      f"{list(zip(*np.where(overlap_cb)))}")

# Verify factorisation at reference values
# pair(v0=refs[i], v1=refs[j]) should reconstruct exactly
# From rows[i, refs[j]-1] since v1_idx=refs[j] gives pair(refs[i], refs[j])
# Actually rows[i, v1_idx] = pair(refs[i], v1_idx) — v1_idx directly indexes into 0..65535
# So rows[i, refs[j], r, c] = pair(refs[i], refs[j])[r, c]

for pi in [0]:  # just check one row-index
    for r, c in list(zip(ov_r.tolist(), ov_c.tolist()))[:3]:
        for pj in [1]:  # just one column-index
            v_ref_i = int(refs[pi])
            v_ref_j = int(refs[pj])
            true_pair = rows[pi, v_ref_j, r, c]
            true_pair_from_cols = cols[pj, v_ref_i, r, c]
            # Reconstruction: phi[v0=v_ref_i, :, r, c] @ psi[:, v1=v_ref_j, r, c]
            recon = float(phi[v_ref_i, :, r, c] @ psi[:, v_ref_j, r, c])
            solo_0_val = solo_0[v_ref_i, r, c]
            solo_1_val = solo_1[v_ref_j, r, c]
            true_cross = true_pair - solo_0_val - solo_1_val
            print(f"    pixel ({r},{c}) v0={v_ref_i} v1={v_ref_j}: "
                  f"true_pair={true_pair:.4f} (from_cols={true_pair_from_cols:.4f}) "
                  f"recon={recon:.4f} true_cross={true_cross:.4f}")

# Check active pixels of pair vs my overlap detection
active_pair = (np.abs(rows).max(axis=(0,1)) > 1e-3) | (np.abs(cols).max(axis=(0,1)) > 1e-3)
ap_r, ap_c = np.where(active_pair)
print(f"\nActive pair pixels (from raw rows/cols): {int(active_pair.sum())} at {list(zip(ap_r.tolist(), ap_c.tolist()))[:10]}")

# Also check: at active_0 & active_1 (both solo active), we expect real cross:
truly_overlap = (np.abs(rows).max(axis=(0,1)) > 1e-3) & active_0 & active_1
tr, tc = np.where(truly_overlap)
print(f"Overlap (active_pair & active_0 & active_1): {int(truly_overlap.sum())} at {list(zip(tr.tolist(), tc.tolist()))[:10]}")
