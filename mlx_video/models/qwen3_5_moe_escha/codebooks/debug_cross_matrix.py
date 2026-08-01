"""Compute cross(v0, v1) at overlap pixels of pair (0,1) for all ref × ref combos."""
import pickle
from pathlib import Path
import numpy as np

D = Path(__file__).parent
with open(D / "full_extract_v4.pkl", "rb") as f:
    d = pickle.load(f)

v = d["gate_up"]
cross_data = v["cross_data"]
solo = v["solo_data"]

pair_key = "(0, 1, 'A_L0_m0')"
entry = cross_data[pair_key]
rows = entry["rows_deltas"].astype(np.float32)
refs = entry["ref_values"]
print(f"refs: {refs}")

solo_0 = solo[0].astype(np.float32)  # (65536, 16, 16)
solo_1 = solo[1].astype(np.float32)

# Overlap pixels
active_0 = np.abs(solo_0).max(axis=0) > 1e-3
active_1 = np.abs(solo_1).max(axis=0) > 1e-3
overlap = active_0 & active_1
ov_r, ov_c = np.where(overlap)

# Also inspect cross at MORE varied (v0, v1) using extracted rows_deltas
# rows[i, v1, r, c] = pair(ref_i, v1)
# For each (ref_i, ref_j), compute cross at overlap pixels
print("\ncross(v0, v1) at overlap pixel (2, 0) for all ref combos:")
for i, v0 in enumerate(refs):
    for j, v1 in enumerate(refs):
        pair_val = rows[i, int(v1) & 0xFFFF, 2, 0]
        cross_val = pair_val - solo_0[int(v0) & 0xFFFF, 2, 0] - solo_1[int(v1) & 0xFFFF, 2, 0]
        print(f"  v0={v0:>6d} v1={v1:>6d}: pair={pair_val:.4f}  cross={cross_val:.4f}")

# Sample cross at OTHER (v0, v1) values by using solo lookups combined with pair-factored recon:
print("\ncross(v0, v1) at overlap pixel (2, 0) for various v1 (v0 = refs[0] = 1):")
i = 0
for v1_test in [1, 2, 3, 4, 5, 10, 100, 1000, 10000, 32000, -1, -2, -100, -1000, -10000]:
    v1_idx = int(v1_test) & 0xFFFF
    pair_val = rows[i, v1_idx, 2, 0]
    cross_val = pair_val - solo_0[1, 2, 0] - solo_1[v1_idx, 2, 0]
    print(f"  v0={refs[i]:>6d} v1={v1_test:>6d}: pair={pair_val:.4f}  cross={cross_val:.4f}")

print("\ncross(v0, v1) at overlap pixel (2, 0) for various v0 (v1 = refs[0] = 1):")
cols = entry["cols_deltas"].astype(np.float32)
j = 0
for v0_test in [1, 2, 3, 4, 5, 10, 100, 1000, 10000, 32000, -1, -2, -100, -1000, -10000]:
    v0_idx = int(v0_test) & 0xFFFF
    pair_val = cols[j, v0_idx, 2, 0]
    cross_val = pair_val - solo_0[v0_idx, 2, 0] - solo_1[1, 2, 0]
    print(f"  v0={v0_test:>6d} v1={refs[j]:>6d}: pair={pair_val:.4f}  cross={cross_val:.4f}")
