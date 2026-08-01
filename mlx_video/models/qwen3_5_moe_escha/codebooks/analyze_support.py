"""Compute actual support patterns from joint_probe_v1.pkl (single-slot deltas)."""

import pickle
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent / "joint_probe_v1.pkl"

with open(DATA, "rb") as f:
    d = pickle.load(f)

p1_meta = d["P1_single_deltas"]["meta"]
p1_deltas = d["P1_single_deltas"]["deltas"].astype(np.float32)

# Union support per k
support_per_k = {}
for i, (k, v) in enumerate(p1_meta):
    active = np.abs(p1_deltas[i]) > 1e-3
    if k not in support_per_k:
        support_per_k[k] = np.zeros((16, 16), dtype=bool)
    support_per_k[k] |= active

print("Actual single-slot support (union across all tested v):")
for k in sorted(support_per_k.keys()):
    rows, cols = np.where(support_per_k[k])
    print(f"  k={k:2d}: {int(support_per_k[k].sum()):3d} pixels  "
          f"rows={sorted(set(rows.tolist()))}  "
          f"cols={sorted(set(cols.tolist()))}")

print("\nPair overlap check (spatial support intersection):")
pairs_to_check = [
    (0,1),(0,2),(0,3),(0,5),(0,7),(0,15),(1,2),(1,4),(2,3),(2,5),(2,7),
    (3,6),(4,5),(4,7),(4,11),(5,8),(6,7),(7,8),(6,9),(7,10),(8,9),(8,11),
    (9,12),(10,11),(10,13),(11,14),(12,13),(12,15),(14,15),(0,14),
]
for k_i, k_j in pairs_to_check:
    if k_i in support_per_k and k_j in support_per_k:
        overlap = support_per_k[k_i] & support_per_k[k_j]
        n = int(overlap.sum())
        rows, cols = np.where(overlap)
        overlap_pixels = list(zip(rows.tolist(), cols.tolist()))
        print(f"  ({k_i:2d},{k_j:2d}): overlap={n:2d} pixels  {overlap_pixels[:5]}")
