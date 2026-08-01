"""Inspect K=3 solo support patterns to figure out slot layout."""
import pickle
from pathlib import Path
import numpy as np

D = Path(__file__).parent
with open(D / "full_extract_v3.pkl", "rb") as f:
    d = pickle.load(f)

v = d["down_proj"]
solo = v["solo_data"]

print("Support (union over v=0..65535) per extracted slot:")
for k in sorted(solo.keys()):
    s = solo[k].astype(np.float32)  # (65536, 16, 16)
    active = (np.abs(s).max(axis=0) > 1e-3)
    rows = sorted(set(np.where(active)[0].tolist()))
    cols = sorted(set(np.where(active)[1].tolist()))
    print(f"  slot k={k:2d}: {int(active.sum()):3d} pixels  rows={rows} cols={cols}")

print("\nCompare K=2 (for reference):")
solo2 = d["gate_up"]["solo_data"]
for k in sorted(solo2.keys()):
    s = solo2[k].astype(np.float32)
    active = (np.abs(s).max(axis=0) > 1e-3)
    rows = sorted(set(np.where(active)[0].tolist()))
    cols = sorted(set(np.where(active)[1].tolist()))
    print(f"  K=2 slot k={k:2d}: {int(active.sum()):3d} pixels  rows={rows} cols={cols}")

# Test all possible shifts for K=3 slot 4 vs slot 0
print("\n\nK=3 slot=4 vs shifts of slot=0:")
s0 = solo[0].astype(np.float32)
s4 = solo[4].astype(np.float32)
for shift_col in range(16):
    diff = float(np.abs(s0 - np.roll(s4, shift=-shift_col, axis=2)).max())
    if diff < 1.0:
        print(f"  shift col by {shift_col}: max_diff = {diff:.4f}")
for shift_row in range(16):
    diff = float(np.abs(s0 - np.roll(s4, shift=-shift_row, axis=1)).max())
    if diff < 1.0:
        print(f"  shift row by {shift_row}: max_diff = {diff:.4f}")
