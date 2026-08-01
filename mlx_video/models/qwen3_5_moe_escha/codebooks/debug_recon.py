"""Debug recon by starting with a known simple code."""
import pickle
from pathlib import Path
import numpy as np

D = Path(__file__).parent
EXTRACT = D / "full_extract_v4.pkl"

with open(EXTRACT, "rb") as f:
    d = pickle.load(f)

# Test 1: verify solo extraction: solo_0(v=1) should equal what my code
# extraction says.
name = "gate_up"
v = d[name]
K = v["K"]
in_f = v["in_f"]
out_f = v["out_f"]
solo = v["solo_data"]
w0 = v["w0"].astype(np.float32)

ref = d["ref_samples"][name]
code = ref["code"]
w_ref = ref["w_ref"].astype(np.float32)

print(f"code shape: {code.shape}, dtype: {code.dtype}")
print(f"code non-zero count: {np.sum(code != 0)}, total: {code.size}")
print(f"code max/min: {code.max()}, {code.min()}")
print(f"code unique count: {len(np.unique(code))}")

# Test naive solo-only recon manually:
bi_max, bj_max, k_max = code.shape
w_recon = w0.copy()
idx = (code.astype(np.int32) & 0xFFFF)
for k in range(k_max):
    tiles = solo[k].astype(np.float32)[idx[:, :, k]]  # (bi, bj, 16, 16)
    # Place into w_recon
    for bi in range(bi_max):
        for bj in range(bj_max):
            w_recon[bi*16:(bi+1)*16, bj*16:(bj+1)*16] += tiles[bi, bj]

diff = np.abs(w_recon - w_ref)
print(f"\nnaive-solo-only recon (with FULL 32 solo functions):")
print(f"|w_recon|max={np.abs(w_recon).max():.3f} |w_ref|max={np.abs(w_ref).max():.3f}")
print(f"max_diff={diff.max():.4f} mean_diff={diff.mean():.6f}")
print(f"rel err = {np.linalg.norm(w_recon-w_ref)/np.linalg.norm(w_ref)*100:.2f}%")

# Now try: for ONE block only, use ONLY solo and see if it's approximately correct
# Take (bi, bj) = (0, 0). Real code has 32 values there for K=2.
# Sum: op(code_with_only_that_block_non_zero) = w0 + delta_from_block(0,0)
# Compare to solo sum for that block

bi, bj = 0, 0
block_ref = w_ref[bi*16:(bi+1)*16, bj*16:(bj+1)*16] - w0[bi*16:(bi+1)*16, bj*16:(bj+1)*16]
block_solo = np.zeros((16, 16), dtype=np.float32)
for k in range(k_max):
    v = int(idx[bi, bj, k])
    block_solo += solo[k].astype(np.float32)[v]

block_diff = block_ref - block_solo
print(f"\nBlock ({bi},{bj}) analysis:")
print(f"  block_ref (target): max={np.abs(block_ref).max():.3f}, l2={np.linalg.norm(block_ref):.3f}")
print(f"  block_solo (sum): max={np.abs(block_solo).max():.3f}, l2={np.linalg.norm(block_solo):.3f}")
print(f"  block_diff (residual): max={np.abs(block_diff).max():.3f}, l2={np.linalg.norm(block_diff):.3f}")

# Sample codes for one slot in the block
print(f"\ncodes at ({bi},{bj}):")
for k in range(k_max):
    print(f"  k={k:2d}: v={int(code[bi, bj, k]):>6d} idx={int(idx[bi, bj, k])}")
