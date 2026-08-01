"""Test: naive additive-only reconstruction vs real op(code)."""
import pickle
from pathlib import Path
import numpy as np

D = Path(__file__).parent
EXTRACT = D / "full_extract_v3.pkl"

with open(EXTRACT, "rb") as f:
    d = pickle.load(f)

for name in ["gate_up", "down_proj"]:
    print(f"\n=== {name} ===")
    v = d[name]
    ref = d["ref_samples"][name]
    K = v["K"]
    in_f = v["in_f"]
    out_f = v["out_f"]
    bi_max = v["cshape"][0]
    bj_max = v["cshape"][1]
    k_max = v["cshape"][2]  # 16*K

    solo = v["solo_data"]  # {k: (65536, 16, 16)}
    print(f"solo_data slot keys: {sorted(solo.keys())}")

    # For each slot 0..k_max-1, we need solo_k. But we only extracted a subset.
    # Try to use col-shift replication from k%4 (for K=2 this works, K=3 unclear).
    # For each K-layer independently, use the same 4 unique row-patterns and shift.

    code = ref["code"]  # (bi_max, bj_max, k_max) int16
    w_ref = ref["w_ref"].astype(np.float32)  # (in_f, out_f)
    w0 = v["w0"].astype(np.float32)

    # Build w_recon = w0 + sum over slots of solo_k(code_k) placed at (bi, bj) block
    w_recon = w0.copy()
    idx = (code.astype(np.int32) & 0xFFFF)  # (bi, bj, k) uint16 index

    for k_slot in range(k_max):
        k_mod = k_slot % 4  # row pattern
        k_div = k_slot // 4  # col shift
        # For gate_up (K=2), solos are for k=0,1,2,3 (K-layer 1) and k=16 (K-layer 2 slot 0)
        # For K-layer determination:
        k_layer = k_slot // 16  # 0..K-1
        slot_in_layer = k_slot % 16
        col_shift_in_layer = slot_in_layer // 4

        # Pick which solo array to use:
        # K-layer 0: use solo[k_mod]
        # K-layer 1: use solo[16 + k_mod] if available
        # K-layer 2: use solo[32 + k_mod] if available
        if k_layer == 0:
            slot_key = k_mod
        elif k_layer == 1:
            slot_key = 16 if k_mod == 0 else None
        elif k_layer == 2:
            slot_key = 32 if k_mod == 0 else None
        else:
            slot_key = None

        if slot_key is None or slot_key not in solo:
            continue  # skip if we don't have this solo

        s = solo[slot_key].astype(np.float32)  # (65536, 16, 16)

        # If we're using solo[k_mod] but slot_in_layer > k_mod (needing col-shift):
        # For K-layer 0: solo[k%4] shifted by k//4 gives solo[k]
        # For K-layer 1: solo[16 + k%4] shifted by (k%16)//4 gives solo[16 + k%16]
        # We only have solo[16] (=slot 16), so K-layer 1 only useful for slot_in_layer==0

        col_shift = col_shift_in_layer  # 0..3

        # Get code indices at this k_slot: (bi_max, bj_max)
        code_k = idx[..., k_slot]  # (bi_max, bj_max)
        # Gather solo tiles: solo_k[code_k] shape (bi_max, bj_max, 16, 16)
        tiles = s[code_k]  # (bi_max, bj_max, 16, 16)
        # Shift cols by col_shift
        if col_shift > 0:
            tiles = np.roll(tiles, shift=col_shift, axis=3)
        # Place in w_recon: block (bi, bj) at rows bi*16..(bi+1)*16, cols bj*16..(bj+1)*16
        # Vectorize by reshaping
        w_recon_blocks = w_recon.reshape(bi_max, 16, bj_max, 16)
        w_recon_blocks[:, :, :, :] += tiles.transpose(0, 2, 1, 3)
        # After transpose, tiles shape (bi_max, 16, bj_max, 16) — matches reshape

    # Compare
    diff = np.abs(w_recon - w_ref)
    print(f"|w_ref|max: {np.abs(w_ref).max():.3f}, |w_recon|max: {np.abs(w_recon).max():.3f}")
    print(f"NAIVE recon: max_diff={diff.max():.4f}, mean_diff={diff.mean():.6f}, "
          f"l2_diff={np.linalg.norm(w_recon - w_ref):.3f} / |w_ref|_2={np.linalg.norm(w_ref):.3f}")
    print(f"Relative error: {np.linalg.norm(w_recon - w_ref)/np.linalg.norm(w_ref)*100:.2f}%")
