"""Test full solo+cross reconstruction — fix: subtract solo from pair-factor to get pure cross."""
import pickle
from pathlib import Path
import numpy as np

D = Path(__file__).parent
CB = np.load(D / "codebook_v3.npz", allow_pickle=False)
REF = np.load(D / "ref_samples_v3.npz", allow_pickle=False)


def reconstruct(name: str, K: int, in_f: int, out_f: int, code: np.ndarray):
    pre = f"{name}__"
    w0 = CB[pre + "w0"].astype(np.float32)
    solo_full = CB[pre + "solo_full"].astype(np.float32)  # (16*K, 65536, 16, 16)
    bi_max = in_f // 16
    bj_max = out_f // 16
    k_max = 16 * K

    idx = (code.astype(np.int32) & 0xFFFF)  # (bi_max, bj_max, k_max)

    # Solo sum
    per_block = np.zeros((bi_max, bj_max, 16, 16), dtype=np.float32)
    for k in range(k_max):
        per_block += solo_full[k, idx[..., k], :, :]

    # Cross: at overlap pixels, add pure cross = pair_factored - solo_i - solo_j
    cross_keys = [k for k in CB.files if k.startswith(pre + "cross_") and k.endswith("_phi")]
    for pk in cross_keys:
        tag = pk[len(pre + "cross_"):-len("_phi")]
        parts = tag.split("_")
        typ = parts[0]
        k_layer = int(parts[1][1:])
        m = int(parts[2][1:])
        base = 16 * k_layer
        if typ == "A":
            k_i, k_j = base + 2*m, base + 2*m + 1
        else:
            k_i, k_j = base + 2*m, base + 2*m + 3
        phi = CB[pre + f"cross_{tag}_phi"].astype(np.float32)  # (65536, R, 16, 16)
        psi = CB[pre + f"cross_{tag}_psi"].astype(np.float32)  # (R, 65536, 16, 16)
        overlap = CB[pre + f"cross_{tag}_overlap"].astype(bool)  # (16, 16)
        v0 = idx[..., k_i]  # (bi, bj)
        v1 = idx[..., k_j]
        phi_gather = phi[v0]  # (bi, bj, R, 16, 16)
        psi_gather = psi[:, v1, :, :].transpose(1, 2, 0, 3, 4)  # (bi, bj, R, 16, 16)
        pair_tile = (phi_gather * psi_gather).sum(axis=2)  # (bi, bj, 16, 16) — full pair
        # solo_i(v0) at all pixels, solo_j(v1) at all pixels
        solo_i_tile = solo_full[k_i, v0, :, :]  # (bi, bj, 16, 16)
        solo_j_tile = solo_full[k_j, v1, :, :]
        cross_tile = pair_tile - solo_i_tile - solo_j_tile  # (bi, bj, 16, 16)
        # Apply only at overlap pixels
        mask = overlap[None, None, :, :]  # (1, 1, 16, 16) bool
        per_block = np.where(mask, per_block + cross_tile, per_block)

    per_block = per_block.transpose(0, 2, 1, 3)
    delta = per_block.reshape(in_f, out_f)
    return delta + w0


for name, K, in_f, out_f in [("gate_up", 2, 2048, 1024), ("down_proj", 3, 512, 2048)]:
    print(f"\n=== {name} K={K} ===")
    code = REF[name + "__code"]
    w_ref = REF[name + "__w_ref"].astype(np.float32)
    w_recon = reconstruct(name, K, in_f, out_f, code)
    diff = np.abs(w_recon - w_ref)
    print(f"|w_ref|max={np.abs(w_ref).max():.3f} |w_recon|max={np.abs(w_recon).max():.3f}")
    print(f"max_diff = {diff.max():.4f}")
    print(f"mean_diff = {diff.mean():.6f}")
    print(f"|delta|_2 / |w_ref|_2 = {np.linalg.norm(w_recon-w_ref) / np.linalg.norm(w_ref) * 100:.3f}%")
    print(f"diff > 1e-2 fraction: {(diff > 1e-2).mean():.4f}")
    print(f"diff > 1e-1 fraction: {(diff > 1e-1).mean():.4f}")
