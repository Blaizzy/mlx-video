"""escham_reconstruct v3 — full joint-lookup MLX implementation.

Structure (from ESCHA_LAYOUT_NOTES.md §3):
  For each 16×16 tile (bi, bj):
    delta[bi, bj] = sum over K-layers m of
                     sum over slot k in 0..15 of solo[m, k, code[bi, bj, 16*m + k]]
                   + sum over 15 interacting pairs (i, j) in the K-layer of
                     cross[m, (i,j), code[bi, bj, 16*m+i], code[bi, bj, 16*m+j]]
  w[in_p, out_p] = w0 + delta

Where cross[m, pair, v0, v1] is a (16, 16) tile with non-zero values only at
"overlap pixels" (where solo_i and solo_j both write). Stored as low-rank
factorization: cross ≈ sum_r phi_r(v0) * psi_r(v1), rank ≤ 4.

Codebook artifact: `codebooks/codebook_v3.npz` produced by
`codebooks/build_codebook_v3.py` from `full_extract_v3.pkl` (Modal A10G).
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np

from .transform import t128


_CB_PATH = Path(__file__).parent / "codebooks" / "codebook_v3.npz"

_CB_CACHE: dict[tuple[int, int, int], dict] = {}


# Interacting pair layout per K-layer (indices 0..15):
# Type A pairs (diff 1, even index first): (0,1), (2,3), ..., (14,15) → 8 pairs
# Type B pairs (diff 3, even index first): (0,3), (2,5), ..., (12,15) → 7 pairs
_PAIRS_A_BASE = [(2*m, 2*m + 1) for m in range(8)]
_PAIRS_B_BASE = [(2*m, 2*m + 3) for m in range(7)]


def _load_codebook_v3(in_features: int, out_features: int, K: int):
    key = (int(in_features), int(out_features), int(K))
    if key in _CB_CACHE:
        return _CB_CACHE[key]
    if not _CB_PATH.exists():
        raise FileNotFoundError(
            f"codebook_v3.npz not found at {_CB_PATH}. "
            f"Run modal_extract_v3.py then build_codebook_v3.py."
        )
    npz = np.load(_CB_PATH, allow_pickle=False)
    if K == 2 and in_features == 2048 and out_features == 1024:
        pre = "gate_up__"
    elif K == 3 and in_features == 512 and out_features == 2048:
        pre = "down_proj__"
    else:
        raise NotImplementedError(f"Unsupported shape: in={in_features}, out={out_features}, K={K}")

    cb = {
        "K": int(npz[pre + "K"][0]),
        "in_f": int(npz[pre + "in_f"][0]),
        "out_f": int(npz[pre + "out_f"][0]),
        "cshape": npz[pre + "cshape"].tolist(),
        "w0": mx.array(npz[pre + "w0"]).astype(mx.bfloat16),  # (in_f, out_f)
        # Solo — stored as full (num_slots, 65536, 16, 16) fp16 array
        # See build_codebook_v3.py for arrangement.
        "solo": mx.array(npz[pre + "solo_full"]).astype(mx.bfloat16),  # (16*K, 65536, 16, 16)
    }

    # Cross data (all interacting pairs)
    n_k_layers = K
    cross_pairs_info = []
    for k_layer in range(n_k_layers):
        for m_a in range(8):
            k_i = 16*k_layer + 2*m_a
            k_j = 16*k_layer + 2*m_a + 1
            tag = f"A_L{k_layer}_m{m_a}"
            cross_pairs_info.append((k_i, k_j, tag))
        for m_b in range(7):
            k_i = 16*k_layer + 2*m_b
            k_j = 16*k_layer + 2*m_b + 3
            tag = f"B_L{k_layer}_m{m_b}"
            cross_pairs_info.append((k_i, k_j, tag))

    cross_bundles = []
    for (k_i, k_j, tag) in cross_pairs_info:
        try:
            phi = mx.array(npz[pre + f"cross_{tag}_phi"]).astype(mx.bfloat16)
            psi = mx.array(npz[pre + f"cross_{tag}_psi"]).astype(mx.bfloat16)
            overlap = np.array(npz[pre + f"cross_{tag}_overlap"], dtype=bool)
        except KeyError:
            continue
        cross_bundles.append({"k_i": k_i, "k_j": k_j, "tag": tag, "phi": phi, "psi": psi, "overlap": overlap})
    cb["cross_pairs"] = cross_bundles

    _CB_CACHE[key] = cb
    return cb


def escham_reconstruct(
    code: mx.array,
    in_features: int,
    out_features: int,
    K: int,
    cb_id: int = 1,
    mul1: bool = False,
) -> mx.array:
    """Decode Escha packed codes to a dense fp16 weight matrix via joint lookup.

    Args:
        code: int16 (in_p/16, out_p/16, 16*K).
        in_features, out_features: padded dims.
        K: residual depth (2 for gate_up, 3 for down).
    """
    if cb_id != 1:
        raise NotImplementedError(f"Only cb_id=1 supported; got {cb_id}")
    cb = _load_codebook_v3(in_features, out_features, K)
    bi_max = in_features // 16
    bj_max = out_features // 16
    k_max = 16 * K
    if code.shape != (bi_max, bj_max, k_max):
        raise ValueError(f"code shape {code.shape} != expected ({bi_max}, {bj_max}, {k_max})")

    # Reinterpret int16 → uint16 index in [0, 65536).
    idx = (code.astype(mx.int32) & 0xFFFF)  # (bi, bj, k_max)

    # ---- solo contribution ----
    # cb['solo']: (k_max, 65536, 16, 16) bf16
    # For each k in 0..k_max-1, gather solo[k, idx[..., k]] → (bi, bj, 16, 16)
    # Flatten (k, code) axis for a single mx.take.
    solo = cb["solo"]  # (k_max, 65536, 16, 16)
    flat_solo = solo.reshape(k_max * 65536, 16, 16)
    idx_k_major = idx.transpose(2, 0, 1)  # (k, bi, bj)
    flat_idx = mx.arange(k_max, dtype=mx.int32).reshape(k_max, 1, 1) * 65536 + idx_k_major
    solo_tiles = mx.take(flat_solo, flat_idx, axis=0)  # (k, bi, bj, 16, 16)
    per_block = solo_tiles.sum(axis=0)  # (bi, bj, 16, 16)

    # ---- cross contribution ----
    for pair in cb["cross_pairs"]:
        k_i, k_j = pair["k_i"], pair["k_j"]
        phi = pair["phi"]  # (65536, R, 16, 16)
        psi = pair["psi"]  # (R, 65536, 16, 16)
        R = phi.shape[1]
        v0 = idx[..., k_i]  # (bi, bj)
        v1 = idx[..., k_j]
        # phi_gather[v0]: (bi, bj, R, 16, 16)
        phi_gather = mx.take(phi, v0, axis=0)
        # psi is (R, 65536, ...). Transpose to (65536, R, ...) then take.
        psi_T = psi.transpose(1, 0, 2, 3)  # (65536, R, 16, 16)
        psi_gather = mx.take(psi_T, v1, axis=0)  # (bi, bj, R, 16, 16)
        # Element-wise product + sum over R
        cross_tile = (phi_gather * psi_gather).sum(axis=2)  # (bi, bj, 16, 16)
        per_block = per_block + cross_tile

    # Reassemble (bi, bj, 16, 16) -> (in, out)
    per_block = per_block.transpose(0, 2, 1, 3)  # (bi, 16, bj, 16)
    delta = per_block.reshape(in_features, out_features)
    return (delta + cb["w0"]).astype(mx.float16)
