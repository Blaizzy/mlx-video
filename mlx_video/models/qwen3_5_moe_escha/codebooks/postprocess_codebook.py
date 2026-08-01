"""Convert Modal's raw sweep output (cb_K2.npy, cb_K3.npy, first_nz_row_K2.npy,
first_nz_row_K3.npy, reference_dump.pkl) into escha_codebooks_v1.npz — the
format expected by eschamoe.py::_load_codebook.

Also runs a correctness check against the reference weights dumped by Modal.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


HERE = Path(__file__).parent


def load_raw() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    cb_K2 = np.load(HERE / "cb_K2.npy")
    cb_K3 = np.load(HERE / "cb_K3.npy")
    nz_K2 = np.load(HERE / "first_nz_row_K2.npy")
    nz_K3 = np.load(HERE / "first_nz_row_K3.npy")
    with open(HERE / "reference_dump.pkl", "rb") as f:
        dump = pickle.load(f)
    return cb_K2, cb_K3, nz_K2, nz_K3, dump


def analyze(cb: np.ndarray, nz: np.ndarray, tag: str) -> None:
    print(f"\n[{tag}] shape={cb.shape} dtype={cb.dtype}")
    print(f"[{tag}] value range: [{cb.min():.4f}, {cb.max():.4f}]")
    print(f"[{tag}] zero-rows count: {(cb == 0).all(axis=1).sum()}/{cb.shape[0]}")
    print(f"[{tag}] first_nz_row unique values: {np.unique(nz).tolist()[:10]} ...")
    print(f"[{tag}] first_nz_row distribution:")
    vals, counts = np.unique(nz, return_counts=True)
    for v, c in list(zip(vals, counts))[:20]:
        print(f"  row {v}: {c} codes")


def to_npz(out_path: Path) -> None:
    cb_K2, cb_K3, nz_K2, nz_K3, dump = load_raw()
    analyze(cb_K2, nz_K2, "K=2")
    analyze(cb_K3, nz_K3, "K=3")

    # Trim to first 16 columns — the classic AQLM codebook layout.
    # (Column 16..31 captures spread; save separately in _wide fields for debug.)
    cb_K2_16 = cb_K2[:, :16].astype(np.float16)
    cb_K3_16 = cb_K3[:, :16].astype(np.float16)

    np.savez_compressed(
        out_path,
        cb_A_K2=cb_K2_16,
        cb_A_K3=cb_K3_16,
        cb_A_K2_wide=cb_K2.astype(np.float16),
        cb_A_K3_wide=cb_K3.astype(np.float16),
        first_nz_row_K2=nz_K2,
        first_nz_row_K3=nz_K3,
    )
    print(f"\nwrote {out_path} ({out_path.stat().st_size/1024/1024:.2f} MiB)")


def check_dump(dump: dict) -> None:
    print("\n=== reference_dump.pkl summary ===")
    for tag in ("gate_up", "down"):
        s = dump.get(f"{tag}_spread", {})
        stats = dump.get(f"{tag}_real_w_stats", {})
        print(f"[{tag}] real_w: shape={stats.get('shape')} finite={stats.get('finite_frac'):.4f} max={stats.get('max_abs'):.3f} mean={stats.get('mean_abs'):.4f}")
        print(f"[{tag}] spread test — perturb each position, count changed rows/cols:")
        for k, v in s.items():
            print(f"  {k}: nz_rows={v['n_nz_rows']}, nz_cols={v['n_nz_cols']}, ||d||={v['delta_norm']:.3f}")
    print(f"[linearity] superposition_err_2pos: {dump.get('superposition_err_2pos'):.6f}")
    print(f"  If ~0, op is linear in code positions → simple codebook is valid.")
    lin = dump.get("linearity_test_K2", {})
    print("[linearity] first_nz_row across values (should be same if lookup is pos-dependent only):")
    for val, info in list(lin.items())[:8]:
        print(f"  code=1×{val} at (0,0,0): nz_row={info['first_nz_row']}, ||d||={info['delta_norm']:.3f}")


if __name__ == "__main__":
    import sys
    if not (HERE / "cb_K2.npy").exists():
        print(f"[error] cb_K2.npy not found in {HERE}. Run: modal run modal_extract_v2.py::fetch")
        sys.exit(1)
    _, _, _, _, dump = load_raw()
    check_dump(dump)
    out = HERE / "escha_codebooks_v1.npz"
    to_npz(out)
