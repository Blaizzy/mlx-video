"""Signature + linearity audit of torch.ops.escha.escham_reconstruct.

Answers:
1. Full op signature (what argument shapes/dtypes are accepted?)
2. Is the op LINEAR in codes? (i.e., op(A + B) == op(A) + op(B) - op(0))
   - If yes -> superposition holds, we can probe many (bi, bj) simultaneously.
3. If we set exactly one code slot to value v, what is the FULL delta pattern?
   - Which (row, col) positions in the output are nonzero?
4. Is the (codebook entry) slot-invariant across (bi, bj)?
   - I.e., does slot (0, 0, 0) v=v produce the same block pattern (offset by 16)
     as slot (1, 1, 0) v=v ?
5. Does the pattern depend on the *tile shape* passed to the op?
   - Compare 128x128 tile vs 2048x1024 real tile.

Writes /vol/op_audit.pkl and /vol/op_audit_report.md.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import modal


WHEEL_REVISION = "1.0.2+qwen3moe"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04",
        add_python="3.12",
    )
    .apt_install("curl", "binutils", "git", "ca-certificates")
    .pip_install("wheel", "pip", "setuptools")
    .pip_install(
        "torch==2.9.*",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install("numpy", "safetensors", "huggingface_hub[cli]")
    .run_commands(
        f"echo escha wheel revision: {WHEEL_REVISION}",
        "mkdir -p /escha",
        "hf download EschaLabs/escha-runtime-qwen3moe --include 'sglang/*' --local-dir /escha",
        "pip install --no-deps /escha/sglang/escha-*.whl",
    )
)

vol = modal.Volume.from_name("escha-codebooks", create_if_missing=True)
app = modal.App("escha-op-audit", image=image)


@app.function(gpu="A10G", timeout=1800, memory=16 * 1024, volumes={"/vol": vol})
def audit() -> dict:
    import inspect
    import time
    import numpy as np
    import torch
    import escha  # noqa: F401

    report_lines: list[str] = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        report_lines.append(msg)

    log("# Escha `escham_reconstruct` — signature + linearity audit")
    log("")

    # ---- (1) Introspection ----
    log("## 1. Introspection")
    log("")
    C = escha._C
    log(f"escha module: `{escha.__file__}`")
    log(f"escha._C: `{C}`")
    log("")
    log("### `dir(escha._C)`")
    for name in sorted(dir(C)):
        if name.startswith("_"):
            continue
        obj = getattr(C, name)
        try:
            sig = str(inspect.signature(obj))
        except Exception:
            sig = "?"
        try:
            doc = (getattr(obj, "__doc__", None) or "").strip().splitlines()[:1]
            doc = doc[0] if doc else ""
        except Exception:
            doc = ""
        log(f"  - `{name}` {sig} — {doc}")
    log("")
    op = torch.ops.escha.escham_reconstruct
    log(f"### torch.ops.escha.escham_reconstruct")
    log(f"  {op}")
    try:
        log(f"  overloads: {op.overloads()}")
    except Exception:
        pass
    try:
        log(f"  default schema: {op._schema}")
    except Exception:
        pass
    for ov_name in ("default",):
        try:
            ov = getattr(op, ov_name)
            log(f"  {ov_name}: {ov._schema}")
        except Exception as e:
            log(f"  {ov_name}: {e}")
    log("")

    # ---- (2) Try different code tensor shapes ----
    log("## 2. Shape acceptance test")
    log("")
    device = "cuda"

    def try_shape(cshape: tuple[int, ...], in_f: int, out_f: int, K: int, tag: str) -> None:
        try:
            p = torch.zeros(cshape, dtype=torch.int16, device=device)
            w = op(p, in_f, out_f, K, True, False)
            log(f"  {tag}: cshape={cshape} in_f={in_f} out_f={out_f} K={K} -> OK, w.shape={tuple(w.shape)} dtype={w.dtype}")
        except Exception as e:
            log(f"  {tag}: cshape={cshape} in_f={in_f} out_f={out_f} K={K} -> {type(e).__name__}: {e}")

    # Minimum: 128x128
    try_shape((8, 8, 32), 128, 128, 2, "min-K2 128x128")
    try_shape((8, 8, 48), 128, 128, 3, "min-K3 128x128")
    # Escha actual
    try_shape((128, 64, 32), 2048, 1024, 2, "escha gate_up (K=2, in=2048/out=1024)")
    try_shape((32, 128, 48), 512, 2048, 3, "escha down (K=3, in=512/out=2048)")
    # Batched leading dim?
    try_shape((2, 8, 8, 32), 128, 128, 2, "leading batch (2, 8, 8, 32) K=2")
    try_shape((16, 8, 8, 32), 128, 128, 2, "leading batch (16, 8, 8, 32) K=2")
    try_shape((4, 128, 64, 32), 2048, 1024, 2, "leading batch (4, 128, 64, 32) K=2")
    log("")

    # ---- (3) Full delta pattern at slot (0,0,0) for various values ----
    log("## 3. Full delta pattern at slot (0,0,0)")
    log("")
    log("For (in=2048, out=1024, K=2) tile: what is the full (row, col) support")
    log("of the delta when we set exactly code[0,0,0] = v, for various v?")
    log("")
    in_f, out_f, K = 2048, 1024, 2
    cshape = (128, 64, 32)
    p0 = torch.zeros(cshape, dtype=torch.int16, device=device)
    w0 = op(p0, in_f, out_f, K, True, False).detach().cpu().numpy().astype(np.float32)
    full_deltas = {}
    for v in [1, 2, 3, 4, 5, 7, 10, 16, 64, 256, 1024, 4096, 16384, 32767, -1, -100, -32768]:
        p = torch.zeros(cshape, dtype=torch.int16, device=device)
        p[0, 0, 0] = np.int16(np.uint16(v & 0xFFFF)) if v >= 0 else np.int16(v)
        w = op(p, in_f, out_f, K, True, False).detach().cpu().numpy().astype(np.float32)
        d = w - w0
        # Find all non-zero (row, col) positions:
        nz_pos = np.argwhere(np.abs(d) > 1e-6)
        # Should be within rows [0..16) × cols [0..16) (or thereabouts) if layout is
        # bi*16 offset. Show ALL positions:
        vals = [(int(r), int(c), float(d[r, c])) for r, c in nz_pos]
        nz_rows_unique = sorted(set(r for r, _, _ in vals))
        nz_cols_unique = sorted(set(c for _, c, _ in vals))
        full_deltas[v] = {
            "positions": vals,
            "nz_rows": nz_rows_unique,
            "nz_cols": nz_cols_unique,
            "n_positions": len(vals),
        }
        log(f"  v={v:6d}: {len(vals):3d} nonzero positions, rows={nz_rows_unique}, cols={nz_cols_unique}")
    log("")

    # ---- (4) Superposition test at real tile shape ----
    log("## 4. Superposition test (LINEARITY in codes)")
    log("")
    log("Test: op(all-zeros with code[bi_a, bj_a, k_a]=v_a AND code[bi_b, bj_b, k_b]=v_b)")
    log("      == op(only code[bi_a, bj_a, k_a]=v_a) + op(only code[bi_b, bj_b, k_b]=v_b) - op(zeros)")
    log("If yes, we can probe many (bi, bj) slots simultaneously in ONE op call.")
    log("")

    def linearity_probe(pairs: list[tuple[tuple[int, int, int], int]], tag: str) -> None:
        # Sum of individual perturbations
        cumulative_delta = np.zeros_like(w0)
        for (pos, v) in pairs:
            pi = torch.zeros(cshape, dtype=torch.int16, device=device)
            pi[pos] = np.int16(np.uint16(v & 0xFFFF)) if v >= 0 else np.int16(v)
            wi = op(pi, in_f, out_f, K, True, False).detach().cpu().numpy().astype(np.float32)
            cumulative_delta += (wi - w0)
        # Combined
        pc = torch.zeros(cshape, dtype=torch.int16, device=device)
        for (pos, v) in pairs:
            pc[pos] = np.int16(np.uint16(v & 0xFFFF)) if v >= 0 else np.int16(v)
        wc = op(pc, in_f, out_f, K, True, False).detach().cpu().numpy().astype(np.float32)
        combined_delta = wc - w0
        # Compare
        diff = np.abs(cumulative_delta - combined_delta)
        norm_combined = np.linalg.norm(combined_delta)
        norm_diff = np.linalg.norm(diff)
        rel = norm_diff / max(norm_combined, 1e-9)
        log(f"  {tag}: |combined|={norm_combined:.3e} |diff|={norm_diff:.3e} rel={rel:.3e}")
        return rel

    # Two positions, different (bi, bj), same k_slot
    linearity_probe([((0, 0, 0), 100), ((1, 1, 0), 200)], "2-pos, distinct (bi,bj), same k")
    # Same block, different k_slot
    linearity_probe([((0, 0, 0), 100), ((0, 0, 5), 200)], "2-pos, same (bi,bj), diff k")
    # Same block, different k_slot, same k-group
    linearity_probe([((0, 0, 0), 100), ((0, 0, 16), 200)], "2-pos, same (bi,bj), diff K-slice")
    # 8 random positions (dense superposition test)
    torch.manual_seed(42)
    pairs = []
    for i in range(8):
        pos = (int(torch.randint(0, 128, (1,))), int(torch.randint(0, 64, (1,))), int(torch.randint(0, 32, (1,))))
        v = int(torch.randint(1, 65536, (1,)))
        pairs.append((pos, v))
    linearity_probe(pairs, "8-pos random")

    # 100 positions (many-slot dense superposition)
    torch.manual_seed(1)
    pairs = []
    used_positions = set()
    for _ in range(100):
        while True:
            pos = (int(torch.randint(0, 128, (1,))), int(torch.randint(0, 64, (1,))), int(torch.randint(0, 32, (1,))))
            if pos not in used_positions:
                used_positions.add(pos)
                break
        v = int(torch.randint(1, 65536, (1,)))
        pairs.append((pos, v))
    linearity_probe(pairs, "100-pos random")
    log("")

    # ---- (5) Slot invariance: does (0,0,0)+v produce same shape as (1,1,0)+v (offset by 16 rows/cols)? ----
    log("## 5. Slot invariance test — is the codebook shared across (bi, bj)?")
    log("")
    log("Compare delta patterns for the SAME value v at DIFFERENT (bi, bj) with the same k_slot.")
    log("If they are identical up to a (bi*16, bj*16) offset, the codebook is (bi, bj)-invariant.")
    log("")
    for v in [1, 100, 32767, -32768]:
        p_a = torch.zeros(cshape, dtype=torch.int16, device=device)
        p_a[0, 0, 0] = np.int16(np.uint16(v & 0xFFFF)) if v >= 0 else np.int16(v)
        w_a = op(p_a, in_f, out_f, K, True, False).detach().cpu().numpy().astype(np.float32)
        d_a = (w_a - w0)[:16, :16]  # extract block at (0..16, 0..16)

        for (bi_test, bj_test) in [(1, 0), (0, 1), (1, 1), (5, 3), (127, 63)]:
            p_b = torch.zeros(cshape, dtype=torch.int16, device=device)
            p_b[bi_test, bj_test, 0] = np.int16(np.uint16(v & 0xFFFF)) if v >= 0 else np.int16(v)
            w_b = op(p_b, in_f, out_f, K, True, False).detach().cpu().numpy().astype(np.float32)
            d_b = (w_b - w0)[bi_test*16:(bi_test+1)*16, bj_test*16:(bj_test+1)*16]
            diff = np.abs(d_a - d_b)
            rel = float(np.linalg.norm(diff) / max(np.linalg.norm(d_a), 1e-9))
            log(f"  v={v:6d} (bi=0,bj=0) vs (bi={bi_test},bj={bj_test}): |diff|={float(np.linalg.norm(diff)):.3e} rel={rel:.3e}")
    log("")

    # ---- (6) k_slot invariance: does (0,0,k)+v have consistent pattern for different k? ----
    log("## 6. k_slot pattern")
    log("")
    log("For each k_slot, what is the row/col support of the (0, 0, k)+v=1 delta?")
    log("")
    ks_patterns: dict = {}
    for K_test in (2, 3):
        cs = (128, 64, 32) if K_test == 2 else (32, 128, 48)
        inf, outf = (2048, 1024) if K_test == 2 else (512, 2048)
        p0k = torch.zeros(cs, dtype=torch.int16, device=device)
        w0k = op(p0k, inf, outf, K_test, True, False).detach().cpu().numpy().astype(np.float32)
        log(f"### K={K_test}, cshape={cs}")
        for k in range(16 * K_test):
            p = torch.zeros(cs, dtype=torch.int16, device=device)
            p[0, 0, k] = 1
            w = op(p, inf, outf, K_test, True, False).detach().cpu().numpy().astype(np.float32)
            d = (w - w0k)[:16, :16]
            nz_pos = np.argwhere(np.abs(d) > 1e-6)
            rows = sorted(set(int(r) for r, _ in nz_pos))
            cols = sorted(set(int(c) for _, c in nz_pos))
            ks_patterns[(K_test, k)] = {"rows": rows, "cols": cols, "n_pos": len(nz_pos),
                                        "dense_block": d.tolist()}
            log(f"  k={k:2d}: rows={rows} cols={cols} n_pos={len(nz_pos)}")
    log("")

    # Save report + full audit data
    Path("/vol").mkdir(parents=True, exist_ok=True)
    result = {
        "full_deltas": full_deltas,
        "ks_patterns": ks_patterns,
        "report_md": "\n".join(report_lines),
    }
    with open("/vol/op_audit.pkl", "wb") as f:
        pickle.dump(result, f)
    with open("/vol/op_audit_report.md", "w") as f:
        f.write("\n".join(report_lines))
    vol.commit()

    return result


@app.local_entrypoint()
def main() -> None:
    result = audit.remote()
    out_pkl = Path("/Users/kaede/mlx-video/mlx_video/models/qwen3_5_moe_escha/codebooks/op_audit.pkl")
    out_md = Path("/Users/kaede/mlx-video/docs/escha_op_signature.md")
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pkl, "wb") as f:
        pickle.dump(result, f)
    with open(out_md, "w") as f:
        f.write(result["report_md"])
    print(f"[local] wrote {out_pkl}")
    print(f"[local] wrote {out_md}")
