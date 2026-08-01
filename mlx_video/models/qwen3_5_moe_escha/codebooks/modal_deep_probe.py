"""Deep probe of the escha _C module: verify whether escham_reconstruct is a
codebook lookup or a deterministic bit-level generator (trellis / QuIP#-style).

Also dumps a REAL-weight reference: given (code, rin, rout) from the actual
safetensors, produces (w_bare, y = xh @ w) for a small canonical x. This lets
the MLX port verify its dequant against ground truth without another Modal run.

Output: single pickle to /Users/kaede/mlx-video/mlx_video/models/qwen3_5_moe_escha/codebooks/escha_reference.pkl
containing:
  - ops_available: list of all torch.ops.escha.* ops with signatures
  - C_symbols: list of interesting _C symbols
  - probe_results: dict of small experiments showing how the op behaves
  - real_weight_ref: {code, rin, rout, w_bare, x, xh, y} for one gate_up expert
"""

import io
import pickle
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
        # --no-deps: escha wheel declares sglang as dep, but we don't need sglang;
        # we only need escha._C which links against torch/CUDA already installed.
        "pip install --no-deps /escha/sglang/escha-*.whl",
    )
)

app = modal.App("escha-deep-probe", image=image)


@app.function(gpu="A10G", timeout=60 * 20, memory=32 * 1024)
def deep_probe(weight_bytes_gu: bytes, weight_bytes_dn: bytes) -> bytes:
    import subprocess
    import numpy as np
    import torch
    import escha  # noqa: F401
    import inspect

    out: dict = {}

    # ---- (1) list all torch.ops.escha.* with real callable check ----
    ops_ns = torch.ops.escha
    ops_list = []
    for name in dir(ops_ns):
        if name.startswith("_"):
            continue
        try:
            obj = getattr(ops_ns, name)
            ops_list.append((name, type(obj).__name__, str(obj)))
        except Exception as e:
            ops_list.append((name, "err", str(e)))
    # Try both the documented names AND some guesses via getattr
    for guess in [
        "escham_reconstruct", "escha_reconstruct", "escham_decode_gemv",
        "escham_moe_linear", "escham_moe_linear_swiglu",
        "escha_dequant", "escha_lut_binary_gemv", "escha_transform",
    ]:
        try:
            op = getattr(ops_ns, guess)
            # Try calling it with plausible zero args -- expect failure but
            # capture the error message; sometimes torch shows the correct schema.
            try:
                op()
                ops_list.append((guess, "callable-0args", "returned"))
            except Exception as e:
                ops_list.append((guess, "callable-schema", str(e)[:400]))
        except Exception as e:
            ops_list.append((guess, "getattr-err", str(e)[:200]))
    out["ops_available"] = ops_list

    # ---- (2) list _C symbols ----
    import escha._C as C
    c_syms = []
    for name in sorted(dir(C)):
        if name.startswith("_"):
            continue
        try:
            obj = getattr(C, name)
            sig = None
            try:
                sig = str(inspect.signature(obj))
            except Exception:
                pass
            doc = getattr(obj, "__doc__", None) or ""
            c_syms.append((name, type(obj).__name__, sig, doc[:300]))
        except Exception as e:
            c_syms.append((name, "err", None, str(e)))
    out["c_symbols"] = c_syms

    # Objdump for constant data section sizes -- codebook indicator
    so = escha._C.__file__
    try:
        objdump = subprocess.check_output(["objdump", "-h", so], text=True)
        rodata_lines = [l for l in objdump.splitlines() if any(s in l for s in (".rodata", ".nv.", ".data.rel.ro"))]
        out["so_sections"] = rodata_lines
    except Exception as e:
        out["so_sections"] = [f"err: {e}"]

    # ---- (3) small probe of escham_reconstruct behavior ----
    # Try to grab the op regardless of dir() visibility
    op = None
    for cand in ("escham_reconstruct", "escha_reconstruct"):
        try:
            op = getattr(ops_ns, cand)
            print(f"[probe] using op {cand}: {op}")
            out["reconstruct_op_name"] = cand
            break
        except Exception:
            continue

    probe_results = {}
    if op is not None:
        device = "cuda"
        # Small canonical shape: in=128, out=128, K=2 -> code shape (8, 8, 32)
        in_f, out_f, K = 128, 128, 2
        cshape = (in_f // 16, out_f // 16, 16 * K)
        # Baseline all-zero
        p0 = torch.zeros(cshape, dtype=torch.int16, device=device)
        w0 = op(p0, in_f, out_f, K, True, False).detach().cpu().numpy().astype(np.float32)
        probe_results["baseline_w0_shape"] = w0.shape
        probe_results["baseline_w0_norm"] = float(np.linalg.norm(w0))
        probe_results["baseline_w0_sample"] = w0[:2, :8].tolist()

        # Perturbation test: change ONE code entry at a specific position
        # and record the delta pattern.
        deltas = {}
        for pos in [
            (0, 0, 0),      # first K-slice, first col
            (0, 0, 15),     # first K-slice, last col
            (0, 0, 16),     # second K-slice, first col
            (0, 0, 31),     # second K-slice, last col
            (0, 1, 0),      # different column-tile
            (1, 0, 0),      # different row-tile
        ]:
            p = torch.zeros(cshape, dtype=torch.int16, device=device)
            p[pos] = 1
            w = op(p, in_f, out_f, K, True, False).detach().cpu().numpy().astype(np.float32)
            delta = w - w0
            nz_rows = np.where(np.any(np.abs(delta) > 1e-6, axis=1))[0].tolist()
            nz_cols = np.where(np.any(np.abs(delta) > 1e-6, axis=0))[0].tolist()
            deltas[f"code[{pos}]=1"] = {
                "nz_rows": nz_rows[:20],
                "n_nz_rows": len(nz_rows),
                "nz_cols": nz_cols[:20],
                "n_nz_cols": len(nz_cols),
                "delta_norm": float(np.linalg.norm(delta)),
                "delta_first_row_first16": delta[nz_rows[0], nz_cols[:16]].tolist() if nz_rows and nz_cols else [],
            }
        probe_results["single_perturbation"] = deltas

        # Sweep the same position (0,0,0) with different values and check linearity
        vals_test = [1, 2, 3, 5, 10, 100, 1000, 32767, -32768, -1]
        val_deltas = {}
        for v in vals_test:
            p = torch.zeros(cshape, dtype=torch.int16, device=device)
            p[0, 0, 0] = v
            w = op(p, in_f, out_f, K, True, False).detach().cpu().numpy().astype(np.float32)
            d = w - w0
            nz = np.where(np.any(np.abs(d) > 1e-6, axis=1))[0].tolist()
            val_deltas[str(v)] = {
                "n_nz_rows": len(nz),
                "delta_norm": float(np.linalg.norm(d)),
                "affected_rows": nz[:10],
                "sample_row_vals": d[nz[0], :16].tolist() if nz else [],
            }
        probe_results["value_sweep_pos_0_0_0"] = val_deltas

        # Sanity: value 1 at (0,0,0), (0,0,1), ..., (0,0,15) — are they all
        # affecting the same 16-col slice? Are the outputs linearly combining?
        col_pattern = {}
        for c in range(16):
            p = torch.zeros(cshape, dtype=torch.int16, device=device)
            p[0, 0, c] = 1
            w = op(p, in_f, out_f, K, True, False).detach().cpu().numpy().astype(np.float32)
            d = w - w0
            col_pattern[c] = {
                "affected_cols": np.where(np.any(np.abs(d) > 1e-6, axis=0))[0].tolist(),
                "affected_rows": np.where(np.any(np.abs(d) > 1e-6, axis=1))[0].tolist(),
                "delta_norm": float(np.linalg.norm(d)),
            }
        probe_results["column_scan_K0"] = col_pattern

    out["probe_results"] = probe_results

    # ---- (4) Real-weight reference dump ----
    # Load actual gate_up_proj expert 0 code from safetensors (uploaded from Mac)
    import io as _io
    gu_data = np.load(_io.BytesIO(weight_bytes_gu), allow_pickle=True)
    dn_data = np.load(_io.BytesIO(weight_bytes_dn), allow_pickle=True)

    def _decode_and_check(name, data, in_f, out_f, K):
        code = torch.from_numpy(data["code"]).contiguous().cuda()
        rin = torch.from_numpy(data["rin"]).to(torch.float16).cuda()
        rout = torch.from_numpy(data["rout"]).to(torch.float16).cuda()

        print(f"[{name}] code={tuple(code.shape)} dtype={code.dtype} "
              f"rin={tuple(rin.shape)} rout={tuple(rout.shape)}")
        w_bare = op(code, in_f, out_f, K, True, False).detach().cpu().numpy()
        print(f"[{name}] w_bare={w_bare.shape} dtype={w_bare.dtype} "
              f"norm={float(np.linalg.norm(w_bare)):.3e}")

        # Small canonical x
        rng = np.random.default_rng(seed=42)
        x_np = rng.standard_normal((4, in_f)).astype(np.float32)
        x = torch.from_numpy(x_np).to(torch.float16).cuda()

        # Apply the escha pipeline: y = t128(x*rin) @ w_bare; y = t128(y)*rout
        from escha.transform import escha_t128
        xh = escha_t128(x, pre_scale=rin)
        y_mid = xh.float() @ torch.from_numpy(w_bare).float().cuda()
        y_mid16 = y_mid.to(torch.float16)
        y_post = escha_t128(y_mid16, post_scale=rout)

        return {
            "code": code.cpu().numpy(),
            "rin": rin.cpu().numpy(),
            "rout": rout.cpu().numpy(),
            "w_bare": w_bare,
            "x": x_np,
            "xh": xh.cpu().numpy().astype(np.float32),
            "y_mid": y_mid.cpu().numpy().astype(np.float32),
            "y_post": y_post.cpu().numpy().astype(np.float32),
            "in_f": in_f, "out_f": out_f, "K": K,
        }

    if op is not None:
        out["gate_up_ref"] = _decode_and_check("gate_up L0 E0", gu_data, 2048, 1024, 2)
        out["down_ref"] = _decode_and_check("down L0 E0", dn_data, 512, 2048, 3)

    return pickle.dumps(out)


@app.local_entrypoint()
def main() -> None:
    import numpy as np
    import io
    from safetensors import safe_open
    import json
    from pathlib import Path

    MODEL_DIR = Path("/Users/kaede/models/Qwen3.6-35B-A3B-Escha-W2")
    idx = json.load(open(MODEL_DIR / "model.safetensors.index.json"))
    wm = idx["weight_map"]

    def load_expert0(parent):
        """Load code[0], rin[0], rout[0] for a given parent (expert 0)."""
        result = {}
        for suf in ("code", "rin", "rout"):
            key = f"{parent}.escha_{suf}"
            shard = wm[key]
            with safe_open(MODEL_DIR / shard, framework="numpy") as f:
                t = f.get_tensor(key)
                result[suf] = t[0]  # expert 0
        buf = io.BytesIO()
        np.savez(buf, **result)
        return buf.getvalue()

    gu = load_expert0("model.language_model.layers.0.mlp.experts.gate_up_proj")
    dn = load_expert0("model.language_model.layers.0.mlp.experts.down_proj")
    print(f"[local] uploading gu={len(gu)/1e6:.1f}MB dn={len(dn)/1e6:.1f}MB")

    payload = deep_probe.remote(gu, dn)
    out = Path("/Users/kaede/mlx-video/mlx_video/models/qwen3_5_moe_escha/codebooks/escha_reference.pkl")
    out.write_bytes(payload)
    print(f"[local] SUCCESS. Wrote {out} ({len(payload)/1e6:.2f} MB)")
