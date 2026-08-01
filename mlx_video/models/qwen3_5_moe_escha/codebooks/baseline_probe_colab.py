"""baseline_probe_colab.py — single-cell Escha "w0 baseline" probe.

Purpose
-------
Escha's `torch.ops.escha.escham_reconstruct(code, in_f, out_f, K, cbA, mul1)`
op is EXACTLY linear in `code`, but adds a shape-dependent additive bias
`w0 = op(all_zeros_code)` that must be subtracted to isolate the codebook
contribution. See `docs/escha_op_signature.md` §4 (linearity) and Escha
port status doc §11 for context.

This script captures `w0` for every (in_f, out_f, K) tile-shape used by
Escha-W2's MoE experts, packages them into a single `baseline_v2.npz`,
and prints the file's base64 encoding so the caller (Chrome-MCP driven
Colab session) can read it off the page and paste back into the MLX port.

Shapes probed (from `docs/escha_w2_tensor_enum.md` + escha_config[9]):
  1. gate_up_proj:  in_f=2048, out_f=1024, K=2   (all 40 layers × 256 experts)
  2. down_proj:     in_f=512,  out_f=2048, K=3   (all 40 layers × 256 experts)

We ALSO probe a few smaller shapes for sanity-cross-check against numbers in
`docs/escha_op_signature.md` §2 (minimal 128×128 tiles for K=2 and K=3).

Usage: paste this whole file into a single Colab cell, hit Shift+Enter.
Runs ~30 sec on a T4. Final `stdout` line is the base64-encoded npz.
"""

# ==== Cell start =============================================================
# 1) Install pinned torch + hf tooling (CUDA 12.8 wheels).
import os, sys, subprocess, base64, io, time, json

def _run(cmd: str, check: bool = True) -> None:
    print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, check=False)
    if check and r.returncode != 0:
        raise SystemExit(f"command failed: {cmd}")

_run("pip install -q torch==2.9.* --index-url https://download.pytorch.org/whl/cu128")
_run("pip install -q numpy safetensors 'huggingface_hub[cli]' hf_transfer")

# 2) Pull the escha wheel + install (no-deps to keep environment clean).
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
_run("mkdir -p /escha")
_run("hf download EschaLabs/escha-runtime-qwen3moe --include 'sglang/*' --local-dir /escha")
_run("pip install -q --no-deps /escha/sglang/escha-*.whl")

# 3) Import + smoke check.
import numpy as np
import torch
import escha  # noqa: F401  (registers torch.ops.escha.*)

assert torch.cuda.is_available(), "CUDA not available — Colab runtime must be GPU."
op = torch.ops.escha.escham_reconstruct
device = "cuda"

# 4) Shapes to probe. Order matters only for readability.
#    (name, in_f, out_f, K, cshape)
SHAPES = [
    # ---- primary Escha-W2 MoE shapes (the two we actually need) ----
    ("gate_up_proj", 2048, 1024, 2, (128, 64, 32)),
    ("down_proj",     512, 2048, 3, ( 32,128, 48)),
    # ---- sanity-check tiles (compare against docs/escha_op_signature.md §2) ----
    ("min_K2_128x128", 128, 128, 2, ( 8,  8, 32)),
    ("min_K3_128x128", 128, 128, 3, ( 8,  8, 48)),
]

# 5) Run baseline probe (all-zeros code) for each shape.
baselines: dict[str, np.ndarray] = {}
meta: dict[str, dict] = {}
t0 = time.time()
for name, in_f, out_f, K, cshape in SHAPES:
    p0 = torch.zeros(cshape, dtype=torch.int16, device=device)
    tic = time.time()
    w0 = op(p0, in_f, out_f, K, True, False)
    torch.cuda.synchronize()
    dt = time.time() - tic
    w0_np = w0.detach().cpu().numpy().astype(np.float16)  # match op output dtype
    key = f"{name}__in{in_f}_out{out_f}_K{K}"
    baselines[key] = w0_np
    meta[key] = {
        "in_f": in_f, "out_f": out_f, "K": K,
        "cshape": list(cshape),
        "out_shape": list(w0_np.shape),
        "out_dtype": str(w0_np.dtype),
        "l2_norm": float(np.linalg.norm(w0_np.astype(np.float32))),
        "abs_max": float(np.abs(w0_np.astype(np.float32)).max()),
        "nnz_frac": float((w0_np != 0).mean()),
        "op_seconds": dt,
    }
    print(
        f"  {key}: shape={w0_np.shape} "
        f"|w0|_2={meta[key]['l2_norm']:.3e} "
        f"|w0|_inf={meta[key]['abs_max']:.3e} "
        f"nnz_frac={meta[key]['nnz_frac']:.3f} "
        f"({dt*1000:.1f} ms)",
        flush=True,
    )
print(f"total probe time: {time.time()-t0:.2f}s", flush=True)

# 6) Save npz + meta.json to /content and print base64.
out_dir = "/content/escha_baseline_v2"
os.makedirs(out_dir, exist_ok=True)
npz_path = f"{out_dir}/baseline_v2.npz"
meta_path = f"{out_dir}/baseline_v2.meta.json"

# Store meta as an extra "meta_json" entry inside the npz so a single file suffices.
save_kwargs = dict(baselines)
save_kwargs["_meta_json"] = np.frombuffer(
    json.dumps({"shapes": meta, "wheel": "EschaLabs/escha-runtime-qwen3moe",
                "torch": torch.__version__}, indent=2).encode("utf-8"),
    dtype=np.uint8,
)
np.savez_compressed(npz_path, **save_kwargs)
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)

size = os.path.getsize(npz_path)
print(f"\nsaved: {npz_path} ({size/1024:.1f} KiB)")
print(f"saved: {meta_path}")

# 7) Emit base64 blob so a page-scraper can copy it back verbatim.
#    We frame with unmistakable BEGIN/END markers.
with open(npz_path, "rb") as f:
    blob = f.read()
b64 = base64.b64encode(blob).decode("ascii")

print("\n=========== BASELINE_V2_NPZ_BASE64_BEGIN ===========")
# Print in fixed-width chunks so it wraps nicely and is easy to reassemble.
CHUNK = 76
for i in range(0, len(b64), CHUNK):
    print(b64[i:i+CHUNK])
print("=========== BASELINE_V2_NPZ_BASE64_END =============")
print(f"base64 length: {len(b64)} chars ({size} bytes)")
# ==== Cell end ===============================================================
