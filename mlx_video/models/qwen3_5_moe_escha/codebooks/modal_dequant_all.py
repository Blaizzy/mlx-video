"""Option B — dense dequantization of every Escha-W2 MoE expert.

For every layer L in [0..39] and every expert E in [0..255], compute the
DENSE bf16 matrix M such that

    escha_expert_forward(x)  ==  x @ M

by pushing the identity through the escha forward pipeline

    y = t128( t128(x, pre=rin) @ w_bare, post=rout )

where w_bare = torch.ops.escha.escham_reconstruct(code, in_f, out_f, K, True, False).

Everything upstream is linear (H128 orthogonal, diag(rin) diag, w_bare a
matmul), so pushing x = I yields M in a single call. We do this per-expert
so the Modal A10G's 24 GB VRAM is plenty (each expert is <10 MB).

Output: one safetensors file per layer at
    /vol/dequant_v1/layer_{L:02d}.safetensors
with keys
    layer_{L}.expert_{E}.gate_up_proj.weight   (bf16, shape (out_f, in_f))
    layer_{L}.expert_{E}.down_proj.weight      (bf16, shape (out_f, in_f))
Weights are stored transposed (out, in) to match torch/MLX Linear convention.

Total size estimate:
  gate_up  : 256 * 40 * (2048 * 1024) * 2 =  ~40 GB
  down     : 256 * 40 * (512  * 2048) * 2 =  ~20 GB
  ~60 GB total; split across 40 shard files (~1.5 GB each).
"""

from __future__ import annotations

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
    .pip_install("numpy", "safetensors", "huggingface_hub[cli]", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_commands(
        f"echo escha wheel revision: {WHEEL_REVISION}",
        "mkdir -p /escha",
        "hf download EschaLabs/escha-runtime-qwen3moe --include 'sglang/*' --local-dir /escha",
        "ls -la /escha/sglang/",
        # --no-deps: escha wheel declares sglang as a dep but we don't need
        # sglang — only escha._C linked against torch/cuda already installed.
        "pip install --no-deps /escha/sglang/escha-*.whl",
        (
            "python -c '"
            "import torch, escha; "
            "print(\"escha OK, ops:\", "
            "[o for o in dir(torch.ops.escha) if not o.startswith(\"_\")])"
            "'"
        ),
    )
)

vol = modal.Volume.from_name("escha-w2-dequant", create_if_missing=True)
app = modal.App("escha-w2-dequant", image=image)


NUM_LAYERS = 40
NUM_EXPERTS = 256
GATE_UP = dict(in_f=2048, out_f=1024, K=2)
DOWN = dict(in_f=512, out_f=2048, K=3)


# ---------------------------------------------------------------------------
# Phase 0 discovery — verifies API + identity trick on 1 expert.
# ---------------------------------------------------------------------------
@app.function(
    gpu="A10G",
    timeout=1800,
    memory=32 * 1024,
    volumes={"/vol": vol},
)
def discover_and_test() -> dict:
    """Sanity-check the API and identity trick on layer 0 / expert 0."""
    import inspect
    import json
    import numpy as np
    import torch
    import escha  # noqa: F401
    from safetensors import safe_open
    from huggingface_hub import snapshot_download

    print(f"[modal] torch {torch.__version__} cuda={torch.cuda.is_available()}")
    print(f"[modal] escha module: {escha.__file__}")

    # Enumerate escha._C and torch.ops.escha for a full picture.
    print("\n=== escha._C ===")
    C = escha._C
    for name in sorted(dir(C)):
        if name.startswith("_"):
            continue
        obj = getattr(C, name)
        try:
            sig = str(inspect.signature(obj))
        except Exception:
            sig = "?"
        print(f"  {name}: {sig}")

    print("\n=== torch.ops.escha ===")
    ops = torch.ops.escha
    print(f"  dir: {[n for n in dir(ops) if not n.startswith('_')]}")

    # escham_reconstruct is exposed as torch.ops.escha.escham_reconstruct
    op = torch.ops.escha.escham_reconstruct
    print(f"  op: {op}")

    # Check for the t128 transform helper.
    print("\n=== escha.transform ===")
    try:
        from escha import transform as et
        print(f"  file: {et.__file__}")
        print(f"  attrs: {[a for a in dir(et) if not a.startswith('_')]}")
        try:
            print(f"  escha_t128 sig: {inspect.signature(et.escha_t128)}")
        except Exception as e:
            print(f"  escha_t128 sig err: {e}")
    except Exception as e:
        print(f"  import failed: {e}")

    # Download the model to /vol so we don't re-download between phases.
    print("\n=== snapshot_download ===")
    model_dir = snapshot_download(
        "EschaLabs/Qwen3.6-35B-A3B-Escha-W2",
        cache_dir="/vol/hf_cache",
    )
    print(f"  model_dir: {model_dir}")

    # Load layer 0 expert 0 gate_up code/rin/rout.
    idx = json.load(open(f"{model_dir}/model.safetensors.index.json"))
    wm = idx["weight_map"]

    def load_e(prefix, expert):
        out = {}
        for suf in ("code", "rin", "rout"):
            key = f"{prefix}.escha_{suf}"
            shard = wm[key]
            with safe_open(f"{model_dir}/{shard}", framework="pt") as f:
                t = f.get_tensor(key)[expert]
                out[suf] = t.cuda()
        return out

    gu = load_e("model.language_model.layers.0.mlp.experts.gate_up_proj", 0)
    print(f"\n=== L0/E0 gate_up ===")
    for k, v in gu.items():
        print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")

    # Run bare escham_reconstruct.
    print("\n=== escham_reconstruct(code) ===")
    w_bare = op(gu["code"], GATE_UP["in_f"], GATE_UP["out_f"], GATE_UP["K"], True, False)
    print(f"  w_bare: shape={tuple(w_bare.shape)} dtype={w_bare.dtype}")
    print(f"  w_bare norm: {float(w_bare.float().norm()):.3e}")
    print(f"  w_bare sample [:2, :4]: {w_bare[:2, :4].float().cpu().numpy()}")

    # Identity trick: M = t128(t128(I, pre=rin) @ w_bare, post=rout)
    print("\n=== identity trick ===")
    from escha.transform import escha_t128

    in_f, out_f = GATE_UP["in_f"], GATE_UP["out_f"]
    I = torch.eye(in_f, dtype=torch.float16, device="cuda")
    xh = escha_t128(I, pre_scale=gu["rin"].to(torch.float16))
    print(f"  xh shape: {xh.shape}, dtype {xh.dtype}, norm {float(xh.float().norm()):.3e}")
    y_mid = xh @ w_bare
    print(f"  y_mid shape: {y_mid.shape}, dtype {y_mid.dtype}, norm {float(y_mid.float().norm()):.3e}")
    M = escha_t128(y_mid, post_scale=gu["rout"].to(torch.float16))
    print(f"  M shape: {M.shape}, dtype {M.dtype}, norm {float(M.float().norm()):.3e}")
    print(f"  M[:2, :4]: {M[:2, :4].float().cpu().numpy()}")

    # Cross-check by pushing a random x through the full pipeline and comparing
    # against x @ M.
    torch.manual_seed(42)
    x = torch.randn(4, in_f, dtype=torch.float16, device="cuda")
    y_via_pipeline = escha_t128(
        escha_t128(x, pre_scale=gu["rin"].to(torch.float16)) @ w_bare,
        post_scale=gu["rout"].to(torch.float16),
    )
    y_via_M = x @ M
    diff = (y_via_pipeline.float() - y_via_M.float()).abs()
    print(f"\n=== cross-check ===")
    print(f"  pipeline y norm: {float(y_via_pipeline.float().norm()):.3e}")
    print(f"  x@M      y norm: {float(y_via_M.float().norm()):.3e}")
    print(f"  diff max: {float(diff.max()):.3e}   diff mean: {float(diff.mean()):.3e}")

    # Also test batched identity (do it in chunks for larger experts).
    print(f"\n=== chunked identity (chunks of 256) ===")
    chunks = []
    chunk = 256
    for i in range(0, in_f, chunk):
        Ic = torch.zeros(chunk, in_f, dtype=torch.float16, device="cuda")
        for j in range(chunk):
            if i + j < in_f:
                Ic[j, i + j] = 1.0
        xh = escha_t128(Ic, pre_scale=gu["rin"].to(torch.float16))
        y_mid = xh @ w_bare
        M_chunk = escha_t128(y_mid, post_scale=gu["rout"].to(torch.float16))
        chunks.append(M_chunk[:min(chunk, in_f - i)])
    M_chunked = torch.cat(chunks, dim=0)
    print(f"  M_chunked shape: {M_chunked.shape}")
    print(f"  chunk vs full diff max: {float((M_chunked.float() - M.float()).abs().max()):.3e}")

    vol.commit()

    return {
        "M_shape": tuple(M.shape),
        "M_dtype": str(M.dtype),
        "pipeline_vs_M_max_diff": float(diff.max()),
        "chunk_vs_full_max_diff": float((M_chunked.float() - M.float()).abs().max()),
    }


# ---------------------------------------------------------------------------
# Phase 1 — full dequant.
# ---------------------------------------------------------------------------
@app.function(
    gpu="A10G",
    timeout=3600 * 3,   # 3 h ceiling; the actual work is ~60 min
    memory=32 * 1024,
    volumes={"/vol": vol},
)
def dequant_all(layer_start: int = 0, layer_end: int = NUM_LAYERS) -> dict:
    """Sweep layers [layer_start, layer_end), write /vol/dequant_v1/layer_XX.safetensors."""
    import json
    import os
    import time
    import torch
    import escha  # noqa: F401
    from safetensors import safe_open
    from safetensors.torch import save_file
    from escha.transform import escha_t128
    from huggingface_hub import snapshot_download

    op = torch.ops.escha.escham_reconstruct
    device = "cuda"

    # Model was pre-downloaded in the discovery pass. If missing, download now.
    print("=== snapshot_download (cached) ===")
    model_dir = snapshot_download(
        "EschaLabs/Qwen3.6-35B-A3B-Escha-W2",
        cache_dir="/vol/hf_cache",
    )
    print(f"  model_dir: {model_dir}")

    out_dir = "/vol/dequant_v1"
    os.makedirs(out_dir, exist_ok=True)

    idx = json.load(open(f"{model_dir}/model.safetensors.index.json"))
    wm = idx["weight_map"]

    # Open each shard exactly once and cache handles.
    open_shards: dict = {}

    def get_shard(name: str):
        shard = wm[name]
        if shard not in open_shards:
            open_shards[shard] = safe_open(f"{model_dir}/{shard}", framework="pt")
        return open_shards[shard]

    def compute_M_layer_proj(prefix: str, in_f: int, out_f: int, K: int) -> dict:
        """Return {expert_id (int) : M (torch.Tensor bf16 on cpu)} for one proj."""
        # Load the WHOLE (256, ...) batched tensor for each of code/rin/rout.
        # Slice per-expert to feed the op.
        t_code = get_shard(f"{prefix}.escha_code").get_tensor(f"{prefix}.escha_code").to(device)
        t_rin  = get_shard(f"{prefix}.escha_rin").get_tensor(f"{prefix}.escha_rin").to(device).to(torch.float16)
        t_rout = get_shard(f"{prefix}.escha_rout").get_tensor(f"{prefix}.escha_rout").to(device).to(torch.float16)
        assert t_code.shape[0] == NUM_EXPERTS, f"expected 256 experts, got {t_code.shape}"

        out: dict = {}
        # Chunk the identity in slabs of ROWS at a time; each expert is a
        # separate M matrix. For gate_up (in_f=2048) we can do the full I
        # in one shot; for down (in_f=512) definitely.
        chunk = min(in_f, 1024)
        for e in range(NUM_EXPERTS):
            code = t_code[e]
            rin = t_rin[e]
            rout = t_rout[e]
            w_bare = op(code, in_f, out_f, K, True, False)   # (in_f, out_f) fp16
            # Identity trick with row-chunking so we never blow up VRAM.
            chunks = []
            for i in range(0, in_f, chunk):
                rows = min(chunk, in_f - i)
                # Build a slab of the identity: I_slab[j, i+j] = 1 for j in [0,rows)
                Ic = torch.zeros(rows, in_f, dtype=torch.float16, device=device)
                # Fast diag fill:
                cols = torch.arange(i, i + rows, device=device)
                Ic[torch.arange(rows, device=device), cols] = 1.0
                xh = escha_t128(Ic, pre_scale=rin)
                y_mid = xh @ w_bare
                M_chunk = escha_t128(y_mid, post_scale=rout)   # (rows, out_f) fp16
                chunks.append(M_chunk)
            M = torch.cat(chunks, dim=0)                       # (in_f, out_f) fp16
            # Store as (out, in) bf16 for MLX/PT Linear convention.
            out[e] = M.T.contiguous().to(torch.bfloat16).cpu()
            del w_bare, chunks, M
        del t_code, t_rin, t_rout
        torch.cuda.empty_cache()
        return out

    t_run_start = time.time()
    for L in range(layer_start, layer_end):
        target = f"{out_dir}/layer_{L:02d}.safetensors"
        if os.path.exists(target):
            sz = os.path.getsize(target)
            if sz > 500 * 1024 * 1024:   # >500 MB → already done
                print(f"[L{L:02d}] SKIP (exists, {sz/1e9:.2f} GB)")
                continue
            print(f"[L{L:02d}] partial ({sz/1e6:.0f} MB), redoing")

        t0 = time.time()
        # Both projections share the same MoE prefix.
        pfx = f"model.language_model.layers.{L}.mlp.experts"
        print(f"[L{L:02d}] gate_up ...", flush=True)
        gu_map = compute_M_layer_proj(f"{pfx}.gate_up_proj", **GATE_UP)
        print(f"[L{L:02d}] down    ...", flush=True)
        dn_map = compute_M_layer_proj(f"{pfx}.down_proj", **DOWN)

        payload: dict = {}
        for e in range(NUM_EXPERTS):
            payload[f"layer_{L}.expert_{e}.gate_up_proj.weight"] = gu_map[e]
            payload[f"layer_{L}.expert_{e}.down_proj.weight"] = dn_map[e]
        save_file(payload, target)
        vol.commit()
        dt = time.time() - t0
        total = time.time() - t_run_start
        sz = os.path.getsize(target) / 1e9
        print(
            f"[L{L:02d}] DONE {sz:.2f} GB in {dt:.0f}s  "
            f"(cumulative {total:.0f}s = {total/60:.1f}min)", flush=True,
        )

    return {"layers_done": layer_end - layer_start}


@app.local_entrypoint()
def main(mode: str = "discover", layer_start: int = 0, layer_end: int = NUM_LAYERS) -> None:
    """Two modes:
       modal run modal_dequant_all.py --mode discover           # 1 expert sanity check
       modal run modal_dequant_all.py --mode dequant            # full sweep
       modal run modal_dequant_all.py --mode dequant --layer-start 20 --layer-end 40
    """
    if mode == "discover":
        result = discover_and_test.remote()
        print(f"\n[local] discover result: {result}")
    elif mode == "dequant":
        result = dequant_all.remote(layer_start=layer_start, layer_end=layer_end)
        print(f"\n[local] dequant result: {result}")
    else:
        raise SystemExit(f"unknown mode {mode!r} (expected 'discover' or 'dequant')")
