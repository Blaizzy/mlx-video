"""modal_baseline_v2.py — Modal wrapper for the w0 baseline probe.

Wraps the same probe logic as `baseline_probe_colab.py` but on Modal's A10G
(sm_86 satisfies escha's sm_80+ requirement).

Runs `op(all_zeros_code)` for each (in_f, out_f, K) tile-shape used by
Escha-W2's MoE experts, packages the outputs into `baseline_v2.npz`, and
saves it both to the Modal volume and returns it as bytes so the local
entrypoint can materialize it into
    codebooks/baseline_v2.npz

Usage:
    cd ~/mlx-video && ~/.venv/bin/modal run \
        mlx_video/models/qwen3_5_moe_escha/codebooks/modal_baseline_v2.py
"""

from __future__ import annotations

import base64
import time
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
    .run_commands(
        f"echo escha wheel revision: {WHEEL_REVISION}",
        "mkdir -p /escha",
        "hf download EschaLabs/escha-runtime-qwen3moe --include 'sglang/*' --local-dir /escha",
        "pip install --no-deps /escha/sglang/escha-*.whl",
    )
)

vol = modal.Volume.from_name("escha-codebooks", create_if_missing=True)
app = modal.App("escha-baseline-v2", image=image)


# Shapes mirror `baseline_probe_colab.py`.
SHAPES = [
    # primary Escha-W2 MoE shapes.
    ("gate_up_proj", 2048, 1024, 2, (128, 64, 32)),
    ("down_proj",     512, 2048, 3, ( 32, 128, 48)),
    # sanity-check tiles.
    ("min_K2_128x128", 128, 128, 2, ( 8,  8, 32)),
    ("min_K3_128x128", 128, 128, 3, ( 8,  8, 48)),
]


@app.function(gpu="A10G", timeout=900, memory=16 * 1024, volumes={"/vol": vol})
def probe() -> bytes:
    """Run baseline probes, save npz to volume, return npz bytes."""
    import json
    import os

    import numpy as np
    import torch
    import escha  # noqa: F401

    assert torch.cuda.is_available(), "CUDA missing on remote container."
    op = torch.ops.escha.escham_reconstruct
    device = "cuda"

    baselines: dict[str, np.ndarray] = {}
    meta: dict[str, dict] = {}
    t0 = time.time()
    for name, in_f, out_f, K, cshape in SHAPES:
        p0 = torch.zeros(cshape, dtype=torch.int16, device=device)
        tic = time.time()
        w0 = op(p0, in_f, out_f, K, True, False)
        torch.cuda.synchronize()
        dt = time.time() - tic
        w0_np = w0.detach().cpu().numpy().astype(np.float16)
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

    # Bundle meta into the npz.
    save_kwargs = dict(baselines)
    save_kwargs["_meta_json"] = np.frombuffer(
        json.dumps(
            {
                "shapes": meta,
                "wheel": "EschaLabs/escha-runtime-qwen3moe",
                "wheel_revision": WHEEL_REVISION,
                "torch": torch.__version__,
            },
            indent=2,
        ).encode("utf-8"),
        dtype=np.uint8,
    )

    os.makedirs("/vol/baseline_v2", exist_ok=True)
    npz_path = "/vol/baseline_v2/baseline_v2.npz"
    np.savez_compressed(npz_path, **save_kwargs)
    size = os.path.getsize(npz_path)
    print(f"saved: {npz_path} ({size/1024:.1f} KiB)", flush=True)

    with open(npz_path, "rb") as f:
        blob = f.read()
    vol.commit()
    return blob


@app.local_entrypoint()
def main() -> None:
    print("[local] launching probe on A10G ...", flush=True)
    blob = probe.remote()
    out_dir = Path("/Users/kaede/mlx-video/mlx_video/models/qwen3_5_moe_escha/codebooks")
    target = out_dir / "baseline_v2.npz"
    target.write_bytes(blob)
    print(f"[local] wrote {target} ({len(blob)/1024:.1f} KiB)")

    # Also emit a base64 shadow for archival + debug re-use.
    b64_target = out_dir / "baseline_v2.b64.txt"
    b64 = base64.b64encode(blob).decode("ascii")
    with open(b64_target, "w") as f:
        f.write("=========== BASELINE_V2_NPZ_BASE64_BEGIN ===========\n")
        CHUNK = 76
        for i in range(0, len(b64), CHUNK):
            f.write(b64[i:i + CHUNK] + "\n")
        f.write("=========== BASELINE_V2_NPZ_BASE64_END =============\n")
    print(f"[local] wrote {b64_target} ({len(b64)} b64 chars)")
