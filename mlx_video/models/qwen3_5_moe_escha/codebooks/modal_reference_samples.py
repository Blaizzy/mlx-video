"""modal_reference_samples.py — dump (code, op(code)) sample pairs for verification.

Grabs a handful of real MoE expert code tensors from the shipped Escha-W2
model, runs `torch.ops.escha.escham_reconstruct` on them, and pickles both
inputs and outputs into `reference_dump.pkl` (with a `samples` list) so
`wire_baseline_v2.py::verify()` can compare the MLX port's output.

Usage:
    cd ~/mlx-video && ~/.venv/bin/modal run \
        mlx_video/models/qwen3_5_moe_escha/codebooks/modal_reference_samples.py
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
    .pip_install("numpy", "safetensors", "huggingface_hub[cli]", "hf_transfer")
    .run_commands(
        f"echo escha wheel revision: {WHEEL_REVISION}",
        "mkdir -p /escha",
        "hf download EschaLabs/escha-runtime-qwen3moe --include 'sglang/*' --local-dir /escha",
        "pip install --no-deps /escha/sglang/escha-*.whl",
    )
)

vol = modal.Volume.from_name("escha-codebooks", create_if_missing=True)
app = modal.App("escha-reference-samples", image=image)


@app.function(gpu="A10G", timeout=1800, memory=32 * 1024, volumes={"/vol": vol})
def collect() -> bytes:
    """Load real code tensors, run op, pickle samples."""
    import json
    import os

    import numpy as np
    import torch
    from safetensors import safe_open
    from huggingface_hub import snapshot_download
    import escha  # noqa: F401

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    op = torch.ops.escha.escham_reconstruct

    print("[remote] downloading model snapshot...", flush=True)
    model_dir = snapshot_download(
        "EschaLabs/Qwen3.6-35B-A3B-Escha-W2",
        cache_dir="/vol/hf_cache",
        allow_patterns=[
            "config.json",
            "model.safetensors.index.json",
            # Only need first-layer shards for MoE expert weights.
            "model-00001-of-*.safetensors",
            "model-00002-of-*.safetensors",
        ],
    )
    idx = json.load(open(f"{model_dir}/model.safetensors.index.json"))
    wm = idx["weight_map"]

    def _load_expert(name: str, expert: int):
        shard = wm[name]
        with safe_open(f"{model_dir}/{shard}", framework="pt") as f:
            return f.get_tensor(name)[expert]

    # Take a mix: 5 experts of gate_up (K=2) and 5 of down (K=3) from layer 0.
    samples = []
    for tag, pfx, in_f, out_f, K in [
        ("gate_up", "model.language_model.layers.0.mlp.experts.gate_up_proj", 2048, 1024, 2),
        ("down",    "model.language_model.layers.0.mlp.experts.down_proj",    512,  2048, 3),
    ]:
        for expert in [0, 1, 7, 32, 128]:
            try:
                code = _load_expert(f"{pfx}.escha_code", expert=expert)
            except Exception as e:
                print(f"[remote] skipping {tag} expert={expert}: {e}", flush=True)
                continue
            code_gpu = code.to("cuda")
            w = op(code_gpu, in_f, out_f, K, True, False)
            torch.cuda.synchronize()
            code_np = code.cpu().numpy()
            w_np = w.detach().cpu().numpy().astype(np.float16)
            samples.append({
                "tag": tag,
                "expert": expert,
                "in_f": in_f,
                "out_f": out_f,
                "K": K,
                "code": code_np,   # int16 (bi, bj, k)
                "w": w_np,         # fp16 (in_f, out_f) — op(code) reference
            })
            print(
                f"[remote] {tag} expert={expert}: code={code_np.shape}{code_np.dtype} "
                f"w={w_np.shape} |w|_inf={float(np.abs(w_np).max()):.3e}",
                flush=True,
            )

    dump = {"samples": samples, "wheel_revision": WHEEL_REVISION,
            "torch": torch.__version__}
    blob = pickle.dumps(dump, protocol=pickle.HIGHEST_PROTOCOL)
    os.makedirs("/vol/reference_dump", exist_ok=True)
    with open("/vol/reference_dump/reference_dump.pkl", "wb") as f:
        f.write(blob)
    vol.commit()
    print(f"[remote] pickled {len(samples)} samples ({len(blob)/1024:.1f} KiB)", flush=True)
    return blob


@app.local_entrypoint()
def main() -> None:
    print("[local] collecting reference samples on A10G ...", flush=True)
    blob = collect.remote()
    out = Path(
        "/Users/kaede/mlx-video/mlx_video/models/qwen3_5_moe_escha/codebooks/"
        "reference_dump.pkl"
    )
    # existing file lacks samples[] format; overwrite.
    out.write_bytes(blob)
    print(f"[local] wrote {out} ({len(blob)/1024:.1f} KiB)")
