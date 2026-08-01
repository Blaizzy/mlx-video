"""Convert original fp32 safetensors to bf16, keep original prefixes."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file as torch_save_file


COPY_EXTS = {
    "config.json",
    "chat_template.jinja",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "LICENSE",
    "ATTRIBUTIONS.md",
    "README.md",
}

VARIANTS = {
    os.path.expanduser("~/models/VisionPsy-Nano-460M"):
        os.path.expanduser("~/models/VisionPsy-Nano-460M-MLX-bf16"),
    os.path.expanduser("~/models/VisionPsy-Nano-460M-Flash"):
        os.path.expanduser("~/models/VisionPsy-Nano-460M-Flash-MLX-bf16"),
}


def convert_one(src_dir: Path, dst_dir: Path) -> None:
    print(f"\n=== {src_dir.name} -> {dst_dir.name} ===")
    dst_dir.mkdir(parents=True, exist_ok=True)

    src_sft = src_dir / "model.safetensors"
    dst_sft = dst_dir / "model.safetensors"

    weights: dict[str, torch.Tensor] = {}
    with safe_open(str(src_sft), framework="pt") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            if t.is_floating_point():
                t = t.to(torch.bfloat16)
            weights[k] = t.contiguous()
    torch_save_file(weights, str(dst_sft))

    src_size = src_sft.stat().st_size
    dst_size = dst_sft.stat().st_size
    print(f"  weights {src_size / 1e9:.3f} GB -> {dst_size / 1e9:.3f} GB")

    for name in COPY_EXTS:
        src_f = src_dir / name
        if src_f.exists():
            shutil.copy2(src_f, dst_dir / name)

    for py in src_dir.glob("*.py"):
        shutil.copy2(py, dst_dir / py.name)
    if (src_dir / "assets").exists():
        shutil.copytree(src_dir / "assets", dst_dir / "assets", dirs_exist_ok=True)


def main() -> None:
    for src, dst in VARIANTS.items():
        convert_one(Path(src), Path(dst))
    print("\nDone.")


if __name__ == "__main__":
    main()
