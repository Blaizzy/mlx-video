"""Inference runtime profiles (eager vs torch.compile + CUDA graphs)."""
from __future__ import annotations

import torch


def apply_eager_profile(model) -> None:
    for cfg in (getattr(model, "cfg", None), getattr(model, "config", None)):
        if cfg is not None and hasattr(cfg, "compile_inference"):
            cfg.compile_inference = False


def apply_deploy_profile(model, device: torch.device) -> None:
    if device.type != "cuda":
        apply_eager_profile(model)
        return
    for cfg in (getattr(model, "cfg", None), getattr(model, "config", None)):
        if cfg is None:
            continue
        if hasattr(cfg, "compile_inference"):
            cfg.compile_inference = True
        if hasattr(cfg, "compile_inference_mode"):
            cfg.compile_inference_mode = "reduce-overhead"
        if hasattr(cfg, "cuda_graphs_cache_quantum"):
            cfg.cuda_graphs_cache_quantum = 128
        if hasattr(cfg, "eos_check_interval"):
            cfg.eos_check_interval = 16


def profile_label(model, device: torch.device) -> str:
    cfg = getattr(model, "cfg", None) or getattr(model, "config", None)
    if device.type != "cuda":
        return "cpu-eager"
    if cfg is not None and getattr(cfg, "compile_inference", False):
        mode = getattr(cfg, "compile_inference_mode", "default")
        return f"cuda-compile-{mode}"
    return "cuda-eager"
