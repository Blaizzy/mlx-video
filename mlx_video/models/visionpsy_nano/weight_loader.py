"""Load PyTorch-flavored VisionPsyNano safetensors into the MLX model.

The reference checkpoint uses these top-level prefixes:
    vision_encoder.*   ->  ViT (patch_embedding, blocks[i].{ln1,ln2,attn,mlp}, layer_norm)
    decoder.*          ->  LanguageModel (token_embedding, blocks[i].{norm1,norm2,attn,mlp}, norm, head)
    MP.proj.*          ->  ModalityProjector.proj

Everything already matches our MLX attribute names 1:1 EXCEPT:
    * Conv2d weight is stored channels-first (out, in, kH, kW); MLX expects
      channels-last (out, kH, kW, in).
    * `decoder.rotary_embd.inv_freq` is a stale buffer — we drop it because
      MLX's `nn.RoPE` computes frequencies on the fly.

We optionally cast to `bf16` (or another dtype) once at load time so the whole
model lives in a smaller footprint (~1 GB instead of ~2 GB for fp32).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import mlx.core as mx
import mlx.nn as nn
from safetensors import safe_open

from .config import VisionPsyNanoConfig
from .visionpsy_nano import VisionPsyNano


# Attribute prefix on disk -> attribute prefix in our MLX module.
# All of these are 1:1 because we deliberately kept the names identical.
_TOP_PREFIXES = {
    "vision_encoder.": "vision_encoder.",
    "decoder.": "decoder.",
    "MP.": "MP.",
}


def _torch_to_mlx_dtype(name: str) -> mx.Dtype:
    return {
        "float32": mx.float32,
        "float16": mx.float16,
        "bfloat16": mx.bfloat16,
    }.get(name, mx.float32)


def _load_raw(weights_path: Path) -> Dict[str, mx.array]:
    """Return the safetensors file as a dict of MLX arrays.

    Uses the torch backend so we can handle bf16 checkpoints (numpy has no
    native bfloat16). Non-bf16 tensors flow through numpy directly; bf16
    tensors are viewed as uint16 and reinterpreted as MLX bfloat16 without
    a lossy fp32 detour.
    """
    import torch

    out: Dict[str, mx.array] = {}
    with safe_open(str(weights_path), framework="pt") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            if t.dtype == torch.bfloat16:
                # torch bf16 -> uint16 view -> numpy -> mlx uint16 -> reinterpret
                arr_u16 = mx.array(t.view(torch.uint16).numpy())
                out[k] = arr_u16.view(mx.bfloat16)
            else:
                out[k] = mx.array(t.numpy())
    return out


def _remap(raw: Dict[str, mx.array], cfg: VisionPsyNanoConfig) -> Dict[str, mx.array]:
    """Rename PyTorch keys into MLX attribute keys and reshape as needed."""
    out: Dict[str, mx.array] = {}
    for k, v in raw.items():
        # Drop rotary-embedding buffers — MLX's nn.RoPE computes freqs itself.
        if k.startswith("decoder.rotary_embd."):
            continue

        # Vision Conv2d: (out, in, kH, kW) -> (out, kH, kW, in).
        if k == "vision_encoder.patch_embedding.conv.weight":
            if v.ndim != 4:
                raise ValueError(
                    f"unexpected patch_embedding conv weight shape {v.shape}"
                )
            v = v.transpose(0, 2, 3, 1)

        # position_embedding is stored as `[1, num_patches, hidden]` — matches
        # our MLX attribute, no reshape needed.

        # Handle top-level prefix mapping (currently 1:1).
        new_key: Optional[str] = None
        for src_prefix, dst_prefix in _TOP_PREFIXES.items():
            if k.startswith(src_prefix):
                new_key = dst_prefix + k[len(src_prefix):]
                break
        if new_key is None:
            # Unknown top-level key: keep verbatim; nn.Module.load_weights will
            # complain if it doesn't correspond to anything.
            new_key = k
        out[new_key] = v
    return out


def load_visionpsy_nano(
    model_dir: str | Path,
    *,
    dtype: mx.Dtype | str | None = mx.bfloat16,
    strict: bool = True,
) -> tuple[VisionPsyNano, VisionPsyNanoConfig]:
    """Load a VisionPsyNano model directory into an MLX-backed VisionPsyNano.

    Parameters
    ----------
    model_dir : path to a directory containing `config.json` and
                `model.safetensors`.
    dtype     : Cast every loaded weight to this MLX dtype (default bf16).
                Pass `None` to keep the safetensors dtype (typically fp32).
    strict    : If True, raise on unexpected / missing keys. If False, log
                warnings but keep loading.

    Returns
    -------
    (model, config)
    """
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(model_dir)

    cfg = VisionPsyNanoConfig.from_pretrained(model_dir)
    weights_path = model_dir / "model.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(weights_path)

    raw = _load_raw(weights_path)
    remapped = _remap(raw, cfg)

    if isinstance(dtype, str):
        dtype = _torch_to_mlx_dtype(dtype)
    if dtype is not None:
        remapped = {k: v.astype(dtype) for k, v in remapped.items()}

    model = VisionPsyNano(cfg)

    # Optionally tie head to embedding table BEFORE loading. Since both keys
    # exist in the checkpoint, we let them both flow through and rely on the
    # tie post-hoc — the two tensors were saved identical.
    model.load_weights(list(remapped.items()), strict=strict)

    # Weight-tie: point head.weight at the embedding table so decode paths
    # share storage after load.
    if cfg.lm_tie_weights:
        model.decoder.head.weight = model.decoder.token_embedding.weight

    mx.eval(model.parameters())
    return model, cfg
