"""Sanity tests for the VisionPsyNano MLX port.

These are opportunistic — they only run if the on-disk model exists at
`~/models/VisionPsy-Nano-460M/`. Otherwise every test skips.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

STANDARD_DIR = Path(os.path.expanduser("~/models/VisionPsy-Nano-460M"))
FLASH_DIR = Path(os.path.expanduser("~/models/VisionPsy-Nano-460M-Flash"))


def _has_model(p: Path) -> bool:
    return (p / "config.json").exists() and (p / "model.safetensors").exists()


require_standard = pytest.mark.skipif(
    not _has_model(STANDARD_DIR), reason=f"missing {STANDARD_DIR}"
)
require_flash = pytest.mark.skipif(
    not _has_model(FLASH_DIR), reason=f"missing {FLASH_DIR}"
)


def test_config_import_shapes():
    from mlx_video.models.visionpsy_nano.config import VisionPsyNanoConfig

    cfg = VisionPsyNanoConfig()
    assert cfg.lm_head_dim == 64
    assert cfg.vit_head_dim == 64
    assert cfg.vit_num_patches == (512 // 16) ** 2 == 1024
    assert cfg.mp_image_token_length == 64  # (32 / 4)^2


@require_standard
def test_standard_config_from_pretrained():
    from mlx_video.models.visionpsy_nano.config import VisionPsyNanoConfig

    cfg = VisionPsyNanoConfig.from_pretrained(STANDARD_DIR)
    assert cfg.is_flash is False
    assert cfg.lm_n_blocks == 32
    assert cfg.lm_hidden_dim == 960
    assert cfg.vit_hidden_dim == 768
    assert cfg.mp_pixel_shuffle_factor == 4


@require_flash
def test_flash_config_from_pretrained():
    from mlx_video.models.visionpsy_nano.config import VisionPsyNanoConfig

    cfg = VisionPsyNanoConfig.from_pretrained(FLASH_DIR)
    assert cfg.is_flash is True
    assert cfg.resize_min_side_len == 512
    assert cfg.resize_to_max_side_len is False


@require_standard
def test_standard_weight_load():
    import mlx.core as mx
    from mlx_video.models.visionpsy_nano import load_visionpsy_nano

    model, cfg = load_visionpsy_nano(STANDARD_DIR, dtype=mx.bfloat16)
    assert model.decoder.token_embedding.weight.shape == (cfg.lm_vocab_size, cfg.lm_hidden_dim)
    assert model.decoder.token_embedding.weight.dtype == mx.bfloat16
    # tied lm head — same shape as embedding
    assert model.decoder.head.weight.shape == model.decoder.token_embedding.weight.shape
    # MP.proj input dim matches the shuffled vision hidden
    assert model.MP.proj.weight.shape[1] == cfg.vit_hidden_dim * cfg.mp_pixel_shuffle_factor ** 2


@require_standard
def test_pixel_shuffle_shape():
    """Standalone check that our MP shape math matches the PyTorch reference."""
    import mlx.core as mx
    from mlx_video.models.visionpsy_nano.config import VisionPsyNanoConfig
    from mlx_video.models.visionpsy_nano.modality_projector import ModalityProjector

    cfg = VisionPsyNanoConfig()
    mp = ModalityProjector(cfg)
    x = mx.zeros((2, 1024, cfg.vit_hidden_dim))  # 32*32 tokens from 512x512
    y = mp.pixel_shuffle(x)
    assert y.shape == (2, 64, cfg.vit_hidden_dim * 16)


@require_standard
def test_processor_and_prefill():
    """Full forward-pass sanity test: does the model run end-to-end?"""
    import mlx.core as mx
    from PIL import Image
    from mlx_video.models.visionpsy_nano import load_visionpsy_nano
    from mlx_video.models.visionpsy_nano.processor import load_processor

    model, cfg = load_visionpsy_nano(STANDARD_DIR, dtype=mx.bfloat16)
    proc = load_processor(STANDARD_DIR, cfg=cfg)

    # A cheap synthetic image (solid grey) so we don't need any real fixtures.
    img = Image.new("RGB", (768, 512), color=(128, 128, 128))
    batch = proc("What color is this image?", image=img)
    logits = model(
        batch["input_ids"],
        pixel_values=batch["pixel_values"],
        image_token_id=batch["image_token_id"],
    )
    mx.eval(logits)
    # Basic well-formedness
    assert logits.shape[0] == 1
    assert logits.shape[-1] == cfg.lm_vocab_size
    # bf16 -> fp32 before numpy: numpy has no native bfloat16 dtype
    last_fp32 = np.asarray(logits[:, -1, :].astype(mx.float32))
    assert not np.any(np.isnan(last_fp32))
    assert float(np.abs(last_fp32).max()) > 0
