"""Helios model loading utilities.

Reuses Wan's T5 encoder and VAE since Helios uses identical components.
"""

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


def load_helios_model(
    model_path: Path,
    config,
    quantization: dict | None = None,
):
    """Load and initialize HeliosModel, with optional quantization.

    Args:
        model_path: Path to model safetensors file
        config: HeliosModelConfig
        quantization: Optional dict with 'bits' and 'group_size' keys.
    """
    from mlx_video.models.helios.transformer import HeliosModel

    model = HeliosModel(config)

    if quantization:
        from mlx_video.convert_helios import _quantize_predicate

        nn.quantize(
            model,
            group_size=quantization["group_size"],
            bits=quantization["bits"],
            class_predicate=lambda path, m: _quantize_predicate(path, m),
        )

    weights = mx.load(str(model_path))
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())
    return model


# Reuse Wan's T5 encoder and VAE loaders since Helios uses the same components
from mlx_video.models.wan.loading import (  # noqa: E402, F401
    _clean_text,
    encode_text,
    load_t5_encoder,
    load_vae_decoder,
    load_vae_encoder,
)
