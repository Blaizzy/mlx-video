"""4x pixel-shuffle projector: (B, N, vit_dim) -> (B, N/16, lm_dim).

Matches the reference `modality_projector.py` exactly:
    input_dim  = vit_hidden_dim * pixel_shuffle_factor**2
    output_dim = lm_hidden_dim
    proj       = Linear(input_dim, output_dim, bias=False)

The pixel_shuffle operation assumes the sequence is a perfect square and that
the side length divides evenly by `scale_factor`.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .config import VisionPsyNanoConfig


class ModalityProjector(nn.Module):
    def __init__(self, cfg: VisionPsyNanoConfig):
        super().__init__()
        self.scale_factor = cfg.mp_pixel_shuffle_factor
        self.input_dim = cfg.vit_hidden_dim * (self.scale_factor ** 2)
        self.output_dim = cfg.lm_hidden_dim
        self.proj = nn.Linear(self.input_dim, self.output_dim, bias=False)

    def pixel_shuffle(self, x: mx.array) -> mx.array:
        # (B, S, D) with sqrt(S) integer, then unfold to h*w with scale factor.
        B, S, D = x.shape
        side = int(round(S ** 0.5))
        if side * side != S:
            raise ValueError(f"expected square token grid; got {S} tokens")
        sf = self.scale_factor
        if side % sf != 0:
            raise ValueError(
                f"grid side {side} not divisible by pixel_shuffle_factor {sf}"
            )
        h_out = w_out = side // sf
        x = x.reshape(B, h_out, sf, w_out, sf, D)
        x = x.transpose(0, 1, 3, 2, 4, 5)  # match reference permute
        x = x.reshape(B, h_out * w_out, D * sf * sf)
        return x

    def __call__(self, x: mx.array) -> mx.array:
        return self.proj(self.pixel_shuffle(x))
