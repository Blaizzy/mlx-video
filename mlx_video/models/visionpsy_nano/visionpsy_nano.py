"""Top-level VisionPsyNano VLM composition + a simple greedy generate().

Forward pass:
    1. tokens -> embeddings via `LanguageModel.embed`
    2. tile batch -> ViT -> hidden [Ntiles, 1024, 768]
    3. ModalityProjector (pixel-shuffle 4x + linear) -> [Ntiles, 64, 960]
    4. `<|image|>` positions in `input_ids` are replaced with the projected
       features (in row-major, tile-major order).
    5. Decoder processes the merged embeddings and returns logits.
"""
from __future__ import annotations

from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import VisionPsyNanoConfig
from .language_model import LanguageModel
from .modality_projector import ModalityProjector
from .vision_transformer import ViT


class VisionPsyNano(nn.Module):
    """VisionPsyNano model: ViT + modality projector + SmolLM2 decoder."""

    def __init__(self, cfg: VisionPsyNanoConfig):
        super().__init__()
        self.cfg = cfg
        self.vision_encoder = ViT(cfg)
        self.decoder = LanguageModel(cfg)
        self.MP = ModalityProjector(cfg)

    # ------------------------------------------------------------------ core
    def encode_images(self, pixel_values: mx.array) -> mx.array:
        """Run ViT + modality projector.

        pixel_values: (Ntiles, 3, 512, 512), values in [0, 1]
        returns:      (Ntiles, mp_image_token_length, lm_hidden_dim)
        """
        hidden = self.vision_encoder(pixel_values)
        return self.MP(hidden)

    def merge_image_features(
        self,
        inputs_embeds: mx.array,
        image_features: mx.array,
        image_token_id: int,
        input_ids: mx.array,
    ) -> mx.array:
        """Scatter the projected image features into the `<|image|>` slots."""
        # inputs_embeds: (B, T, D); image_features: (Ntiles, S, D)
        image_features_flat = image_features.reshape(-1, image_features.shape[-1])
        # Cast to Python int list of positions using numpy for correctness.
        input_ids_np = np.asarray(input_ids)
        # We expect batch size 1 in the smoke path.
        if input_ids_np.ndim == 2:
            flat_ids = input_ids_np.reshape(-1)
        else:
            flat_ids = input_ids_np
        positions = np.where(flat_ids == image_token_id)[0]

        n_needed = int(image_features_flat.shape[0])
        n_slots = int(positions.shape[0])
        if n_needed != n_slots:
            raise ValueError(
                f"image feature/token count mismatch: features={n_needed}, "
                f"image_tokens_in_prompt={n_slots}"
            )

        # Flat view of inputs_embeds so we can index it with the token
        # positions. Assumes batch=1 which is our current inference path.
        D = inputs_embeds.shape[-1]
        flat_embeds = inputs_embeds.reshape(-1, D)
        pos_arr = mx.array(positions.astype(np.int32))
        flat_embeds[pos_arr] = image_features_flat.astype(flat_embeds.dtype)
        return flat_embeds.reshape(inputs_embeds.shape)

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        image_token_id: Optional[int] = None,
        cache: Optional[list] = None,
    ) -> mx.array:
        token_embeds = self.decoder.embed(input_ids)

        if pixel_values is not None:
            if image_token_id is None:
                raise ValueError("image_token_id is required when pixel_values are provided")
            image_features = self.encode_images(pixel_values)
            token_embeds = self.merge_image_features(
                token_embeds, image_features, image_token_id, input_ids
            )

        return self.decoder(token_embeds, cache=cache)

    # ---------------------------------------------------------------- generate
    def generate(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        image_token_id: Optional[int] = None,
        max_new_tokens: int = 64,
        eos_token_id: Optional[int] = None,
    ):
        """Greedy generation. Yields token ids one at a time."""
        cache = self.decoder.make_cache()

        # 1. Prefill with the full prompt + image embeds.
        logits = self(input_ids, pixel_values, image_token_id, cache=cache)
        next_id = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(next_id)

        yield int(next_id.item())
        if eos_token_id is not None and int(next_id.item()) == eos_token_id:
            return

        # 2. Decode step-by-step with the KV cache.
        for _ in range(max_new_tokens - 1):
            next_input = next_id.reshape(1, 1)
            emb = self.decoder.embed(next_input)
            logits = self.decoder(emb, cache=cache)
            next_id = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(next_id)
            tok = int(next_id.item())
            yield tok
            if eos_token_id is not None and tok == eos_token_id:
                return
