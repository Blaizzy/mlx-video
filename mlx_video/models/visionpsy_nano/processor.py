"""Image + text preprocessing for VisionPsy-Nano (MLX).

Matches the reference `processing_visionpsynano.py`:

1. PIL -> RGB -> DynamicResize:
     - Standard variant: force the long side to `max_img_size` (2048), then
       round the short side to a multiple of the tile size (512).
     - Flash variant: preserve aspect, resize so the short side is at least
       `resize_min_side_len` (512), long side capped at `max_img_size`.
   In both cases the final size is a multiple of the tile size (512).

2. ToTensor: [0, 1] float, channels-first (C, H, W).

3. GlobalAndSplitImages: if the image is bigger than a single tile, prepend a
   global 512x512 tile (bilinear resize) followed by the row/column tiles.
   Returns (Ntiles, 3, 512, 512) and the tile grid `(n_h, n_w)`.

4. Prompt string:
      <|global_image|> <|image|>*64  (if global present)
      + for each tile (i, j):  <row_{i+1}_col_{j+1}> <|image|>*64
     ...followed by the user prompt, wrapped by the chat template's
     `<|im_start|>user ... <|im_end|>\n<|im_start|>assistant\n`.

Only PIL images / tensors are handled; no batching support in this port —
inference is single-sample (batch=1).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import mlx.core as mx
import numpy as np
from PIL import Image

from .config import DEFAULT_EXTRA_TOKENS, VisionPsyNanoConfig


TILE_SIZE = 512  # vit_img_size
IMAGE_TOKEN = "<|image|>"
GLOBAL_IMAGE_TOKEN = "<|global_image|>"


def _row_col_token(i: int, j: int) -> str:
    return f"<row_{i}_col_{j}>"


def _compute_target_hw(
    h: int,
    w: int,
    tile: int,
    max_side: int,
    resize_to_max_side_len: bool,
    min_side_len: Optional[int],
) -> Tuple[int, int]:
    """Reproduce DynamicResize._get_new_hw from the reference implementation.

    Returns `(new_h, new_w)` — both multiples of `tile`.
    """
    long, short = (w, h) if w >= h else (h, w)

    if min_side_len and not resize_to_max_side_len and short < min_side_len:
        den = short * tile
        # -(-a // b) = ceil(a / b) trick from the reference code
        target_long = min(max_side, -(-(long * min_side_len) // den) * tile)
        target_short = max(-(-(short * min_side_len) // den) * tile, tile)
        return (target_short, target_long) if w >= h else (target_long, target_short)

    if resize_to_max_side_len:
        target_long = max_side
    else:
        target_long = min(max_side, math.ceil(long / tile) * tile)

    scale = target_long / long
    target_short = math.ceil(short * scale / tile) * tile
    target_short = max(target_short, tile)
    return (target_short, target_long) if w >= h else (target_long, target_short)


def _pil_resize(img: Image.Image, new_h: int, new_w: int) -> Image.Image:
    return img.resize((new_w, new_h), Image.BICUBIC)


def _to_tensor_chw(img: Image.Image) -> np.ndarray:
    """PIL RGB -> float32 CHW in [0, 1]."""
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)  # HWC -> CHW


def _split_tiles(chw: np.ndarray, tile: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Split a (3, H, W) tensor into (n_h * n_w, 3, tile, tile) tiles."""
    _, H, W = chw.shape
    if H % tile or W % tile:
        raise ValueError(f"image size ({H}, {W}) not divisible by tile {tile}")
    n_h, n_w = H // tile, W // tile
    tiles = chw.reshape(3, n_h, tile, n_w, tile)
    tiles = tiles.transpose(1, 3, 0, 2, 4)  # (n_h, n_w, 3, tile, tile)
    tiles = tiles.reshape(n_h * n_w, 3, tile, tile)
    return tiles, (n_h, n_w)


def _global_tile(chw: np.ndarray, tile: int) -> np.ndarray:
    """Bilinear resize a (3, H, W) tensor down to (3, tile, tile) via PIL.

    We convert back to PIL to reuse the same resampling behaviour as the
    reference implementation (torchvision.functional.resize with default bilinear).
    """
    hwc = (chw.transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(hwc, mode="RGB")
    img = img.resize((tile, tile), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)


@dataclass
class ImageProcessed:
    pixel_values: mx.array  # (Ntiles, 3, tile, tile)
    grid: Tuple[int, int]  # (n_h, n_w)
    has_global: bool


class VisionPsyNanoProcessor:
    """Combined image + text processor for VisionPsy-Nano."""

    def __init__(
        self,
        cfg: VisionPsyNanoConfig,
        tokenizer,
        *,
        chat_template: Optional[str] = None,
    ):
        self.cfg = cfg
        self.tokenizer = tokenizer

        # Ensure the tokenizer knows about our extra tokens as named attributes
        # (matches the reference `_attach_extra_token_attrs` behaviour).
        for name, tok_str in (cfg.vlm_extra_tokens or DEFAULT_EXTRA_TOKENS).items():
            if not hasattr(tokenizer, name):
                setattr(tokenizer, name, tok_str)
        # image_token_id
        try:
            self.image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
            tokenizer.image_token = IMAGE_TOKEN
            tokenizer.image_token_id = self.image_token_id
        except Exception:
            self.image_token_id = None

        self.chat_template = chat_template or cfg.lm_chat_template
        if self.chat_template is not None:
            tokenizer.chat_template = self.chat_template

    # ------------------------------------------------------------------ image
    def _process_image(self, image: Image.Image) -> ImageProcessed:
        img = image.convert("RGB")
        w, h = img.size
        new_h, new_w = _compute_target_hw(
            h, w,
            tile=TILE_SIZE,
            max_side=int(self.cfg.inference_max_img_size or self.cfg.max_img_size),
            resize_to_max_side_len=bool(self.cfg.resize_to_max_side_len),
            min_side_len=self.cfg.resize_min_side_len,
        )
        img = _pil_resize(img, new_h, new_w)
        chw = _to_tensor_chw(img)
        tiles, grid = _split_tiles(chw, TILE_SIZE)
        n_h, n_w = grid
        if (n_h, n_w) == (1, 1):
            pixel_values = tiles
            has_global = False
        else:
            g = _global_tile(chw, TILE_SIZE)[None, ...]  # (1, 3, tile, tile)
            pixel_values = np.concatenate([g, tiles], axis=0)
            has_global = True
        return ImageProcessed(
            pixel_values=mx.array(pixel_values),
            grid=grid,
            has_global=has_global,
        )

    # ------------------------------------------------------------------- text
    def _build_image_string(self, grid: Tuple[int, int], has_global: bool) -> str:
        n_h, n_w = grid
        img_tokens_per_tile = IMAGE_TOKEN * self.cfg.mp_image_token_length
        parts: List[str] = []
        if has_global:
            parts.append(GLOBAL_IMAGE_TOKEN)
            parts.append(img_tokens_per_tile)
        for i in range(n_h):
            for j in range(n_w):
                parts.append(_row_col_token(i + 1, j + 1))
                parts.append(img_tokens_per_tile)
        # If there's a single tile we still want the row/col tag AND the global
        # tile — that matches the reference when `global_image_token` is
        # available (see processors.get_image_string).
        # For the (1, 1) no-global case we only emit the single tile block.
        if not has_global and (n_h, n_w) == (1, 1):
            # parts already contains the r1c1 tag + tokens
            pass
        return "".join(parts)

    def _apply_chat_template(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Fallback to hand-rolled template so we're never blocked by
            # tokenizer-config edge cases.
            return (
                "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"
            )

    # ------------------------------------------------------------------- api
    def __call__(
        self,
        text: str,
        image: Optional[Image.Image] = None,
    ) -> dict:
        if image is not None:
            processed = self._process_image(image)
            image_string = self._build_image_string(processed.grid, processed.has_global)
            full_prompt = self._apply_chat_template(image_string + text)
            token_ids = self.tokenizer(full_prompt, return_tensors=None, add_special_tokens=False)["input_ids"]
            input_ids = mx.array(np.asarray(token_ids, dtype=np.int64)).reshape(1, -1)
            return {
                "input_ids": input_ids,
                "pixel_values": processed.pixel_values,
                "image_token_id": self.image_token_id,
                "grid": processed.grid,
                "has_global": processed.has_global,
                "prompt": full_prompt,
            }

        # Text-only path (still useful for sanity checks).
        full_prompt = self._apply_chat_template(text)
        token_ids = self.tokenizer(full_prompt, return_tensors=None, add_special_tokens=False)["input_ids"]
        input_ids = mx.array(np.asarray(token_ids, dtype=np.int64)).reshape(1, -1)
        return {
            "input_ids": input_ids,
            "pixel_values": None,
            "image_token_id": self.image_token_id,
            "grid": None,
            "has_global": False,
            "prompt": full_prompt,
        }

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


def load_processor(
    model_dir: str | Path,
    cfg: Optional[VisionPsyNanoConfig] = None,
) -> VisionPsyNanoProcessor:
    """Build a processor by loading the tokenizer from `model_dir`.

    Requires `transformers` for the tokenizer. The tokenizer files
    (`tokenizer.json`, `tokenizer_config.json`) live alongside the model.
    """
    from transformers import AutoTokenizer

    model_dir = Path(model_dir)
    cfg = cfg or VisionPsyNanoConfig.from_pretrained(model_dir)

    extra_tokens = cfg.vlm_extra_tokens or DEFAULT_EXTRA_TOKENS
    tok = AutoTokenizer.from_pretrained(
        str(model_dir),
        use_fast=True,
        extra_special_tokens=extra_tokens,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return VisionPsyNanoProcessor(cfg, tok, chat_template=cfg.lm_chat_template)
