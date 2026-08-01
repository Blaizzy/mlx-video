from typing import Optional

from transformers import AutoTokenizer
import torchvision.transforms as transforms

from .custom_transforms import DynamicResize, GlobalAndSplitImages

TOKENIZERS_CACHE = {}
FLASH_MIN_SIDE_LEN = 512


# Local copies of configuration_visionpsynano.{resolve_is_flash,apply_flash_preprocess}:
# the data/ tree is vendored standalone (vLLM plugin), so it must not import from
# the HF package layout. Keep in sync with configuration_visionpsynano.py.
def resolve_is_flash(
    *,
    is_flash: Optional[bool] = None,
    variant: Optional[str] = None,
    resize_to_max_side_len: Optional[bool] = None,
) -> bool:
    """Resolve Flash preprocess mode from explicit flag, legacy variant, or resize policy."""
    if is_flash is not None:
        return bool(is_flash)
    if variant is not None:
        v = str(variant).lower().strip()
        if v in ("flash", "nano-flash", "visionpsy-nano-flash"):
            return True
        if v in ("nano", "plain", "nano-plain", "visionpsy-nano"):
            return False
        raise ValueError(
            "legacy variant must be 'nano' or 'flash' "
            f"(or aliases); got {variant!r}. Prefer is_flash=True/False."
        )
    if resize_to_max_side_len is not None:
        return not bool(resize_to_max_side_len)
    return False


def apply_flash_preprocess(
    *,
    is_flash: bool,
    resize_to_max_side_len: Optional[bool] = None,
    resize_min_side_len: Optional[int] = None,
) -> tuple[bool, Optional[int]]:
    """Return (resize_to_max_side_len, resize_min_side_len) for the given Flash mode."""
    if resize_to_max_side_len is None:
        resize_to_max_side_len = not is_flash
    resize_to_max_side_len = bool(resize_to_max_side_len)
    if is_flash:
        resize_min_side_len = max(int(resize_min_side_len or 0), FLASH_MIN_SIDE_LEN)
    elif resize_to_max_side_len:
        resize_min_side_len = None
    return resize_to_max_side_len, resize_min_side_len


def apply_model_preprocess(cfg, *, flash: bool = False) -> bool:
    is_flash = resolve_is_flash(
        is_flash=True if flash else getattr(cfg, "is_flash", None),
        variant=getattr(cfg, "variant", None),
        resize_to_max_side_len=(
            False if flash else getattr(cfg, "resize_to_max_side_len", None)
        ),
    )
    resize_to_max, resize_min = apply_flash_preprocess(
        is_flash=is_flash,
        resize_to_max_side_len=None if flash else getattr(cfg, "resize_to_max_side_len", None),
        resize_min_side_len=getattr(cfg, "resize_min_side_len", None),
    )
    cfg.resize_to_max_side_len = resize_to_max
    cfg.resize_min_side_len = resize_min
    if hasattr(cfg, "is_flash"):
        try:
            cfg.is_flash = is_flash
        except AttributeError:
            pass
    return is_flash


def get_tokenizer(name, extra_special_tokens=None, chat_template=None):
    cache_key = (
        name,
        tuple(sorted((extra_special_tokens or {}).items())),
        chat_template,
    )
    if cache_key not in TOKENIZERS_CACHE:
        tokenizer_init_kwargs = {"use_fast": True}
        if extra_special_tokens is not None:
            tokenizer_init_kwargs["extra_special_tokens"] = extra_special_tokens
        if chat_template is not None:
            tokenizer_init_kwargs["chat_template"] = chat_template
        tokenizer = AutoTokenizer.from_pretrained(name, **tokenizer_init_kwargs,)
        tokenizer.pad_token = tokenizer.eos_token
        TOKENIZERS_CACHE[cache_key] = tokenizer
    return TOKENIZERS_CACHE[cache_key]

def get_image_processor(max_img_size, splitted_image_size, resize_to_max_side_len=False, min_side_len=None):
    return transforms.Compose([
        DynamicResize(splitted_image_size, max_img_size, resize_to_max_side_len, min_side_len),
        transforms.ToTensor(),
        GlobalAndSplitImages(splitted_image_size),
    ])

def get_image_string(tokenizer, splitted_image_counts, mp_image_token_length):
    image_string = ""

    for idx, (n_h, n_w) in enumerate(splitted_image_counts):
        if len(splitted_image_counts) > 1:
            image_string += f"<image: {idx}>"
        if hasattr(tokenizer, "global_image_token"):
            image_string += tokenizer.global_image_token
            image_string += tokenizer.image_token * mp_image_token_length
            if n_h == 1 and n_w == 1:
                continue
        for i in range(n_h):
            for j in range(n_w):
                image_string += getattr(tokenizer, f'r{i+1}c{j+1}')
                image_string += tokenizer.image_token * mp_image_token_length
    return image_string
