"""Wan2.2 Text-to-Video generation pipeline for MLX."""

import argparse
import gc
import math
import random
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from tqdm import tqdm

from mlx_video.models.wan_2.i2v_utils import build_i2v_mask, preprocess_image
from mlx_video.models.wan_2.utils import (
    encode_text,
    load_t5_encoder,
    load_vae_decoder,
    load_vae_encoder,
    load_wan_model,
)
from mlx_video.models.wan_2.postprocess import save_video


class Colors:
    """ANSI color codes for terminal output."""

    CYAN = "\033[96m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


# Backward-compat alias (tests and external code may use the old name)
_build_i2v_mask = build_i2v_mask


def _best_output_size(w, h, dw, dh, max_area):
    """Compute the best output resolution that fits within max_area while
    preserving the input aspect ratio and satisfying alignment constraints.
    Matches the reference implementation's best_output_size().
    """
    ratio = w / h
    ow = (max_area * ratio) ** 0.5
    oh = max_area / ow

    # Option 1: process width first
    ow1 = int(ow // dw * dw)
    oh1 = int(max_area / ow1 // dh * dh)
    ratio1 = ow1 / oh1

    # Option 2: process height first
    oh2 = int(oh // dh * dh)
    ow2 = int(max_area / oh2 // dw * dw)
    ratio2 = ow2 / oh2

    if max(ratio / ratio1, ratio1 / ratio) < max(ratio / ratio2, ratio2 / ratio):
        return ow1, oh1
    return ow2, oh2


def generate_video(
    model_dir: str,
    prompt: str,
    negative_prompt: str | None = None,
    image: str | None = None,
    width: int = 1280,
    height: int = 704,
    num_frames: int = 81,
    steps: int = None,
    guide_scale: str | float | tuple = None,
    shift: float = None,
    seed: int = -1,
    output_path: str = "output.mp4",
    scheduler: str = "unipc",
    loras: list | None = None,
    loras_high: list | None = None,
    loras_low: list | None = None,
    tiling: str = "auto",
    no_compile: bool = False,
    trim_first_frames: int = 0,
    debug_latents: bool = False,
):
    """Generate video using Wan pipeline (supports T2V and I2V).

    Args:
        model_dir: Path to converted MLX model directory
        prompt: Text prompt
        negative_prompt: Negative prompt (None = use config default, "" = no negative prompt)
        image: Path to input image for I2V (None = T2V mode)
        width: Video width
        height: Video height
        num_frames: Number of frames (must be 4n+1)
        steps: Number of diffusion steps (None = use config default)
        guide_scale: Guidance scale: float for single, (low,high) for dual (None = config default)
        shift: Noise schedule shift (None = use config default)
        seed: Random seed (-1 for random)
        output_path: Output video path
        scheduler: Solver type: 'euler', 'dpm++', or 'unipc' (default)
        loras: Optional list of (path, strength) tuples applied to all models
        loras_high: Optional list of (path, strength) tuples for high-noise model only
        loras_low: Optional list of (path, strength) tuples for low-noise model only
        tiling: Tiling mode for VAE decoding. Options:
            - "auto": Automatically determine tiling based on video size (default)
            - "none": Disable tiling
            - "default", "aggressive", "conservative": Preset tiling configs
            - "spatial": Spatial tiling only
            - "temporal": Temporal tiling only
        no_compile: If True, skip mx.compile on models (useful for debugging)
        trim_first_frames: Number of temporal latent positions to generate extra
            and discard from the start. Each position = 4 pixel frames. Use 1
            to fix first-frame artifacts on 14B models (generates 4 extra frames,
            discards first 4). Use 2 for more aggressive trimming. Default: 0.
        debug_latents: If True, print per-temporal-position latent statistics
            after denoising for diagnosing first-frame artifacts.
    """
    import json

    from mlx_video.models.wan_2.config import WanModelConfig
    from mlx_video.models.wan_2.scheduler import (
        FlowDPMPP2MScheduler,
        FlowMatchEulerScheduler,
        FlowUniPCScheduler,
    )

    model_dir = Path(model_dir)

    # Load config from model dir if available, otherwise auto-detect
    config_path = model_dir / "config.json"
    quantization = None
    if config_path.exists():
        with open(config_path) as f:
            config_dict = json.load(f)
        # Extract quantization config (not a model config field)
        quantization = config_dict.pop("quantization", None)
        # Handle tuple fields stored as lists in JSON
        for key in ("patch_size", "vae_stride", "window_size", "sample_guide_scale"):
            if key in config_dict and isinstance(config_dict[key], list):
                config_dict[key] = tuple(config_dict[key])
        config = WanModelConfig(
            **{
                k: v
                for k, v in config_dict.items()
                if k in WanModelConfig.__dataclass_fields__
            }
        )
    else:
        # Auto-detect: dual model files → 2.2, single model → 2.1
        if (model_dir / "low_noise_model.safetensors").exists():
            config = WanModelConfig.wan22_t2v_14b()
        else:
            # Detect 1.3B vs 14B from weight shapes
            model_path = model_dir / "model.safetensors"
            if model_path.exists():
                probe = mx.load(str(model_path), return_metadata=False)
                for k, v in probe.items():
                    if "patch_embedding_proj.weight" in k:
                        dim = v.shape[0]
                        if dim <= 2048:
                            config = WanModelConfig.wan21_t2v_1_3b()
                        else:
                            config = WanModelConfig.wan21_t2v_14b()
                        break
                else:
                    config = WanModelConfig.wan21_t2v_14b()
                del probe
            else:
                config = WanModelConfig.wan21_t2v_14b()

    is_dual = config.dual_model
    is_i2v = image is not None

    # Validate config against actual weights (handles mismatched config.json)
    if not is_dual:
        model_path = model_dir / "model.safetensors"
        if model_path.exists():
            probe = mx.load(str(model_path), return_metadata=False)
            for k, v in probe.items():
                if "patch_embedding_proj.weight" in k:
                    actual_dim = v.shape[0]
                    if actual_dim != config.dim:
                        print(
                            f"{Colors.YELLOW}  Config dim={config.dim} doesn't match weights dim={actual_dim}, auto-correcting...{Colors.RESET}"
                        )
                        if actual_dim <= 2048:
                            config = WanModelConfig.wan21_t2v_1_3b()
                        else:
                            config = WanModelConfig.wan21_t2v_14b()
                    break
            del probe

    # Auto-correct Wan2.2 VAE params from stale configs
    if config.in_dim == 48 and config.vae_z_dim != 48:
        print(
            f"{Colors.YELLOW}  Auto-correcting Wan2.2 VAE params (in_dim=48 but vae_z_dim={config.vae_z_dim}){Colors.RESET}"
        )
        config = WanModelConfig(
            **{
                **{
                    f.name: getattr(config, f.name)
                    for f in config.__dataclass_fields__.values()
                },
                "vae_z_dim": 48,
                "vae_stride": (4, 16, 16),
                "sample_fps": 24,
            }
        )

    # Apply defaults from config if not overridden
    if steps is None:
        steps = config.sample_steps
    if shift is None:
        shift = config.sample_shift
    if guide_scale is None:
        guide_scale = config.sample_guide_scale

    # Normalize guide_scale
    if isinstance(guide_scale, (int, float)):
        guide_scale = float(guide_scale)
    elif isinstance(guide_scale, str):
        parts = [float(x) for x in guide_scale.split(",")]
        guide_scale = tuple(parts) if len(parts) > 1 else parts[0]

    # Detect CFG-disabled mode (guide_scale=1.0 for all models → skip uncond pass for 2x speedup)
    if isinstance(guide_scale, tuple):
        cfg_disabled = all(gs <= 1.0 for gs in guide_scale)
    else:
        cfg_disabled = guide_scale <= 1.0

    # Validate frame count
    assert (num_frames - 1) % 4 == 0, f"num_frames must be 4n+1, got {num_frames}"

    gen_frames = num_frames
    if trim_first_frames > 0:
        gen_frames = num_frames + trim_first_frames * 4
        print(
            f"{Colors.DIM}  Trim: generating {gen_frames} frames, will discard first {trim_first_frames * 4}{Colors.RESET}"
        )

    version_str = f"Wan{config.model_version}"
    mode_str = "dual-model" if is_dual else "single-model"
    pipeline_str = "Image-to-Video" if is_i2v else "Text-to-Video"
    # Resolve negative prompt: explicit user value > config default
    # The official Wan2.2 uses a Chinese negative prompt (config.sample_neg_prompt)
    # that prevents oversaturation, artifacts, and comic look. We use it by default.
    # Text cleaning (_clean_text) normalizes fullwidth chars to match official tokenization.
    if negative_prompt is None:
        neg_prompt_resolved = config.sample_neg_prompt
    else:
        neg_prompt_resolved = negative_prompt
    print(f"{Colors.CYAN}{'='*60}")
    print(f"  {version_str} {pipeline_str} Generation (MLX, {mode_str})")
    print(f"{'='*60}{Colors.RESET}")
    print(f"{Colors.DIM}  Prompt: {prompt}")
    if is_i2v:
        print(f"  Image: {image}")
    if neg_prompt_resolved and neg_prompt_resolved.strip():
        neg_display = (
            neg_prompt_resolved[:60] + "..."
            if len(neg_prompt_resolved) > 60
            else neg_prompt_resolved
        )
        print(f"  Neg prompt: {neg_display}")
    print(f"  Size: {width}x{height}, Frames: {num_frames}")
    print(
        f"  Steps: {steps}, Guide: {guide_scale}, Shift: {shift}, Solver: {scheduler}"
    )
    if cfg_disabled:
        print(f"  CFG: disabled (guide_scale≤1 → B=1 fast path, 2x denoising speedup)")
    print(f"{Colors.RESET}")

    # Seed
    if seed < 0:
        seed = random.randint(0, 2**32 - 1)
    mx.random.seed(seed)
    np.random.seed(seed)
    print(f"{Colors.DIM}  Seed: {seed}{Colors.RESET}")

    # Align dimensions to patch_size * vae_stride (required for patchify)
    vae_stride = config.vae_stride
    patch_size = config.patch_size
    align_h = patch_size[1] * vae_stride[1]  # e.g. 2*16=32
    align_w = patch_size[2] * vae_stride[2]
    if height % align_h != 0 or width % align_w != 0:
        old_h, old_w = height, width
        height = (height // align_h) * align_h
        width = (width // align_w) * align_w
        if height == 0:
            height = align_h
        if width == 0:
            width = align_w
        print(
            f"{Colors.DIM}  Aligned {old_w}x{old_h} → {width}x{height} (must be divisible by {align_w}x{align_h}){Colors.RESET}"
        )

    # Enforce max_area constraint (model-specific resolution limit)
    if config.max_area > 0 and height * width > config.max_area:
        old_h, old_w = height, width
        width, height = _best_output_size(
            width, height, align_w, align_h, config.max_area
        )
        print(
            f"{Colors.YELLOW}  ⚠ Resolution {old_w}x{old_h} exceeds model's max area "
            f"({config.max_area:,}px). Adjusted → {width}x{height}{Colors.RESET}"
        )

    # Compute target latent shape
    z_dim = config.vae_z_dim
    t_latent = (gen_frames - 1) // vae_stride[0] + 1
    h_latent = height // vae_stride[1]
    w_latent = width // vae_stride[2]
    target_shape = (z_dim, t_latent, h_latent, w_latent)

    # Sequence length for transformer
    seq_len = math.ceil(
        (h_latent * w_latent) / (patch_size[1] * patch_size[2]) * t_latent
    )

    print(f"{Colors.DIM}  Latent shape: {target_shape}")
    print(f"  Sequence length: {seq_len}{Colors.RESET}")

    # Load T5 encoder
    t1 = time.time()
    print(f"\n{Colors.BLUE}Loading T5 encoder...{Colors.RESET}")
    t5_path = model_dir / "t5_encoder.safetensors"
    t5_encoder = load_t5_encoder(t5_path, config)

    # Load tokenizer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("google/umt5-xxl")

    # Encode prompts
    print(f"{Colors.BLUE}Encoding text...{Colors.RESET}")
    context = encode_text(t5_encoder, tokenizer, prompt, config.text_len)
    if cfg_disabled:
        context_null = None
        mx.eval(context)
    else:
        context_null = encode_text(
            t5_encoder, tokenizer, neg_prompt_resolved, config.text_len
        )
        mx.eval(context, context_null)

    # Free T5 from memory
    del t5_encoder
    gc.collect()
    mx.clear_cache()
    print(f"{Colors.DIM}  T5 encoding: {time.time() - t1:.1f}s{Colors.RESET}")

    # I2V: encode image to latent space
    z_img = None
    i2v_mask = None
    i2v_mask_tokens = None
    y_i2v = None
    is_i2v_channel_concat = is_i2v and config.model_type == "i2v"
    is_i2v_mask_blend = is_i2v and config.model_type != "i2v"
    if is_i2v:
        print(f"\n{Colors.BLUE}Encoding input image...{Colors.RESET}")
        t_img = time.time()

        vae_path = model_dir / "vae.safetensors"

        if is_i2v_channel_concat:
            # I2V-14B: encode full video (first frame = image, rest = zeros)
            # and construct y tensor with mask + encoded latents
            from PIL import Image

            img = Image.open(image).convert("RGB")
            scale = max(width / img.width, height / img.height)
            img = img.resize(
                (round(img.width * scale), round(img.height * scale)), Image.LANCZOS
            )
            x1, y1 = (img.width - width) // 2, (img.height - height) // 2
            img = img.crop((x1, y1, x1 + width, y1 + height))
            img_arr = mx.array(
                np.array(img, dtype=np.float32) / 255.0 * 2.0 - 1.0
            )  # [H, W, 3]
            img_chw = img_arr.transpose(2, 0, 1)  # [3, H, W]

            # Build video: first frame = image, rest = zeros -> [3, F, H, W]
            # Chunked encoding processes 1-frame + 4-frame chunks with temporal caching
            video = mx.concatenate(
                [
                    img_chw[:, None, :, :],
                    mx.zeros((3, num_frames - 1, height, width)),
                ],
                axis=1,
            )

            # Encode through Wan2.1 VAE -> [1, z_dim, T_lat, H_lat, W_lat]
            vae_enc = load_vae_encoder(vae_path, config)
            z_video = vae_enc.encode(video[None])  # [1, 16, T_lat, H_lat, W_lat]
            mx.eval(z_video)
            z_video = z_video[0]  # [16, T_lat, H_lat, W_lat]

            # Build mask: 1 for first frame, 0 for rest -> rearrange to [4, T_lat, H, W]
            msk = mx.ones((1, num_frames, h_latent, w_latent))
            msk = mx.concatenate(
                [msk[:, :1], mx.zeros((1, num_frames - 1, h_latent, w_latent))], axis=1
            )
            # Repeat first frame 4x, concat rest: [1, 4 + (F-1), H_lat, W_lat]
            msk = mx.concatenate(
                [
                    mx.repeat(msk[:, :1], 4, axis=1),
                    msk[:, 1:],
                ],
                axis=1,
            )
            # Reshape to [1, T_lat, 4, H_lat, W_lat] then transpose -> [4, T_lat, H_lat, W_lat]
            msk = msk.reshape(1, msk.shape[1] // 4, 4, h_latent, w_latent)
            msk = msk.transpose(0, 2, 1, 3, 4)[0]  # [4, T_lat, H_lat, W_lat]

            # y = concat([mask, encoded_video]) -> [20, T_lat, H_lat, W_lat]
            y_i2v = mx.concatenate([msk, z_video], axis=0)
            mx.eval(y_i2v)

            del vae_enc, img_arr, img_chw, video, z_video, msk
        else:
            # TI2V-5B: encode single image, blend with noise via mask
            img_tensor = preprocess_image(image, width, height)
            mx.eval(img_tensor)

            vae_enc = load_vae_encoder(vae_path, config)
            z_img = vae_enc.encode(img_tensor)  # [1, 1, H_lat, W_lat, z_dim]
            mx.eval(z_img)
            z_img = z_img[0].transpose(3, 0, 1, 2)  # [z_dim, 1, H_lat, W_lat]
            i2v_mask, i2v_mask_tokens = build_i2v_mask(target_shape, config.patch_size)

            del vae_enc, img_tensor

        gc.collect()
        mx.clear_cache()
        print(f"{Colors.DIM}  Image encoding: {time.time() - t_img:.1f}s{Colors.RESET}")

    # Load transformer models
    print(f"\n{Colors.BLUE}Loading transformer model(s)...{Colors.RESET}")
    if quantization:
        print(
            f"{Colors.DIM}  Using {quantization['bits']}-bit quantized weights (group_size={quantization['group_size']}){Colors.RESET}"
        )
    t2 = time.time()

    # Merge per-model LoRAs with shared LoRAs
    _loras_low = (loras or []) + (loras_low or []) or None
    _loras_high = (loras or []) + (loras_high or []) or None
    _loras_single = loras

    if is_dual:
        low_noise_path = model_dir / "low_noise_model.safetensors"
        high_noise_path = model_dir / "high_noise_model.safetensors"
        low_noise_model = load_wan_model(
            low_noise_path, config, quantization, loras=_loras_low
        )
        high_noise_model = load_wan_model(
            high_noise_path, config, quantization, loras=_loras_high
        )
    else:
        single_model = load_wan_model(
            model_dir / "model.safetensors", config, quantization, loras=_loras_single
        )
    print(f"{Colors.DIM}  Models loaded: {time.time() - t2:.1f}s{Colors.RESET}")

    # Precompute text embeddings once (avoids redundant MLP in every step)
    # Each model has its own text_embedding weights, so dual models need separate embeddings
    if cfg_disabled:
        # No CFG: only compute cond embeddings (B=1 forward pass, 2x faster)
        if is_dual:
            context_emb_low = low_noise_model.embed_text([context])
            context_emb_high = high_noise_model.embed_text([context])
            mx.eval(context_emb_low, context_emb_high)
            context_cond_low = context_emb_low[0:1]
            context_cond_high = context_emb_high[0:1]
        else:
            context_emb = single_model.embed_text([context])
            mx.eval(context_emb)
            context_cond = context_emb[0:1]
    else:
        if is_dual:
            context_emb_low = low_noise_model.embed_text([context, context_null])
            context_emb_high = high_noise_model.embed_text([context, context_null])
            mx.eval(context_emb_low, context_emb_high)
            context_cfg_low = mx.concatenate(
                [context_emb_low[0:1], context_emb_low[1:2]], axis=0
            )
            context_cfg_high = mx.concatenate(
                [context_emb_high[0:1], context_emb_high[1:2]], axis=0
            )
        else:
            context_emb = single_model.embed_text([context, context_null])
            mx.eval(context_emb)
            context_cfg = mx.concatenate([context_emb[0:1], context_emb[1:2]], axis=0)

    # Precompute cross-attention K/V caches (constant across all steps)
    if cfg_disabled:
        if is_dual:
            cross_kv_low = low_noise_model.prepare_cross_kv(context_cond_low)
            cross_kv_high = high_noise_model.prepare_cross_kv(context_cond_high)
            mx.eval(cross_kv_low, cross_kv_high)
        else:
            cross_kv = single_model.prepare_cross_kv(context_cond)
            mx.eval(cross_kv)
    else:
        if is_dual:
            cross_kv_low = low_noise_model.prepare_cross_kv(context_cfg_low)
            cross_kv_high = high_noise_model.prepare_cross_kv(context_cfg_high)
            mx.eval(cross_kv_low, cross_kv_high)
        else:
            cross_kv = single_model.prepare_cross_kv(context_cfg)
            mx.eval(cross_kv)

    # Precompute RoPE frequencies (grid sizes are constant across all steps)
    f_grid = t_latent // patch_size[0]
    h_grid = h_latent // patch_size[1]
    w_grid = w_latent // patch_size[2]
    if cfg_disabled:
        rope_grid_sizes = [(f_grid, h_grid, w_grid)]
    else:
        rope_grid_sizes = [(f_grid, h_grid, w_grid), (f_grid, h_grid, w_grid)]
    if is_dual:
        rope_cos_sin_low = low_noise_model.prepare_rope(rope_grid_sizes)
        rope_cos_sin_high = high_noise_model.prepare_rope(rope_grid_sizes)
        mx.eval(rope_cos_sin_low, rope_cos_sin_high)
    else:
        rope_cos_sin = single_model.prepare_rope(rope_grid_sizes)
        mx.eval(rope_cos_sin)

    # Setup scheduler
    _schedulers = {
        "euler": FlowMatchEulerScheduler,
        "dpm++": FlowDPMPP2MScheduler,
        "unipc": FlowUniPCScheduler,
    }
    sched_cls = _schedulers.get(scheduler, FlowUniPCScheduler)
    sched = sched_cls(num_train_timesteps=config.num_train_timesteps)
    sched.set_timesteps(steps, shift=shift)

    # Generate initial noise
    noise = mx.random.normal(target_shape)

    # I2V initialization: TI2V-5B blends image with noise, I2V-14B uses pure noise
    if is_i2v_mask_blend:
        latents = (1.0 - i2v_mask) * z_img + i2v_mask * noise
    else:
        latents = noise

    # Boundary for model switching (dual model only)
    boundary = (config.boundary * config.num_train_timesteps) if is_dual else None

    # Diffusion loop
    print(f"\n{Colors.GREEN}Denoising ({steps} steps)...{Colors.RESET}")
    t3 = time.time()

    # Compile model forward for faster denoising
    if not no_compile:
        models_to_compile = (
            [high_noise_model, low_noise_model] if is_dual else [single_model]
        )
        for m in models_to_compile:
            m._compiled = mx.compile(m)

    # Pre-convert timesteps to Python list to avoid .item() sync each step
    timestep_list = sched.timesteps.tolist()

    for i, t in enumerate(tqdm(range(steps), desc="Diffusion")):
        timestep_val = timestep_list[i]

        # Select model, cached K/V, and precomputed RoPE
        if is_dual:
            if timestep_val >= boundary:
                model = high_noise_model
                kv = cross_kv_high
                rcs = rope_cos_sin_high
            else:
                model = low_noise_model
                kv = cross_kv_low
                rcs = rope_cos_sin_low
        else:
            model = single_model
            kv = cross_kv
            rcs = rope_cos_sin

        # Use compiled forward when available (faster after first trace)
        _call = getattr(model, "_compiled", model)

        if cfg_disabled:
            # No CFG: B=1 forward pass (2x faster than B=2 CFG batch)
            if is_i2v_mask_blend:
                t_tokens = i2v_mask_tokens * timestep_val
                pad_len = seq_len - t_tokens.shape[1]
                if pad_len > 0:
                    t_tokens = mx.concatenate(
                        [t_tokens, mx.full((1, pad_len), timestep_val)], axis=1
                    )
                t_batch = t_tokens  # [1, L]
            else:
                t_batch = mx.array([timestep_val])

            y_arg = [y_i2v] if is_i2v_channel_concat else None

            if is_dual:
                ctx = (
                    context_cond_high if timestep_val >= boundary else context_cond_low
                )
            else:
                ctx = context_cond
            preds = _call(
                [latents],
                t=t_batch,
                context=ctx,
                seq_len=seq_len,
                cross_kv_caches=kv,
                y=y_arg,
                rope_cos_sin=rcs,
            )
            noise_pred = preds[0]
            del preds
        else:
            # CFG: batch cond + uncond into single B=2 forward pass
            if is_dual:
                gs = guide_scale[1] if timestep_val >= boundary else guide_scale[0]
            else:
                gs = (
                    guide_scale
                    if isinstance(guide_scale, (int, float))
                    else guide_scale[0]
                )

            if is_i2v_mask_blend:
                t_tokens = i2v_mask_tokens * timestep_val
                pad_len = seq_len - t_tokens.shape[1]
                if pad_len > 0:
                    t_tokens = mx.concatenate(
                        [t_tokens, mx.full((1, pad_len), timestep_val)], axis=1
                    )
                t_batch = mx.concatenate([t_tokens, t_tokens], axis=0)
            else:
                t_batch = mx.array([timestep_val, timestep_val])

            y_arg = [y_i2v, y_i2v] if is_i2v_channel_concat else None

            ctx = (
                context_cfg
                if not is_dual
                else (context_cfg_high if timestep_val >= boundary else context_cfg_low)
            )
            preds = _call(
                [latents, latents],
                t=t_batch,
                context=ctx,
                seq_len=seq_len,
                cross_kv_caches=kv,
                y=y_arg,
                rope_cos_sin=rcs,
            )
            noise_pred_cond, noise_pred_uncond = preds[0], preds[1]
            noise_pred = noise_pred_uncond + gs * (noise_pred_cond - noise_pred_uncond)
            del noise_pred_cond, noise_pred_uncond, preds

        latents = sched.step(noise_pred[None], timestep_val, latents[None]).squeeze(0)

        # TI2V-5B: re-apply mask to keep first frame frozen
        if is_i2v_mask_blend:
            latents = (1.0 - i2v_mask) * z_img + i2v_mask * latents

        # Release temporaries before eval to free memory for graph execution
        del noise_pred
        mx.eval(latents)

    print(f"{Colors.DIM}  Denoising: {time.time() - t3:.1f}s{Colors.RESET}")

    # Diagnostic: per-temporal-position latent statistics
    if debug_latents:
        lat_np = np.array(latents)  # [C, T, H, W]
        n_t = lat_np.shape[1]
        print(
            f"\n{Colors.CYAN}  Latent diagnostics (shape {lat_np.shape}):{Colors.RESET}"
        )
        print(
            f"  {'Pos':>4s}  {'Mean':>8s}  {'Std':>8s}  {'Min':>8s}  {'Max':>8s}  {'AbsMean':>8s}"
        )
        for t_pos in range(min(n_t, 8)):
            frame = lat_np[:, t_pos, :, :]
            print(
                f"  {t_pos:4d}  {frame.mean():8.4f}  {frame.std():8.4f}  "
                f"{frame.min():8.4f}  {frame.max():8.4f}  {np.abs(frame).mean():8.4f}"
            )
        if n_t > 8:
            interior = lat_np[:, 4:, :, :]
            print(
                f"  {'4+':>4s}  {interior.mean():8.4f}  {interior.std():8.4f}  "
                f"{interior.min():8.4f}  {interior.max():8.4f}  {np.abs(interior).mean():8.4f}"
            )
        print()

    # Free transformer models and text embeddings
    if is_dual:
        del low_noise_model, high_noise_model, cross_kv_low, cross_kv_high
        if cfg_disabled:
            del context_cond_low, context_cond_high
        else:
            del context_cfg_low, context_cfg_high
    else:
        del single_model, cross_kv
        if cfg_disabled:
            del context_cond
        else:
            del context_cfg
    del model, kv, context
    if context_null is not None:
        del context_null
    gc.collect()
    mx.clear_cache()

    # Load VAE and decode
    print(f"\n{Colors.BLUE}Decoding with VAE...{Colors.RESET}")
    t4 = time.time()
    vae_path = model_dir / "vae.safetensors"
    vae = load_vae_decoder(vae_path, config)

    is_wan22_vae = config.vae_z_dim == 48

    # Temporal extend: prepend reflected latent frames to the VAE input so that
    # the CausalConv3d zero-padding artifacts fall on the prefix (which we crop).
    # This gives the first real frame a full temporal receptive field of real data.
    # Select tiling configuration
    from mlx_video.models.ltx_2.video_vae.tiling import TilingConfig

    if tiling == "none":
        tiling_config = None
    elif tiling == "auto":
        tiling_config = TilingConfig.auto(height, width, num_frames)
    elif tiling == "default":
        tiling_config = TilingConfig.default()
    elif tiling == "aggressive":
        tiling_config = TilingConfig.aggressive()
    elif tiling == "conservative":
        tiling_config = TilingConfig.conservative()
    elif tiling == "spatial":
        tiling_config = TilingConfig.spatial_only()
    elif tiling == "temporal":
        tiling_config = TilingConfig.temporal_only()
    else:
        print(
            f"{Colors.YELLOW}  Unknown tiling mode '{tiling}', using auto{Colors.RESET}"
        )
        tiling_config = TilingConfig.auto(height, width, num_frames)

    if tiling_config is not None:
        spatial_info = (
            f"{tiling_config.spatial_config.tile_size_in_pixels}px"
            if tiling_config.spatial_config
            else "none"
        )
        temporal_info = (
            f"{tiling_config.temporal_config.tile_size_in_frames}f"
            if tiling_config.temporal_config
            else "none"
        )
        print(
            f"{Colors.DIM}  Tiling ({tiling}): spatial={spatial_info}, temporal={temporal_info}{Colors.RESET}"
        )

    if is_wan22_vae:
        from mlx_video.models.wan_2.vae22 import denormalize_latents

        # latents: [C, T, H, W] → [1, T, H, W, C] (channels-last for Wan2.2 VAE)
        z = latents.transpose(1, 2, 3, 0)[None]
        z = denormalize_latents(z)
        if tiling_config is not None:
            video = vae.decode_tiled(z, tiling_config)
        else:
            video = vae(z)
        mx.eval(video)
        print(f"{Colors.DIM}  VAE decode: {time.time() - t4:.1f}s{Colors.RESET}")

        video = np.array(video[0])  # [T', H', W', 3]
        video = (video + 1.0) / 2.0
        video = np.clip(video * 255.0, 0, 255).astype(np.uint8)
    else:
        if tiling_config is not None:
            video = vae.decode_tiled(latents[None], tiling_config)
        else:
            video = vae.decode(latents[None])
        mx.eval(video)
        print(f"{Colors.DIM}  VAE decode: {time.time() - t4:.1f}s{Colors.RESET}")

        video = np.array(video[0])  # [3, T', H, W]
        video = (video + 1.0) / 2.0
        video = np.clip(video * 255.0, 0, 255).astype(np.uint8)
        video = video.transpose(1, 2, 3, 0)  # [T, H, W, 3]

    # Trim first N temporal chunks if requested (avoids first-frame artifacts)
    if trim_first_frames > 0:
        trim_pixels = trim_first_frames * 4
        video = video[trim_pixels:]
        print(
            f"{Colors.DIM}  Trimmed first {trim_pixels} frames ({video.shape[0]} remaining){Colors.RESET}"
        )

    save_video(video, output_path, fps=config.sample_fps)
    print(f"\n{Colors.GREEN}✓ Video saved to {output_path}{Colors.RESET}")
    print(f"{Colors.DIM}  Total time: {time.time() - t1:.1f}s{Colors.RESET}")


def generate_s2v_video(
    model_dir: str,
    prompt: str,
    image: str,
    audio: str,
    negative_prompt: str | None = None,
    width: int = 512,
    height: int = 512,
    num_frames: int = 81,
    steps: int | None = None,
    guide_scale: float | tuple | None = None,
    shift: float | None = None,
    seed: int = -1,
    output_path: str = "output_s2v.mp4",
    scheduler: str = "unipc",
    tiling: str = "auto",
    no_compile: bool = False,
):
    """End-to-end Wan 2.2 Speech-to-Video generation.

    Args:
        model_dir: converted MLX S2V model directory (must contain
            ``model.safetensors``, ``t5_encoder.safetensors``,
            ``vae.safetensors``, ``config.json``).
        prompt: text prompt describing the scene.
        image: path to the reference (portrait) image.
        audio: path to the driving audio (.wav, mono, 16 kHz preferred).
        negative_prompt: passed to CFG. Defaults to the model's Chinese
            negative prompt when None.
        width/height: video resolution (aligned to VAE stride * patch).
        num_frames: number of pixel frames to render (must be 4n+1).
        steps: diffusion steps. Defaults to ``config.sample_steps``.
        guide_scale: CFG scale. Defaults to ``config.sample_guide_scale``.
        shift: flow-matching shift. Defaults to ``config.sample_shift``.
        seed: RNG seed. -1 = random.
        output_path: MP4 output path.
        scheduler: 'unipc' | 'dpm++' | 'euler'.
        tiling: VAE-decode tiling mode (see ``generate_video``).
        no_compile: skip ``mx.compile`` on the transformer.

    This function is untested end-to-end in the CI sandbox. Run on Mac to
    produce an actual MP4. Known limitations tagged with ``TODO(verify)``:
      * Ref-image RoPE position (design says index 30) is currently *not*
        applied — appended tokens skip RoPE. First-frame results may show
        artifacts until the ref-position RoPE is wired up.
      * Motion-history / framepack is not passed by default here (talking
        head first-clip case). Extend by adding a ``motion_history`` arg.
    """
    import json

    from mlx_video.models.wan_2.config import WanModelConfig
    from mlx_video.models.wan_2.scheduler import (
        FlowDPMPP2MScheduler,
        FlowMatchEulerScheduler,
        FlowUniPCScheduler,
    )
    from mlx_video.models.wan_2.audio_encoder import extract_wav2vec_features

    model_dir = Path(model_dir)

    # ------ config ------
    config_path = model_dir / "config.json"
    quantization = None
    if config_path.exists():
        with open(config_path) as f:
            cfg_dict = json.load(f)
        quantization = cfg_dict.pop("quantization", None)
        for k in (
            "patch_size", "vae_stride", "window_size", "sample_guide_scale",
            "audio_inject_layers",
        ):
            if k in cfg_dict and isinstance(cfg_dict[k], list):
                cfg_dict[k] = tuple(cfg_dict[k])
        config = WanModelConfig(
            **{k: v for k, v in cfg_dict.items()
               if k in WanModelConfig.__dataclass_fields__}
        )
    else:
        config = WanModelConfig.wan22_s2v_14b()

    if config.model_type != "s2v":
        print(f"{Colors.YELLOW}  Overriding config.model_type='{config.model_type}' → 's2v'{Colors.RESET}")
        config = WanModelConfig.wan22_s2v_14b()

    if steps is None:
        steps = config.sample_steps
    if shift is None:
        shift = config.sample_shift
    if guide_scale is None:
        guide_scale = config.sample_guide_scale
    if isinstance(guide_scale, tuple):
        cfg_disabled = all(gs <= 1.0 for gs in guide_scale)
    else:
        cfg_disabled = float(guide_scale) <= 1.0

    if negative_prompt is None:
        neg_prompt_resolved = config.sample_neg_prompt
    else:
        neg_prompt_resolved = negative_prompt

    assert (num_frames - 1) % 4 == 0, f"num_frames must be 4n+1, got {num_frames}"

    print(f"{Colors.CYAN}{'='*60}")
    print(f"  Wan{config.model_version} Speech-to-Video (MLX, Phase-2)")
    print(f"{'='*60}{Colors.RESET}")
    print(f"{Colors.DIM}  Model dir: {model_dir}")
    print(f"  Prompt: {prompt}")
    print(f"  Ref image: {image}")
    print(f"  Audio: {audio}")
    print(f"  Size: {width}x{height}, Frames: {num_frames}")
    print(f"  Steps: {steps}, Guide: {guide_scale}, Shift: {shift}{Colors.RESET}")

    if seed < 0:
        seed = random.randint(0, 2**32 - 1)
    mx.random.seed(seed)
    np.random.seed(seed)
    print(f"{Colors.DIM}  Seed: {seed}{Colors.RESET}")

    # Align dimensions.
    vae_stride = config.vae_stride
    patch_size = config.patch_size
    align_h = patch_size[1] * vae_stride[1]
    align_w = patch_size[2] * vae_stride[2]
    if height % align_h != 0 or width % align_w != 0:
        height = max(align_h, (height // align_h) * align_h)
        width = max(align_w, (width // align_w) * align_w)
        print(f"{Colors.DIM}  Aligned to {width}x{height}{Colors.RESET}")
    if config.max_area > 0 and height * width > config.max_area:
        width, height = _best_output_size(width, height, align_w, align_h, config.max_area)
        print(f"{Colors.YELLOW}  Clamped to {width}x{height}{Colors.RESET}")

    z_dim = config.vae_z_dim
    t_latent = (num_frames - 1) // vae_stride[0] + 1
    h_latent = height // vae_stride[1]
    w_latent = width // vae_stride[2]
    target_shape = (z_dim, t_latent, h_latent, w_latent)
    seq_len = math.ceil((h_latent * w_latent) / (patch_size[1] * patch_size[2]) * t_latent)

    # ------ text encoder ------
    t1 = time.time()
    print(f"\n{Colors.BLUE}Loading T5 encoder...{Colors.RESET}")
    t5_path = model_dir / "t5_encoder.safetensors"
    t5_encoder = load_t5_encoder(t5_path, config)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/umt5-xxl")
    print(f"{Colors.BLUE}Encoding text...{Colors.RESET}")
    context = encode_text(t5_encoder, tokenizer, prompt, config.text_len)
    if cfg_disabled:
        context_null = None
        mx.eval(context)
    else:
        context_null = encode_text(t5_encoder, tokenizer, neg_prompt_resolved, config.text_len)
        mx.eval(context, context_null)
    del t5_encoder
    gc.collect()
    mx.clear_cache()
    print(f"{Colors.DIM}  T5 encoding: {time.time() - t1:.1f}s{Colors.RESET}")

    # ------ VAE encode reference image ------
    print(f"\n{Colors.BLUE}Encoding reference image...{Colors.RESET}")
    vae_path = model_dir / "vae.safetensors"
    # preprocess_image returns channels-LAST (1, 1, H, W, 3), but WanVAE.encode
    # expects channels-FIRST (B, 3, T=1, H, W). Transpose accordingly.
    img_tensor = preprocess_image(image, width, height)  # (1, 1, H, W, 3)
    img_tensor = mx.transpose(img_tensor, (0, 4, 1, 2, 3))  # (1, 3, 1, H, W)
    mx.eval(img_tensor)
    vae_enc = load_vae_encoder(vae_path, config)
    z_ref = vae_enc.encode(img_tensor)  # (1, z_dim, 1, H_lat, W_lat) channel-first
    mx.eval(z_ref)
    # to (z_dim, 1, H_lat, W_lat)
    z_ref = z_ref[0]  # drop batch dim → (z_dim, 1, H_lat, W_lat)
    del vae_enc, img_tensor
    gc.collect()
    mx.clear_cache()

    # ------ wav2vec2 features ------
    print(f"\n{Colors.BLUE}Extracting wav2vec2 features...{Colors.RESET}")
    # Compute the video frames the S2V transformer sees (post-patchify F axis).
    F_vid = t_latent // patch_size[0]
    # Prefer the wav2vec2 bundled with the S2V checkpoint — those weights are
    # 100x smaller magnitude than facebook/wav2vec2-large-xlsr-53 (which is the
    # unfine-tuned base) and produce meaningful audio embeddings.
    w2v_candidates = [
        model_dir / "wav2vec2-large-xlsr-53-english",
        model_dir.parent / "Wan2.2-S2V-14B" / "wav2vec2-large-xlsr-53-english",
        Path.home() / "mlx-video/checkpoints/Wan2.2-S2V-14B/wav2vec2-large-xlsr-53-english",
    ]
    w2v_model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
    for cand in w2v_candidates:
        if cand.exists() and (cand / "config.json").exists():
            w2v_model_name = str(cand)
            break
    print(f"{Colors.DIM}  wav2vec2: {w2v_model_name}{Colors.RESET}")
    audio_feat = extract_wav2vec_features(
        wav_path=audio,
        num_video_frames=F_vid,
        num_audio_token=config.num_audio_token,
        model_name=w2v_model_name,
    )  # (1, 25, 1024, F_vid * num_audio_token)
    mx.eval(audio_feat)

    # ------ transformer ------
    print(f"\n{Colors.BLUE}Loading S2V transformer...{Colors.RESET}")
    t2 = time.time()
    single_model = load_wan_model(
        model_dir / "model.safetensors", config, quantization
    )
    print(f"{Colors.DIM}  Model loaded: {time.time() - t2:.1f}s{Colors.RESET}")

    # Precompute text embeddings + cross-attn K/V.
    if cfg_disabled:
        context_emb = single_model.embed_text([context])
        mx.eval(context_emb)
        context_cond = context_emb[0:1]
        cross_kv = single_model.prepare_cross_kv(context_cond)
    else:
        context_emb = single_model.embed_text([context, context_null])
        mx.eval(context_emb)
        context_cfg = mx.concatenate([context_emb[0:1], context_emb[1:2]], axis=0)
        cross_kv = single_model.prepare_cross_kv(context_cfg)
    mx.eval(cross_kv)

    # Multi-segment RoPE: covers [noise, ref, motion] in a single (cos, sin)
    # so ref gets t=max(30, F+9) and motion buckets t=-1/-3/-19. Verified
    # against kijai wanvideo/modules/model.py::rope_encode_comfy (line 2705).
    f_grid = t_latent // patch_size[0]
    h_grid = h_latent // patch_size[1]
    w_grid = w_latent // patch_size[2]
    # Reference latent grid: single frame at the same H_lat, W_lat.
    ref_grid = (
        (z_ref.shape[1], h_grid, w_grid) if z_ref is not None else None
    )
    # No motion history in first-clip talking-head generation.
    motion_shapes = None
    rope_cos_sin = single_model.prepare_rope_s2v(
        noise_grid=(f_grid, h_grid, w_grid),
        ref_grid=ref_grid,
        motion_shapes=motion_shapes,
    )
    mx.eval(rope_cos_sin)

    # Scheduler
    _schedulers = {
        "euler": FlowMatchEulerScheduler,
        "dpm++": FlowDPMPP2MScheduler,
        "unipc": FlowUniPCScheduler,
    }
    sched_cls = _schedulers.get(scheduler, FlowUniPCScheduler)
    sched = sched_cls(num_train_timesteps=config.num_train_timesteps)
    sched.set_timesteps(steps, shift=shift)

    latents = mx.random.normal(target_shape)

    # ------ diffusion loop ------
    print(f"\n{Colors.GREEN}Denoising ({steps} steps)...{Colors.RESET}")
    t3 = time.time()
    timestep_list = sched.timesteps.tolist()
    for i in tqdm(range(steps), desc="Diffusion"):
        timestep_val = timestep_list[i]
        if cfg_disabled:
            t_batch = mx.array([timestep_val])
            preds = single_model(
                [latents],
                t=t_batch,
                context=context_cond,
                seq_len=seq_len,
                audio_input=audio_feat,
                ref_image_latent=z_ref,
                motion_history_latent=None,
                cross_kv_caches=cross_kv,
                rope_cos_sin=rope_cos_sin,
            )
            noise_pred = preds[0]
        else:
            t_batch = mx.array([timestep_val, timestep_val])
            # Broadcast audio + ref to batch 2.
            audio_b = mx.broadcast_to(audio_feat, (2,) + audio_feat.shape[1:])
            gs = guide_scale if isinstance(guide_scale, (int, float)) else guide_scale[0]
            preds = single_model(
                [latents, latents],
                t=t_batch,
                context=context_cfg,
                seq_len=seq_len,
                audio_input=audio_b,
                ref_image_latent=z_ref,
                motion_history_latent=None,
                cross_kv_caches=cross_kv,
                rope_cos_sin=rope_cos_sin,
            )
            noise_pred_cond, noise_pred_uncond = preds[0], preds[1]
            noise_pred = noise_pred_uncond + gs * (noise_pred_cond - noise_pred_uncond)

        latents = sched.step(noise_pred[None], timestep_val, latents[None]).squeeze(0)
        mx.eval(latents)
    print(f"{Colors.DIM}  Denoising: {time.time() - t3:.1f}s{Colors.RESET}")

    del single_model, cross_kv
    gc.collect()
    mx.clear_cache()

    # ------ VAE decode ------
    print(f"\n{Colors.BLUE}Decoding with VAE...{Colors.RESET}")
    t4 = time.time()
    vae = load_vae_decoder(vae_path, config)
    from mlx_video.models.ltx_2.video_vae.tiling import TilingConfig
    if tiling == "none":
        tiling_config = None
    elif tiling == "auto":
        tiling_config = TilingConfig.auto(height, width, num_frames)
    else:
        tiling_config = TilingConfig.default()

    if tiling_config is not None:
        video = vae.decode_tiled(latents[None], tiling_config)
    else:
        video = vae.decode(latents[None])
    mx.eval(video)
    print(f"{Colors.DIM}  VAE decode: {time.time() - t4:.1f}s{Colors.RESET}")
    video = np.array(video[0])  # (3, T', H, W)
    video = (video + 1.0) / 2.0
    video = np.clip(video * 255.0, 0, 255).astype(np.uint8)
    video = video.transpose(1, 2, 3, 0)  # (T, H, W, 3)

    save_video(video, output_path, fps=config.sample_fps)
    print(f"\n{Colors.GREEN}✓ Video saved to {output_path}{Colors.RESET}")
    print(f"{Colors.DIM}  Total time: {time.time() - t1:.1f}s{Colors.RESET}")


def _dispatch_s2v(args):
    """CLI entry point for Wan 2.2 S2V."""
    guide_scale = None
    if args.guide_scale is not None:
        parts = [float(x) for x in args.guide_scale.split(",")]
        guide_scale = tuple(parts) if len(parts) > 1 else parts[0]

    neg_prompt = args.negative_prompt
    if args.no_negative_prompt:
        neg_prompt = ""

    if not args.image:
        raise ValueError("--image PATH is required for S2V (reference image).")
    if not args.audio:
        raise ValueError("--audio PATH is required for S2V (driving audio).")

    generate_s2v_video(
        model_dir=args.model_dir,
        prompt=args.prompt,
        image=args.image,
        audio=args.audio,
        negative_prompt=neg_prompt,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        steps=args.steps,
        guide_scale=guide_scale,
        shift=args.shift,
        seed=args.seed,
        output_path=args.output_path,
        scheduler=args.scheduler,
        tiling=args.tiling,
        no_compile=args.no_compile,
    )


def main():
    parser = argparse.ArgumentParser(description="Wan Text-to-Video Generation (MLX)")
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to converted MLX model directory",
    )
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to input image for I2V or S2V reference image (omit for T2V mode)",
    )
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Path to input audio (.wav) for S2V speech-to-video mode",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        choices=["t2v", "i2v", "ti2v", "s2v"],
        help="Override auto-detected model type (default: from config.json)",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=None,
        help="Negative prompt for CFG (default: official Chinese prompt from config)",
    )
    parser.add_argument(
        "--no-negative-prompt",
        action="store_true",
        help="Disable negative prompt (use empty string instead of config default)",
    )
    parser.add_argument(
        "--width", type=int, default=1280, help="Video width (default: 1280)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=704,
        help="Video height (default: 704; 720p models use 704)",
    )
    parser.add_argument(
        "--num-frames", type=int, default=81, help="Number of frames (must be 4n+1)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of diffusion steps (default: from config)",
    )
    parser.add_argument(
        "--guide-scale",
        type=str,
        default=None,
        help="Guidance scale: single float or low,high pair",
    )
    parser.add_argument(
        "--shift",
        type=float,
        default=None,
        help="Noise schedule shift (default: from config)",
    )
    parser.add_argument("--seed", type=int, default=-1, help="Random seed")
    parser.add_argument(
        "--output-path", type=str, default="output.mp4", help="Output video path"
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="unipc",
        choices=["euler", "dpm++", "unipc"],
        help="Diffusion solver: euler (1st order), dpm++ (2nd order), unipc (2nd order PC, default/official)",
    )
    parser.add_argument(
        "--lora",
        nargs=2,
        action="append",
        metavar=("PATH", "STRENGTH"),
        help="Apply a LoRA to all models (repeatable). Format: --lora path.safetensors 0.8",
    )
    parser.add_argument(
        "--lora-high",
        nargs=2,
        action="append",
        metavar=("PATH", "STRENGTH"),
        help="Apply a LoRA to high-noise model only (dual-model, repeatable)",
    )
    parser.add_argument(
        "--lora-low",
        nargs=2,
        action="append",
        metavar=("PATH", "STRENGTH"),
        help="Apply a LoRA to low-noise model only (dual-model, repeatable)",
    )
    parser.add_argument(
        "--tiling",
        type=str,
        default="auto",
        choices=[
            "auto",
            "none",
            "default",
            "aggressive",
            "conservative",
            "spatial",
            "temporal",
        ],
        help="VAE tiling mode to reduce memory during decoding (default: auto)",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable mx.compile on models (for debugging)",
    )
    parser.add_argument(
        "--trim-first-frames",
        type=int,
        default=0,
        metavar="N",
        help="Generate N extra temporal chunks (N×4 frames) and discard them from the start. "
        "Fixes first-frame color/lighting artifacts on 14B models. Try 1 first (4 frames). "
        "Default: 0 (disabled)",
    )
    parser.add_argument(
        "--debug-latents",
        action="store_true",
        help="Print per-temporal-position latent statistics after denoising (diagnostic)",
    )
    args = parser.parse_args()

    # Parse guide scale
    guide_scale = None
    if args.guide_scale is not None:
        parts = [float(x) for x in args.guide_scale.split(",")]
        guide_scale = tuple(parts) if len(parts) > 1 else parts[0]

    # Handle negative prompt: --no-negative-prompt forces empty, otherwise pass through
    neg_prompt = args.negative_prompt
    if args.no_negative_prompt:
        neg_prompt = ""

    # Parse LoRA configs: convert [path, strength_str] → (path, float)
    def _parse_lora_args(lora_list):
        if not lora_list:
            return None
        return [(path, float(strength)) for path, strength in lora_list]

    # ---------- S2V dispatch ----------
    # If user passed --audio or --model-type s2v, route to the S2V pipeline
    # (Phase 1: loads the model, then raises NotImplementedError from forward).
    if args.audio is not None or args.model_type == "s2v":
        _dispatch_s2v(args)
        return

    generate_video(
        model_dir=args.model_dir,
        prompt=args.prompt,
        negative_prompt=neg_prompt,
        image=args.image,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        steps=args.steps,
        guide_scale=guide_scale,
        shift=args.shift,
        seed=args.seed,
        output_path=args.output_path,
        scheduler=args.scheduler,
        loras=_parse_lora_args(args.lora),
        loras_high=_parse_lora_args(args.lora_high),
        loras_low=_parse_lora_args(args.lora_low),
        tiling=args.tiling,
        no_compile=args.no_compile,
        trim_first_frames=args.trim_first_frames,
        debug_latents=args.debug_latents,
    )


if __name__ == "__main__":
    main()
