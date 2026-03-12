"""Helios Text-to-Video generation pipeline for MLX.

Autoregressive chunk-based video generation with multi-scale history memory.
Supports the Helios-Distilled model (x0-prediction, no CFG, 2-3 steps/chunk).
"""

import argparse
import gc
import json
import math
import random
import sys
import time
import warnings
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import cv2
from tqdm import tqdm

from mlx_video.models.helios.loading import (
    _clean_text,
    encode_text,
    load_helios_model,
    load_t5_encoder,
    load_vae_decoder,
)
from mlx_video.models.wan.postprocess import save_video
from mlx_video.generate_wan import Colors


def sample_block_noise(
    batch_size: int,
    channels: int,
    num_frames: int,
    height: int,
    width: int,
    patch_size: tuple[int, int, int],
    gamma: float,
) -> mx.array:
    """Generate structured per-patch noise using correlated multivariate normal.

    This reduces block artifacts by ensuring spatially adjacent latents within
    each patch have correlated noise values.

    Returns:
        Noise tensor of shape (channels, num_frames, height, width).
    """
    _, ph, pw = patch_size
    block_size = ph * pw

    # Build covariance matrix: I*(1+gamma) - ones*gamma + eps*I
    cov = np.eye(block_size, dtype=np.float64) * (1 + gamma) - np.ones((block_size, block_size), dtype=np.float64) * gamma
    cov += np.eye(block_size, dtype=np.float64) * 1e-6
    L = np.linalg.cholesky(cov)

    # Sample standard normal and transform
    block_count = batch_size * channels * num_frames * (height // ph) * (width // pw)
    z = np.random.randn(block_count, block_size)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        samples = (z @ L.T).astype(np.float32)

    # Reshape to spatial layout
    samples = samples.reshape(batch_size, channels, num_frames, height // ph, width // pw, ph, pw)
    samples = samples.transpose(0, 1, 2, 3, 5, 4, 6)  # interleave patches
    samples = samples.reshape(batch_size, channels, num_frames, height, width)

    # Return as (C, F, H, W) — drop batch dim since we always use batch=1
    return mx.array(samples[0])


def _spatial_reshape(x: mx.array, num_frames: int, channels: int) -> mx.array:
    """Reshape (C, F, H, W) → (F, C, H, W) for spatial operations."""
    return x.transpose(1, 0, 2, 3)  # (F, C, H, W)


def _spatial_unreshape(
    x: mx.array, num_frames: int, channels: int, h: int, w: int
) -> mx.array:
    """Reshape (F, C, H, W) → (C, F, H, W)."""
    return x.transpose(1, 0, 2, 3)  # (C, F, H, W)


def _bilinear_downsample_2d(x: mx.array, target_h: int, target_w: int) -> mx.array:
    """Bilinear interpolation downsample matching F.interpolate(mode='bilinear').

    For 2× integer downsampling, PyTorch's bilinear interpolation with
    align_corners=False samples at the centers of the output cells using a
    triangular (tent) filter over the 2×2 input neighborhood.  With a scale
    factor of exactly 0.5 this reduces to a weighted average:
        [1/4, 1/4, 1/4, 1/4] — i.e. the same as area averaging.

    Input: (F, C, H, W).
    """
    F, C, H, W = x.shape
    scale_h = H // target_h
    scale_w = W // target_w
    x = x.reshape(F, C, target_h, scale_h, target_w, scale_w)
    x = x.mean(axis=(3, 5))
    return x


def _nearest_upsample_2d(x: mx.array, target_h: int, target_w: int) -> mx.array:
    """Nearest-neighbor 2x upsample. Input: (F, C, H, W)."""
    F, C, H, W = x.shape
    scale_h = target_h // H
    scale_w = target_w // W
    # Repeat along spatial dims
    x = mx.repeat(x, scale_h, axis=2)
    x = mx.repeat(x, scale_w, axis=3)
    return x


def _downsample_history(hist: mx.array, factor: int) -> mx.array:
    """Downsample history latents spatially by factor. Input: (C, F, H, W)."""
    C, F, H, W = hist.shape
    target_h = H // factor
    target_w = W // factor
    hist = hist.reshape(C, F, target_h, factor, target_w, factor)
    hist = hist.mean(axis=(3, 5))
    return hist


def _debug_stats(name: str, x: mx.array) -> str:
    """Return a compact stats string for a tensor."""
    x_f = x.astype(mx.float32)
    return (
        f"{name}: shape={list(x.shape)} dtype={x.dtype} "
        f"mean={x_f.mean().item():.6f} std={x_f.std().item():.6f} "
        f"min={x_f.min().item():.6f} max={x_f.max().item():.6f}"
    )


def generate_video(
    model_dir: str,
    prompt: str,
    width: int = 640,
    height: int = 384,
    num_frames: int = 99,
    pyramid_steps: list[int] | None = None,
    seed: int = -1,
    output_path: str = "output_helios.mp4",
    tiling: str = "auto",
    amplify_first_chunk: bool = True,
    guidance_scale: float = 1.0,
    negative_prompt: str = "",
    chunk_blend: int = 0,
    crossfade_frames: int = 0,
    anti_drifting: bool = False,
    anti_drift_blend: float = 0.5,
    debug: bool = False,
    no_compile: bool = False,
):
    """Generate video using Helios autoregressive pipeline with pyramid denoising.

    Args:
        model_dir: Path to converted MLX model directory
        prompt: Text prompt
        width: Video width (must be divisible by 16)
        height: Video height (must be divisible by 16)
        num_frames: Number of frames (auto-rounded to multiple of 33)
        pyramid_steps: Steps per pyramid stage (default: [2, 2, 2] for distilled)
        seed: Random seed (-1 for random)
        output_path: Output video path
        tiling: VAE tiling mode: auto, none, default, aggressive, conservative
        amplify_first_chunk: Double steps for first chunk (recommended for distilled model)
        guidance_scale: CFG guidance scale (1.0 = no CFG, 5.0 = default)
        negative_prompt: Negative prompt for CFG (empty string = unconditional)
        chunk_blend: Number of latent frames to blend at chunk boundaries (0 to disable)
        crossfade_frames: Number of pixel frames to cross-fade between chunks (0 to disable)
        anti_drifting: Enable adaptive anti-drifting for temporal consistency
        anti_drift_blend: How much to normalize history toward EMA (0=off, 0.5=half, 1.0=full)
        no_compile: If True, skip mx.compile on models (useful for debugging)
    """
    from mlx_video.models.helios.config import HeliosModelConfig

    if pyramid_steps is None:
        pyramid_steps = [2, 2, 2]

    model_dir = Path(model_dir)
    t1 = time.time()

    # Load config
    config_path = model_dir / "config.json"
    quantization = None
    if config_path.exists():
        with open(config_path) as f:
            config_dict = json.load(f)
        quantization = config_dict.pop("quantization", None)
        for key in ("patch_size", "vae_stride", "rope_dim", "history_sizes", "stage_range"):
            if key in config_dict and isinstance(config_dict[key], list):
                config_dict[key] = tuple(config_dict[key]) if key in ("patch_size", "vae_stride", "rope_dim") else config_dict[key]
        config = HeliosModelConfig(**{
            k: v for k, v in config_dict.items()
            if k in HeliosModelConfig.__dataclass_fields__
        })
    else:
        config = HeliosModelConfig.helios_distilled()

    # Frame and dimension alignment
    vae_stride_t, vae_stride_h, vae_stride_w = config.vae_stride
    frames_per_chunk = 33  # (num_latent_frames_per_chunk - 1) * vae_stride_t + 1
    num_latent_per_chunk = config.num_latent_frames_per_chunk  # 9

    # Round num_frames to nearest multiple of frames_per_chunk
    num_chunks = max(1, (num_frames + frames_per_chunk - 1) // frames_per_chunk)
    num_frames = num_chunks * frames_per_chunk
    total_latent_frames = num_chunks * num_latent_per_chunk

    # Align spatial dimensions for pyramid: need latent H,W divisible by
    # 2^(stages-1) * patch = 4*2 = 8, so pixel dims by 8*vae_stride = 64
    num_stages = len(pyramid_steps)
    pyramid_factor = 2 ** (num_stages - 1)  # 4 for 3-stage
    align_h = config.patch_size[1] * pyramid_factor * vae_stride_h  # 2*4*8 = 64
    align_w = config.patch_size[2] * pyramid_factor * vae_stride_w  # 2*4*8 = 64
    height = ((height + align_h - 1) // align_h) * align_h
    width = ((width + align_w - 1) // align_w) * align_w

    h_latent = height // vae_stride_h
    w_latent = width // vae_stride_w

    if seed < 0:
        seed = random.randint(0, 2**32 - 1)
    mx.random.seed(seed)

    print(f"\n{Colors.CYAN}Helios Video Generation{Colors.RESET}")
    print(f"  Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
    print(f"  Resolution: {width}x{height}, {num_frames} frames ({num_chunks} chunks)")
    print(f"  Pyramid steps: {pyramid_steps} ({sum(pyramid_steps)} total/chunk), Seed: {seed}, Guidance: {guidance_scale}")
    if quantization:
        print(f"  Quantization: {quantization['bits']}-bit, group_size={quantization['group_size']}")

    # 1. Load T5 text encoder and encode prompt
    print(f"\n{Colors.BLUE}Loading text encoder...{Colors.RESET}")
    t2 = time.time()
    t5_path = model_dir / "t5_encoder.safetensors"
    tokenizer_path = model_dir / "tokenizer"

    # Try to find tokenizer
    if tokenizer_path.exists():
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("google/umt5-xxl")

    encoder = load_t5_encoder(t5_path, config)
    context = encode_text(encoder, tokenizer, prompt, text_len=config.text_len)
    mx.eval(context)

    do_cfg = guidance_scale > 1.0
    negative_context = None
    if do_cfg:
        negative_context = encode_text(encoder, tokenizer, negative_prompt, text_len=config.text_len)
        mx.eval(negative_context)

    print(f"{Colors.DIM}  T5 encode: {time.time() - t2:.1f}s, tokens: {context.shape[0]}{', CFG enabled' if do_cfg else ''}{Colors.RESET}")

    del encoder
    gc.collect()
    mx.clear_cache()

    # 2. Load transformer
    print(f"\n{Colors.BLUE}Loading Helios transformer...{Colors.RESET}")
    t3 = time.time()
    model_path = model_dir / "model.safetensors"
    model = load_helios_model(model_path, config, quantization=quantization)
    print(f"{Colors.DIM}  Model load: {time.time() - t3:.1f}s{Colors.RESET}")

    # 3. Pre-compute text embeddings and cross-attention KV caches
    context_embedded = model.embed_text([context])
    mx.eval(context_embedded)
    cross_kv_caches = model.prepare_cross_kv(context_embedded)
    mx.eval(*[v for kv in cross_kv_caches for v in kv])

    negative_context_embedded = None
    negative_cross_kv_caches = None
    if do_cfg:
        negative_context_embedded = model.embed_text([negative_context])
        mx.eval(negative_context_embedded)
        negative_cross_kv_caches = model.prepare_cross_kv(negative_context_embedded)
        mx.eval(*[v for kv in negative_cross_kv_caches for v in kv])

    print(f"{Colors.DIM}  Text embedding + KV cache: ready{Colors.RESET}")

    # Compile model for faster inference via kernel fusion
    if not no_compile:
        model._compiled = mx.compile(model)

    # 4. History setup (keep_first_frame=True matching reference)
    history_sizes = config.history_sizes  # [16, 2, 1]
    num_history_frames = sum(history_sizes)  # 19 latent frames of history
    history_latents = mx.zeros((config.in_dim, num_history_frames, h_latent, w_latent))

    # Frame indices with prefix: [prefix | history_long | history_mid | history_1x | current]
    # Reference uses keep_first_frame=True which adds a prefix frame to short history
    total_indices = 1 + sum(history_sizes) + num_latent_per_chunk  # +1 for prefix
    indices = mx.arange(total_indices)
    idx_prefix = indices[:1]                                       # [0]
    idx_long = indices[1:1 + history_sizes[0]]                     # [1..16]
    idx_mid = indices[1 + history_sizes[0]:1 + history_sizes[0] + history_sizes[1]]  # [17..18]
    idx_1x = indices[1 + history_sizes[0] + history_sizes[1]:1 + sum(history_sizes)]  # [19]
    idx_short = mx.concatenate([idx_prefix, idx_1x])               # [0, 19]
    idx_current = indices[1 + sum(history_sizes):]                  # [20..28]

    # 5. Initialize scheduler
    from mlx_video.models.helios.scheduler import HeliosScheduler

    scheduler = HeliosScheduler(
        num_train_timesteps=1000,
        shift=1.0,
        stages=3,
        gamma=1 / 3,
        use_dynamic_shifting=True,
    )

    total_steps = sum(pyramid_steps) * num_chunks
    print(f"\n{Colors.BLUE}Generating {num_chunks} chunks ({sum(pyramid_steps)} steps/chunk, 3-stage pyramid)...{Colors.RESET}")
    all_latent_chunks = []
    total_generated = 0
    image_latents_prefix = None  # Set after first chunk for keep_first_frame

    # Adaptive anti-drifting: EMA of per-channel latent statistics
    drift_global_mean = None
    drift_global_var = None
    drift_rho = 0.9  # EMA momentum

    for chunk_idx in range(num_chunks):
        t_chunk = time.time()
        is_first = chunk_idx == 0

        # Prepare history from accumulated latents (keep_first_frame=True)
        hist_long, hist_mid, hist_1x = mx.split(
            history_latents[:, -num_history_frames:],
            [history_sizes[0], history_sizes[0] + history_sizes[1]],
            axis=1,
        )

        # Prefix is zero for first chunk (no image conditioning), otherwise first frame
        if is_first:
            latents_prefix = mx.zeros((config.in_dim, 1, h_latent, w_latent))
        else:
            latents_prefix = image_latents_prefix

        # Short history = prefix + 1x history (2 frames)
        hist_short = mx.concatenate([latents_prefix, hist_1x], axis=1)

        # Initialize noise for this chunk at full resolution
        noise = mx.random.normal((config.in_dim, num_latent_per_chunk, h_latent, w_latent))

        # Downsample to 1/4 resolution (2 halvings for 3-stage pyramid)
        cur_h, cur_w = h_latent, w_latent
        latents = _spatial_reshape(noise, num_latent_per_chunk, config.in_dim)
        for _ in range(scheduler.stages - 1):
            cur_h //= 2
            cur_w //= 2
            latents = _bilinear_downsample_2d(latents, cur_h, cur_w) * 2
        latents = _spatial_unreshape(latents, num_latent_per_chunk, config.in_dim, cur_h, cur_w)

        # Track per-stage start points for DMD re-noising
        start_point_list = [latents]

        if debug:
            mx.eval(latents)
            print(f"\n[DEBUG] Chunk {chunk_idx}: initial noise → 1/4 res")
            print(f"  {_debug_stats('start_point[0]', latents)}")

        is_amplified = amplify_first_chunk and is_first
        total_steps = sum(s * 2 if is_amplified else s for s in pyramid_steps)
        pbar = tqdm(
            total=total_steps,
            desc=f"  Chunk {chunk_idx + 1}/{num_chunks}",
            leave=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

        for i_s in range(scheduler.stages):
            # Compute image_seq_len at current resolution for dynamic shift
            image_seq_len = (
                num_latent_per_chunk * cur_h * cur_w
                // math.prod(config.patch_size)
            )

            scheduler.set_timesteps(
                pyramid_steps[i_s],
                stage_index=i_s,
                image_seq_len=image_seq_len,
                is_amplify_first_chunk=(amplify_first_chunk and is_first),
            )
            timesteps = scheduler.timesteps

            if debug:
                mx.eval(latents)
                print(f"\n[DEBUG] Stage {i_s}: res={cur_h}x{cur_w}, seq_len={image_seq_len}")
                print(f"  sigmas: {[f'{s:.6f}' for s in scheduler.sigmas.tolist()]}")
                print(f"  timesteps: {[f'{t:.1f}' for t in timesteps.tolist()]}")
                print(f"  {_debug_stats('latents_in', latents)}")

            if i_s > 0:
                # Upsample 2x with nearest-neighbor
                cur_h *= 2
                cur_w *= 2
                latents = _spatial_reshape(latents, num_latent_per_chunk, config.in_dim)
                latents = _nearest_upsample_2d(latents, cur_h, cur_w)
                latents = _spatial_unreshape(latents, num_latent_per_chunk, config.in_dim, cur_h, cur_w)

                # Alpha/beta noise mixing to reduce block artifacts
                ori_sigma = 1 - scheduler.ori_start_sigmas[i_s]
                gamma = scheduler.gamma
                alpha = 1 / (math.sqrt(1 + (1 / gamma)) * (1 - ori_sigma) + ori_sigma)
                beta = alpha * (1 - ori_sigma) / math.sqrt(gamma)

                block_noise = sample_block_noise(
                    1, config.in_dim, num_latent_per_chunk, cur_h, cur_w,
                    config.patch_size, gamma,
                )
                latents = alpha * latents + beta * block_noise
                start_point_list.append(latents)

                if debug:
                    mx.eval(latents)
                    print(f"  After upsample+mix: alpha={alpha:.4f} beta={beta:.4f} ori_sigma={ori_sigma:.4f}")
                    print(f"  {_debug_stats('start_point[' + str(i_s) + ']', latents)}")

            # History is always passed at full resolution — the Conv3d
            # patchifiers handle the spatial mismatch between history and
            # current latents since they are concatenated in sequence dim.
            h_short, h_mid, h_long = hist_short, hist_mid, hist_long

            # Scale frame indices to match current spatial resolution
            cur_idx = idx_current  # [20..28] with prefix offset
            cur_idx_short = idx_short
            cur_idx_mid = idx_mid
            cur_idx_long = idx_long

            _call = getattr(model, '_compiled', model)

            for idx, t in enumerate(timesteps):
                # Reference casts timestep to int64 before model call
                timestep = mx.array(int(t.item()), dtype=mx.int32)
                # Cast to bfloat16 to match reference (model trained with bf16 activations)
                noise_pred = _call(
                    latents=latents.astype(mx.bfloat16),
                    timestep=timestep,
                    encoder_hidden_states=context_embedded,
                    frame_indices=cur_idx,
                    history_short=h_short.astype(mx.bfloat16),
                    history_mid=h_mid.astype(mx.bfloat16),
                    history_long=h_long.astype(mx.bfloat16),
                    history_short_indices=cur_idx_short,
                    history_mid_indices=cur_idx_mid,
                    history_long_indices=cur_idx_long,
                    cross_kv_caches=cross_kv_caches,
                )
                mx.eval(noise_pred)

                if debug:
                    sigma_t = scheduler.sigmas[idx].item()
                    print(f"\n  [Step {idx}] t={int(t.item())} sigma={sigma_t:.6f}")
                    print(f"    {_debug_stats('model_in', latents)}")
                    print(f"    {_debug_stats('noise_pred', noise_pred)}")

                if do_cfg:
                    noise_uncond = _call(
                        latents=latents.astype(mx.bfloat16),
                        timestep=timestep,
                        encoder_hidden_states=negative_context_embedded,
                        frame_indices=cur_idx,
                        history_short=h_short.astype(mx.bfloat16),
                        history_mid=h_mid.astype(mx.bfloat16),
                        history_long=h_long.astype(mx.bfloat16),
                        history_short_indices=cur_idx_short,
                        history_mid_indices=cur_idx_mid,
                        history_long_indices=cur_idx_long,
                        cross_kv_caches=negative_cross_kv_caches,
                    )
                    mx.eval(noise_uncond)
                    noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

                latents = scheduler.step_dmd(
                    model_output=noise_pred,
                    sample=latents,
                    cur_step=idx,
                    noisy_start=start_point_list[i_s],
                )
                mx.eval(latents)

                if debug:
                    print(f"    {_debug_stats('latents_out', latents)}")

                pbar.update(1)

            if debug:
                mx.eval(latents)
                print(f"\n[DEBUG] Stage {i_s} complete:")
                print(f"  {_debug_stats('stage_output', latents)}")

        pbar.close()
        mx.eval(latents)

        # Adaptive anti-drifting: normalize history latent statistics to prevent
        # color/style drift between chunks. Clean latents are kept for decoding;
        # only the history copy is normalized toward the running EMA.
        history_latents_chunk = latents  # default: same as output
        if anti_drifting and num_chunks > 1:
            lat_f32 = latents.astype(mx.float32)
            # Per-channel stats: latents is [C, F, H, W]
            cur_mean = mx.mean(lat_f32, axis=(1, 2, 3))  # [C]
            cur_var = mx.var(lat_f32, axis=(1, 2, 3))  # [C]
            mx.eval(cur_mean, cur_var)

            if drift_global_mean is None:
                drift_global_mean = cur_mean
                drift_global_var = cur_var
            else:
                # Update EMA BEFORE detection (matching reference order)
                drift_global_mean = drift_rho * drift_global_mean + (1 - drift_rho) * cur_mean
                drift_global_var = drift_rho * drift_global_var + (1 - drift_rho) * cur_var

                # Detect drift: L2 norm of deviation from updated EMA
                mean_drift = float(mx.sqrt(mx.sum((cur_mean - drift_global_mean) ** 2)).item())
                var_drift = float(mx.sqrt(mx.sum((cur_var - drift_global_var) ** 2)).item())
                has_drift = mean_drift > 0.15 and var_drift > 0.15

                if has_drift and chunk_idx < num_chunks - 1:
                    # Normalize history copy toward EMA (deterministic, no noise)
                    # Per-channel: shift mean and scale variance
                    cur_mean_4d = cur_mean[:, None, None, None]
                    cur_std_4d = mx.sqrt(mx.maximum(cur_var, mx.array(1e-8)))[:, None, None, None]
                    global_mean_4d = drift_global_mean[:, None, None, None]
                    global_std_4d = mx.sqrt(mx.maximum(drift_global_var, mx.array(1e-8)))[:, None, None, None]

                    # Standardize, then rescale to target stats
                    normalized = (latents - cur_mean_4d) / cur_std_4d * global_std_4d + global_mean_4d
                    # Blend: 0 = keep raw, 1 = fully normalize to EMA
                    history_latents_chunk = (1 - anti_drift_blend) * latents + anti_drift_blend * normalized
                    history_latents_chunk = history_latents_chunk.astype(latents.dtype)
                    mx.eval(history_latents_chunk)
                    print(f"{Colors.DIM}  ⚠ Drift detected (mean={mean_drift:.3f}, var={var_drift:.3f}), normalized history{Colors.RESET}")
                elif debug:
                    print(f"  [drift] mean={mean_drift:.3f}, var={var_drift:.3f}, threshold=0.15")

        all_latent_chunks.append(latents)  # clean latents for decoding

        # Update history: use potentially normalized chunk for conditioning
        total_generated += num_latent_per_chunk
        history_latents = mx.concatenate([history_latents, history_latents_chunk], axis=1)

        # After first chunk, save first frame as prefix for subsequent chunks
        if is_first and image_latents_prefix is None:
            image_latents_prefix = latents[:, 0:1, :, :]

        chunk_time = time.time() - t_chunk
        step_count = sum(pyramid_steps)
        print(f"{Colors.DIM}  Chunk {chunk_idx + 1}/{num_chunks} done: {chunk_time:.1f}s ({chunk_time / step_count:.2f}s/step){Colors.RESET}")

    # Free transformer
    del model
    gc.collect()
    mx.clear_cache()

    # 6. VAE decode
    print(f"\n{Colors.BLUE}Decoding with VAE...{Colors.RESET}")
    t4 = time.time()
    vae_path = model_dir / "vae.safetensors"
    vae = load_vae_decoder(vae_path, config)

    # Select tiling config
    from mlx_video.models.ltx.video_vae.tiling import TilingConfig

    if tiling == "none":
        tiling_config = None
    elif tiling == "auto":
        tiling_config = TilingConfig.auto(height, width, frames_per_chunk)
    elif tiling == "default":
        tiling_config = TilingConfig.default()
    elif tiling == "aggressive":
        tiling_config = TilingConfig.aggressive()
    elif tiling == "conservative":
        tiling_config = TilingConfig.conservative()
    else:
        tiling_config = TilingConfig.auto(height, width, frames_per_chunk)

    # Optional: smooth chunk boundaries in latent space (off by default).
    # When enabled, blends first N latent frames of each new chunk toward
    # the previous chunk's last frame to reduce quality discontinuity.
    if chunk_blend > 0 and num_chunks > 1:
        blend_n = min(chunk_blend, num_latent_per_chunk - 1)
        for b in range(1, num_chunks):
            ref_np = np.array(all_latent_chunks[b - 1][:, -1])  # [C, H, W]
            chunk_np = np.array(all_latent_chunks[b])  # [C, F, H, W]
            for k in range(min(blend_n, chunk_np.shape[1])):
                target = chunk_np[:, k]
                ref_weight = 0.4 * (blend_n - k) / blend_n
                blended = (1 - ref_weight) * target + ref_weight * ref_np
                for c in range(blended.shape[0]):
                    blended[c] += target[c].mean() - blended[c].mean()
                chunk_np[:, k] = blended
            all_latent_chunks[b] = mx.array(chunk_np)
        print(f"{Colors.DIM}  Applied chunk boundary blend ({blend_n} latent frames){Colors.RESET}")

    # Decode each chunk independently (matching reference behavior).
    # Per-chunk decoding avoids cross-chunk VAE temporal convolution artifacts
    # that occur when the quality discontinuity at boundaries hits the causal conv.
    video_chunks = []
    for ci, chunk_latents in enumerate(all_latent_chunks):
        z = chunk_latents[None, :, :, :, :]  # [1, C, 9, H_lat, W_lat]
        if tiling_config is not None:
            chunk_video = vae.decode_tiled(z, tiling_config)
        else:
            chunk_video = vae.decode(z)
        mx.eval(chunk_video)

        chunk_np = np.array(chunk_video[0])  # [3, T_decoded, H, W]
        # Trim VAE warmup frames (causal padding produces stride_t-1 garbage at start)
        valid = (num_latent_per_chunk - 1) * vae_stride_t + 1  # 33
        trim = chunk_np.shape[1] - valid
        if trim > 0:
            chunk_np = chunk_np[:, trim:]
        # Drop first pixel frame: it's the overlap/conditioning frame from history
        # (distorted duplicate of previous chunk's last frame). 33→32 = exact 2s at 16fps.
        chunk_np = chunk_np[:, 1:]
        video_chunks.append(chunk_np)

        del chunk_video, z
        gc.collect()
        mx.clear_cache()

    print(f"{Colors.DIM}  VAE decode: {time.time() - t4:.1f}s{Colors.RESET}")

    # Correct brightness/contrast discontinuity at chunk boundaries caused by VAE
    # causal padding warmup. Two-stage correction:
    # 1. Spatially-varying brightness: match low-frequency (blurred) brightness per
    #    channel to the previous chunk's last frame, fixing the "face darkens while
    #    background brightens" effect from the VAE's spatial redistribution.
    # 2. Per-channel contrast: scale std dev to match, fixing the ~7% contrast drop.
    if len(video_chunks) > 1:
        blend_n = 6  # frames over which to ramp correction
        blur_size = 16  # downscale factor for low-frequency brightness map
        for i in range(1, len(video_chunks)):
            ref_frame = video_chunks[i - 1][:, -1]  # [3, H, W]
            _, fh, fw = ref_frame.shape
            # Pre-compute low-frequency brightness map of reference
            small_h, small_w = max(fh // blur_size, 1), max(fw // blur_size, 1)
            ref_lf = np.zeros((3, small_h, small_w), dtype=np.float32)
            for c in range(3):
                ref_lf[c] = cv2.resize(ref_frame[c], (small_w, small_h), interpolation=cv2.INTER_AREA)
            # Per-channel global stats
            ref_std = ref_frame.std(axis=(1, 2), keepdims=True)

            for k in range(min(blend_n, video_chunks[i].shape[1])):
                frame = video_chunks[i][:, k]
                ramp = 1.0 - k / blend_n  # 1.0 → 0.0

                # Stage 1: spatially-varying brightness correction
                for c in range(3):
                    cur_lf = cv2.resize(frame[c], (small_w, small_h), interpolation=cv2.INTER_AREA)
                    diff_small = ref_lf[c] - cur_lf
                    diff_full = cv2.resize(diff_small, (fw, fh), interpolation=cv2.INTER_LINEAR)
                    frame[c] = frame[c] + ramp * diff_full

                # Stage 2: per-channel contrast correction
                cur_std = frame.std(axis=(1, 2), keepdims=True)
                cur_std = np.maximum(cur_std, 1e-6)
                target_std = cur_std + ramp * (ref_std - cur_std)
                cur_mean = frame.mean(axis=(1, 2), keepdims=True)
                video_chunks[i][:, k] = (frame - cur_mean) * (target_std / cur_std) + cur_mean

    # Pixel-space cross-fade at chunk boundaries to smooth transitions.
    # Unlike latent-space blending, this is clean — no grid artifacts since
    # the VAE decode has already resolved block noise patterns.
    if crossfade_frames > 0 and len(video_chunks) > 1:
        cf = min(crossfade_frames, video_chunks[0].shape[1] - 1)
        for i in range(1, len(video_chunks)):
            for k in range(cf):
                # Linear ramp: weight 1→0 for previous chunk, 0→1 for current
                w = (k + 1) / (cf + 1)
                video_chunks[i][:, k] = (1 - w) * video_chunks[i - 1][:, -(cf - k)] + w * video_chunks[i][:, k]
        print(f"{Colors.DIM}  Applied pixel cross-fade ({cf} frames at each boundary){Colors.RESET}")

    # Concatenate pixel frames from all chunks
    video = np.concatenate(video_chunks, axis=1)  # [3, T_total, H, W]

    video = (video + 1.0) / 2.0
    video = np.clip(video * 255.0, 0, 255).astype(np.uint8)
    video = video.transpose(1, 2, 3, 0)  # [T, H, W, 3]

    # Trim to requested frame count
    video = video[:num_frames]

    save_video(video, output_path, fps=config.sample_fps)
    print(f"\n{Colors.GREEN}✓ Video saved to {output_path}{Colors.RESET}")
    print(f"{Colors.DIM}  Total time: {time.time() - t1:.1f}s{Colors.RESET}")


def main():
    parser = argparse.ArgumentParser(description="Helios Text-to-Video Generation (MLX)")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to converted MLX model directory")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--width", type=int, default=640, help="Video width")
    parser.add_argument("--height", type=int, default=384, help="Video height")
    parser.add_argument("--num-frames", type=int, default=99, help="Number of frames (auto-rounded to multiple of 33)")
    parser.add_argument(
        "--pyramid-steps", type=int, nargs="+", default=[2, 2, 2],
        help="Steps per pyramid stage (default: 2 2 2 for distilled, total 6 forward passes)",
    )
    parser.add_argument("--amplify-first-chunk", action="store_true", default=True, help="Double steps for first chunk (default: on, recommended for distilled)")
    parser.add_argument("--no-amplify-first-chunk", action="store_false", dest="amplify_first_chunk", help="Disable first chunk amplification")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed")
    parser.add_argument("--output-path", type=str, default="output_helios.mp4", help="Output video path")
    parser.add_argument(
        "--tiling", type=str, default="auto",
        choices=["auto", "none", "default", "aggressive", "conservative"],
        help="VAE tiling mode for memory efficiency",
    )
    parser.add_argument("--guidance-scale", type=float, default=1.0, help="CFG guidance scale (1.0 = no CFG, default for distilled)")
    parser.add_argument("--negative-prompt", type=str, default="", help="Negative prompt for CFG")
    parser.add_argument("--chunk-blend", type=int, default=0, help="Latent frames to blend at chunk boundaries (0=off, default=0)")
    parser.add_argument("--crossfade-frames", type=int, default=0, help="Pixel frames to cross-fade between chunks (0=off, default=0)")
    parser.add_argument("--anti-drifting", action="store_true", help="Enable adaptive anti-drifting for temporal consistency between chunks")
    parser.add_argument("--anti-drift-blend", type=float, default=0.5, help="How much to normalize history toward EMA stats (0=off, 0.5=half, 1.0=full; default=0.5)")
    parser.add_argument("--debug", action="store_true", help="Print per-step latent statistics for debugging")
    parser.add_argument("--no-compile", action="store_true", help="Disable mx.compile on models (for debugging)")
    args = parser.parse_args()

    generate_video(
        model_dir=args.model_dir,
        prompt=args.prompt,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        pyramid_steps=args.pyramid_steps,
        seed=args.seed,
        output_path=args.output_path,
        tiling=args.tiling,
        amplify_first_chunk=args.amplify_first_chunk,
        guidance_scale=args.guidance_scale,
        negative_prompt=args.negative_prompt,
        chunk_blend=args.chunk_blend,
        crossfade_frames=args.crossfade_frames,
        anti_drifting=args.anti_drifting,
        anti_drift_blend=args.anti_drift_blend,
        debug=args.debug,
        no_compile=args.no_compile,
    )


if __name__ == "__main__":
    main()
