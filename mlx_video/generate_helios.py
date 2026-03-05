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
from tqdm import tqdm

from mlx_video.models.helios.loading import (
    _clean_text,
    encode_text,
    load_helios_model,
    load_t5_encoder,
    load_vae_decoder,
)
from mlx_video.postprocess import save_video
from mlx_video.utils import Colors


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
    """Bilinear interpolation downsample. Input: (F, C, H, W)."""
    F, C, H, W = x.shape
    # MLX doesn't have F.interpolate — use manual bilinear via grid sampling
    # Simple approach: reshape to (F*C, 1, H, W) and use average pooling
    scale_h = H // target_h
    scale_w = W // target_w
    # Use reshape-based area averaging (equivalent to bilinear for integer factors)
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
    amplify_first_chunk: bool = False,
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
        amplify_first_chunk: Double steps for first chunk (better quality)
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

    # Align spatial dimensions
    align_h = config.patch_size[1] * vae_stride_h  # 2 * 8 = 16
    align_w = config.patch_size[2] * vae_stride_w  # 2 * 8 = 16
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
    print(f"  Pyramid steps: {pyramid_steps} ({sum(pyramid_steps)} total/chunk), Seed: {seed}")
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
    print(f"{Colors.DIM}  T5 encode: {time.time() - t2:.1f}s, tokens: {context.shape[0]}{Colors.RESET}")

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
    print(f"{Colors.DIM}  Text embedding + KV cache: ready{Colors.RESET}")

    # 4. History setup
    history_sizes = config.history_sizes  # [16, 2, 1]
    num_history_frames = sum(history_sizes)  # 19 latent frames of history
    history_latents = mx.zeros((config.in_dim, num_history_frames, h_latent, w_latent))

    # Frame indices: [history_long | history_mid | history_short | current]
    total_indices = sum(history_sizes) + num_latent_per_chunk
    indices = mx.arange(total_indices)
    idx_long = indices[:history_sizes[0]]  # [0..15]
    idx_mid = indices[history_sizes[0]:history_sizes[0] + history_sizes[1]]  # [16..17]
    idx_short = indices[history_sizes[0] + history_sizes[1]:sum(history_sizes)]  # [18]
    idx_current = indices[sum(history_sizes):]  # [19..27]

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

    for chunk_idx in range(num_chunks):
        t_chunk = time.time()
        is_first = chunk_idx == 0

        # Prepare history from accumulated latents
        hist_long, hist_mid, hist_short = mx.split(
            history_latents[:, -num_history_frames:],
            [history_sizes[0], history_sizes[0] + history_sizes[1]],
            axis=1,
        )

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

        pbar = tqdm(
            total=sum(pyramid_steps),
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

            # Recompute history at current resolution
            ds_factor = h_latent // cur_h
            if ds_factor > 1:
                h_short = _downsample_history(hist_short, ds_factor)
                h_mid = _downsample_history(hist_mid, ds_factor)
                h_long = _downsample_history(hist_long, ds_factor)
            else:
                h_short, h_mid, h_long = hist_short, hist_mid, hist_long

            # Scale frame indices to match current spatial resolution
            cur_idx = mx.arange(num_latent_per_chunk) + sum(history_sizes)
            cur_idx_short = idx_short
            cur_idx_mid = idx_mid
            cur_idx_long = idx_long

            for idx, t in enumerate(timesteps):
                timestep = t
                model_output = model(
                    latents=latents,
                    timestep=timestep,
                    encoder_hidden_states=context_embedded,
                    frame_indices=cur_idx,
                    history_short=h_short,
                    history_mid=h_mid,
                    history_long=h_long,
                    history_short_indices=cur_idx_short,
                    history_mid_indices=cur_idx_mid,
                    history_long_indices=cur_idx_long,
                    cross_kv_caches=cross_kv_caches,
                )
                mx.eval(model_output)

                latents = scheduler.step_dmd(
                    model_output=model_output,
                    sample=latents,
                    cur_step=idx,
                    noisy_start=start_point_list[i_s],
                )
                pbar.update(1)

        pbar.close()
        mx.eval(latents)
        all_latent_chunks.append(latents)

        # Update history: append this chunk's latents
        total_generated += num_latent_per_chunk
        history_latents = mx.concatenate([history_latents, latents], axis=1)

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
        tiling_config = TilingConfig.auto(height, width, num_frames)
    elif tiling == "default":
        tiling_config = TilingConfig.default()
    elif tiling == "aggressive":
        tiling_config = TilingConfig.aggressive()
    elif tiling == "conservative":
        tiling_config = TilingConfig.conservative()
    else:
        tiling_config = TilingConfig.auto(height, width, num_frames)

    # Concatenate all chunks: each is [C, F_lat, H_lat, W_lat]
    all_latents = mx.concatenate(all_latent_chunks, axis=1)  # [C, total_F_lat, H_lat, W_lat]

    # Decode: WanVAE expects [B, C, T, H, W], handles denormalization internally
    z = all_latents[None, :, :, :, :]  # [1, C, T, H, W]
    if tiling_config is not None:
        video = vae.decode_tiled(z, tiling_config)
    else:
        video = vae.decode(z)
    mx.eval(video)
    print(f"{Colors.DIM}  VAE decode: {time.time() - t4:.1f}s{Colors.RESET}")

    # Convert to numpy: video is [1, 3, T, H, W] in [-1, 1]
    video = np.array(video[0])  # [3, T, H, W]
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
    parser.add_argument("--amplify-first-chunk", action="store_true", help="Double steps for first chunk (better quality)")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed")
    parser.add_argument("--output-path", type=str, default="output_helios.mp4", help="Output video path")
    parser.add_argument(
        "--tiling", type=str, default="auto",
        choices=["auto", "none", "default", "aggressive", "conservative"],
        help="VAE tiling mode for memory efficiency",
    )
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
    )


if __name__ == "__main__":
    main()
