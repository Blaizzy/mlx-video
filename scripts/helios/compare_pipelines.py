#!/usr/bin/env python3
"""Compare Helios pipeline mechanics: PyTorch reference vs MLX.

Runs both schedulers with identical inputs and fixed dummy model outputs to
isolate pipeline logic differences (downsampling, upsampling, alpha/beta
blending, DMD stepping). No model weights needed.

This was used to verify that the MLX scheduler produces numerically identical
results to the PyTorch reference, ruling out pipeline mechanics as a source of
output quality differences.

Requirements:
    - MLX video package (this repo)
    - Reference Helios repo on sys.path (--helios-dir)
    - PyTorch + diffusers

Usage:
    python scripts/helios/compare_pipelines.py \
        --helios-dir /path/to/Helios

    # Custom parameters
    python scripts/helios/compare_pipelines.py \
        --helios-dir /path/to/Helios \
        --seed 123 --stages 3 --steps 2 2 2
"""

import argparse
import math
import sys

import numpy as np


def calculate_shift(
    image_seq_len,
    base_seq_len=256,
    max_seq_len=4096,
    base_shift=0.5,
    max_shift=1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def main():
    parser = argparse.ArgumentParser(description="Compare pipeline mechanics")
    parser.add_argument("--helios-dir", required=True, help="Path to reference Helios repo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stages", type=int, default=3)
    parser.add_argument("--steps", type=int, nargs="+", default=[2, 2, 2])
    args = parser.parse_args()

    sys.path.insert(0, args.helios_dir)
    sys.path.insert(0, ".")

    import torch
    import torch.nn.functional as F
    import mlx.core as mx

    from helios.diffusers_version.scheduling_helios_diffusers import (
        HeliosScheduler as PTScheduler,
    )
    from mlx_video.generate_helios import (
        _bilinear_downsample_2d,
        _nearest_upsample_2d,
        _spatial_reshape,
        _spatial_unreshape,
    )
    from mlx_video.models.helios.scheduler import HeliosScheduler as MLXScheduler

    C, NL, H, W = 16, 9, 48, 80
    PATCH_SIZE = (1, 2, 2)
    GAMMA = 1 / 3
    STAGES = args.stages
    PYRAMID_STEPS = args.steps

    # Create identical initial noise from numpy
    rng = np.random.RandomState(args.seed)
    noise_np = rng.randn(C, NL, H, W).astype(np.float32)
    mx_latents = mx.array(noise_np)
    pt_latents = torch.from_numpy(noise_np).unsqueeze(0)

    print(f"Initial noise: mean={noise_np.mean():.6f} std={noise_np.std():.6f}")

    # Downsample — MLX
    cur_h, cur_w = H, W
    mx_flat = _spatial_reshape(mx_latents, NL, C)
    for _ in range(STAGES - 1):
        cur_h //= 2
        cur_w //= 2
        mx_flat = _bilinear_downsample_2d(mx_flat, cur_h, cur_w) * 2
    mx_latents = _spatial_unreshape(mx_flat, NL, C, cur_h, cur_w)
    mx.eval(mx_latents)

    # Downsample — PyTorch
    cur_h_pt, cur_w_pt = H, W
    pt_flat = pt_latents.permute(0, 2, 1, 3, 4).reshape(NL, C, H, W)
    for _ in range(STAGES - 1):
        cur_h_pt //= 2
        cur_w_pt //= 2
        pt_flat = F.interpolate(pt_flat, size=(cur_h_pt, cur_w_pt), mode="bilinear") * 2
    pt_latents = pt_flat.reshape(1, NL, C, cur_h_pt, cur_w_pt).permute(0, 2, 1, 3, 4)

    mx_np = np.array(mx_latents)
    pt_np = pt_latents.squeeze(0).numpy()
    diff = np.abs(mx_np - pt_np)
    print(f"After downsample to {cur_h}×{cur_w}: diff max={diff.max():.8f} mean={diff.mean():.8f}")

    # Initialize schedulers
    mlx_sched = MLXScheduler(
        num_train_timesteps=1000, shift=1.0, stages=STAGES,
        gamma=GAMMA, use_dynamic_shifting=True,
    )
    pt_sched = PTScheduler(
        num_train_timesteps=1000, shift=1.0, stages=STAGES,
        gamma=GAMMA, use_dynamic_shifting=True, scheduler_type="dmd",
    )

    mx_start_points = [mx_latents]
    pt_start_points = [pt_latents.clone()]
    max_diff = 0.0

    for i_s in range(STAGES):
        seq_len = (NL * cur_h * cur_w) // math.prod(PATCH_SIZE)
        mu = calculate_shift(seq_len)

        mlx_sched.set_timesteps(PYRAMID_STEPS[i_s], stage_index=i_s, image_seq_len=seq_len)
        pt_sched.set_timesteps(PYRAMID_STEPS[i_s], i_s, device="cpu", mu=mu)

        print(f"\nStage {i_s}: {cur_h}×{cur_w}, seq_len={seq_len}")
        print(f"  MLX sigmas:     {mlx_sched.sigmas.tolist()}")
        print(f"  PT  sigmas:     {pt_sched.sigmas.tolist()}")

        if i_s > 0:
            cur_h *= 2
            cur_w *= 2
            cur_h_pt *= 2
            cur_w_pt *= 2

            # Upsample
            mx_flat = _spatial_reshape(mx_latents, NL, C)
            mx_flat = _nearest_upsample_2d(mx_flat, cur_h, cur_w)
            mx_latents = _spatial_unreshape(mx_flat, NL, C, cur_h, cur_w)

            pt_flat = pt_latents.permute(0, 2, 1, 3, 4).reshape(NL, C, cur_h_pt // 2, cur_w_pt // 2)
            pt_flat = F.interpolate(pt_flat, size=(cur_h_pt, cur_w_pt), mode="nearest")
            pt_latents = pt_flat.reshape(1, NL, C, cur_h_pt, cur_w_pt).permute(0, 2, 1, 3, 4)

            # Alpha/beta blending with same noise
            ori_sigma = 1 - mlx_sched.ori_start_sigmas[i_s]
            alpha = 1 / (math.sqrt(1 + (1 / GAMMA)) * (1 - ori_sigma) + ori_sigma)
            beta = alpha * (1 - ori_sigma) / math.sqrt(GAMMA)

            noise_np2 = np.random.RandomState(args.seed + i_s * 1000).randn(
                C, NL, cur_h, cur_w
            ).astype(np.float32)
            mx_latents = alpha * mx_latents + beta * mx.array(noise_np2)
            pt_latents = alpha * pt_latents + beta * torch.from_numpy(noise_np2).unsqueeze(0)

            mx.eval(mx_latents)
            mx_start_points.append(mx_latents)
            pt_start_points.append(pt_latents.clone())

            d = np.abs(np.array(mx_latents) - pt_latents.squeeze(0).numpy())
            print(f"  After upsample+mix: diff max={d.max():.8f} mean={d.mean():.8f}")

        # DMD steps with fixed model output
        for idx in range(len(mlx_sched.timesteps)):
            t_pt = pt_sched.timesteps[idx]

            mx_pred = mx.full(mx_latents.shape, 0.05)
            pt_pred = torch.full(pt_latents.shape, 0.05)

            mx_latents = mlx_sched.step_dmd(mx_pred, mx_latents, idx, mx_start_points[i_s])
            mx.eval(mx_latents)

            pt_latents = pt_sched.step(
                pt_pred, t_pt, pt_latents, return_dict=False,
                cur_sampling_step=idx,
                dmd_noisy_tensor=pt_start_points[i_s],
                dmd_sigmas=pt_sched.sigmas,
                dmd_timesteps=pt_sched.timesteps,
                all_timesteps=pt_sched.timesteps,
            )[0]

            d = np.abs(np.array(mx_latents) - pt_latents.squeeze(0).numpy())
            max_diff = max(max_diff, d.max())
            print(
                f"  Step {idx}: MLX mean={mx_latents.mean().item():.6f} "
                f"PT mean={pt_latents.mean():.6f} diff max={d.max():.8f}"
            )

    print(f"\n{'=' * 50}")
    print(f"Maximum difference across all stages/steps: {max_diff:.8f}")
    if max_diff < 1e-4:
        print("✓ Pipelines are numerically equivalent")
    elif max_diff < 1e-2:
        print("⚠ Small differences (likely floating point precision)")
    else:
        print("✗ Significant differences detected — investigate!")


if __name__ == "__main__":
    main()
