#!/usr/bin/env python3
"""Run the reference Helios pipeline on MPS for comparison.

Generates a video using the original PyTorch/diffusers Helios pipeline on Apple
MPS, with necessary float64→float32 patches for MPS compatibility. Useful for
comparing output quality against the MLX implementation.

Requirements:
    pip install diffusers transformers torch accelerate

Usage:
    python mlx_video/models/helios/scripts/run_reference.py \
        --model-dir /path/to/Helios-Distilled \
        --prompt "A golden retriever running on a sunny beach" \
        --output /tmp/helios_ref.mp4

    # Compare against MLX output
    python -m mlx_video.generate_helios \
        --model-dir /path/to/Helios-Distilled-MLX \
        --prompt "A golden retriever running on a sunny beach" \
        --output-path /tmp/helios_mlx.mp4
    python mlx_video/models/helios/scripts/analyze_boundaries.py \
        /tmp/helios_ref.mp4 /tmp/helios_mlx.mp4
"""

import argparse

import cv2
import numpy as np
import torch


def patch_scheduler_for_mps():
    """Patch the Helios DMD scheduler to work on MPS (no float64 support)."""
    import diffusers.schedulers.scheduling_helios_dmd as sched_mod

    _orig_set_ts = sched_mod.HeliosDMDScheduler.set_timesteps

    def _patched_set_ts(
        self,
        num_inference_steps,
        stage_index=None,
        device=None,
        sigmas=None,
        mu=None,
        is_amplify_first_chunk=False,
    ):
        real_device = device
        _orig_set_ts(
            self,
            num_inference_steps,
            stage_index=stage_index,
            device="cpu",
            sigmas=sigmas,
            mu=mu,
            is_amplify_first_chunk=is_amplify_first_chunk,
        )
        self.timesteps = self.timesteps.float()
        self.sigmas = self.sigmas.float()
        if real_device is not None and str(real_device) != "cpu":
            self.timesteps = self.timesteps.to(real_device)
            self.sigmas = self.sigmas.to(real_device)

    sched_mod.HeliosDMDScheduler.set_timesteps = _patched_set_ts

    def _patched_convert_flow(self, flow_pred, xt, timestep, sigmas, timesteps):
        original_dtype = flow_pred.dtype
        device = flow_pred.device
        flow_pred, xt, sigmas, timesteps = (
            x.float().to(device) for x in (flow_pred, xt, sigmas, timesteps)
        )
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1
        )
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    sched_mod.HeliosDMDScheduler.convert_flow_pred_to_x0 = _patched_convert_flow


def main():
    parser = argparse.ArgumentParser(description="Run Helios reference pipeline on MPS")
    parser.add_argument("--model-dir", required=True, help="Path to Helios-Distilled weights")
    parser.add_argument("--prompt", required=True, help="Text prompt")
    parser.add_argument("--output", default="/tmp/helios_ref.mp4", help="Output video path")
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num-frames", type=int, default=99, help="Total frames (33 per chunk)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=16)
    args = parser.parse_args()

    print("Patching scheduler for MPS compatibility...")
    patch_scheduler_for_mps()

    print("Loading pipeline...")
    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float16,
    ).to("mps")

    generator = torch.Generator("mps").manual_seed(args.seed)

    print(f"Generating {args.num_frames} frames...")
    video = pipe(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        guidance_scale=1.0,
        generator=generator,
        pyramid_num_inference_steps_list=[2, 2, 2],
        is_amplify_first_chunk=True,
    ).frames

    frames = video[0]
    print(f"Got {len(frames)} frames, size: {frames[0].size}")

    out = cv2.VideoWriter(
        args.output, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, frames[0].size
    )
    for f in frames:
        out.write(cv2.cvtColor(np.array(f), cv2.COLOR_RGB2BGR))
    out.release()
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
