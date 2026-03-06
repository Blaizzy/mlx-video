#!/usr/bin/env python3
"""Cross-framework model comparison: feed identical inputs to MLX and PyTorch models.

Saves intermediate tensors from the MLX pipeline, loads them into the PyTorch
reference model, and compares outputs. Used to verify the MLX transformer
produces numerically equivalent flow predictions.

Workflow:
    1. Run MLX generation with --debug to save /tmp/helios_model_inputs.npz
       and /tmp/helios_mlx_output.npy
    2. Run this script to load inputs into PyTorch model and compare

Requirements:
    - Reference Helios weights (original PyTorch format)
    - diffusers, torch
    - Saved inputs from MLX debug run

Usage:
    # Step 1: Generate with debug to save inputs
    python -m mlx_video.generate_helios \
        --model-dir /path/to/Helios-Distilled-MLX \
        --prompt "A beautiful sunset over the ocean" \
        --debug --num-frames 33 \
        --output-path /tmp/debug_test.mp4

    # Step 2: Compare with PyTorch
    python scripts/helios/compare_models.py \
        --model-dir /path/to/Helios-Distilled \
        --prompt "A beautiful sunset over the ocean" \
        --inputs /tmp/helios_model_inputs.npz \
        --mlx-output /tmp/helios_mlx_output.npy
"""

import argparse
import sys

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description="Compare MLX vs PyTorch model outputs")
    parser.add_argument("--model-dir", required=True, help="Path to original Helios weights")
    parser.add_argument("--prompt", required=True, help="Same prompt used for MLX debug run")
    parser.add_argument("--inputs", default="/tmp/helios_model_inputs.npz", help="Saved MLX inputs")
    parser.add_argument("--mlx-output", default="/tmp/helios_mlx_output.npy", help="Saved MLX output")
    args = parser.parse_args()

    # Load saved MLX inputs
    data = np.load(args.inputs)
    print("Loaded inputs:")
    for k in data.files:
        print(f"  {k}: shape={data[k].shape}, dtype={data[k].dtype}")

    # Load reference model
    print("\nLoading reference pipeline...")
    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(
        args.model_dir, torch_dtype=torch.float16
    ).to("mps")
    transformer = pipe.transformer

    # Convert inputs to torch tensors (MLX: [C,F,H,W] → PT: [B,C,F,H,W])
    latents_pt = torch.from_numpy(data["latents"]).unsqueeze(0).to("mps")
    timestep_pt = torch.tensor([int(data["timestep"][0])], dtype=torch.int64, device="mps")
    hist_short = torch.from_numpy(data["hist_short"]).unsqueeze(0).to("mps")
    hist_mid = torch.from_numpy(data["hist_mid"]).unsqueeze(0).to("mps")
    hist_long = torch.from_numpy(data["hist_long"]).unsqueeze(0).to("mps")
    idx_current = torch.from_numpy(data["idx_current"]).unsqueeze(0).to("mps")
    idx_short = torch.from_numpy(data["idx_short"]).unsqueeze(0).to("mps")
    idx_mid = torch.from_numpy(data["idx_mid"]).unsqueeze(0).to("mps")
    idx_long = torch.from_numpy(data["idx_long"]).unsqueeze(0).to("mps")

    # Encode prompt with reference text encoder
    print("\nEncoding prompt with reference T5...")
    prompt_embeds, _ = pipe.encode_prompt(
        prompt=args.prompt, do_classifier_free_guidance=False, device="mps"
    )
    print(f"  prompt_embeds: {prompt_embeds.shape}")

    # Run reference forward pass
    print("Running reference model...")
    transformer.eval()
    with torch.no_grad():
        output = transformer(
            hidden_states=latents_pt.half(),
            timestep=timestep_pt,
            encoder_hidden_states=prompt_embeds.half(),
            return_dict=False,
            indices_hidden_states=idx_current,
            indices_latents_history_short=idx_short,
            indices_latents_history_mid=idx_mid,
            indices_latents_history_long=idx_long,
            latents_history_short=hist_short.half(),
            latents_history_mid=hist_mid.half(),
            latents_history_long=hist_long.half(),
        )

    pt_output = output[0].float().cpu().numpy().squeeze(0)

    # Load MLX output
    mlx_output = np.load(args.mlx_output)

    print(f"\n{'=' * 50}")
    print(f"MLX:  shape={mlx_output.shape}, mean={mlx_output.mean():.6f}, std={mlx_output.std():.6f}")
    print(f"PT:   shape={pt_output.shape}, mean={pt_output.mean():.6f}, std={pt_output.std():.6f}")

    diff = mlx_output - pt_output
    rmse = np.sqrt(np.mean(diff**2))
    mae = np.mean(np.abs(diff))
    cos_sim = np.sum(mlx_output * pt_output) / (
        np.linalg.norm(mlx_output) * np.linalg.norm(pt_output)
    )

    print(f"\nRMSE:              {rmse:.6f}")
    print(f"MAE:               {mae:.6f}")
    print(f"Cosine similarity: {cos_sim:.6f}")
    print(f"Max abs diff:      {np.abs(diff).max():.6f}")

    if cos_sim > 0.999:
        print("\n✓ Models produce equivalent outputs")
    elif cos_sim > 0.99:
        print("\n⚠ Minor differences (likely precision-related)")
    else:
        print("\n✗ Significant differences — investigate!")

    # Per-channel breakdown
    print(f"\nPer-channel (first 4):")
    for c in range(min(4, mlx_output.shape[0])):
        c_cos = np.sum(mlx_output[c] * pt_output[c]) / (
            np.linalg.norm(mlx_output[c]) * np.linalg.norm(pt_output[c]) + 1e-8
        )
        print(
            f"  Ch {c}: MLX mean={mlx_output[c].mean():.4f} "
            f"PT mean={pt_output[c].mean():.4f} cos={c_cos:.4f}"
        )


if __name__ == "__main__":
    main()
