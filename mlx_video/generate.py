import argparse
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

# ANSI color codes
class Colors:
    CYAN = "\033[96m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

from mlx_video.models.ltx.config import LTXModelConfig, LTXModelType, LTXRopeType
from mlx_video.models.ltx.ltx import LTXModel
from mlx_video.models.ltx.transformer import Modality
from mlx_video.convert import sanitize_transformer_weights
from mlx_video.utils import to_denoised
from mlx_video.models.ltx.video_vae.decoder import load_vae_decoder
from mlx_video.models.ltx.upsampler import load_upsampler, upsample_latents

from mlx_video.utils import get_model_path


# Distilled sigma schedules
STAGE_1_SIGMAS = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]
STAGE_2_SIGMAS = [0.909375, 0.725, 0.421875, 0.0]


def create_position_grid(
    batch_size: int,
    num_frames: int,
    height: int,
    width: int,
    temporal_scale: int = 8,
    spatial_scale: int = 32,
    fps: float = 24.0,
    causal_fix: bool = True,
) -> mx.array:
    """Create position grid for RoPE in pixel space.

    Args:
        batch_size: Batch size
        num_frames: Number of frames (latent)
        height: Height (latent)
        width: Width (latent)
        temporal_scale: VAE temporal scale factor (default 8)
        spatial_scale: VAE spatial scale factor (default 32)
        fps: Frames per second (default 24.0)
        causal_fix: Apply causal fix for first frame (default True)

    Returns:
        Position grid of shape (B, 3, num_patches, 2) in pixel space
        where dim 2 is [start, end) bounds for each patch
    """
    # Patch size is (1, 1, 1) for LTX-2 - no spatial patching
    patch_size_t, patch_size_h, patch_size_w = 1, 1, 1

    # Generate grid coordinates for each dimension (frame, height, width)
    t_coords = np.arange(0, num_frames, patch_size_t)
    h_coords = np.arange(0, height, patch_size_h)
    w_coords = np.arange(0, width, patch_size_w)

    # Create meshgrid with indexing='ij' for (frame, height, width) order
    t_grid, h_grid, w_grid = np.meshgrid(t_coords, h_coords, w_coords, indexing='ij')

    # Stack to get shape (3, grid_t, grid_h, grid_w)
    patch_starts = np.stack([t_grid, h_grid, w_grid], axis=0)

    # Calculate end coordinates (start + patch_size)
    patch_size_delta = np.array([patch_size_t, patch_size_h, patch_size_w]).reshape(3, 1, 1, 1)
    patch_ends = patch_starts + patch_size_delta

    # Stack start and end: shape (3, grid_t, grid_h, grid_w, 2)
    latent_coords = np.stack([patch_starts, patch_ends], axis=-1)

    # Flatten spatial/temporal dims: (3, num_patches, 2)
    num_patches = num_frames * height * width
    latent_coords = latent_coords.reshape(3, num_patches, 2)

    # Broadcast to batch: (batch, 3, num_patches, 2)
    latent_coords = np.tile(latent_coords[np.newaxis, ...], (batch_size, 1, 1, 1))

    # Convert latent coords to pixel coords by scaling with VAE factors
    scale_factors = np.array([temporal_scale, spatial_scale, spatial_scale]).reshape(1, 3, 1, 1)
    pixel_coords = (latent_coords * scale_factors).astype(np.float32)

    # Apply causal fix for first frame temporal axis
    if causal_fix:
        # VAE temporal stride for first frame is 1 instead of temporal_scale
        pixel_coords[:, 0, :, :] = np.clip(
            pixel_coords[:, 0, :, :] + 1 - temporal_scale,
            a_min=0,
            a_max=None
        )

    # Convert temporal to time in seconds by dividing by fps
    pixel_coords[:, 0, :, :] = pixel_coords[:, 0, :, :] / fps

    return mx.array(pixel_coords, dtype=mx.float32)


def denoise(
    latents: mx.array,
    positions: mx.array,
    text_embeddings: mx.array,
    transformer: LTXModel,
    sigmas: list,
) -> mx.array:
    """Run denoising loop."""
    for i in range(len(sigmas) - 1):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]

        b, c, f, h, w = latents.shape
        latents_flat = mx.transpose(mx.reshape(latents, (b, c, -1)), (0, 2, 1))

        video_modality = Modality(
            latent=latents_flat,
            timesteps=mx.full((1,), sigma),
            positions=positions,
            context=text_embeddings,
            context_mask=None,
            enabled=True,
        )

        velocity, _ = transformer(video=video_modality, audio=None)
        mx.eval(velocity)

        velocity = mx.reshape(mx.transpose(velocity, (0, 2, 1)), (b, c, f, h, w))
        denoised = to_denoised(latents, velocity, sigma)
        mx.eval(denoised)

        if sigma_next > 0:
            latents = denoised + sigma_next * (latents - denoised) / sigma
        else:
            latents = denoised
        mx.eval(latents)

    return latents


def generate_video(
    model_repo: str,
    prompt: str,
    text_encoder_repo: str = None,
    height: int = 512,
    width: int = 512,
    num_frames: int = 33,
    seed: int = 42,
    fps: int = 24,
    output_path: str = "output.mp4",
    save_frames: bool = False,
):
    """Generate video from text prompt.

    Args:
        model_repo: HuggingFace repo ID or local path for the main model
        prompt: Text description of the video to generate
        text_encoder_repo: Optional custom text encoder repo or local path.
                          If None, uses model_repo. Supports both HuggingFace
                          repos (e.g., 'user/custom-encoder') and local paths.
        height: Output video height (must be divisible by 64)
        width: Output video width (must be divisible by 64)
        num_frames: Number of frames (must be 1 + 8*k, e.g., 33, 65, 97)
        seed: Random seed for reproducibility
        fps: Frames per second for output video
        output_path: Path to save the output video
        save_frames: Whether to save individual frames as images
    """
    start_time = time.time()

    # Validate dimensions
    assert height % 64 == 0, f"Height must be divisible by 64, got {height}"
    assert width % 64 == 0, f"Width must be divisible by 64, got {width}"

    print(f"{Colors.BOLD}{Colors.CYAN}🎬 Generating {width}x{height} video with {num_frames} frames{Colors.RESET}")
    print(f"{Colors.DIM}Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}{Colors.RESET}")

    # Get model path
    model_path = get_model_path(model_repo)

    # Calculate latent dimensions
    stage1_h, stage1_w = height // 2 // 32, width // 2 // 32
    stage2_h, stage2_w = height // 32, width // 32
    latent_frames = 1 + (num_frames - 1) // 8

    mx.random.seed(seed)

    # Load text encoder (from custom repo if specified, otherwise from model_repo)
    print(f"{Colors.BLUE}📝 Loading text encoder...{Colors.RESET}")
    text_encoder_path = get_model_path(text_encoder_repo) if text_encoder_repo else model_path
    if text_encoder_repo:
        print(f"{Colors.DIM}Using custom text encoder: {text_encoder_repo}{Colors.RESET}")
    from mlx_video.models.ltx.text_encoder import LTX2TextEncoder
    text_encoder = LTX2TextEncoder(model_path=str(text_encoder_path))
    # Load transformer weights from the main model repository
    text_encoder.load(str(model_path))
    mx.eval(text_encoder.parameters())

    text_embeddings, _ = text_encoder(prompt)
    mx.eval(text_embeddings)

    del text_encoder
    mx.clear_cache()

    # Load transformer
    print(f"{Colors.BLUE}🤖 Loading transformer...{Colors.RESET}")
    raw_weights = mx.load(str(model_path / 'ltx-2-19b-distilled.safetensors'))
    sanitized = sanitize_transformer_weights(raw_weights)

    config = LTXModelConfig(
        model_type=LTXModelType.VideoOnly,
        num_attention_heads=32,
        attention_head_dim=128,
        in_channels=128,
        out_channels=128,
        num_layers=48,
        cross_attention_dim=4096,
        caption_channels=3840,
        rope_type=LTXRopeType.SPLIT,
        double_precision_rope=True,
        positional_embedding_theta=10000.0,
        positional_embedding_max_pos=[20, 2048, 2048],
        use_middle_indices_grid=True,
        timestep_scale_multiplier=1000,
    )

    transformer = LTXModel(config)
    transformer.load_weights(list(sanitized.items()), strict=False)
    mx.eval(transformer.parameters())

    # Stage 1: Generate at half resolution
    print(f"{Colors.YELLOW}⚡ Stage 1: Generating at {width//2}x{height//2} (8 steps)...{Colors.RESET}")
    mx.random.seed(seed)
    latents = mx.random.normal((1, 128, latent_frames, stage1_h, stage1_w))
    mx.eval(latents)

    positions = create_position_grid(1, latent_frames, stage1_h, stage1_w)
    mx.eval(positions)

    latents = denoise(latents, positions, text_embeddings, transformer, STAGE_1_SIGMAS)

    # Upsample latents
    print(f"{Colors.MAGENTA}🔍 Upsampling latents 2x...{Colors.RESET}")
    upsampler = load_upsampler(str(model_path / 'ltx-2-spatial-upscaler-x2-1.0.safetensors'))
    mx.eval(upsampler.parameters())

    vae_decoder = load_vae_decoder(
        str(model_path / 'ltx-2-19b-distilled.safetensors'),
        timestep_conditioning=True
    )

    latents = upsample_latents(latents, upsampler, vae_decoder.latents_mean, vae_decoder.latents_std)
    mx.eval(latents)

    del upsampler
    mx.clear_cache()

    # Stage 2: Refine at full resolution
    print(f"{Colors.YELLOW}⚡ Stage 2: Refining at {width}x{height} (3 steps)...{Colors.RESET}")
    positions = create_position_grid(1, latent_frames, stage2_h, stage2_w)
    mx.eval(positions)

    # Add noise for refinement
    noise_scale = STAGE_2_SIGMAS[0]
    noise = mx.random.normal(latents.shape)
    latents = noise * noise_scale + latents * (1 - noise_scale)
    mx.eval(latents)

    latents = denoise(latents, positions, text_embeddings, transformer, STAGE_2_SIGMAS)

    del transformer
    mx.clear_cache()

    # Decode to video
    print(f"{Colors.BLUE}🎞️  Decoding video...{Colors.RESET}")
    video = vae_decoder(latents)
    mx.eval(video)
    mx.clear_cache()

    # Convert to uint8 frames
    video = mx.squeeze(video, axis=0)  # (C, F, H, W)
    video = mx.transpose(video, (1, 2, 3, 0))  # (F, H, W, C)
    video = mx.clip((video + 1.0) / 2.0, 0.0, 1.0)
    video = (video * 255).astype(mx.uint8)
    video_np = np.array(video)

    # Save outputs
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import cv2
        height, width = video_np.shape[1], video_np.shape[2]
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        for frame in video_np:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()
        print(f"{Colors.GREEN}✅ Saved video to {output_path}{Colors.RESET}")
    except Exception as e:
        print(f"{Colors.RED}❌ Could not save video: {e}{Colors.RESET}")

    if save_frames:
        frames_dir = output_path.parent / f"{output_path.stem}_frames"
        frames_dir.mkdir(exist_ok=True)
        for i, frame in enumerate(video_np):
            Image.fromarray(frame).save(frames_dir / f"frame_{i:04d}.png")
        print(f"{Colors.GREEN}✅ Saved {len(video_np)} frames to {frames_dir}{Colors.RESET}")

    elapsed = time.time() - start_time
    print(f"{Colors.BOLD}{Colors.GREEN}🎉 Done! Generated in {elapsed:.1f}s ({elapsed/num_frames:.2f}s/frame){Colors.RESET}")

    return video_np


def main():
    parser = argparse.ArgumentParser(
        description="Generate videos with MLX LTX-2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m mlx_video.generate --prompt "A cat walking on grass"
  python -m mlx_video.generate --prompt "Ocean waves at sunset" --height 768 --width 768
  python -m mlx_video.generate --prompt "..." --num-frames 65 --seed 123 --output my_video.mp4
        """
    )

    parser.add_argument(
        "--prompt", "-p",
        type=str,
        required=True,
        help="Text description of the video to generate"
    )
    parser.add_argument(
        "--height", "-H",
        type=int,
        default=512,
        help="Output video height (default: 512, must be divisible by 32)"
    )
    parser.add_argument(
        "--width", "-W",
        type=int,
        default=512,
        help="Output video width (default: 512, must be divisible by 32)"
    )
    parser.add_argument(
        "--num-frames", "-n",
        type=int,
        default=100,
        help="Number of frames (default: 100)"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Frames per second for output video (default: 24)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output.mp4",
        help="Output video path (default: output.mp4)"
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Save individual frames as images"
    )
    parser.add_argument(
        "--model-repo",
        type=str,
        default="Lightricks/LTX-2",
        help="Model repository to use (default: Lightricks/LTX-2)"
    )
    parser.add_argument(
        "--text-encoder-repo",
        type=str,
        default=None,
        help="Custom text encoder repository or local path (defaults to --model-repo)"
    )
    args = parser.parse_args()

    generate_video(
        model_repo=args.model_repo,
        prompt=args.prompt,
        text_encoder_repo=args.text_encoder_repo,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        seed=args.seed,
        fps=args.fps,
        output_path=args.output,
        save_frames=args.save_frames,
    )


if __name__ == "__main__":
    main()
