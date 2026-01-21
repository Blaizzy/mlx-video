"""
Copyright (c) 2026, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-video)

LTX-2 Dev Model Generation Pipeline

This module provides a single-stage video generation pipeline using the LTX-2 19B dev model.
Unlike the distilled model which uses fixed sigma schedules, the dev model uses:
- Dynamic sigma scheduling via LTX2Scheduler
- Classifier-Free Guidance (CFG) for better prompt adherence
- More inference steps (default 40)
"""

import argparse
import math
import time
from pathlib import Path
from typing import Optional

import mlx.core as mx
import numpy as np
from PIL import Image
from tqdm import tqdm

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
from mlx_video.convert import sanitize_transformer_weights, sanitize_audio_vae_weights, sanitize_vocoder_weights
from mlx_video.utils import to_denoised, load_image, prepare_image_for_encoding
from mlx_video.models.ltx.video_vae.decoder import load_vae_decoder
from mlx_video.models.ltx.video_vae.encoder import load_vae_encoder
from mlx_video.models.ltx.video_vae.tiling import TilingConfig
from mlx_video.conditioning import VideoConditionByLatentIndex, apply_conditioning
from mlx_video.conditioning.latent import LatentState, apply_denoise_mask
from mlx_video.utils import get_model_path


# Default values matching PyTorch implementation
DEFAULT_NEGATIVE_PROMPT = (
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, "
    "grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, "
    "deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, "
    "wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of "
    "field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent "
    "lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny "
    "valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, "
    "mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, "
    "off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward "
    "pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, "
    "inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."
)

BASE_SHIFT_ANCHOR = 1024
MAX_SHIFT_ANCHOR = 4096

# Audio constants
AUDIO_SAMPLE_RATE = 24000  # Output audio sample rate
AUDIO_LATENT_SAMPLE_RATE = 16000  # VAE internal sample rate
AUDIO_HOP_LENGTH = 160
AUDIO_LATENT_DOWNSAMPLE_FACTOR = 4
AUDIO_LATENT_CHANNELS = 8  # Latent channels before patchifying
AUDIO_MEL_BINS = 16
AUDIO_LATENTS_PER_SECOND = AUDIO_LATENT_SAMPLE_RATE / AUDIO_HOP_LENGTH / AUDIO_LATENT_DOWNSAMPLE_FACTOR  # 25


def ltx2_scheduler(
    steps: int,
    num_tokens: Optional[int] = None,
    max_shift: float = 2.05,
    base_shift: float = 0.95,
    stretch: bool = True,
    terminal: float = 0.1,
) -> mx.array:
    """
    LTX-2 scheduler for sigma generation.

    Generates a sigma schedule with token-count-dependent shifting and optional
    stretching to a terminal value.

    Args:
        steps: Number of inference steps
        num_tokens: Number of latent tokens (F*H*W). If None, uses MAX_SHIFT_ANCHOR
        max_shift: Maximum shift factor
        base_shift: Base shift factor
        stretch: Whether to stretch sigmas to terminal value
        terminal: Terminal sigma value for stretching

    Returns:
        Array of sigma values of shape (steps + 1,)
    """
    tokens = num_tokens if num_tokens is not None else MAX_SHIFT_ANCHOR
    sigmas = np.linspace(1.0, 0.0, steps + 1)

    # Compute shift based on token count
    x1 = BASE_SHIFT_ANCHOR
    x2 = MAX_SHIFT_ANCHOR
    mm = (max_shift - base_shift) / (x2 - x1)
    b = base_shift - mm * x1
    sigma_shift = tokens * mm + b

    # Apply shift transformation
    power = 1
    sigmas = np.where(
        sigmas != 0,
        math.exp(sigma_shift) / (math.exp(sigma_shift) + (1 / sigmas - 1) ** power),
        0,
    )

    # Stretch sigmas to terminal value
    if stretch:
        non_zero_mask = sigmas != 0
        non_zero_sigmas = sigmas[non_zero_mask]
        one_minus_z = 1.0 - non_zero_sigmas
        scale_factor = one_minus_z[-1] / (1.0 - terminal)
        stretched = 1.0 - (one_minus_z / scale_factor)
        sigmas[non_zero_mask] = stretched

    return mx.array(sigmas, dtype=mx.float32)


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

    # Always return float32 for RoPE precision - bfloat16 causes quality degradation
    return mx.array(pixel_coords, dtype=mx.float32)


def create_audio_position_grid(
    batch_size: int,
    audio_frames: int,
    sample_rate: int = AUDIO_LATENT_SAMPLE_RATE,
    hop_length: int = AUDIO_HOP_LENGTH,
    downsample_factor: int = AUDIO_LATENT_DOWNSAMPLE_FACTOR,
    is_causal: bool = True,
) -> mx.array:
    """Create temporal position grid for audio RoPE.

    Audio positions are timestamps in seconds, shape (B, 1, T, 2).
    Matches PyTorch's AudioPatchifier.get_patch_grid_bounds exactly.

    Args:
        batch_size: Batch size
        audio_frames: Number of audio latent frames
        sample_rate: Audio sample rate (default 16000)
        hop_length: Hop length for mel spectrogram (default 160)
        downsample_factor: Latent downsample factor (default 4)
        is_causal: Whether to use causal alignment (default True)

    Returns:
        Position grid of shape (B, 1, T, 2)
    """
    def get_audio_latent_time_in_sec(start_idx: int, end_idx: int) -> np.ndarray:
        """Convert latent indices to seconds."""
        latent_frame = np.arange(start_idx, end_idx, dtype=np.float32)
        mel_frame = latent_frame * downsample_factor
        if is_causal:
            mel_frame = np.clip(mel_frame + 1 - downsample_factor, 0, None)
        return mel_frame * hop_length / sample_rate

    start_times = get_audio_latent_time_in_sec(0, audio_frames)
    end_times = get_audio_latent_time_in_sec(1, audio_frames + 1)

    positions = np.stack([start_times, end_times], axis=-1)
    positions = positions[np.newaxis, np.newaxis, :, :]  # (1, 1, T, 2)
    positions = np.tile(positions, (batch_size, 1, 1, 1))

    return mx.array(positions, dtype=mx.float32)


def compute_audio_frames(num_video_frames: int, fps: float) -> int:
    """Compute number of audio latent frames given video duration."""
    duration = num_video_frames / fps
    return round(duration * AUDIO_LATENTS_PER_SECOND)


def cfg_delta(cond: mx.array, uncond: mx.array, scale: float) -> mx.array:
    """Compute CFG (Classifier-Free Guidance) delta.

    Args:
        cond: Conditioned prediction
        uncond: Unconditioned prediction
        scale: Guidance scale (1.0 = no guidance)

    Returns:
        CFG delta to add to conditioned prediction
    """
    return (scale - 1.0) * (cond - uncond)


def load_audio_decoder(model_path: Path):
    """Load audio VAE decoder."""
    from mlx_video.models.ltx.audio_vae import AudioDecoder, CausalityAxis, NormType

    decoder = AudioDecoder(
        ch=128,
        out_ch=2,  # stereo
        ch_mult=(1, 2, 4),
        num_res_blocks=2,
        attn_resolutions={8, 16, 32},
        resolution=256,
        z_channels=AUDIO_LATENT_CHANNELS,
        norm_type=NormType.PIXEL,
        causality_axis=CausalityAxis.HEIGHT,
        mel_bins=64,  # Output mel bins
    )

    # Load weights - try dev model first, fall back to distilled
    weight_file = model_path / "ltx-2-19b-dev.safetensors"
    if not weight_file.exists():
        weight_file = model_path / "ltx-2-19b-distilled.safetensors"

    if weight_file.exists():
        raw_weights = mx.load(str(weight_file))
        sanitized = sanitize_audio_vae_weights(raw_weights)
        if sanitized:
            decoder.load_weights(list(sanitized.items()), strict=False)

            # Manually load per-channel statistics
            if "per_channel_statistics._mean_of_means" in sanitized:
                decoder.per_channel_statistics._mean_of_means = sanitized["per_channel_statistics._mean_of_means"]
            if "per_channel_statistics._std_of_means" in sanitized:
                decoder.per_channel_statistics._std_of_means = sanitized["per_channel_statistics._std_of_means"]

    return decoder


def load_vocoder(model_path: Path):
    """Load vocoder for mel to waveform conversion."""
    from mlx_video.models.ltx.audio_vae import Vocoder

    vocoder = Vocoder(
        resblock_kernel_sizes=[3, 7, 11],
        upsample_rates=[6, 5, 2, 2, 2],
        upsample_kernel_sizes=[16, 15, 8, 4, 4],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_initial_channel=1024,
        stereo=True,
        output_sample_rate=AUDIO_SAMPLE_RATE,
    )

    # Load weights - try dev model first, fall back to distilled
    weight_file = model_path / "ltx-2-19b-dev.safetensors"
    if not weight_file.exists():
        weight_file = model_path / "ltx-2-19b-distilled.safetensors"

    if weight_file.exists():
        raw_weights = mx.load(str(weight_file))
        sanitized = sanitize_vocoder_weights(raw_weights)
        if sanitized:
            vocoder.load_weights(list(sanitized.items()), strict=False)

    return vocoder


def save_audio(audio: np.ndarray, path: Path, sample_rate: int = AUDIO_SAMPLE_RATE):
    """Save audio to WAV file."""
    import wave

    # Ensure audio is in correct format (channels, samples) or (samples,)
    if audio.ndim == 2:
        # (channels, samples) -> (samples, channels)
        audio = audio.T

    # Normalize and convert to int16
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)

    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(2 if audio_int16.ndim == 2 else 1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


def mux_video_audio(video_path: Path, audio_path: Path, output_path: Path) -> bool:
    """Combine video and audio into final output using ffmpeg."""
    import subprocess

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        str(output_path)
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"{Colors.RED}FFmpeg error: {e.stderr.decode()}{Colors.RESET}")
        return False
    except FileNotFoundError:
        print(f"{Colors.RED}FFmpeg not found. Please install ffmpeg.{Colors.RESET}")
        return False


def denoise_with_cfg(
    latents: mx.array,
    positions: mx.array,
    text_embeddings_pos: mx.array,
    text_embeddings_neg: mx.array,
    transformer: LTXModel,
    sigmas: mx.array,
    cfg_scale: float = 4.0,
    verbose: bool = True,
    state: Optional[LatentState] = None,
) -> mx.array:
    """Run denoising loop with CFG (Classifier-Free Guidance).

    Uses separate forward passes for positive and negative conditioning
    to match PyTorch implementation behavior (avoids potential issues with
    batched attention patterns).

    Args:
        latents: Noisy latent tensor (B, C, F, H, W)
        positions: Position embeddings
        text_embeddings_pos: Positive (prompt) text conditioning embeddings
        text_embeddings_neg: Negative prompt text conditioning embeddings
        transformer: LTX model
        sigmas: Array of sigma values for denoising schedule
        cfg_scale: Guidance scale (default 4.0, 1.0 = no guidance)
        verbose: Whether to show progress bar
        state: Optional LatentState for I2V conditioning

    Returns:
        Denoised latent tensor
    """
    from mlx_video.models.ltx.rope import precompute_freqs_cis

    dtype = latents.dtype
    if state is not None:
        latents = state.latent

    sigmas_list = sigmas.tolist()
    use_cfg = cfg_scale != 1.0

    # Precompute RoPE once (expensive operation due to NumPy conversion for double precision)
    # This avoids recomputing it every forward pass
    precomputed_rope = precompute_freqs_cis(
        positions,
        dim=transformer.inner_dim,
        theta=transformer.positional_embedding_theta,
        max_pos=transformer.positional_embedding_max_pos,
        use_middle_indices_grid=transformer.use_middle_indices_grid,
        num_attention_heads=transformer.num_attention_heads,
        rope_type=transformer.rope_type,
        double_precision=transformer.config.double_precision_rope,
    )
    mx.eval(precomputed_rope)

    for i in tqdm(range(len(sigmas_list) - 1), desc="Denoising", disable=not verbose):
        sigma = sigmas_list[i]
        sigma_next = sigmas_list[i + 1]

        b, c, f, h, w = latents.shape
        num_tokens = f * h * w
        latents_flat = mx.transpose(mx.reshape(latents, (b, c, -1)), (0, 2, 1))

        # Compute per-token timesteps
        if state is not None:
            denoise_mask_flat = mx.reshape(state.denoise_mask, (b, 1, f, 1, 1))
            denoise_mask_flat = mx.broadcast_to(denoise_mask_flat, (b, 1, f, h, w))
            denoise_mask_flat = mx.reshape(denoise_mask_flat, (b, num_tokens))
            timesteps = mx.array(sigma, dtype=dtype) * denoise_mask_flat
        else:
            timesteps = mx.full((b, num_tokens), sigma, dtype=dtype)

        # First forward pass: positive conditioning
        video_modality_pos = Modality(
            latent=latents_flat,
            timesteps=timesteps,
            positions=positions,
            context=text_embeddings_pos,
            context_mask=None,
            enabled=True,
            positional_embeddings=precomputed_rope,
        )
        velocity_pos, _ = transformer(video=video_modality_pos, audio=None)

        if use_cfg:
            # Second forward pass: negative conditioning
            video_modality_neg = Modality(
                latent=latents_flat,
                timesteps=timesteps,
                positions=positions,
                context=text_embeddings_neg,
                context_mask=None,
                enabled=True,
                positional_embeddings=precomputed_rope,
            )
            velocity_neg, _ = transformer(video=video_modality_neg, audio=None)

            # Apply CFG: velocity = pos + (scale - 1) * (pos - neg)
            velocity_flat = velocity_pos + (cfg_scale - 1.0) * (velocity_pos - velocity_neg)
        else:
            velocity_flat = velocity_pos

        # Reshape back to 5D
        velocity = mx.reshape(mx.transpose(velocity_flat, (0, 2, 1)), (b, c, f, h, w))
        denoised = to_denoised(latents, velocity, sigma)

        # Apply conditioning mask if state is provided
        if state is not None:
            denoised = apply_denoise_mask(denoised, state.clean_latent, state.denoise_mask)

        # Euler step
        if sigma_next > 0:
            sigma_next_arr = mx.array(sigma_next, dtype=dtype)
            sigma_arr = mx.array(sigma, dtype=dtype)
            latents = denoised + sigma_next_arr * (latents - denoised) / sigma_arr
        else:
            latents = denoised

        # Single eval at end of step (lazy evaluation handles the rest)
        mx.eval(latents)

    return latents


def denoise_av_with_cfg(
    video_latents: mx.array,
    audio_latents: mx.array,
    video_positions: mx.array,
    audio_positions: mx.array,
    video_embeddings_pos: mx.array,
    video_embeddings_neg: mx.array,
    audio_embeddings_pos: mx.array,
    audio_embeddings_neg: mx.array,
    transformer: LTXModel,
    sigmas: mx.array,
    cfg_scale: float = 4.0,
    verbose: bool = True,
    video_state: Optional[LatentState] = None,
) -> tuple[mx.array, mx.array]:
    """Run denoising loop for audio-video generation with CFG.

    Uses separate forward passes for positive and negative CFG to ensure
    correct audio-video cross-attention behavior (matching PyTorch implementation).

    Args:
        video_latents: Video latent tensor (B, C, F, H, W)
        audio_latents: Audio latent tensor (B, C, T, F)
        video_positions: Video position embeddings
        audio_positions: Audio position embeddings
        video_embeddings_pos: Positive video text embeddings
        video_embeddings_neg: Negative video text embeddings
        audio_embeddings_pos: Positive audio text embeddings
        audio_embeddings_neg: Negative audio text embeddings
        transformer: LTX model
        sigmas: Array of sigma values for denoising schedule
        cfg_scale: Guidance scale (default 4.0, 1.0 = no guidance)
        verbose: Whether to show progress bar
        video_state: Optional LatentState for I2V conditioning

    Returns:
        Tuple of (video_latents, audio_latents)
    """
    from mlx_video.models.ltx.rope import precompute_freqs_cis

    dtype = video_latents.dtype
    if video_state is not None:
        video_latents = video_state.latent

    sigmas_list = sigmas.tolist()
    use_cfg = cfg_scale != 1.0

    # Precompute video RoPE (single batch, not doubled for CFG)
    precomputed_video_rope = precompute_freqs_cis(
        video_positions,
        dim=transformer.inner_dim,
        theta=transformer.positional_embedding_theta,
        max_pos=transformer.positional_embedding_max_pos,
        use_middle_indices_grid=transformer.use_middle_indices_grid,
        num_attention_heads=transformer.num_attention_heads,
        rope_type=transformer.rope_type,
        double_precision=transformer.config.double_precision_rope,
    )

    # Precompute audio RoPE (1D positions)
    precomputed_audio_rope = precompute_freqs_cis(
        audio_positions,
        dim=transformer.audio_inner_dim,
        theta=transformer.positional_embedding_theta,
        max_pos=transformer.audio_positional_embedding_max_pos,
        use_middle_indices_grid=transformer.use_middle_indices_grid,
        num_attention_heads=transformer.audio_num_attention_heads,
        rope_type=transformer.rope_type,
        double_precision=transformer.config.double_precision_rope,
    )
    mx.eval(precomputed_video_rope, precomputed_audio_rope)

    for i in tqdm(range(len(sigmas_list) - 1), desc="Denoising A/V", disable=not verbose):
        sigma = sigmas_list[i]
        sigma_next = sigmas_list[i + 1]

        # Flatten video latents
        b, c, f, h, w = video_latents.shape
        num_video_tokens = f * h * w
        video_flat = mx.transpose(mx.reshape(video_latents, (b, c, -1)), (0, 2, 1))

        # Flatten audio latents: (B, C, T, F) -> (B, T, C*F)
        ab, ac, at, af = audio_latents.shape
        audio_flat = mx.transpose(audio_latents, (0, 2, 1, 3))  # (B, T, C, F)
        audio_flat = mx.reshape(audio_flat, (ab, at, ac * af))

        # Compute per-token timesteps for video
        if video_state is not None:
            denoise_mask_flat = mx.reshape(video_state.denoise_mask, (b, 1, f, 1, 1))
            denoise_mask_flat = mx.broadcast_to(denoise_mask_flat, (b, 1, f, h, w))
            denoise_mask_flat = mx.reshape(denoise_mask_flat, (b, num_video_tokens))
            video_timesteps = mx.array(sigma, dtype=dtype) * denoise_mask_flat
        else:
            video_timesteps = mx.full((b, num_video_tokens), sigma, dtype=dtype)

        audio_timesteps = mx.full((ab, at), sigma, dtype=dtype)

        # First forward pass: positive conditioning
        video_modality_pos = Modality(
            latent=video_flat,
            timesteps=video_timesteps,
            positions=video_positions,
            context=video_embeddings_pos,
            context_mask=None,
            enabled=True,
            positional_embeddings=precomputed_video_rope,
        )

        audio_modality_pos = Modality(
            latent=audio_flat,
            timesteps=audio_timesteps,
            positions=audio_positions,
            context=audio_embeddings_pos,
            context_mask=None,
            enabled=True,
            positional_embeddings=precomputed_audio_rope,
        )

        video_vel_pos, audio_vel_pos = transformer(video=video_modality_pos, audio=audio_modality_pos)

        if use_cfg:
            # Second forward pass: negative conditioning
            video_modality_neg = Modality(
                latent=video_flat,
                timesteps=video_timesteps,
                positions=video_positions,
                context=video_embeddings_neg,
                context_mask=None,
                enabled=True,
                positional_embeddings=precomputed_video_rope,
            )

            audio_modality_neg = Modality(
                latent=audio_flat,
                timesteps=audio_timesteps,
                positions=audio_positions,
                context=audio_embeddings_neg,
                context_mask=None,
                enabled=True,
                positional_embeddings=precomputed_audio_rope,
            )

            video_vel_neg, audio_vel_neg = transformer(video=video_modality_neg, audio=audio_modality_neg)

            # Apply CFG: denoised = pos + (scale - 1) * (pos - neg)
            video_velocity_flat = video_vel_pos + (cfg_scale - 1.0) * (video_vel_pos - video_vel_neg)
            audio_velocity_flat = audio_vel_pos + (cfg_scale - 1.0) * (audio_vel_pos - audio_vel_neg)
        else:
            video_velocity_flat = video_vel_pos
            audio_velocity_flat = audio_vel_pos

        # Reshape velocities back
        video_velocity = mx.reshape(mx.transpose(video_velocity_flat, (0, 2, 1)), (b, c, f, h, w))
        audio_velocity = mx.reshape(audio_velocity_flat, (ab, at, ac, af))
        audio_velocity = mx.transpose(audio_velocity, (0, 2, 1, 3))  # (B, C, T, F)

        # Compute denoised
        video_denoised = to_denoised(video_latents, video_velocity, sigma)
        audio_denoised = to_denoised(audio_latents, audio_velocity, sigma)

        # Apply conditioning mask for video if state is provided
        if video_state is not None:
            video_denoised = apply_denoise_mask(video_denoised, video_state.clean_latent, video_state.denoise_mask)

        # Euler step
        if sigma_next > 0:
            sigma_next_arr = mx.array(sigma_next, dtype=dtype)
            sigma_arr = mx.array(sigma, dtype=dtype)
            video_latents = video_denoised + sigma_next_arr * (video_latents - video_denoised) / sigma_arr
            audio_latents = audio_denoised + sigma_next_arr * (audio_latents - audio_denoised) / sigma_arr
        else:
            video_latents = video_denoised
            audio_latents = audio_denoised

        mx.eval(video_latents, audio_latents)

    return video_latents, audio_latents


def generate_video_dev(
    model_repo: str,
    text_encoder_repo: str,
    prompt: str,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    height: int = 512,
    width: int = 768,
    num_frames: int = 33,
    num_inference_steps: int = 40,
    cfg_scale: float = 4.0,
    seed: int = 42,
    fps: int = 24,
    output_path: str = "output.mp4",
    output_audio_path: Optional[str] = None,
    save_frames: bool = False,
    verbose: bool = True,
    enhance_prompt: bool = False,
    max_tokens: int = 512,
    temperature: float = 0.7,
    image: Optional[str] = None,
    image_strength: float = 1.0,
    image_frame_idx: int = 0,
    tiling: str = "none",
    audio: bool = False,
):
    """Generate video using LTX-2 dev model with CFG.

    This is a single-stage pipeline that uses the full dev model with
    Classifier-Free Guidance for better prompt adherence.

    Args:
        model_repo: Model repository ID
        text_encoder_repo: Text encoder repository ID
        prompt: Text description of the video to generate
        negative_prompt: Negative prompt for CFG
        height: Output video height (must be divisible by 32)
        width: Output video width (must be divisible by 32)
        num_frames: Number of frames (must be 1 + 8*k, e.g., 33, 65, 97)
        num_inference_steps: Number of denoising steps (default 40)
        cfg_scale: Guidance scale for CFG (default 4.0)
        seed: Random seed for reproducibility
        fps: Frames per second for output video
        output_path: Path to save the output video
        output_audio_path: Path to save audio (if audio=True)
        save_frames: Whether to save individual frames as images
        verbose: Whether to print progress
        enhance_prompt: Whether to enhance prompt using Gemma
        max_tokens: Max tokens for prompt enhancement
        temperature: Temperature for prompt enhancement
        image: Path to conditioning image for I2V (Image-to-Video)
        image_strength: Conditioning strength (1.0 = full denoise, 0.0 = keep original)
        image_frame_idx: Frame index to condition (0 = first frame)
        tiling: Tiling mode for VAE decoding
        audio: Whether to generate synchronized audio
    """
    start_time = time.time()

    # Validate dimensions
    assert height % 32 == 0, f"Height must be divisible by 32, got {height}"
    assert width % 32 == 0, f"Width must be divisible by 32, got {width}"

    if num_frames % 8 != 1:
        adjusted_num_frames = round((num_frames - 1) / 8) * 8 + 1
        print(f"{Colors.YELLOW}Warning: Number of frames must be 1 + 8*k. Using nearest valid value: {adjusted_num_frames}{Colors.RESET}")
        num_frames = adjusted_num_frames

    # Calculate audio frames if audio is enabled
    audio_frames = compute_audio_frames(num_frames, fps) if audio else 0

    is_i2v = image is not None
    mode_str = "I2V" if is_i2v else "T2V"
    if audio:
        mode_str += "+Audio"
    print(f"{Colors.BOLD}{Colors.CYAN}[DEV] [{mode_str}] Generating {width}x{height} video with {num_frames} frames{Colors.RESET}")
    print(f"{Colors.DIM}Steps: {num_inference_steps}, CFG: {cfg_scale}{Colors.RESET}")
    if audio:
        print(f"{Colors.DIM}Audio: {audio_frames} latent frames @ {AUDIO_SAMPLE_RATE}Hz{Colors.RESET}")
    print(f"{Colors.DIM}Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}{Colors.RESET}")
    if is_i2v:
        print(f"{Colors.DIM}Image: {image} (strength={image_strength}, frame={image_frame_idx}){Colors.RESET}")

    # Get model path
    model_path = get_model_path(model_repo)
    text_encoder_path = model_path if text_encoder_repo is None else get_model_path(text_encoder_repo)

    # Calculate latent dimensions (single-stage, no upsampling)
    latent_h, latent_w = height // 32, width // 32
    latent_frames = 1 + (num_frames - 1) // 8

    mx.random.seed(seed)

    # Load text encoder
    print(f"{Colors.BLUE}Loading text encoder...{Colors.RESET}")
    from mlx_video.models.ltx.text_encoder import LTX2TextEncoder
    text_encoder = LTX2TextEncoder()
    text_encoder.load(model_path=model_path, text_encoder_path=text_encoder_path)
    mx.eval(text_encoder.parameters())

    # Optionally enhance the prompt
    if enhance_prompt:
        print(f"{Colors.MAGENTA}Enhancing prompt...{Colors.RESET}")
        prompt = text_encoder.enhance_t2v(prompt, max_tokens=max_tokens, temperature=temperature, seed=seed, verbose=verbose)
        print(f"{Colors.DIM}Enhanced: {prompt[:150]}{'...' if len(prompt) > 150 else ''}{Colors.RESET}")

    # Encode both positive and negative prompts
    if audio:
        video_embeddings_pos, audio_embeddings_pos = text_encoder(prompt, return_audio_embeddings=True)
        video_embeddings_neg, audio_embeddings_neg = text_encoder(negative_prompt, return_audio_embeddings=True)
        model_dtype = video_embeddings_pos.dtype
        mx.eval(video_embeddings_pos, video_embeddings_neg, audio_embeddings_pos, audio_embeddings_neg)
    else:
        video_embeddings_pos, _ = text_encoder(prompt, return_audio_embeddings=False)
        video_embeddings_neg, _ = text_encoder(negative_prompt, return_audio_embeddings=False)
        audio_embeddings_pos = None
        audio_embeddings_neg = None
        model_dtype = video_embeddings_pos.dtype
        mx.eval(video_embeddings_pos, video_embeddings_neg)

    del text_encoder
    mx.clear_cache()

    # Load transformer (dev model)
    print(f"{Colors.BLUE}Loading dev transformer{' (A/V mode)' if audio else ''}...{Colors.RESET}")
    raw_weights = mx.load(str(model_path / 'ltx-2-19b-dev.safetensors'))
    sanitized = sanitize_transformer_weights(raw_weights)
    # Convert transformer weights to bfloat16 for memory efficiency
    sanitized = {k: v.astype(mx.bfloat16) if v.dtype == mx.float32 else v for k, v in sanitized.items()}

    if audio:
        config = LTXModelConfig(
            model_type=LTXModelType.AudioVideo,
            num_attention_heads=32,
            attention_head_dim=128,
            in_channels=128,
            out_channels=128,
            num_layers=48,
            cross_attention_dim=4096,
            caption_channels=3840,
            # Audio config
            audio_num_attention_heads=32,
            audio_attention_head_dim=64,
            audio_in_channels=AUDIO_LATENT_CHANNELS * AUDIO_MEL_BINS,  # 8 * 16 = 128
            audio_out_channels=AUDIO_LATENT_CHANNELS * AUDIO_MEL_BINS,
            audio_cross_attention_dim=2048,
            rope_type=LTXRopeType.SPLIT,
            double_precision_rope=True,
            positional_embedding_theta=10000.0,
            positional_embedding_max_pos=[20, 2048, 2048],
            audio_positional_embedding_max_pos=[20],
            use_middle_indices_grid=True,
            timestep_scale_multiplier=1000,
        )
    else:
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

    # Load VAE encoder for I2V
    image_latent = None
    if is_i2v:
        print(f"{Colors.BLUE}Loading VAE encoder and encoding image...{Colors.RESET}")
        vae_encoder = load_vae_encoder(str(model_path / 'ltx-2-19b-dev.safetensors'))
        mx.eval(vae_encoder.parameters())

        input_image = load_image(image, height=height, width=width, dtype=model_dtype)
        image_tensor = prepare_image_for_encoding(input_image, height, width, dtype=model_dtype)
        image_latent = vae_encoder(image_tensor)
        mx.eval(image_latent)
        print(f"  Image latent: {image_latent.shape}")

        del vae_encoder
        mx.clear_cache()

    # Generate sigma schedule
    num_tokens = latent_frames * latent_h * latent_w
    sigmas = ltx2_scheduler(
        steps=num_inference_steps,
        num_tokens=num_tokens,
    )
    mx.eval(sigmas)
    print(f"{Colors.DIM}Sigma schedule: {sigmas[0].item():.4f} -> {sigmas[-2].item():.4f} -> {sigmas[-1].item():.4f}{Colors.RESET}")

    # Create position grids
    print(f"{Colors.YELLOW}Generating at {width}x{height} ({num_inference_steps} steps, CFG={cfg_scale})...{Colors.RESET}")
    mx.random.seed(seed)

    video_positions = create_position_grid(1, latent_frames, latent_h, latent_w)
    mx.eval(video_positions)

    if audio:
        audio_positions = create_audio_position_grid(1, audio_frames)
        mx.eval(audio_positions)
    else:
        audio_positions = None

    # Initialize latents with optional I2V conditioning
    video_state = None
    video_latent_shape = (1, 128, latent_frames, latent_h, latent_w)
    if is_i2v and image_latent is not None:
        video_state = LatentState(
            latent=mx.zeros(video_latent_shape, dtype=model_dtype),
            clean_latent=mx.zeros(video_latent_shape, dtype=model_dtype),
            denoise_mask=mx.ones((1, 1, latent_frames, 1, 1), dtype=model_dtype),
        )
        conditioning = VideoConditionByLatentIndex(
            latent=image_latent,
            frame_idx=image_frame_idx,
            strength=image_strength,
        )
        video_state = apply_conditioning(video_state, [conditioning])

        # Apply noiser
        noise = mx.random.normal(video_latent_shape, dtype=model_dtype)
        noise_scale = sigmas[0]
        scaled_mask = video_state.denoise_mask * noise_scale

        video_state = LatentState(
            latent=noise * scaled_mask + video_state.latent * (mx.array(1.0, dtype=model_dtype) - scaled_mask),
            clean_latent=video_state.clean_latent,
            denoise_mask=video_state.denoise_mask,
        )
        video_latents = video_state.latent
        mx.eval(video_latents)
    else:
        # T2V: just use random noise
        video_latents = mx.random.normal(video_latent_shape, dtype=model_dtype)
        mx.eval(video_latents)

    # Initialize audio latents if audio is enabled
    if audio:
        audio_latents = mx.random.normal((1, AUDIO_LATENT_CHANNELS, audio_frames, AUDIO_MEL_BINS), dtype=model_dtype)
        mx.eval(audio_latents)
    else:
        audio_latents = None

    # Denoise with CFG
    if audio:
        video_latents, audio_latents = denoise_av_with_cfg(
            video_latents, audio_latents,
            video_positions, audio_positions,
            video_embeddings_pos, video_embeddings_neg,
            audio_embeddings_pos, audio_embeddings_neg,
            transformer, sigmas, cfg_scale=cfg_scale, verbose=verbose, video_state=video_state
        )
    else:
        video_latents = denoise_with_cfg(
            video_latents, video_positions, video_embeddings_pos, video_embeddings_neg,
            transformer, sigmas, cfg_scale=cfg_scale, verbose=verbose, state=video_state
        )

    del transformer
    mx.clear_cache()

    # Decode to video
    print(f"{Colors.BLUE}Decoding video...{Colors.RESET}")
    vae_decoder = load_vae_decoder(
        str(model_path / 'ltx-2-19b-dev.safetensors'),
        timestep_conditioning=None
    )
    mx.eval(vae_decoder.parameters())

    # Select tiling configuration
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
        print(f"{Colors.YELLOW}  Unknown tiling mode '{tiling}', using auto{Colors.RESET}")
        tiling_config = TilingConfig.auto(height, width, num_frames)

    if tiling_config is not None:
        spatial_info = f"{tiling_config.spatial_config.tile_size_in_pixels}px" if tiling_config.spatial_config else "none"
        temporal_info = f"{tiling_config.temporal_config.tile_size_in_frames}f" if tiling_config.temporal_config else "none"
        print(f"{Colors.DIM}  Tiling ({tiling}): spatial={spatial_info}, temporal={temporal_info}{Colors.RESET}")
        video = vae_decoder.decode_tiled(video_latents, tiling_config=tiling_config, tiling_mode=tiling, debug=verbose)
    else:
        print(f"{Colors.DIM}  Tiling: disabled{Colors.RESET}")
        video = vae_decoder(video_latents)
    mx.eval(video)

    del vae_decoder
    mx.clear_cache()

    # Decode audio if enabled
    audio_np = None
    if audio and audio_latents is not None:
        print(f"{Colors.BLUE}Decoding audio...{Colors.RESET}")

        # Load audio decoder
        audio_decoder = load_audio_decoder(model_path)
        mx.eval(audio_decoder.parameters())

        # Decode audio latents to mel spectrogram
        mel_spectrogram = audio_decoder(audio_latents)
        mx.eval(mel_spectrogram)

        del audio_decoder
        mx.clear_cache()

        # Load vocoder and convert mel to waveform
        vocoder = load_vocoder(model_path)
        mx.eval(vocoder.parameters())

        audio_waveform = vocoder(mel_spectrogram)
        mx.eval(audio_waveform)

        del vocoder
        mx.clear_cache()

        # Convert to numpy
        audio_np = np.array(audio_waveform)
        if audio_np.ndim == 3:
            audio_np = audio_np[0]  # Remove batch dim

        print(f"{Colors.DIM}  Audio shape: {audio_np.shape}, duration: {audio_np.shape[-1] / AUDIO_SAMPLE_RATE:.2f}s{Colors.RESET}")

    # Convert video to uint8 frames
    video = mx.squeeze(video, axis=0)  # (C, F, H, W)
    video = mx.transpose(video, (1, 2, 3, 0))  # (F, H, W, C)
    video = mx.clip((video + 1.0) / 2.0, 0.0, 1.0)
    video = (video * 255).astype(mx.uint8)
    video_np = np.array(video)

    # Save outputs
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine audio output path
    if audio and audio_np is not None:
        if output_audio_path is None:
            audio_output = output_path.parent / f"{output_path.stem}.wav"
        else:
            audio_output = Path(output_audio_path)

        # Save audio
        save_audio(audio_np, audio_output)
        print(f"{Colors.GREEN}Saved audio to{Colors.RESET} {audio_output}")

    # Save video (to temp file if we need to mux with audio)
    if audio and audio_np is not None:
        # Save video to temp file, then mux with audio
        temp_video_path = output_path.parent / f"{output_path.stem}_temp.mp4"
        video_save_path = temp_video_path
    else:
        video_save_path = output_path

    try:
        import cv2
        h, w = video_np.shape[1], video_np.shape[2]
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(str(video_save_path), fourcc, fps, (w, h))
        for frame in video_np:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()

        if audio and audio_np is not None:
            # Mux video and audio
            print(f"{Colors.BLUE}Muxing video and audio...{Colors.RESET}")
            if mux_video_audio(temp_video_path, audio_output, output_path):
                print(f"{Colors.GREEN}Saved video with audio to{Colors.RESET} {output_path}")
                # Clean up temp file
                temp_video_path.unlink(missing_ok=True)
            else:
                # Fallback: keep separate files
                print(f"{Colors.YELLOW}Could not mux, keeping separate files{Colors.RESET}")
                temp_video_path.rename(output_path.parent / f"{output_path.stem}_video.mp4")
        else:
            print(f"{Colors.GREEN}Saved video to{Colors.RESET} {output_path}")
    except Exception as e:
        print(f"{Colors.RED}Could not save video: {e}{Colors.RESET}")

    if save_frames:
        frames_dir = output_path.parent / f"{output_path.stem}_frames"
        frames_dir.mkdir(exist_ok=True)
        for i, frame in enumerate(video_np):
            Image.fromarray(frame).save(frames_dir / f"frame_{i:04d}.png")
        print(f"{Colors.GREEN}Saved {len(video_np)} frames to {frames_dir}{Colors.RESET}")

    elapsed = time.time() - start_time
    print(f"{Colors.BOLD}{Colors.GREEN}Done! Generated in {elapsed:.1f}s ({elapsed/num_frames:.2f}s/frame){Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}Peak memory: {mx.get_peak_memory() / (1024 ** 3):.2f}GB{Colors.RESET}")

    return video_np


def main():
    parser = argparse.ArgumentParser(
        description="Generate videos with MLX LTX-2 Dev Model (with CFG)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Text-to-Video (T2V) with dev model
  python -m mlx_video.generate_dev --prompt "A cat walking on grass"
  python -m mlx_video.generate_dev --prompt "Ocean waves at sunset" --cfg-scale 6.0 --steps 50

  # With custom negative prompt
  python -m mlx_video.generate_dev --prompt "..." --negative-prompt "blurry, low quality"

  # Image-to-Video (I2V)
  python -m mlx_video.generate_dev --prompt "A person dancing" --image photo.jpg

  # With synchronized audio
  python -m mlx_video.generate_dev --prompt "Ocean waves crashing on rocks" --audio
  python -m mlx_video.generate_dev --prompt "A busy city street" --audio --output-audio street.wav
        """
    )

    parser.add_argument(
        "--prompt", "-p",
        type=str,
        required=True,
        help="Text description of the video to generate"
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=DEFAULT_NEGATIVE_PROMPT,
        help="Negative prompt for CFG guidance"
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
        default=768,
        help="Output video width (default: 768, must be divisible by 32)"
    )
    parser.add_argument(
        "--num-frames", "-n",
        type=int,
        default=33,
        help="Number of frames (default: 33)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=40,
        help="Number of inference steps (default: 40)"
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=4.0,
        help="CFG guidance scale (default: 4.0, 1.0 = no guidance)"
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
        "--output-path",
        type=str,
        default="output_dev.mp4",
        help="Output video path (default: output_dev.mp4)"
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Save individual frames as images"
    )
    parser.add_argument(
        "--model-repo",
        type=str,
        default="mlx-community/LTX-2-dev-bf16",
        help="Model repository to use (default: mlx-community/LTX-2-dev-bf16)"
    )
    parser.add_argument(
        "--text-encoder-repo",
        type=str,
        default=None,
        help="Text encoder repository to use (default: None)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--enhance-prompt",
        action="store_true",
        help="Enhance the prompt using Gemma before generation"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate (default: 512)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for prompt enhancement (default: 0.7)"
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        default=None,
        help="Path to conditioning image for I2V (Image-to-Video) generation"
    )
    parser.add_argument(
        "--image-strength",
        type=float,
        default=1.0,
        help="Conditioning strength for I2V (1.0 = full denoise, 0.0 = keep original, default: 1.0)"
    )
    parser.add_argument(
        "--image-frame-idx",
        type=int,
        default=0,
        help="Frame index to condition for I2V (0 = first frame, default: 0)"
    )
    parser.add_argument(
        "--tiling",
        type=str,
        default="none",
        choices=["none", "auto", "default", "aggressive", "conservative", "spatial", "temporal"],
        help="Tiling mode for VAE decoding (default: none, faster on high-memory systems)"
    )
    parser.add_argument(
        "--audio",
        action="store_true",
        help="Generate synchronized audio with the video"
    )
    parser.add_argument(
        "--output-audio",
        type=str,
        default=None,
        help="Output audio path (default: same as video with .wav extension)"
    )
    args = parser.parse_args()

    generate_video_dev(
        model_repo=args.model_repo,
        text_encoder_repo=args.text_encoder_repo,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
        fps=args.fps,
        output_path=args.output_path,
        output_audio_path=args.output_audio,
        save_frames=args.save_frames,
        verbose=args.verbose,
        enhance_prompt=args.enhance_prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        image=args.image,
        image_strength=args.image_strength,
        image_frame_idx=args.image_frame_idx,
        tiling=args.tiling,
        audio=args.audio,
    )


if __name__ == "__main__":
    main()
