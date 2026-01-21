"""Audio-Video generation pipeline for LTX-2."""

import argparse
import time
from pathlib import Path
from typing import Optional

import mlx.core as mx
import numpy as np
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
from mlx_video.utils import to_denoised, get_model_path, load_image, prepare_image_for_encoding
from mlx_video.models.ltx.video_vae.decoder import load_vae_decoder
from mlx_video.models.ltx.video_vae.encoder import load_vae_encoder
from mlx_video.models.ltx.video_vae.tiling import TilingConfig
from mlx_video.models.ltx.upsampler import load_upsampler, upsample_latents
from mlx_video.conditioning import VideoConditionByLatentIndex, apply_conditioning
from mlx_video.conditioning.latent import LatentState, apply_denoise_mask


# Distilled sigma schedules
STAGE_1_SIGMAS = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]
STAGE_2_SIGMAS = [0.909375, 0.725, 0.421875, 0.0]

# Audio constants
AUDIO_SAMPLE_RATE = 24000  # Output audio sample rate
AUDIO_LATENT_SAMPLE_RATE = 16000  # VAE internal sample rate
AUDIO_HOP_LENGTH = 160
AUDIO_LATENT_DOWNSAMPLE_FACTOR = 4
AUDIO_LATENT_CHANNELS = 8  # Latent channels before patchifying
AUDIO_MEL_BINS = 16
AUDIO_LATENTS_PER_SECOND = AUDIO_LATENT_SAMPLE_RATE / AUDIO_HOP_LENGTH / AUDIO_LATENT_DOWNSAMPLE_FACTOR  # 25


def create_video_position_grid(
    batch_size: int,
    num_frames: int,
    height: int,
    width: int,
    temporal_scale: int = 8,
    spatial_scale: int = 32,
    fps: float = 24.0,
    causal_fix: bool = True,
) -> mx.array:
    """Create position grid for video RoPE in pixel space."""
    patch_size_t, patch_size_h, patch_size_w = 1, 1, 1

    t_coords = np.arange(0, num_frames, patch_size_t)
    h_coords = np.arange(0, height, patch_size_h)
    w_coords = np.arange(0, width, patch_size_w)

    t_grid, h_grid, w_grid = np.meshgrid(t_coords, h_coords, w_coords, indexing='ij')
    patch_starts = np.stack([t_grid, h_grid, w_grid], axis=0)

    patch_size_delta = np.array([patch_size_t, patch_size_h, patch_size_w]).reshape(3, 1, 1, 1)
    patch_ends = patch_starts + patch_size_delta

    latent_coords = np.stack([patch_starts, patch_ends], axis=-1)
    num_patches = num_frames * height * width
    latent_coords = latent_coords.reshape(3, num_patches, 2)
    latent_coords = np.tile(latent_coords[np.newaxis, ...], (batch_size, 1, 1, 1))

    scale_factors = np.array([temporal_scale, spatial_scale, spatial_scale]).reshape(1, 3, 1, 1)
    pixel_coords = (latent_coords * scale_factors).astype(np.float32)

    if causal_fix:
        pixel_coords[:, 0, :, :] = np.clip(
            pixel_coords[:, 0, :, :] + 1 - temporal_scale,
            a_min=0,
            a_max=None
        )

    pixel_coords[:, 0, :, :] = pixel_coords[:, 0, :, :] / fps

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
    """
    def get_audio_latent_time_in_sec(start_idx: int, end_idx: int) -> np.ndarray:
        """Convert latent indices to seconds (matching PyTorch's _get_audio_latent_time_in_sec)."""
        latent_frame = np.arange(start_idx, end_idx, dtype=np.float32)
        mel_frame = latent_frame * downsample_factor
        if is_causal:
            # Frame offset for causal alignment (PyTorch uses +1 - downsample_factor)
            mel_frame = np.clip(mel_frame + 1 - downsample_factor, 0, None)
        return mel_frame * hop_length / sample_rate

    # Start times: latent indices 0 to audio_frames
    start_times = get_audio_latent_time_in_sec(0, audio_frames)

    # End times: latent indices 1 to audio_frames+1 (shifted by 1)
    end_times = get_audio_latent_time_in_sec(1, audio_frames + 1)

    # Shape: (B, 1, T, 2)
    positions = np.stack([start_times, end_times], axis=-1)
    positions = positions[np.newaxis, np.newaxis, :, :]  # (1, 1, T, 2)
    positions = np.tile(positions, (batch_size, 1, 1, 1))

    return mx.array(positions, dtype=mx.float32)


def compute_audio_frames(num_video_frames: int, fps: float) -> int:
    """Compute number of audio latent frames given video duration."""
    duration = num_video_frames / fps
    return round(duration * AUDIO_LATENTS_PER_SECOND)


def denoise_av(
    video_latents: mx.array,
    audio_latents: mx.array,
    video_positions: mx.array,
    audio_positions: mx.array,
    video_embeddings: mx.array,
    audio_embeddings: mx.array,
    transformer: LTXModel,
    sigmas: list,
    verbose: bool = True,
    video_state: Optional[LatentState] = None,
) -> tuple[mx.array, mx.array]:
    """Run denoising loop for audio-video generation with optional I2V conditioning.

    Args:
        video_latents: Video latent tensor (B, C, F, H, W)
        audio_latents: Audio latent tensor (B, C, T, F)
        video_positions: Video position embeddings
        audio_positions: Audio position embeddings
        video_embeddings: Video text embeddings
        audio_embeddings: Audio text embeddings
        transformer: LTX model
        sigmas: List of sigma values
        verbose: Whether to show progress bar
        video_state: Optional LatentState for I2V conditioning

    Returns:
        Tuple of (video_latents, audio_latents)
    """
    dtype = video_latents.dtype
    # If video state is provided, use its latent
    if video_state is not None:
        video_latents = video_state.latent

    for i in tqdm(range(len(sigmas) - 1), desc="Denoising A/V", disable=not verbose):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]

        # Flatten video latents
        b, c, f, h, w = video_latents.shape
        num_video_tokens = f * h * w
        video_flat = mx.transpose(mx.reshape(video_latents, (b, c, -1)), (0, 2, 1))

        # Flatten audio latents: (B, C, T, F) -> (B, T, C*F)
        ab, ac, at, af = audio_latents.shape
        audio_flat = mx.transpose(audio_latents, (0, 2, 1, 3))  # (B, T, C, F)
        audio_flat = mx.reshape(audio_flat, (ab, at, ac * af))

        # Compute per-token timesteps for video
        # For I2V: conditioned tokens get timestep=0 (mask=0), unconditioned get timestep=sigma (mask=1)
        if video_state is not None:
            # Reshape denoise_mask from (B, 1, F, 1, 1) to (B, num_tokens)
            denoise_mask_flat = mx.reshape(video_state.denoise_mask, (b, 1, f, 1, 1))
            denoise_mask_flat = mx.broadcast_to(denoise_mask_flat, (b, 1, f, h, w))
            denoise_mask_flat = mx.reshape(denoise_mask_flat, (b, num_video_tokens))
            # Per-token timesteps: sigma * mask
            video_timesteps = mx.array(sigma, dtype=dtype) * denoise_mask_flat
        else:
            # All tokens get the same timestep
            video_timesteps = mx.full((b, num_video_tokens), sigma, dtype=dtype)

        video_modality = Modality(
            latent=video_flat,
            timesteps=video_timesteps,
            positions=video_positions,
            context=video_embeddings,
            context_mask=None,
            enabled=True,
        )

        audio_modality = Modality(
            latent=audio_flat,
            timesteps=mx.full((ab, at), sigma, dtype=dtype),
            positions=audio_positions,
            context=audio_embeddings,
            context_mask=None,
            enabled=True,
        )

        video_velocity, audio_velocity = transformer(video=video_modality, audio=audio_modality)
        mx.eval(video_velocity, audio_velocity)

        # Reshape velocities back
        video_velocity = mx.reshape(mx.transpose(video_velocity, (0, 2, 1)), (b, c, f, h, w))
        audio_velocity = mx.reshape(audio_velocity, (ab, at, ac, af))
        audio_velocity = mx.transpose(audio_velocity, (0, 2, 1, 3))  # (B, C, T, F)

        # Compute denoised
        video_denoised = to_denoised(video_latents, video_velocity, sigma)
        audio_denoised = to_denoised(audio_latents, audio_velocity, sigma)

        # Apply conditioning mask for video if state is provided
        if video_state is not None:
            video_denoised = apply_denoise_mask(video_denoised, video_state.clean_latent, video_state.denoise_mask)

        mx.eval(video_denoised, audio_denoised)

        # Euler step - use dtype-preserving arrays to avoid float32 promotion
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

    # Load weights from main model file
    weight_file = model_path / "ltx-2-19b-distilled.safetensors"
    if weight_file.exists():
        raw_weights = mx.load(str(weight_file))
        sanitized = sanitize_audio_vae_weights(raw_weights)
        if sanitized:
            decoder.load_weights(list(sanitized.items()), strict=False)

            # Manually load per-channel statistics (they're plain mx.array, not tracked by load_weights)
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

    # Load weights
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


def mux_video_audio(video_path: Path, audio_path: Path, output_path: Path):
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


def generate_video_with_audio(
    model_repo: str,
    text_encoder_repo: Optional[str],
    prompt: str,
    height: int = 512,
    width: int = 512,
    num_frames: int = 33,
    seed: int = 42,
    fps: int = 24,
    output_path: str = "output_av.mp4",
    output_audio_path: Optional[str] = None,
    verbose: bool = True,
    enhance_prompt: bool = False,
    max_tokens: int = 512,
    temperature: float = 0.7,
    image: Optional[str] = None,
    image_strength: float = 1.0,
    image_frame_idx: int = 0,
    tiling: str = "auto",
):
    """Generate video with synchronized audio from text prompt, optionally conditioned on an image.

    Args:
        model_repo: Model repository ID
        text_encoder_repo: Text encoder repository ID
        prompt: Text description of the video to generate
        height: Output video height (must be divisible by 64)
        width: Output video width (must be divisible by 64)
        num_frames: Number of frames
        seed: Random seed
        fps: Frames per second
        output_path: Output video path
        output_audio_path: Output audio path
        verbose: Whether to print progress
        enhance_prompt: Whether to enhance prompt using Gemma
        max_tokens: Max tokens for prompt enhancement
        temperature: Temperature for prompt enhancement
        image: Path to conditioning image for I2V
        image_strength: Conditioning strength (1.0 = full denoise)
        image_frame_idx: Frame index to condition (0 = first frame)
        tiling: Tiling mode for VAE decoding (auto/none/default/aggressive/conservative/spatial/temporal)
    """
    start_time = time.time()

    # Validate dimensions
    assert height % 64 == 0, f"Height must be divisible by 64, got {height}"
    assert width % 64 == 0, f"Width must be divisible by 64, got {width}"

    if num_frames % 8 != 1:
        adjusted_num_frames = round((num_frames - 1) / 8) * 8 + 1
        print(f"{Colors.YELLOW}⚠️  Adjusted frames to {adjusted_num_frames}{Colors.RESET}")
        num_frames = adjusted_num_frames

    # Calculate audio frames
    audio_frames = compute_audio_frames(num_frames, fps)

    is_i2v = image is not None
    mode_str = "I2V+Audio" if is_i2v else "T2V+Audio"
    print(f"{Colors.BOLD}{Colors.CYAN}🎬 [{mode_str}] Generating {width}x{height} video with {num_frames} frames + audio{Colors.RESET}")
    print(f"{Colors.DIM}Audio: {audio_frames} latent frames @ {AUDIO_SAMPLE_RATE}Hz{Colors.RESET}")
    print(f"{Colors.DIM}Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}{Colors.RESET}")
    if is_i2v:
        print(f"{Colors.DIM}Image: {image} (strength={image_strength}, frame={image_frame_idx}){Colors.RESET}")

    model_path = get_model_path(model_repo)
    text_encoder_path = model_path if text_encoder_repo is None else get_model_path(text_encoder_repo)

    # Calculate latent dimensions
    stage1_h, stage1_w = height // 2 // 32, width // 2 // 32
    stage2_h, stage2_w = height // 32, width // 32
    latent_frames = 1 + (num_frames - 1) // 8

    mx.random.seed(seed)

    # Load text encoder with audio embeddings
    print(f"{Colors.BLUE}📝 Loading text encoder...{Colors.RESET}")
    from mlx_video.models.ltx.text_encoder import LTX2TextEncoder
    text_encoder = LTX2TextEncoder()
    text_encoder.load(model_path=model_path, text_encoder_path=text_encoder_path)
    mx.eval(text_encoder.parameters())

    # Optionally enhance prompt
    if enhance_prompt:
        print(f"{Colors.MAGENTA}✨ Enhancing prompt...{Colors.RESET}")
        prompt = text_encoder.enhance_t2v(prompt, max_tokens=max_tokens, temperature=temperature, seed=seed, verbose=verbose)
        print(f"{Colors.DIM}Enhanced: {prompt[:150]}{'...' if len(prompt) > 150 else ''}{Colors.RESET}")

    # Get both video and audio embeddings
    video_embeddings, audio_embeddings = text_encoder(prompt)
    model_dtype = video_embeddings.dtype  # bfloat16 from text encoder
    mx.eval(video_embeddings, audio_embeddings)

    del text_encoder
    mx.clear_cache()

    # Load transformer with AudioVideo config
    print(f"{Colors.BLUE}🤖 Loading transformer (A/V mode)...{Colors.RESET}")
    raw_weights = mx.load(str(model_path / 'ltx-2-19b-distilled.safetensors'))
    sanitized = sanitize_transformer_weights(raw_weights)

    # Convert transformer weights to bfloat16 for memory efficiency
    sanitized = {k: v.astype(mx.bfloat16) if v.dtype == mx.float32 else v for k, v in sanitized.items()}

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

    transformer = LTXModel(config)
    transformer.load_weights(list(sanitized.items()), strict=False)
    mx.eval(transformer.parameters())

    # Load VAE encoder and encode image for I2V conditioning
    stage1_image_latent = None
    stage2_image_latent = None
    if is_i2v:
        print(f"{Colors.BLUE}🖼️  Loading VAE encoder and encoding image...{Colors.RESET}")
        vae_encoder = load_vae_encoder(str(model_path / 'ltx-2-19b-distilled.safetensors'))
        mx.eval(vae_encoder.parameters())

        # Load and prepare image for stage 1 (half resolution)
        input_image = load_image(image, height=height // 2, width=width // 2, dtype=model_dtype)
        stage1_image_tensor = prepare_image_for_encoding(input_image, height // 2, width // 2, dtype=model_dtype)
        stage1_image_latent = vae_encoder(stage1_image_tensor)
        mx.eval(stage1_image_latent)

        # Load and prepare image for stage 2 (full resolution)
        input_image = load_image(image, height=height, width=width, dtype=model_dtype)
        stage2_image_tensor = prepare_image_for_encoding(input_image, height, width, dtype=model_dtype)
        stage2_image_latent = vae_encoder(stage2_image_tensor)
        mx.eval(stage2_image_latent)

        del vae_encoder
        mx.clear_cache()

    # Initialize latents
    print(f"{Colors.YELLOW}⚡ Stage 1: Generating at {width//2}x{height//2} (8 steps)...{Colors.RESET}")
    mx.random.seed(seed)

    # Create position grids - MUST stay float32 for RoPE precision
    # bfloat16 positions cause quality degradation due to precision loss in sin/cos calculations
    video_positions = create_video_position_grid(1, latent_frames, stage1_h, stage1_w)  # float32
    audio_positions = create_audio_position_grid(1, audio_frames)  # float32
    mx.eval(video_positions, audio_positions)

    # Apply I2V conditioning for stage 1 if provided
    video_state1 = None
    video_latent_shape = (1, 128, latent_frames, stage1_h, stage1_w)
    if is_i2v and stage1_image_latent is not None:
        # PyTorch flow: create zeros -> apply conditioning -> apply noiser
        video_state1 = LatentState(
            latent=mx.zeros(video_latent_shape, dtype=model_dtype),
            clean_latent=mx.zeros(video_latent_shape, dtype=model_dtype),
            denoise_mask=mx.ones((1, 1, latent_frames, 1, 1), dtype=model_dtype),
        )
        conditioning = VideoConditionByLatentIndex(
            latent=stage1_image_latent,
            frame_idx=image_frame_idx,
            strength=image_strength,
        )
        video_state1 = apply_conditioning(video_state1, [conditioning])

        # Apply noiser: latent = noise * (mask * noise_scale) + latent * (1 - mask * noise_scale)
        noise = mx.random.normal(video_latent_shape).astype(model_dtype)
        noise_scale = mx.array(STAGE_1_SIGMAS[0], dtype=model_dtype)  # 1.0
        scaled_mask = video_state1.denoise_mask * noise_scale
        video_state1 = LatentState(
            latent=noise * scaled_mask + video_state1.latent * (mx.array(1.0, dtype=model_dtype) - scaled_mask),
            clean_latent=video_state1.clean_latent,
            denoise_mask=video_state1.denoise_mask,
        )
        video_latents = video_state1.latent
        mx.eval(video_latents)
    else:
        # T2V: just use random noise
        video_latents = mx.random.normal(video_latent_shape).astype(model_dtype)
        mx.eval(video_latents)

    # Audio always uses pure noise (no I2V for audio)
    audio_latents = mx.random.normal((1, AUDIO_LATENT_CHANNELS, audio_frames, AUDIO_MEL_BINS)).astype(model_dtype)
    mx.eval(audio_latents)

    # Stage 1 denoising
    video_latents, audio_latents = denoise_av(
        video_latents, audio_latents,
        video_positions, audio_positions,
        video_embeddings, audio_embeddings,
        transformer, STAGE_1_SIGMAS, verbose=verbose,
        video_state=video_state1
    )

    # Upsample video latents
    print(f"{Colors.MAGENTA}🔍 Upsampling video latents 2x...{Colors.RESET}")
    upsampler = load_upsampler(str(model_path / 'ltx-2-spatial-upscaler-x2-1.0.safetensors'))
    mx.eval(upsampler.parameters())

    vae_decoder = load_vae_decoder(
        str(model_path / 'ltx-2-19b-distilled.safetensors'),
        timestep_conditioning=None  # Auto-detect from model metadata
    )

    video_latents = upsample_latents(video_latents, upsampler, vae_decoder.latents_mean, vae_decoder.latents_std)
    mx.eval(video_latents)

    del upsampler
    mx.clear_cache()

    # Stage 2: Refine at full resolution
    print(f"{Colors.YELLOW}⚡ Stage 2: Refining at {width}x{height} (3 steps)...{Colors.RESET}")
    # Position grids stay float32 for RoPE precision
    video_positions = create_video_position_grid(1, latent_frames, stage2_h, stage2_w)  # float32
    mx.eval(video_positions)

    # Apply I2V conditioning for stage 2 if provided
    video_state2 = None
    if is_i2v and stage2_image_latent is not None:
        # PyTorch flow: start with upscaled latent -> apply conditioning -> apply noiser
        video_state2 = LatentState(
            latent=video_latents,  # Start with upscaled latent
            clean_latent=mx.zeros_like(video_latents),
            denoise_mask=mx.ones((1, 1, latent_frames, 1, 1), dtype=model_dtype),
        )
        conditioning = VideoConditionByLatentIndex(
            latent=stage2_image_latent,
            frame_idx=image_frame_idx,
            strength=image_strength,
        )
        video_state2 = apply_conditioning(video_state2, [conditioning])

        # Apply noiser: conditioned frames (mask=0) keep image latent, unconditioned get partial noise
        video_noise = mx.random.normal(video_latents.shape).astype(model_dtype)
        noise_scale = mx.array(STAGE_2_SIGMAS[0], dtype=model_dtype)
        scaled_mask = video_state2.denoise_mask * noise_scale
        video_state2 = LatentState(
            latent=video_noise * scaled_mask + video_state2.latent * (mx.array(1.0, dtype=model_dtype) - scaled_mask),
            clean_latent=video_state2.clean_latent,
            denoise_mask=video_state2.denoise_mask,
        )
        video_latents = video_state2.latent
        mx.eval(video_latents)

        # Audio still gets noise (no I2V for audio)
        audio_noise = mx.random.normal(audio_latents.shape).astype(model_dtype)
        one_minus_scale = mx.array(1.0, dtype=model_dtype) - noise_scale
        audio_latents = audio_noise * noise_scale + audio_latents * one_minus_scale
        mx.eval(audio_latents)
    else:
        # T2V: add noise to all frames for refinement
        noise_scale = mx.array(STAGE_2_SIGMAS[0], dtype=model_dtype)
        one_minus_scale = mx.array(1.0, dtype=model_dtype) - noise_scale
        video_noise = mx.random.normal(video_latents.shape).astype(model_dtype)
        audio_noise = mx.random.normal(audio_latents.shape).astype(model_dtype)
        video_latents = video_noise * noise_scale + video_latents * one_minus_scale
        audio_latents = audio_noise * noise_scale + audio_latents * one_minus_scale
        mx.eval(video_latents, audio_latents)

    video_latents, audio_latents = denoise_av(
        video_latents, audio_latents,
        video_positions, audio_positions,
        video_embeddings, audio_embeddings,
        transformer, STAGE_2_SIGMAS, verbose=verbose,
        video_state=video_state2
    )

    del transformer
    mx.clear_cache()

    # Decode video with tiling
    print(f"{Colors.BLUE}🎞️  Decoding video...{Colors.RESET}")

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
        video = vae_decoder.decode_tiled(video_latents, tiling_config=tiling_config, debug=verbose)
    else:
        print(f"{Colors.DIM}  Tiling: disabled{Colors.RESET}")
        video = vae_decoder(video_latents)
    mx.eval(video)

    # Convert video to uint8 frames
    video = mx.squeeze(video, axis=0)
    video = mx.transpose(video, (1, 2, 3, 0))
    video = mx.clip((video + 1.0) / 2.0, 0.0, 1.0)
    video = (video * 255).astype(mx.uint8)
    video_np = np.array(video)

    # Decode audio
    print(f"{Colors.BLUE}🔊 Decoding audio...{Colors.RESET}")
    audio_decoder = load_audio_decoder(model_path)
    vocoder = load_vocoder(model_path)
    mx.eval(audio_decoder.parameters(), vocoder.parameters())

    mel_spectrogram = audio_decoder(audio_latents)
    mx.eval(mel_spectrogram)

    # Audio decoder output is already in vocoder format (B, C, T, F)
    audio_waveform = vocoder(mel_spectrogram)
    mx.eval(audio_waveform)

    audio_np = np.array(audio_waveform)
    if audio_np.ndim == 3:
        audio_np = audio_np[0]  # Remove batch dim

    del audio_decoder, vocoder, vae_decoder
    mx.clear_cache()

    # Save outputs
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save video (temporary without audio)
    temp_video_path = output_path.with_suffix('.temp.mp4')

    try:
        import cv2
        h, w = video_np.shape[1], video_np.shape[2]
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(str(temp_video_path), fourcc, fps, (w, h))
        for frame in video_np:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()
        print(f"{Colors.GREEN}✅ Video encoded{Colors.RESET}")
    except Exception as e:
        print(f"{Colors.RED}❌ Video encoding failed: {e}{Colors.RESET}")
        return None, None

    # Save audio
    audio_path = output_path.with_suffix('.wav') if output_audio_path is None else Path(output_audio_path)
    save_audio(audio_np, audio_path, AUDIO_SAMPLE_RATE)
    print(f"{Colors.GREEN}✅ Saved audio to{Colors.RESET} {audio_path}")

    # Mux video and audio
    print(f"{Colors.BLUE}🎬 Combining video and audio...{Colors.RESET}")
    if mux_video_audio(temp_video_path, audio_path, output_path):
        print(f"{Colors.GREEN}✅ Saved video with audio to{Colors.RESET} {output_path}")
        temp_video_path.unlink()  # Remove temp file
    else:
        # Fallback: keep video without audio
        temp_video_path.rename(output_path)
        print(f"{Colors.YELLOW}⚠️  Saved video without audio to{Colors.RESET} {output_path}")

    elapsed = time.time() - start_time
    print(f"{Colors.BOLD}{Colors.GREEN}🎉 Done! Generated in {elapsed:.1f}s{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}✨ Peak memory: {mx.get_peak_memory() / (1024 ** 3):.2f}GB{Colors.RESET}")

    return video_np, audio_np


def main():
    parser = argparse.ArgumentParser(
        description="Generate videos with synchronized audio using MLX LTX-2 (T2V and I2V)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Text-to-Video with Audio (T2V+Audio)
  python -m mlx_video.generate_av --prompt "Ocean waves crashing on a beach"
  python -m mlx_video.generate_av --prompt "A jazz band playing" --enhance-prompt
  python -m mlx_video.generate_av --prompt "..." --output my_video.mp4 --output-audio my_audio.wav

  # Image-to-Video with Audio (I2V+Audio)
  python -m mlx_video.generate_av --prompt "A person dancing" --image photo.jpg
  python -m mlx_video.generate_av --prompt "Waves crashing" --image beach.png --image-strength 0.8
        """
    )

    parser.add_argument("--prompt", "-p", type=str, required=True,
                        help="Text description of the video/audio to generate")
    parser.add_argument("--height", "-H", type=int, default=512,
                        help="Output video height (default: 512)")
    parser.add_argument("--width", "-W", type=int, default=512,
                        help="Output video width (default: 512)")
    parser.add_argument("--num-frames", "-n", type=int, default=65,
                        help="Number of frames (default: 65)")
    parser.add_argument("--seed", "-s", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--fps", type=int, default=24,
                        help="Frames per second (default: 24)")
    parser.add_argument("--output-path", type=str, default="output_av.mp4",
                        help="Output video path (default: output_av.mp4)")
    parser.add_argument("--output-audio", type=str, default=None,
                        help="Output audio path (default: same as video with .wav)")
    parser.add_argument("--model-repo", type=str, default="mlx-community/LTX-2-distilled-bf16",
                        help="Model repository (default: mlx-community/LTX-2-distilled-bf16)")
    parser.add_argument("--text-encoder-repo", type=str, default=None,
                        help="Text encoder repository")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output")
    parser.add_argument("--enhance-prompt", action="store_true",
                        help="Enhance prompt using Gemma")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max tokens for prompt enhancement")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for prompt enhancement")
    parser.add_argument("--image", "-i", type=str, default=None,
                        help="Path to conditioning image for I2V (Image-to-Video) generation")
    parser.add_argument("--image-strength", type=float, default=1.0,
                        help="Conditioning strength for I2V (1.0 = full denoise, 0.0 = keep original, default: 1.0)")
    parser.add_argument("--image-frame-idx", type=int, default=0,
                        help="Frame index to condition for I2V (0 = first frame, default: 0)")
    parser.add_argument("--tiling", type=str, default="auto",
                        choices=["auto", "none", "default", "aggressive", "conservative", "spatial", "temporal"],
                        help="Tiling mode for VAE decoding (default: auto). "
                             "auto=based on size, none=disabled, default=512px/64f, "
                             "aggressive=256px/32f (lowest memory), conservative=768px/96f")

    args = parser.parse_args()

    generate_video_with_audio(
        model_repo=args.model_repo,
        text_encoder_repo=args.text_encoder_repo,
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        seed=args.seed,
        fps=args.fps,
        output_path=args.output_path,
        output_audio_path=args.output_audio,
        verbose=args.verbose,
        enhance_prompt=args.enhance_prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        image=args.image,
        image_strength=args.image_strength,
        image_frame_idx=args.image_frame_idx,
        tiling=args.tiling,
    )


if __name__ == "__main__":
    main()
