"""Unified video and audio-video generation pipeline for LTX-2.

Supports both distilled (two-stage with upsampling) and dev (single-stage with CFG) pipelines.
"""

import argparse
import math
import time
from enum import Enum
from pathlib import Path
from typing import Optional

import mlx.core as mx
import numpy as np
from PIL import Image
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

# Rich console for styled output
console = Console()


from mlx_video.models.ltx_2.conditioning import (
    VideoConditionByLatentIndex,
    apply_conditioning,
)
from mlx_video.models.ltx_2.conditioning.latent import LatentState, apply_denoise_mask
from mlx_video.models.ltx_2.ltx_2 import LTXModel
from mlx_video.models.ltx_2.transformer import Modality
from mlx_video.models.ltx_2.upsampler import load_upsampler, upsample_latents
from mlx_video.models.ltx_2.video_vae import VideoEncoder
from mlx_video.models.ltx_2.video_vae.decoder import VideoDecoder
from mlx_video.models.ltx_2.video_vae.tiling import TilingConfig
from mlx_video.utils import (
    get_model_path,
    load_image,
    prepare_image_for_encoding,
)


class PipelineType(Enum):
    """Pipeline type selector."""

    DISTILLED = "distilled"  # Two-stage with upsampling, fixed sigmas, no CFG
    DEV = "dev"  # Single-stage, dynamic sigmas, CFG
    DEV_TWO_STAGE = (
        "dev-two-stage"  # Two-stage: dev (half res, CFG) + distilled LoRA (full res)
    )
    DEV_TWO_STAGE_HQ = "dev-two-stage-hq"  # Two-stage: res_2s sampler, LoRA both stages


# Distilled model sigma schedules
STAGE_1_SIGMAS = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]
STAGE_2_SIGMAS = [0.909375, 0.725, 0.421875, 0.0]

# Dev model scheduling constants
BASE_SHIFT_ANCHOR = 1024
MAX_SHIFT_ANCHOR = 4096

# Audio constants
AUDIO_SAMPLE_RATE = 24000  # Output audio sample rate
AUDIO_LATENT_SAMPLE_RATE = 16000  # VAE internal sample rate
AUDIO_HOP_LENGTH = 160
AUDIO_LATENT_DOWNSAMPLE_FACTOR = 4
AUDIO_LATENT_CHANNELS = 8  # Latent channels before patchifying
AUDIO_MEL_BINS = 16
AUDIO_LATENTS_PER_SECOND = (
    AUDIO_LATENT_SAMPLE_RATE / AUDIO_HOP_LENGTH / AUDIO_LATENT_DOWNSAMPLE_FACTOR
)  # 25

# Default negative prompt for CFG (dev pipeline)
# Matches PyTorch LTX-2 reference DEFAULT_NEGATIVE_PROMPT from constants.py
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


def load_and_merge_lora(
    model: LTXModel,
    lora_path: str,
    strength: float = 1.0,
) -> None:
    """Load LoRA weights and merge them into the transformer model in-place.

    Supports two formats:
    - Raw PyTorch: keys like diffusion_model.{module}.lora_A.weight (needs sanitization)
    - Pre-converted MLX: keys like {module}.lora_A.weight (already sanitized)

    Merge formula: weight += (lora_B * strength) @ lora_A

    Args:
        model: The LTXModel transformer to merge into
        lora_path: Path to the LoRA safetensors file or directory containing one
        strength: LoRA strength/coefficient (default 1.0)
    """
    # Resolve path: local file/dir or HuggingFace repo
    lora_file = Path(lora_path)
    if lora_file.is_file():
        pass  # direct file path
    elif lora_file.is_dir():
        # Local directory: find safetensors inside
        candidates = sorted(lora_file.glob("*.safetensors"))
        if not candidates:
            raise FileNotFoundError(f"No .safetensors files found in {lora_path}")
        # Prefer distilled-lora files over full model weights
        lora_candidates = [c for c in candidates if "distilled-lora" in c.name]
        lora_file = lora_candidates[0] if lora_candidates else candidates[0]
        console.print(f"[dim]Using LoRA file: {lora_file.name}[/]")
    else:
        # Treat as HuggingFace repo ID
        lora_dir = get_model_path(lora_path)
        candidates = sorted(lora_dir.glob("*.safetensors"))
        if not candidates:
            raise FileNotFoundError(f"No .safetensors files found in {lora_dir}")
        # Prefer distilled-lora files over full model weights
        lora_candidates = [c for c in candidates if "distilled-lora" in c.name]
        lora_file = lora_candidates[0] if lora_candidates else candidates[0]
        console.print(f"[dim]Using LoRA from repo: {lora_path} ({lora_file.name})[/]")

    # Load LoRA weights
    lora_weights = mx.load(str(lora_file))

    # Detect format: raw PyTorch has 'diffusion_model.' prefix
    has_prefix = any(k.startswith("diffusion_model.") for k in lora_weights)

    # Group into A/B pairs by module name
    lora_pairs = {}
    for key in lora_weights:
        module_key = key
        if has_prefix:
            if not key.startswith("diffusion_model."):
                continue
            module_key = key.replace("diffusion_model.", "")

        if module_key.endswith(".lora_A.weight"):
            base_key = module_key.replace(".lora_A.weight", "")
            lora_pairs.setdefault(base_key, {})["A"] = lora_weights[key]
        elif module_key.endswith(".lora_B.weight"):
            base_key = module_key.replace(".lora_B.weight", "")
            lora_pairs.setdefault(base_key, {})["B"] = lora_weights[key]

    # Apply key sanitization only for raw PyTorch format
    # Replacements handle both mid-string and end-of-string positions
    # since LoRA base keys end at the module name without trailing dot
    _LORA_KEY_REPLACEMENTS = [
        (".to_out.0", ".to_out"),
        (".ff.net.0.proj", ".ff.proj_in"),
        (".ff.net.2", ".ff.proj_out"),
        (".audio_ff.net.0.proj", ".audio_ff.proj_in"),
        (".audio_ff.net.2", ".audio_ff.proj_out"),
        (".linear_1", ".linear1"),
        (".linear_2", ".linear2"),
    ]
    if has_prefix:
        sanitized_pairs = {}
        for key, pair in lora_pairs.items():
            new_key = key
            for old, new in _LORA_KEY_REPLACEMENTS:
                if new_key.endswith(old):
                    new_key = new_key[: -len(old)] + new
                else:
                    new_key = new_key.replace(old + ".", new + ".")
            sanitized_pairs[new_key] = pair
    else:
        sanitized_pairs = lora_pairs

    # Get current model weights as a flat dict (references, not copies)
    def flatten_params(params, prefix=""):
        flat = {}
        for k, v in params.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                flat.update(flatten_params(v, full_key))
            else:
                flat[full_key] = v
        return flat

    flat_weights = flatten_params(dict(model.parameters()))

    # Merge LoRA deltas in batches to avoid doubling memory
    merged_count = 0
    batch = []
    batch_size = 100  # merge 100 weights at a time, then eval to free intermediates

    for module_key, pair in sanitized_pairs.items():
        if "A" not in pair or "B" not in pair:
            continue

        weight_key = f"{module_key}.weight"
        if weight_key not in flat_weights:
            continue

        lora_a = pair["A"].astype(mx.float32)  # (rank, in_features)
        lora_b = pair["B"].astype(mx.float32)  # (out_features, rank)

        # delta = (lora_B * strength) @ lora_A
        delta = (lora_b * strength) @ lora_a

        base_weight = flat_weights.pop(weight_key)
        merged_weight = (base_weight.astype(mx.float32) + delta).astype(
            base_weight.dtype
        )
        batch.append((weight_key, merged_weight))
        del base_weight
        merged_count += 1

        if len(batch) >= batch_size:
            model.load_weights(batch, strict=False)
            mx.eval(model.parameters())
            batch.clear()

    if batch:
        model.load_weights(batch, strict=False)
        mx.eval(model.parameters())
        batch.clear()

    del flat_weights, lora_weights
    mx.clear_cache()
    console.print(f"[green]✓[/] Merged {merged_count} LoRA pairs (strength={strength})")


def cfg_delta(cond: mx.array, uncond: mx.array, scale: float) -> mx.array:
    """Compute CFG delta for classifier-free guidance.

    Args:
        cond: Conditional prediction
        uncond: Unconditional prediction
        scale: CFG guidance scale

    Returns:
        Delta to add to unconditional for CFG: (scale - 1) * (cond - uncond)
    """
    return (scale - 1.0) * (cond - uncond)


def apg_delta(
    cond: mx.array,
    uncond: mx.array,
    scale: float,
    eta: float = 1.0,
    norm_threshold: float = 0.0,
) -> mx.array:
    """Compute APG (Adaptive Projected Guidance) delta.

    Decomposes guidance into parallel and orthogonal components relative to
    the conditional prediction, providing more stable guidance for I2V.

    Based on: https://arxiv.org/abs/2407.12173

    Args:
        cond: Conditional prediction (x0_pos)
        uncond: Unconditional prediction (x0_neg)
        scale: Guidance strength (same as CFG scale)
        eta: Weight for parallel component (1.0 = keep full parallel)
        norm_threshold: Clamp guidance norm to this value (0 = no clamping)

    Returns:
        Delta to add to unconditional for APG guidance
    """
    guidance = cond - uncond

    # Optionally clamp guidance norm for stability
    if norm_threshold > 0:
        guidance_norm = mx.sqrt(
            mx.sum(guidance**2, axis=(-1, -2, -3), keepdims=True) + 1e-8
        )
        scale_factor = mx.minimum(
            mx.ones_like(guidance_norm), norm_threshold / guidance_norm
        )
        guidance = guidance * scale_factor

    # Project guidance onto cond direction
    batch_size = cond.shape[0]
    cond_flat = mx.reshape(cond, (batch_size, -1))
    guidance_flat = mx.reshape(guidance, (batch_size, -1))

    # Projection coefficient: (guidance · cond) / (cond · cond)
    dot_product = mx.sum(guidance_flat * cond_flat, axis=1, keepdims=True)
    squared_norm = mx.sum(cond_flat**2, axis=1, keepdims=True) + 1e-8
    proj_coeff = dot_product / squared_norm

    # Reshape back and compute parallel/orthogonal components
    proj_coeff = mx.reshape(proj_coeff, (batch_size,) + (1,) * (cond.ndim - 1))
    g_parallel = proj_coeff * cond
    g_orth = guidance - g_parallel

    # Combine with eta weighting parallel component
    g_apg = g_parallel * eta + g_orth

    return g_apg * (scale - 1.0)


def ltx2_scheduler(
    steps: int,
    num_tokens: Optional[int] = None,
    max_shift: float = 2.05,
    base_shift: float = 0.95,
    stretch: bool = True,
    terminal: float = 0.1,
) -> mx.array:
    """LTX-2 scheduler for sigma generation (dev model).

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
    with np.errstate(divide="ignore", invalid="ignore"):
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
    patch_size_t, patch_size_h, patch_size_w = 1, 1, 1

    t_coords = np.arange(0, num_frames, patch_size_t)
    h_coords = np.arange(0, height, patch_size_h)
    w_coords = np.arange(0, width, patch_size_w)

    t_grid, h_grid, w_grid = np.meshgrid(t_coords, h_coords, w_coords, indexing="ij")
    patch_starts = np.stack([t_grid, h_grid, w_grid], axis=0)

    patch_size_delta = np.array([patch_size_t, patch_size_h, patch_size_w]).reshape(
        3, 1, 1, 1
    )
    patch_ends = patch_starts + patch_size_delta

    latent_coords = np.stack([patch_starts, patch_ends], axis=-1)
    num_patches = num_frames * height * width
    latent_coords = latent_coords.reshape(3, num_patches, 2)
    latent_coords = np.tile(latent_coords[np.newaxis, ...], (batch_size, 1, 1, 1))

    scale_factors = np.array([temporal_scale, spatial_scale, spatial_scale]).reshape(
        1, 3, 1, 1
    )
    pixel_coords = (latent_coords * scale_factors).astype(np.float32)

    if causal_fix:
        pixel_coords[:, 0, :, :] = np.clip(
            pixel_coords[:, 0, :, :] + 1 - temporal_scale, a_min=0, a_max=None
        )

    # Divide temporal coords by fps
    pixel_coords[:, 0, :, :] = pixel_coords[:, 0, :, :] / fps

    # Cast entire position grid through bfloat16 to match PyTorch's behavior.
    # PyTorch does: positions = positions.to(bfloat16) on ALL coordinates before
    # passing to the transformer/RoPE. This quantization is what the model was
    # trained with, so we must replicate it for numerical fidelity.
    positions_bf16 = mx.array(pixel_coords, dtype=mx.bfloat16)
    mx.eval(positions_bf16)
    return positions_bf16.astype(mx.float32)


def create_audio_position_grid(
    batch_size: int,
    audio_frames: int,
    sample_rate: int = AUDIO_LATENT_SAMPLE_RATE,
    hop_length: int = AUDIO_HOP_LENGTH,
    downsample_factor: int = AUDIO_LATENT_DOWNSAMPLE_FACTOR,
    is_causal: bool = True,
) -> mx.array:
    """Create temporal position grid for audio RoPE."""

    def get_audio_latent_time_in_sec(start_idx: int, end_idx: int) -> np.ndarray:
        latent_frame = np.arange(start_idx, end_idx, dtype=np.float32)
        mel_frame = latent_frame * downsample_factor
        if is_causal:
            mel_frame = np.clip(mel_frame + 1 - downsample_factor, 0, None)
        return mel_frame * hop_length / sample_rate

    start_times = get_audio_latent_time_in_sec(0, audio_frames)
    end_times = get_audio_latent_time_in_sec(1, audio_frames + 1)

    positions = np.stack([start_times, end_times], axis=-1)
    positions = positions[np.newaxis, np.newaxis, :, :]
    positions = np.tile(positions, (batch_size, 1, 1, 1))

    # Cast through bfloat16 to match PyTorch's precision behavior
    positions_bf16 = mx.array(positions, dtype=mx.bfloat16)
    mx.eval(positions_bf16)
    return positions_bf16.astype(mx.float32)


def compute_audio_frames(num_video_frames: int, fps: float) -> int:
    """Compute number of audio latent frames given video duration."""
    duration = num_video_frames / fps
    return round(duration * AUDIO_LATENTS_PER_SECOND)


# =============================================================================
# Distilled Pipeline Denoising (no CFG, fixed sigmas)
# =============================================================================


def denoise_distilled(
    latents: mx.array,
    positions: mx.array,
    text_embeddings: mx.array,
    transformer: LTXModel,
    sigmas: list,
    verbose: bool = True,
    state: Optional[LatentState] = None,
    audio_latents: Optional[mx.array] = None,
    audio_positions: Optional[mx.array] = None,
    audio_embeddings: Optional[mx.array] = None,
    audio_frozen: bool = False,
    guiding_tokens: Optional[mx.array] = None,
    guiding_positions: Optional[mx.array] = None,
    guiding_strength: float = 1.0,
) -> tuple[mx.array, Optional[mx.array]]:
    """Run denoising loop for distilled pipeline (no CFG).

    guiding_tokens/guiding_positions: end-frame conditioning via token appending
    (LTX-2 KeyframeIndex approach). Tokens are appended to the video sequence
    before the transformer so the model attends to them without hard-replacing
    any video token, producing smoother transitions to the target end frame.
    """
    dtype = latents.dtype
    enable_audio = audio_latents is not None

    if state is not None:
        latents = state.latent

    # Keep latents in float32 throughout to avoid quantization noise accumulation.
    latents = latents.astype(mx.float32)
    if enable_audio:
        audio_latents = audio_latents.astype(mx.float32)

    desc = "[cyan]Denoising A/V[/]" if enable_audio else "[cyan]Denoising[/]"
    num_steps = len(sigmas) - 1

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        disable=not verbose,
    ) as progress:
        task = progress.add_task(desc, total=num_steps)

        for i in range(num_steps):
            sigma, sigma_next = sigmas[i], sigmas[i + 1]

            b, c, f, h, w = latents.shape
            num_tokens = f * h * w
            # Cast to model dtype for transformer input
            latents_flat = mx.transpose(
                mx.reshape(latents, (b, c, -1)), (0, 2, 1)
            ).astype(dtype)

            if state is not None:
                denoise_mask_flat = mx.reshape(state.denoise_mask, (b, 1, f, 1, 1))
                denoise_mask_flat = mx.broadcast_to(denoise_mask_flat, (b, 1, f, h, w))
                denoise_mask_flat = mx.reshape(denoise_mask_flat, (b, num_tokens))
                timesteps = mx.array(sigma, dtype=dtype) * denoise_mask_flat
            else:
                timesteps = mx.full((b, num_tokens), sigma, dtype=dtype)

            # Append guiding tokens (end-frame keyframe conditioning).
            # Tokens are appended to the sequence so the transformer attends to
            # them without replacing any video token — avoids the hard boundary
            # that causes abrupt appearance changes near the end frame.
            if guiding_tokens is not None:
                n_cond = guiding_tokens.shape[1]
                cond_ts = mx.full(
                    (b, n_cond),
                    sigma * (1.0 - guiding_strength),
                    dtype=dtype,
                )
                seq_latent = mx.concatenate(
                    [latents_flat, guiding_tokens.astype(dtype)], axis=1
                )
                seq_timesteps = mx.concatenate([timesteps, cond_ts], axis=1)
                seq_positions = mx.concatenate([positions, guiding_positions], axis=2)
            else:
                seq_latent = latents_flat
                seq_timesteps = timesteps
                seq_positions = positions

            video_modality = Modality(
                latent=seq_latent,
                timesteps=seq_timesteps,
                positions=seq_positions,
                context=text_embeddings,
                context_mask=None,
                enabled=True,
                sigma=mx.full((b,), sigma, dtype=dtype),
            )

            audio_modality = None
            if enable_audio:
                ab, ac, at, af = audio_latents.shape
                audio_flat = mx.transpose(audio_latents, (0, 2, 1, 3))
                audio_flat = mx.reshape(audio_flat, (ab, at, ac * af)).astype(dtype)

                # A2V: frozen audio uses timesteps=0 (tells model audio is clean)
                a_ts = (
                    mx.zeros((ab, at), dtype=dtype)
                    if audio_frozen
                    else mx.full((ab, at), sigma, dtype=dtype)
                )
                a_sig = (
                    mx.zeros((ab,), dtype=dtype)
                    if audio_frozen
                    else mx.full((ab,), sigma, dtype=dtype)
                )
                audio_modality = Modality(
                    latent=audio_flat,
                    timesteps=a_ts,
                    positions=audio_positions,
                    context=audio_embeddings,
                    context_mask=None,
                    enabled=True,
                    sigma=a_sig,
                )

            velocity, audio_velocity = transformer(
                video=video_modality, audio=audio_modality
            )
            # Drop guiding token outputs — only use velocity for the video tokens.
            if guiding_tokens is not None:
                velocity = velocity[:, :num_tokens, :]
            mx.eval(velocity)
            if audio_velocity is not None:
                mx.eval(audio_velocity)

            # Compute denoised (x0) using per-token timesteps in float32
            sigma_f32 = mx.array(sigma, dtype=mx.float32)
            latents_flat_f32 = mx.transpose(mx.reshape(latents, (b, c, -1)), (0, 2, 1))
            timesteps_f32 = mx.expand_dims(timesteps.astype(mx.float32), axis=-1)
            x0_f32 = latents_flat_f32 - timesteps_f32 * velocity.astype(mx.float32)
            denoised = mx.reshape(mx.transpose(x0_f32, (0, 2, 1)), (b, c, f, h, w))

            audio_denoised = None
            if enable_audio and audio_velocity is not None and not audio_frozen:
                ab, ac, at, af = audio_latents.shape
                audio_velocity = mx.reshape(audio_velocity, (ab, at, ac, af))
                audio_velocity = mx.transpose(audio_velocity, (0, 2, 1, 3))
                audio_denoised = audio_latents - sigma_f32 * audio_velocity.astype(
                    mx.float32
                )

            if state is not None:
                denoised = apply_denoise_mask(
                    denoised, state.clean_latent.astype(mx.float32), state.denoise_mask
                )

            mx.eval(denoised)
            if audio_denoised is not None:
                mx.eval(audio_denoised)

            # Euler step in float32
            if sigma_next > 0:
                sigma_next_f32 = mx.array(sigma_next, dtype=mx.float32)
                latents = denoised + sigma_next_f32 * (latents - denoised) / sigma_f32
                if enable_audio and audio_denoised is not None and not audio_frozen:
                    audio_latents = (
                        audio_denoised
                        + sigma_next_f32 * (audio_latents - audio_denoised) / sigma_f32
                    )
            else:
                latents = denoised
                if enable_audio and audio_denoised is not None and not audio_frozen:
                    audio_latents = audio_denoised

            mx.eval(latents)
            if enable_audio:
                mx.eval(audio_latents)

            progress.advance(task)

    return latents.astype(dtype), audio_latents.astype(dtype) if enable_audio else None


# =============================================================================
# Dev Pipeline Denoising (with CFG, dynamic sigmas)
# =============================================================================


def denoise_dev(
    latents: mx.array,
    positions: mx.array,
    text_embeddings_pos: mx.array,
    text_embeddings_neg: mx.array,
    transformer: LTXModel,
    sigmas: mx.array,
    cfg_scale: float = 4.0,
    cfg_rescale: float = 0.0,
    verbose: bool = True,
    state: Optional[LatentState] = None,
    use_apg: bool = False,
    apg_eta: float = 1.0,
    apg_norm_threshold: float = 0.0,
    stg_scale: float = 0.0,
    stg_blocks: Optional[list] = None,
) -> mx.array:
    """Run denoising loop for dev pipeline with CFG/APG and optional STG guidance.

    Args:
        cfg_rescale: Rescale factor for CFG (0.0-1.0). Normalizes guided prediction
                     variance relative to conditional prediction to reduce over-saturation.
                     PyTorch default is 0.7. Set to 0.0 to disable.
        use_apg: Use Adaptive Projected Guidance instead of standard CFG.
                 APG decomposes guidance into parallel/orthogonal components
                 for more stable I2V generation.
        apg_eta: APG parallel component weight (1.0 = keep full parallel)
        apg_norm_threshold: APG guidance norm clamp (0 = no clamping)
        stg_scale: STG (Spatiotemporal Guidance) scale. 0.0 = disabled.
        stg_blocks: Transformer block indices for STG perturbation.
    """
    from mlx_video.models.ltx_2.rope import precompute_freqs_cis

    dtype = latents.dtype
    if state is not None:
        latents = state.latent

    # Keep latents in float32 throughout the denoising loop to avoid
    # quantization noise accumulation over many steps.
    # Model input is cast to model dtype; all denoising math stays in float32.
    latents = latents.astype(mx.float32)

    sigmas_list = sigmas.tolist()
    use_cfg = cfg_scale != 1.0
    use_stg = stg_scale != 0.0 and stg_blocks is not None
    num_steps = len(sigmas_list) - 1

    # Precompute RoPE once
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

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        disable=not verbose,
    ) as progress:
        passes = ["CFG"] if use_cfg else []
        if use_stg:
            passes.append("STG")
        label = "+".join(passes) if passes else "uncond"
        task = progress.add_task(f"[cyan]Denoising ({label})[/]", total=num_steps)

        for i in range(num_steps):
            sigma = sigmas_list[i]
            sigma_next = sigmas_list[i + 1]

            b, c, f, h, w = latents.shape
            num_tokens = f * h * w
            # Cast to model dtype for transformer input
            latents_flat = mx.transpose(
                mx.reshape(latents, (b, c, -1)), (0, 2, 1)
            ).astype(dtype)

            if state is not None:
                denoise_mask_flat = mx.reshape(state.denoise_mask, (b, 1, f, 1, 1))
                denoise_mask_flat = mx.broadcast_to(denoise_mask_flat, (b, 1, f, h, w))
                denoise_mask_flat = mx.reshape(denoise_mask_flat, (b, num_tokens))
                timesteps = mx.array(sigma, dtype=dtype) * denoise_mask_flat
            else:
                timesteps = mx.full((b, num_tokens), sigma, dtype=dtype)

            sigma_array = mx.full((b,), sigma, dtype=dtype)

            # Positive conditioning pass
            video_modality_pos = Modality(
                latent=latents_flat,
                timesteps=timesteps,
                positions=positions,
                context=text_embeddings_pos,
                context_mask=None,
                enabled=True,
                positional_embeddings=precomputed_rope,
                sigma=sigma_array,
            )
            velocity_pos, _ = transformer(video=video_modality_pos, audio=None)

            # Convert velocity to x0 (denoised) using per-token timesteps
            # Matches PyTorch's X0Model: x0 = latent - timestep * velocity
            # For conditioned tokens (timestep=0): x0 = latent (correct regardless of velocity)
            # For unconditioned tokens (timestep=sigma): x0 = latent - sigma * velocity
            latents_flat_f32 = mx.transpose(mx.reshape(latents, (b, c, -1)), (0, 2, 1))
            timesteps_f32 = mx.expand_dims(timesteps.astype(mx.float32), axis=-1)
            x0_pos_f32 = latents_flat_f32 - timesteps_f32 * velocity_pos.astype(
                mx.float32
            )

            # Start with positive prediction
            x0_guided_f32 = x0_pos_f32

            if use_cfg:
                # Negative conditioning pass
                video_modality_neg = Modality(
                    latent=latents_flat,
                    timesteps=timesteps,
                    positions=positions,
                    context=text_embeddings_neg,
                    context_mask=None,
                    enabled=True,
                    positional_embeddings=precomputed_rope,
                    sigma=sigma_array,
                )
                velocity_neg, _ = transformer(video=video_modality_neg, audio=None)

                # Convert negative velocity to x0 using per-token timesteps
                x0_neg_f32 = latents_flat_f32 - timesteps_f32 * velocity_neg.astype(
                    mx.float32
                )

                # Apply guidance to x0 predictions
                # For conditioned tokens: x0_pos = x0_neg = latent, so delta = 0
                if use_apg:
                    # APG: decompose into parallel/orthogonal components for stability
                    x0_guided_f32 = x0_pos_f32 + apg_delta(
                        x0_pos_f32,
                        x0_neg_f32,
                        cfg_scale,
                        eta=apg_eta,
                        norm_threshold=apg_norm_threshold,
                    )
                else:
                    # Standard CFG
                    x0_guided_f32 = x0_pos_f32 + (cfg_scale - 1.0) * (
                        x0_pos_f32 - x0_neg_f32
                    )

            # STG pass: skip self-attention at specified blocks
            if use_stg:
                velocity_ptb, _ = transformer(
                    video=video_modality_pos,
                    audio=None,
                    stg_video_blocks=stg_blocks,
                )
                mx.eval(velocity_ptb)

                x0_ptb_f32 = latents_flat_f32 - timesteps_f32 * velocity_ptb.astype(
                    mx.float32
                )
                x0_guided_f32 = x0_guided_f32 + stg_scale * (x0_pos_f32 - x0_ptb_f32)

            # Apply CFG rescale if enabled (std-ratio rescaling to reduce over-saturation)
            # factor = rescale * (cond_std / pred_std) + (1 - rescale)
            # pred = pred * factor
            if cfg_rescale > 0.0 and (use_cfg or use_stg):
                v_factor = x0_pos_f32.std() / (x0_guided_f32.std() + 1e-8)
                v_factor = cfg_rescale * v_factor + (1.0 - cfg_rescale)
                x0_guided_f32 = x0_guided_f32 * v_factor

            # Reshape x0 from token space (b, tokens, c) to spatial (b, c, f, h, w)
            denoised = mx.reshape(
                mx.transpose(x0_guided_f32, (0, 2, 1)), (b, c, f, h, w)
            )

            sigma_f32 = mx.array(sigma, dtype=mx.float32)

            if state is not None:
                denoised = apply_denoise_mask(
                    denoised, state.clean_latent.astype(mx.float32), state.denoise_mask
                )

            # Euler step in float32 (latents stay in float32)
            if sigma_next > 0:
                sigma_next_f32 = mx.array(sigma_next, dtype=mx.float32)
                latents = denoised + sigma_next_f32 * (latents - denoised) / sigma_f32
            else:
                latents = denoised

            mx.eval(latents)
            progress.advance(task)

    return latents.astype(dtype)


def denoise_dev_av(
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
    audio_cfg_scale: float = 7.0,
    cfg_rescale: float = 0.0,
    verbose: bool = True,
    video_state: Optional[LatentState] = None,
    use_apg: bool = False,
    apg_eta: float = 1.0,
    apg_norm_threshold: float = 0.0,
    stg_scale: float = 0.0,
    stg_video_blocks: Optional[list] = None,
    stg_audio_blocks: Optional[list] = None,
    modality_scale: float = 1.0,
    audio_frozen: bool = False,
    guiding_tokens: Optional[mx.array] = None,
    guiding_positions: Optional[mx.array] = None,
    guiding_strength: float = 1.0,
) -> tuple[mx.array, mx.array]:
    """Run denoising loop for dev pipeline with CFG/APG, STG, modality guidance, and audio.

    Args:
        audio_cfg_scale: Separate CFG scale for audio (PyTorch default: 7.0).
        cfg_rescale: Rescale factor for CFG (0.0-1.0). Normalizes guided prediction
                     variance to reduce artifacts. Default 0.0 means no rescaling.
        use_apg: Use Adaptive Projected Guidance instead of standard CFG for video.
        apg_eta: APG parallel component weight (1.0 = keep full parallel)
        apg_norm_threshold: APG guidance norm clamp (0 = no clamping)
        stg_scale: STG (Spatiotemporal Guidance) scale. 0.0 = disabled.
        stg_video_blocks: Transformer block indices for video STG perturbation.
        stg_audio_blocks: Transformer block indices for audio STG perturbation.
        modality_scale: Cross-modal guidance scale. 1.0 = disabled.
    """
    from mlx_video.models.ltx_2.rope import precompute_freqs_cis

    dtype = video_latents.dtype
    if video_state is not None:
        video_latents = video_state.latent

    # Keep latents in float32 throughout the denoising loop for precision.
    video_latents = video_latents.astype(mx.float32)
    audio_latents = audio_latents.astype(mx.float32)

    sigmas_list = sigmas.tolist()
    use_cfg = cfg_scale != 1.0
    use_stg = stg_scale != 0.0 and stg_video_blocks is not None
    use_modality = modality_scale != 1.0
    num_steps = len(sigmas_list) - 1

    # Precompute video RoPE
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

    # Precompute audio RoPE
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

    # Pre-compute extended RoPE covering video + guiding tokens (once, outside loop).
    n_guiding = 0
    extended_video_rope = precomputed_video_rope
    if guiding_tokens is not None:
        n_guiding = guiding_tokens.shape[1]
        extended_positions = mx.concatenate([video_positions, guiding_positions], axis=2)
        extended_video_rope = precompute_freqs_cis(
            extended_positions,
            dim=transformer.inner_dim,
            theta=transformer.positional_embedding_theta,
            max_pos=transformer.positional_embedding_max_pos,
            use_middle_indices_grid=transformer.use_middle_indices_grid,
            num_attention_heads=transformer.num_attention_heads,
            rope_type=transformer.rope_type,
            double_precision=transformer.config.double_precision_rope,
        )
        mx.eval(extended_video_rope)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        disable=not verbose,
    ) as progress:
        passes = ["CFG"] if use_cfg else []
        if use_stg:
            passes.append("STG")
        if use_modality:
            passes.append("Mod")
        label = "+".join(passes) if passes else "uncond"
        task = progress.add_task(f"[cyan]Denoising A/V ({label})[/]", total=num_steps)

        for i in range(num_steps):
            sigma = sigmas_list[i]
            sigma_next = sigmas_list[i + 1]

            # Flatten video latents (cast to model dtype for transformer input)
            b, c, f, h, w = video_latents.shape
            num_video_tokens = f * h * w
            video_flat = mx.transpose(
                mx.reshape(video_latents, (b, c, -1)), (0, 2, 1)
            ).astype(dtype)

            # Flatten audio latents (cast to model dtype for transformer input)
            ab, ac, at, af = audio_latents.shape
            audio_flat = mx.transpose(audio_latents, (0, 2, 1, 3))
            audio_flat = mx.reshape(audio_flat, (ab, at, ac * af)).astype(dtype)

            # Compute timesteps
            if video_state is not None:
                denoise_mask_flat = mx.reshape(
                    video_state.denoise_mask, (b, 1, f, 1, 1)
                )
                denoise_mask_flat = mx.broadcast_to(denoise_mask_flat, (b, 1, f, h, w))
                denoise_mask_flat = mx.reshape(denoise_mask_flat, (b, num_video_tokens))
                video_timesteps = mx.array(sigma, dtype=dtype) * denoise_mask_flat
            else:
                video_timesteps = mx.full((b, num_video_tokens), sigma, dtype=dtype)

            # A2V: frozen audio uses timesteps=0 (tells model audio is clean)
            audio_timesteps = (
                mx.zeros((ab, at), dtype=dtype)
                if audio_frozen
                else mx.full((ab, at), sigma, dtype=dtype)
            )

            # Positive conditioning pass
            sigma_array = mx.full((b,), sigma, dtype=dtype)
            audio_sigma_array = (
                mx.zeros((ab,), dtype=dtype)
                if audio_frozen
                else mx.full((ab,), sigma, dtype=dtype)
            )

            # Build extended sequence if end-frame guiding tokens are active.
            if guiding_tokens is not None:
                cond_ts = mx.full(
                    (b, n_guiding), sigma * (1.0 - guiding_strength), dtype=dtype
                )
                seq_video_flat = mx.concatenate(
                    [video_flat, guiding_tokens.astype(dtype)], axis=1
                )
                seq_video_timesteps = mx.concatenate([video_timesteps, cond_ts], axis=1)
            else:
                seq_video_flat = video_flat
                seq_video_timesteps = video_timesteps

            video_modality_pos = Modality(
                latent=seq_video_flat,
                timesteps=seq_video_timesteps,
                positions=video_positions,
                context=video_embeddings_pos,
                context_mask=None,
                enabled=True,
                positional_embeddings=extended_video_rope,
                sigma=sigma_array,
            )
            audio_modality_pos = Modality(
                latent=audio_flat,
                timesteps=audio_timesteps,
                positions=audio_positions,
                context=audio_embeddings_pos,
                context_mask=None,
                enabled=True,
                positional_embeddings=precomputed_audio_rope,
                sigma=audio_sigma_array,
            )
            video_vel_pos, audio_vel_pos = transformer(
                video=video_modality_pos, audio=audio_modality_pos
            )
            if guiding_tokens is not None:
                video_vel_pos = video_vel_pos[:, :num_video_tokens, :]
            mx.eval(video_vel_pos, audio_vel_pos)

            # Convert velocity to denoised (x0) using per-token timesteps
            # This matches PyTorch's X0ModelWrapper: x0 = latent - timestep * velocity
            # For conditioned tokens (timestep=0): x0 = latent (velocity is irrelevant)
            # For unconditioned tokens (timestep=sigma): x0 = latent - sigma * velocity
            video_flat_f32 = mx.transpose(
                mx.reshape(video_latents, (b, c, -1)), (0, 2, 1)
            )
            audio_flat_f32 = mx.reshape(
                mx.transpose(audio_latents, (0, 2, 1, 3)), (ab, at, ac * af)
            )
            video_timesteps_f32 = mx.expand_dims(
                video_timesteps.astype(mx.float32), axis=-1
            )
            audio_timesteps_f32 = mx.expand_dims(
                audio_timesteps.astype(mx.float32), axis=-1
            )

            video_x0_pos_f32 = (
                video_flat_f32 - video_timesteps_f32 * video_vel_pos.astype(mx.float32)
            )
            audio_x0_pos_f32 = (
                audio_flat_f32 - audio_timesteps_f32 * audio_vel_pos.astype(mx.float32)
            )

            # Start with positive prediction
            video_x0_guided_f32 = video_x0_pos_f32
            audio_x0_guided_f32 = audio_x0_pos_f32

            # Pass 2: CFG (negative conditioning)
            if use_cfg:
                video_modality_neg = Modality(
                    latent=seq_video_flat,
                    timesteps=seq_video_timesteps,
                    positions=video_positions,
                    context=video_embeddings_neg,
                    context_mask=None,
                    enabled=True,
                    positional_embeddings=extended_video_rope,
                    sigma=sigma_array,
                )
                audio_modality_neg = Modality(
                    latent=audio_flat,
                    timesteps=audio_timesteps,
                    positions=audio_positions,
                    context=audio_embeddings_neg,
                    context_mask=None,
                    enabled=True,
                    positional_embeddings=precomputed_audio_rope,
                    sigma=audio_sigma_array,
                )
                video_vel_neg, audio_vel_neg = transformer(
                    video=video_modality_neg, audio=audio_modality_neg
                )
                if guiding_tokens is not None:
                    video_vel_neg = video_vel_neg[:, :num_video_tokens, :]
                mx.eval(video_vel_neg, audio_vel_neg)

                video_x0_neg_f32 = (
                    video_flat_f32
                    - video_timesteps_f32 * video_vel_neg.astype(mx.float32)
                )
                audio_x0_neg_f32 = (
                    audio_flat_f32
                    - audio_timesteps_f32 * audio_vel_neg.astype(mx.float32)
                )

                if use_apg:
                    video_x0_guided_f32 = video_x0_pos_f32 + apg_delta(
                        video_x0_pos_f32,
                        video_x0_neg_f32,
                        cfg_scale,
                        eta=apg_eta,
                        norm_threshold=apg_norm_threshold,
                    )
                else:
                    video_x0_guided_f32 = video_x0_pos_f32 + (cfg_scale - 1.0) * (
                        video_x0_pos_f32 - video_x0_neg_f32
                    )
                audio_x0_guided_f32 = audio_x0_pos_f32 + (audio_cfg_scale - 1.0) * (
                    audio_x0_pos_f32 - audio_x0_neg_f32
                )

            # Pass 3: STG (self-attention perturbation at specified blocks)
            if use_stg:
                video_vel_ptb, audio_vel_ptb = transformer(
                    video=video_modality_pos,
                    audio=audio_modality_pos,
                    stg_video_blocks=stg_video_blocks,
                    stg_audio_blocks=stg_audio_blocks,
                )
                if guiding_tokens is not None:
                    video_vel_ptb = video_vel_ptb[:, :num_video_tokens, :]
                mx.eval(video_vel_ptb, audio_vel_ptb)

                video_x0_ptb_f32 = (
                    video_flat_f32
                    - video_timesteps_f32 * video_vel_ptb.astype(mx.float32)
                )
                audio_x0_ptb_f32 = (
                    audio_flat_f32
                    - audio_timesteps_f32 * audio_vel_ptb.astype(mx.float32)
                )

                video_x0_guided_f32 = video_x0_guided_f32 + stg_scale * (
                    video_x0_pos_f32 - video_x0_ptb_f32
                )
                audio_x0_guided_f32 = audio_x0_guided_f32 + stg_scale * (
                    audio_x0_pos_f32 - audio_x0_ptb_f32
                )

            # Pass 4: Modality isolation (skip all cross-modal attention)
            if use_modality:
                video_vel_iso, audio_vel_iso = transformer(
                    video=video_modality_pos,
                    audio=audio_modality_pos,
                    skip_cross_modal=True,
                )
                if guiding_tokens is not None:
                    video_vel_iso = video_vel_iso[:, :num_video_tokens, :]
                mx.eval(video_vel_iso, audio_vel_iso)

                video_x0_iso_f32 = (
                    video_flat_f32
                    - video_timesteps_f32 * video_vel_iso.astype(mx.float32)
                )
                audio_x0_iso_f32 = (
                    audio_flat_f32
                    - audio_timesteps_f32 * audio_vel_iso.astype(mx.float32)
                )

                video_x0_guided_f32 = video_x0_guided_f32 + (modality_scale - 1.0) * (
                    video_x0_pos_f32 - video_x0_iso_f32
                )
                audio_x0_guided_f32 = audio_x0_guided_f32 + (modality_scale - 1.0) * (
                    audio_x0_pos_f32 - audio_x0_iso_f32
                )

            # Apply CFG rescale (std-ratio rescaling to reduce over-saturation)
            if cfg_rescale > 0.0 and (use_cfg or use_stg or use_modality):
                v_factor = video_x0_pos_f32.std() / (video_x0_guided_f32.std() + 1e-8)
                v_factor = cfg_rescale * v_factor + (1.0 - cfg_rescale)
                video_x0_guided_f32 = video_x0_guided_f32 * v_factor
                a_factor = audio_x0_pos_f32.std() / (audio_x0_guided_f32.std() + 1e-8)
                a_factor = cfg_rescale * a_factor + (1.0 - cfg_rescale)
                audio_x0_guided_f32 = audio_x0_guided_f32 * a_factor

            # Reshape x0 from token space (b, tokens, c) to spatial (b, c, f, h, w)
            video_denoised_f32 = mx.reshape(
                mx.transpose(video_x0_guided_f32, (0, 2, 1)), (b, c, f, h, w)
            )
            audio_denoised_f32 = mx.reshape(audio_x0_guided_f32, (ab, at, ac, af))
            audio_denoised_f32 = mx.transpose(audio_denoised_f32, (0, 2, 1, 3))

            # Post-process: blend denoised with clean latent using mask
            # Matches PyTorch's post_process_latent: denoised * mask + clean * (1 - mask)
            sigma_f32 = mx.array(sigma, dtype=mx.float32)

            if video_state is not None:
                clean_f32 = video_state.clean_latent.astype(mx.float32)
                mask_f32 = video_state.denoise_mask.astype(mx.float32)
                video_denoised_f32 = video_denoised_f32 * mask_f32 + clean_f32 * (
                    1.0 - mask_f32
                )

            mx.eval(video_denoised_f32, audio_denoised_f32)

            # Euler step: sample + velocity * dt (float32)
            if sigma_next > 0:
                sigma_next_f32 = mx.array(sigma_next, dtype=mx.float32)
                dt_f32 = sigma_next_f32 - sigma_f32

                video_velocity_f32 = (video_latents - video_denoised_f32) / sigma_f32
                video_latents = video_latents + video_velocity_f32 * dt_f32

                if not audio_frozen:
                    audio_velocity_f32 = (
                        audio_latents - audio_denoised_f32
                    ) / sigma_f32
                    audio_latents = audio_latents + audio_velocity_f32 * dt_f32
            else:
                video_latents = video_denoised_f32
                if not audio_frozen:
                    audio_latents = audio_denoised_f32

            mx.eval(video_latents, audio_latents)
            progress.advance(task)

    return video_latents, audio_latents


def denoise_res2s_av(
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
    cfg_scale: float = 3.0,
    audio_cfg_scale: float = 7.0,
    cfg_rescale: float = 0.45,
    audio_cfg_rescale: Optional[float] = None,
    verbose: bool = True,
    video_state: Optional[LatentState] = None,
    stg_scale: float = 0.0,
    stg_video_blocks: Optional[list] = None,
    stg_audio_blocks: Optional[list] = None,
    modality_scale: float = 1.0,
    noise_seed: int = 42,
    bongmath: bool = True,
    bongmath_max_iter: int = 100,
    audio_frozen: bool = False,
    guiding_tokens: Optional[mx.array] = None,
    guiding_positions: Optional[mx.array] = None,
    guiding_strength: float = 1.0,
) -> tuple[mx.array, mx.array]:
    """Run res_2s second-order denoising loop with CFG/STG/modality guidance.

    Two model evaluations per step (current point + midpoint), with SDE noise
    injection and optional bong iteration for anchor refinement.

    Args:
        audio_cfg_rescale: Separate rescale for audio. If None, uses cfg_rescale.
        noise_seed: Seed for SDE noise generators.
        bongmath: Enable iterative anchor refinement.
        bongmath_max_iter: Max bong iterations per step.
    """
    from mlx_video.models.ltx_2.rope import precompute_freqs_cis
    from mlx_video.models.ltx_2.samplers import (
        get_new_noise,
        get_res2s_coefficients,
        sde_noise_step,
    )

    if audio_cfg_rescale is None:
        audio_cfg_rescale = cfg_rescale

    dtype = video_latents.dtype
    if video_state is not None:
        video_latents = video_state.latent

    video_latents = video_latents.astype(mx.float32)
    audio_latents = audio_latents.astype(mx.float32)

    sigmas_list = sigmas.tolist()
    use_cfg = cfg_scale != 1.0
    use_stg = stg_scale != 0.0 and stg_video_blocks is not None
    use_modality = modality_scale != 1.0
    n_full_steps = len(sigmas_list) - 1

    # Pad sigmas if last is 0 (avoid division by zero in RK steps)
    if sigmas_list[-1] == 0:
        sigmas_list = sigmas_list[:-1] + [0.0011, 0.0]

    # Compute step sizes in log-space for the main loop steps only.
    # After padding, sigmas_list may have an extra [0.0011, 0.0] tail;
    # we only need hs for the n_full_steps pairs the loop actually uses.
    hs = [-math.log(sigmas_list[i + 1] / sigmas_list[i]) for i in range(n_full_steps)]

    # Precompute RoPE
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

    # Pre-compute extended RoPE for guiding tokens (once, outside loop).
    n_guiding = 0
    extended_video_rope = precomputed_video_rope
    if guiding_tokens is not None:
        n_guiding = guiding_tokens.shape[1]
        extended_positions = mx.concatenate([video_positions, guiding_positions], axis=2)
        extended_video_rope = precompute_freqs_cis(
            extended_positions,
            dim=transformer.inner_dim,
            theta=transformer.positional_embedding_theta,
            max_pos=transformer.positional_embedding_max_pos,
            use_middle_indices_grid=transformer.use_middle_indices_grid,
            num_attention_heads=transformer.num_attention_heads,
            rope_type=transformer.rope_type,
            double_precision=transformer.config.double_precision_rope,
        )
        mx.eval(extended_video_rope)

    phi_cache = {}
    c2 = 0.5

    # Noise key management: step noise and substep noise use different keys
    step_noise_key = mx.random.key(noise_seed)
    substep_noise_key = mx.random.key(noise_seed + 10000)

    def _eval_guided_denoise(v_latents, a_latents, sigma):
        """Run all guidance passes and return (video_denoised, audio_denoised) in float32 spatial format."""
        b, c, f, h, w = v_latents.shape
        num_video_tokens = f * h * w
        video_flat = mx.transpose(mx.reshape(v_latents, (b, c, -1)), (0, 2, 1)).astype(
            dtype
        )

        ab, ac, at, af = a_latents.shape
        audio_flat = mx.transpose(a_latents, (0, 2, 1, 3))
        audio_flat = mx.reshape(audio_flat, (ab, at, ac * af)).astype(dtype)

        # Timesteps
        if video_state is not None:
            denoise_mask_flat = mx.reshape(video_state.denoise_mask, (b, 1, f, 1, 1))
            denoise_mask_flat = mx.broadcast_to(denoise_mask_flat, (b, 1, f, h, w))
            denoise_mask_flat = mx.reshape(denoise_mask_flat, (b, num_video_tokens))
            video_timesteps = mx.array(sigma, dtype=dtype) * denoise_mask_flat
        else:
            video_timesteps = mx.full((b, num_video_tokens), sigma, dtype=dtype)
        audio_timesteps = (
            mx.zeros((ab, at), dtype=dtype)
            if audio_frozen
            else mx.full((ab, at), sigma, dtype=dtype)
        )

        sigma_array = mx.full((b,), sigma, dtype=dtype)
        audio_sigma_array = (
            mx.zeros((ab,), dtype=dtype)
            if audio_frozen
            else mx.full((ab,), sigma, dtype=dtype)
        )

        # Build extended sequence if end-frame guiding tokens are active.
        if guiding_tokens is not None:
            cond_ts = mx.full(
                (b, n_guiding), sigma * (1.0 - guiding_strength), dtype=dtype
            )
            seq_video_flat = mx.concatenate(
                [video_flat, guiding_tokens.astype(dtype)], axis=1
            )
            seq_video_timesteps = mx.concatenate([video_timesteps, cond_ts], axis=1)
        else:
            seq_video_flat = video_flat
            seq_video_timesteps = video_timesteps

        # Pass 1: Positive conditioning
        video_modality_pos = Modality(
            latent=seq_video_flat,
            timesteps=seq_video_timesteps,
            positions=video_positions,
            context=video_embeddings_pos,
            context_mask=None,
            enabled=True,
            positional_embeddings=extended_video_rope,
            sigma=sigma_array,
        )
        audio_modality_pos = Modality(
            latent=audio_flat,
            timesteps=audio_timesteps,
            positions=audio_positions,
            context=audio_embeddings_pos,
            context_mask=None,
            enabled=True,
            positional_embeddings=precomputed_audio_rope,
            sigma=audio_sigma_array,
        )
        video_vel_pos, audio_vel_pos = transformer(
            video=video_modality_pos, audio=audio_modality_pos
        )
        if guiding_tokens is not None:
            video_vel_pos = video_vel_pos[:, :num_video_tokens, :]
        mx.eval(video_vel_pos, audio_vel_pos)

        # Convert velocity to x0
        video_flat_f32 = mx.transpose(mx.reshape(v_latents, (b, c, -1)), (0, 2, 1))
        audio_flat_f32 = mx.reshape(
            mx.transpose(a_latents, (0, 2, 1, 3)), (ab, at, ac * af)
        )
        video_ts_f32 = mx.expand_dims(video_timesteps.astype(mx.float32), axis=-1)
        audio_ts_f32 = mx.expand_dims(audio_timesteps.astype(mx.float32), axis=-1)

        video_x0_pos = video_flat_f32 - video_ts_f32 * video_vel_pos.astype(mx.float32)
        audio_x0_pos = audio_flat_f32 - audio_ts_f32 * audio_vel_pos.astype(mx.float32)

        video_x0_guided = video_x0_pos
        audio_x0_guided = audio_x0_pos

        # Pass 2: CFG
        if use_cfg:
            video_modality_neg = Modality(
                latent=seq_video_flat,
                timesteps=seq_video_timesteps,
                positions=video_positions,
                context=video_embeddings_neg,
                context_mask=None,
                enabled=True,
                positional_embeddings=extended_video_rope,
                sigma=sigma_array,
            )
            audio_modality_neg = Modality(
                latent=audio_flat,
                timesteps=audio_timesteps,
                positions=audio_positions,
                context=audio_embeddings_neg,
                context_mask=None,
                enabled=True,
                positional_embeddings=precomputed_audio_rope,
                sigma=audio_sigma_array,
            )
            video_vel_neg, audio_vel_neg = transformer(
                video=video_modality_neg, audio=audio_modality_neg
            )
            if guiding_tokens is not None:
                video_vel_neg = video_vel_neg[:, :num_video_tokens, :]
            mx.eval(video_vel_neg, audio_vel_neg)

            video_x0_neg = video_flat_f32 - video_ts_f32 * video_vel_neg.astype(
                mx.float32
            )
            audio_x0_neg = audio_flat_f32 - audio_ts_f32 * audio_vel_neg.astype(
                mx.float32
            )

            video_x0_guided = video_x0_pos + (cfg_scale - 1.0) * (
                video_x0_pos - video_x0_neg
            )
            audio_x0_guided = audio_x0_pos + (audio_cfg_scale - 1.0) * (
                audio_x0_pos - audio_x0_neg
            )

        # Pass 3: STG
        if use_stg:
            video_vel_ptb, audio_vel_ptb = transformer(
                video=video_modality_pos,
                audio=audio_modality_pos,
                stg_video_blocks=stg_video_blocks,
                stg_audio_blocks=stg_audio_blocks,
            )
            if guiding_tokens is not None:
                video_vel_ptb = video_vel_ptb[:, :num_video_tokens, :]
            mx.eval(video_vel_ptb, audio_vel_ptb)

            video_x0_ptb = video_flat_f32 - video_ts_f32 * video_vel_ptb.astype(
                mx.float32
            )
            audio_x0_ptb = audio_flat_f32 - audio_ts_f32 * audio_vel_ptb.astype(
                mx.float32
            )

            video_x0_guided = video_x0_guided + stg_scale * (
                video_x0_pos - video_x0_ptb
            )
            audio_x0_guided = audio_x0_guided + stg_scale * (
                audio_x0_pos - audio_x0_ptb
            )

        # Pass 4: Modality isolation
        if use_modality:
            video_vel_iso, audio_vel_iso = transformer(
                video=video_modality_pos,
                audio=audio_modality_pos,
                skip_cross_modal=True,
            )
            if guiding_tokens is not None:
                video_vel_iso = video_vel_iso[:, :num_video_tokens, :]
            mx.eval(video_vel_iso, audio_vel_iso)

            video_x0_iso = video_flat_f32 - video_ts_f32 * video_vel_iso.astype(
                mx.float32
            )
            audio_x0_iso = audio_flat_f32 - audio_ts_f32 * audio_vel_iso.astype(
                mx.float32
            )

            video_x0_guided = video_x0_guided + (modality_scale - 1.0) * (
                video_x0_pos - video_x0_iso
            )
            audio_x0_guided = audio_x0_guided + (modality_scale - 1.0) * (
                audio_x0_pos - audio_x0_iso
            )

        # Rescale (separate factors for video and audio)
        if cfg_rescale > 0.0 and (use_cfg or use_stg or use_modality):
            v_factor = video_x0_pos.std() / (video_x0_guided.std() + 1e-8)
            v_factor = cfg_rescale * v_factor + (1.0 - cfg_rescale)
            video_x0_guided = video_x0_guided * v_factor
        if audio_cfg_rescale > 0.0 and (use_cfg or use_stg or use_modality):
            a_factor = audio_x0_pos.std() / (audio_x0_guided.std() + 1e-8)
            a_factor = audio_cfg_rescale * a_factor + (1.0 - audio_cfg_rescale)
            audio_x0_guided = audio_x0_guided * a_factor

        # Reshape to spatial
        video_denoised = mx.reshape(
            mx.transpose(video_x0_guided, (0, 2, 1)), (b, c, f, h, w)
        )
        audio_denoised = mx.reshape(audio_x0_guided, (ab, at, ac, af))
        audio_denoised = mx.transpose(audio_denoised, (0, 2, 1, 3))

        # Post-process with mask
        if video_state is not None:
            clean_f32 = video_state.clean_latent.astype(mx.float32)
            mask_f32 = video_state.denoise_mask.astype(mx.float32)
            video_denoised = video_denoised * mask_f32 + clean_f32 * (1.0 - mask_f32)

        mx.eval(video_denoised, audio_denoised)
        return video_denoised, audio_denoised

    # Main res_2s loop
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        disable=not verbose,
    ) as progress:
        passes = ["res2s"]
        if use_cfg:
            passes.append("CFG")
        if use_stg:
            passes.append("STG")
        if use_modality:
            passes.append("Mod")
        label = "+".join(passes)
        task = progress.add_task(
            f"[cyan]Denoising A/V ({label})[/]", total=n_full_steps
        )

        for step_idx in range(n_full_steps):
            sigma = sigmas_list[step_idx]
            sigma_next = sigmas_list[step_idx + 1]
            h = hs[step_idx]

            # Initialize anchor
            x_anchor_video = video_latents
            x_anchor_audio = audio_latents

            # ============================================================
            # Stage 1: Evaluate denoiser at current sigma
            # ============================================================
            denoised_video_1, denoised_audio_1 = _eval_guided_denoise(
                video_latents, audio_latents, sigma
            )

            # RK coefficients
            a21, b1, b2 = get_res2s_coefficients(h, phi_cache, c2)

            # Substep sigma (geometric midpoint for c2=0.5)
            sub_sigma = math.sqrt(sigma * sigma_next)

            # Compute midpoint
            eps_1_video = denoised_video_1 - x_anchor_video
            x_mid_video = x_anchor_video + h * a21 * eps_1_video

            if not audio_frozen:
                eps_1_audio = denoised_audio_1 - x_anchor_audio
                x_mid_audio = x_anchor_audio + h * a21 * eps_1_audio
            else:
                eps_1_audio = None
                x_mid_audio = audio_latents  # frozen: pass through unchanged

            # SDE noise injection at substep
            substep_noise_key, key1, key2 = mx.random.split(substep_noise_key, 3)
            substep_noise_v = get_new_noise(video_latents.shape, key1)

            x_mid_video = sde_noise_step(
                x_anchor_video, x_mid_video, sigma, sub_sigma, substep_noise_v
            )
            if not audio_frozen:
                substep_noise_a = get_new_noise(audio_latents.shape, key2)
                x_mid_audio = sde_noise_step(
                    x_anchor_audio, x_mid_audio, sigma, sub_sigma, substep_noise_a
                )
            mx.eval(x_mid_video, x_mid_audio)

            # ============================================================
            # Bong iteration: refine anchor (pure arithmetic, no model calls)
            # ============================================================
            if bongmath and h < 0.5 and sigma > 0.03:
                for _ in range(bongmath_max_iter):
                    x_anchor_video = x_mid_video - h * a21 * eps_1_video
                    eps_1_video = denoised_video_1 - x_anchor_video
                    if not audio_frozen:
                        x_anchor_audio = x_mid_audio - h * a21 * eps_1_audio
                        eps_1_audio = denoised_audio_1 - x_anchor_audio
                if audio_frozen:
                    mx.eval(x_anchor_video, eps_1_video)
                else:
                    mx.eval(x_anchor_video, x_anchor_audio, eps_1_video, eps_1_audio)

            # ============================================================
            # Stage 2: Evaluate denoiser at midpoint sigma
            # ============================================================
            denoised_video_2, denoised_audio_2 = _eval_guided_denoise(
                x_mid_video.astype(mx.float32),
                x_mid_audio.astype(mx.float32),
                sub_sigma,
            )

            # ============================================================
            # Final combination with RK coefficients
            # ============================================================
            eps_2_video = denoised_video_2 - x_anchor_video
            x_next_video = x_anchor_video + h * (b1 * eps_1_video + b2 * eps_2_video)

            # SDE noise injection at step level
            step_noise_key, key1, key2 = mx.random.split(step_noise_key, 3)
            step_noise_v = get_new_noise(video_latents.shape, key1)
            x_next_video = sde_noise_step(
                x_anchor_video, x_next_video, sigma, sigma_next, step_noise_v
            )

            video_latents = x_next_video.astype(mx.float32)
            if not audio_frozen:
                eps_2_audio = denoised_audio_2 - x_anchor_audio
                x_next_audio = x_anchor_audio + h * (
                    b1 * eps_1_audio + b2 * eps_2_audio
                )
                step_noise_a = get_new_noise(audio_latents.shape, key2)
                x_next_audio = sde_noise_step(
                    x_anchor_audio, x_next_audio, sigma, sigma_next, step_noise_a
                )
                audio_latents = x_next_audio.astype(mx.float32)

            mx.eval(video_latents, audio_latents)
            progress.advance(task)

    # Final clean step if original schedule ended at 0
    if sigmas.tolist()[-1] == 0:
        denoised_video, denoised_audio = _eval_guided_denoise(
            video_latents, audio_latents, sigmas_list[n_full_steps]
        )
        video_latents = denoised_video
        if not audio_frozen:
            audio_latents = denoised_audio
        mx.eval(video_latents, audio_latents)

    return video_latents, audio_latents


# =============================================================================
# Audio Loading and Processing
# =============================================================================


def load_audio_decoder(model_path: Path, pipeline: PipelineType):
    """Load audio VAE decoder."""
    from mlx_video.models.ltx_2.audio_vae import AudioDecoder

    decoder = AudioDecoder.from_pretrained(model_path / "audio_vae" / "decoder")

    return decoder


def load_vocoder_model(model_path: Path, pipeline: PipelineType):
    """Load vocoder for mel to waveform conversion.

    Automatically detects HiFi-GAN (LTX-2) or BigVGAN+BWE (LTX-2.3).
    """
    from mlx_video.models.ltx_2.audio_vae.vocoder import load_vocoder as _load_vocoder

    return _load_vocoder(model_path / "vocoder")


def save_audio(audio: np.ndarray, path: Path, sample_rate: int = AUDIO_SAMPLE_RATE):
    """Save audio to WAV file."""
    import wave

    if audio.ndim == 2:
        audio = audio.T

    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2 if audio_int16.ndim == 2 else 1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


def mux_video_audio(video_path: Path, audio_path: Path, output_path: Path):
    """Combine video and audio into final output using ffmpeg."""
    import subprocess

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(output_path),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        console.print(f"[red]FFmpeg error: {e.stderr.decode()}[/]")
        return False
    except FileNotFoundError:
        console.print("[red]FFmpeg not found. Please install ffmpeg.[/]")
        return False


# =============================================================================
# Unified Generate Function
# =============================================================================


def _build_i2v_conditionings(
    image_latent,
    image_frame_idx: int,
    image_strength: float,
):
    """Build a list of VideoConditionByLatentIndex for first-frame I2V conditioning."""
    conditionings = []
    if image_latent is not None:
        conditionings.append(
            VideoConditionByLatentIndex(
                latent=image_latent, frame_idx=image_frame_idx, strength=image_strength
            )
        )
    return conditionings


def _prepare_guiding_tokens(
    cond_latent: mx.array,
    positions: mx.array,
    latent_frames: int,
    h: int,
    w: int,
) -> tuple:
    """Prepare end-frame conditioning tokens for token-appending guidance.

    Flattens the conditioning latent to token form and extracts the positional
    encoding of the last latent frame so they can be appended to the video
    sequence before the transformer forward pass (no hard token replacement,
    which avoids abrupt appearance changes near the end frame).

    Args:
        cond_latent: Encoded end-image latent, shape (1, C, 1, h, w)
        positions: Full position grid, shape (B, 3, F*h*w, 2)
        latent_frames: Total number of latent frames F
        h: Latent height
        w: Latent width

    Returns:
        tokens: (1, h*w, C) conditioning tokens
        frame_positions: (B, 3, h*w, 2) positions for the last latent frame
    """
    c = cond_latent.shape[1]
    tokens = mx.transpose(mx.reshape(cond_latent, (1, c, -1)), (0, 2, 1))
    start = (latent_frames - 1) * h * w
    end = latent_frames * h * w
    frame_positions = positions[:, :, start:end, :]
    return tokens, frame_positions


def generate_video(
    model_repo: str,
    text_encoder_repo: str,
    prompt: str,
    pipeline: PipelineType = PipelineType.DISTILLED,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    height: int = 512,
    width: int = 512,
    num_frames: int = 33,
    num_inference_steps: int = 40,
    cfg_scale: float = 4.0,
    audio_cfg_scale: float = 7.0,
    cfg_rescale: float = 0.0,
    seed: int = 42,
    fps: int = 24,
    output_path: str = "output.mp4",
    save_frames: bool = False,
    verbose: bool = True,
    enhance_prompt: bool = False,
    max_tokens: int = 512,
    temperature: float = 0.7,
    image: Optional[str] = None,
    image_strength: float = 1.0,
    image_frame_idx: int = 0,
    end_image: Optional[str] = None,
    end_image_strength: Optional[float] = None,
    tiling: str = "auto",
    stream: bool = False,
    audio: bool = False,
    output_audio_path: Optional[str] = None,
    use_apg: bool = False,
    apg_eta: float = 1.0,
    apg_norm_threshold: float = 0.0,
    stg_scale: float = 1.0,
    stg_blocks: Optional[list] = None,
    modality_scale: float = 3.0,
    lora_path: Optional[str] = None,
    lora_strength: float = 1.0,
    lora_strength_stage_1: Optional[float] = None,
    lora_strength_stage_2: Optional[float] = None,
    audio_file: Optional[str] = None,
    audio_start_time: float = 0.0,
    spatial_upscaler: Optional[str] = None,
):
    """Generate video using LTX-2 models.

    Supports four pipelines:
    - DISTILLED: Two-stage generation with upsampling, fixed sigma schedules, no CFG
    - DEV: Single-stage generation with dynamic sigmas and CFG
    - DEV_TWO_STAGE: Stage 1 dev (half res, CFG) + upsample + stage 2 distilled with LoRA (full res, no CFG)
    - DEV_TWO_STAGE_HQ: res_2s sampler, LoRA both stages (0.25/0.5), lower rescale

    Args:
        model_repo: Model repository ID
        text_encoder_repo: Text encoder repository ID
        prompt: Text description of the video to generate
        pipeline: Pipeline type (DISTILLED or DEV)
        negative_prompt: Negative prompt for CFG (dev pipeline only)
        height: Output video height (must be divisible by 32/64)
        width: Output video width (must be divisible by 32/64)
        num_frames: Number of frames (must be 1 + 8*k)
        num_inference_steps: Number of denoising steps (dev pipeline only)
        cfg_scale: Guidance scale for CFG (dev pipeline only)
        seed: Random seed for reproducibility
        fps: Frames per second for output video
        output_path: Path to save the output video
        save_frames: Whether to save individual frames as images
        verbose: Whether to print progress
        enhance_prompt: Whether to enhance prompt using Gemma
        max_tokens: Max tokens for prompt enhancement
        temperature: Temperature for prompt enhancement
        image: Path to conditioning image for I2V (first frame by default)
        image_strength: Conditioning strength for I2V
        image_frame_idx: Frame index to condition for I2V (ignored when end_image is set)
        end_image: Path to conditioning image for the last frame (I2V end-frame control)
        end_image_strength: Conditioning strength for end frame (defaults to image_strength)
        tiling: Tiling mode for VAE decoding
        stream: Stream frames to output as they're decoded
        audio: Enable synchronized audio generation
        output_audio_path: Path to save audio file
        use_apg: Use Adaptive Projected Guidance instead of CFG (more stable for I2V)
        apg_eta: APG parallel component weight (1.0 = keep full parallel)
        apg_norm_threshold: APG guidance norm clamp (0 = no clamping)
    """
    start_time = time.time()

    # Validate dimensions
    is_two_stage = pipeline in (
        PipelineType.DISTILLED,
        PipelineType.DEV_TWO_STAGE,
        PipelineType.DEV_TWO_STAGE_HQ,
    )
    divisor = 64 if is_two_stage else 32
    assert height % divisor == 0, f"Height must be divisible by {divisor}, got {height}"
    assert width % divisor == 0, f"Width must be divisible by {divisor}, got {width}"

    if num_frames % 8 != 1:
        adjusted_num_frames = round((num_frames - 1) / 8) * 8 + 1
        console.print(
            f"[yellow]⚠️  Number of frames must be 1 + 8*k. Using: {adjusted_num_frames}[/]"
        )
        num_frames = adjusted_num_frames

    is_i2v = image is not None or end_image is not None
    has_end_image = end_image is not None
    if end_image_strength is None:
        end_image_strength = image_strength
    is_a2v = audio_file is not None
    if is_a2v and audio:
        raise ValueError(
            "Cannot use both --audio-file (A2V) and --audio (generate audio). Choose one."
        )
    # A2V implicitly enables audio path through the transformer
    if is_a2v:
        audio = True
    mode_str = "I2V" if is_i2v else "T2V"
    if has_end_image and image is not None:
        mode_str = "I2V(first+last)"
    elif has_end_image:
        mode_str = "I2V(last)"
    if is_a2v:
        mode_str = "A2V" + ("+I2V" if is_i2v else "")
    elif audio:
        mode_str += "+Audio"

    pipeline_names = {
        PipelineType.DISTILLED: "DISTILLED",
        PipelineType.DEV: "DEV",
        PipelineType.DEV_TWO_STAGE: "DEV-TWO-STAGE",
        PipelineType.DEV_TWO_STAGE_HQ: "DEV-TWO-STAGE-HQ",
    }
    pipeline_name = pipeline_names[pipeline]
    header = f"[bold cyan]🎬 [{pipeline_name}] [{mode_str}] {width}x{height} • {num_frames} frames[/]"
    console.print(Panel(header, expand=False))
    console.print(f"[dim]Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}[/]")

    if pipeline in (
        PipelineType.DEV,
        PipelineType.DEV_TWO_STAGE,
        PipelineType.DEV_TWO_STAGE_HQ,
    ):
        audio_cfg_info = f", Audio CFG: {audio_cfg_scale}" if audio else ""
        stg_info = f", STG: {stg_scale} blocks={stg_blocks}" if stg_scale != 0.0 else ""
        mod_info = f", Modality: {modality_scale}" if modality_scale != 1.0 else ""
        console.print(
            f"[dim]Steps: {num_inference_steps}, CFG: {cfg_scale}{audio_cfg_info}, Rescale: {cfg_rescale}{stg_info}{mod_info}[/]"
        )

    if is_i2v:
        if image is not None:
            console.print(
                f"[dim]First image: {image} (strength={image_strength}, frame={image_frame_idx})[/]"
            )
        if has_end_image:
            console.print(
                f"[dim]Last image: {end_image} (strength={end_image_strength}, frame=-1)[/]"
            )

    # Always compute audio frames - PyTorch distilled pipeline unconditionally
    # generates audio alongside video (model was trained with joint audio-video).
    # The --audio flag only controls whether audio is decoded and saved to output.
    audio_frames = compute_audio_frames(num_frames, fps)
    if audio:
        console.print(
            f"[dim]Audio: {audio_frames} latent frames @ {AUDIO_SAMPLE_RATE}Hz[/]"
        )

    # Get model path
    model_path = get_model_path(model_repo)
    text_encoder_path = (
        model_path if text_encoder_repo is None else get_model_path(text_encoder_repo)
    )

    # Resolve spatial upscaler path for two-stage pipelines
    upscaler_path = None
    upscaler_scale = 2.0
    if is_two_stage:
        if spatial_upscaler is not None:
            # User-specified upscaler file
            upscaler_path = (
                model_path / spatial_upscaler
                if not Path(spatial_upscaler).is_absolute()
                else Path(spatial_upscaler)
            )
            if not upscaler_path.exists():
                # Try as a filename within model_path
                upscaler_path = model_path / spatial_upscaler
            # Detect scale from filename
            if "x1.5" in str(upscaler_path):
                upscaler_scale = 1.5
            elif "x2" in str(upscaler_path):
                upscaler_scale = 2.0
        else:
            # Auto-detect: prefer x2 upscaler
            upscaler_files = sorted(
                model_path.glob("*spatial-upscaler-x2*.safetensors")
            )
            if upscaler_files:
                upscaler_path = upscaler_files[0]
                upscaler_scale = 2.0

    # Calculate latent dimensions
    if is_two_stage:
        # Stage 1 always at half resolution (matches PyTorch)
        stage1_h, stage1_w = height // 2 // 32, width // 2 // 32
        # Stage 2 resolution = stage 1 * upscaler scale
        stage2_h = int(stage1_h * upscaler_scale)
        stage2_w = int(stage1_w * upscaler_scale)
    else:
        latent_h, latent_w = height // 32, width // 32
    latent_frames = 1 + (num_frames - 1) // 8

    mx.random.seed(seed)

    # Read transformer config to detect model version
    import json

    transformer_config_path = model_path / "transformer" / "config.json"
    has_prompt_adaln = False
    if transformer_config_path.exists():
        with open(transformer_config_path) as f:
            has_prompt_adaln = json.load(f).get("has_prompt_adaln", False)

    # Load text encoder
    with console.status("[blue]📝 Loading text encoder...[/]", spinner="dots"):
        from mlx_video.models.ltx_2.text_encoder import LTX2TextEncoder

        text_encoder = LTX2TextEncoder(has_prompt_adaln=has_prompt_adaln)
        text_encoder.load(model_path=model_path, text_encoder_path=text_encoder_path)
        mx.eval(text_encoder.parameters())
    console.print("[green]✓[/] Text encoder loaded")

    # Optionally enhance the prompt
    if enhance_prompt:
        console.print("[bold magenta]✨ Enhancing prompt[/]")
        prompt = text_encoder.enhance_t2v(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            verbose=verbose,
        )
        console.print(
            f"[dim]Enhanced: {prompt[:150]}{'...' if len(prompt) > 150 else ''}[/]"
        )

    # Encode prompts - always get audio embeddings since the model was trained
    # with joint audio-video processing (PyTorch unconditionally generates audio)
    if pipeline in (
        PipelineType.DEV,
        PipelineType.DEV_TWO_STAGE,
        PipelineType.DEV_TWO_STAGE_HQ,
    ):
        # Dev/dev-two-stage pipelines need positive and negative embeddings for CFG
        video_embeddings_pos, audio_embeddings_pos = text_encoder(
            prompt, return_audio_embeddings=True
        )
        video_embeddings_neg, audio_embeddings_neg = text_encoder(
            negative_prompt, return_audio_embeddings=True
        )
        model_dtype = video_embeddings_pos.dtype
        mx.eval(
            video_embeddings_pos,
            video_embeddings_neg,
            audio_embeddings_pos,
            audio_embeddings_neg,
        )
        # For dev-two-stage, stage 2 uses single positive embedding (no CFG)
        if pipeline in (PipelineType.DEV_TWO_STAGE, PipelineType.DEV_TWO_STAGE_HQ):
            text_embeddings = video_embeddings_pos
    else:
        # Distilled pipeline - single embedding
        text_embeddings, audio_embeddings = text_encoder(
            prompt, return_audio_embeddings=True
        )
        mx.eval(text_embeddings, audio_embeddings)
        model_dtype = text_embeddings.dtype

    del text_encoder
    mx.clear_cache()

    # Load transformer
    transformer_desc = f"🤖 Loading {pipeline_name.lower()} transformer{' (A/V mode)' if audio else ''}..."
    with console.status(f"[blue]{transformer_desc}[/]", spinner="dots"):
        transformer = LTXModel.from_pretrained(
            model_path=model_path / "transformer", strict=True
        )

    console.print("[green]✓[/] Transformer loaded")

    # Auto-detect stg_blocks from transformer config if not explicitly provided.
    # LTX-2.3 (has_prompt_adaln=True) uses block 28; LTX-2 uses block 29.
    if stg_blocks is None and stg_scale != 0.0:
        if transformer.config.has_prompt_adaln:
            stg_blocks = [28]
        else:
            stg_blocks = [29]
        console.print(
            f"[dim]Auto-detected STG blocks: {stg_blocks} (model={'2.3' if transformer.config.has_prompt_adaln else '2'})[/]"
        )

    # ==========================================================================
    # A2V: Encode input audio to frozen latents
    # ==========================================================================
    a2v_audio_latents = None
    a2v_waveform = None
    a2v_sr = None
    if is_a2v:
        from mlx_video.models.ltx_2.audio_vae import AudioEncoder
        from mlx_video.models.ltx_2.audio_vae.audio_processor import (
            ensure_stereo,
            load_audio,
            waveform_to_mel,
        )
        from mlx_video.models.ltx_2.utils import convert_audio_encoder

        with console.status(
            "[blue]Loading and encoding input audio (A2V)...[/]", spinner="dots"
        ):
            video_duration = num_frames / fps

            # Load audio
            waveform, sr = load_audio(
                audio_file,
                target_sr=AUDIO_LATENT_SAMPLE_RATE,
                start_time=audio_start_time,
                max_duration=video_duration,
            )
            waveform = ensure_stereo(waveform)
            a2v_waveform = waveform.copy()
            a2v_sr = sr

            # Compute mel-spectrogram
            mel = waveform_to_mel(
                waveform,
                sample_rate=sr,
                n_fft=1024,
                hop_length=AUDIO_HOP_LENGTH,
                n_mels=64,
            )

            # Convert audio encoder weights if needed, then load
            encoder_dir = convert_audio_encoder(
                model_path, source_repo="Lightricks/LTX-2"
            )
            audio_encoder = AudioEncoder.from_pretrained(encoder_dir)
            mx.eval(audio_encoder.parameters())

            # Encode: (1, 2, time, 64) -> normalized latents
            encoded = audio_encoder(mel)
            mx.eval(encoded)

            # encoded is in MLX format (B, T', mel_bins', z_channels) = (1, T', 16, 8)
            # Convert to PyTorch-style format for consistency: (B, C, T, mel_bins)
            a2v_audio_latents = mx.transpose(encoded, (0, 3, 1, 2)).astype(model_dtype)

            # Trim/pad to match expected audio_frames
            t_encoded = a2v_audio_latents.shape[2]
            if t_encoded > audio_frames:
                a2v_audio_latents = a2v_audio_latents[:, :, :audio_frames, :]
            elif t_encoded < audio_frames:
                pad_size = audio_frames - t_encoded
                padding = mx.zeros(
                    (1, AUDIO_LATENT_CHANNELS, pad_size, AUDIO_MEL_BINS),
                    dtype=model_dtype,
                )
                a2v_audio_latents = mx.concatenate([a2v_audio_latents, padding], axis=2)
            mx.eval(a2v_audio_latents)

            del audio_encoder
            mx.clear_cache()

        console.print(
            f"[green]✓[/] Audio encoded ({a2v_audio_latents.shape[2]} frames from {audio_file})"
        )

    # ==========================================================================
    # Pipeline-specific generation logic
    # ==========================================================================

    if pipeline == PipelineType.DISTILLED:
        # ======================================================================
        # DISTILLED PIPELINE: Two-stage with upsampling
        # ======================================================================

        # Load VAE encoder for I2V
        stage1_image_latent = None
        stage2_image_latent = None
        stage1_end_image_latent = None
        stage2_end_image_latent = None
        if is_i2v:
            with console.status(
                "[blue]🖼️  Loading VAE encoder and encoding image(s)...[/]", spinner="dots"
            ):
                vae_encoder = VideoEncoder.from_pretrained(
                    model_path / "vae" / "encoder"
                )

                s1_h, s1_w = stage1_h * 32, stage1_w * 32
                s2_h, s2_w = stage2_h * 32, stage2_w * 32

                if image is not None:
                    input_image = load_image(image, height=s1_h, width=s1_w, dtype=model_dtype)
                    stage1_image_latent = vae_encoder(prepare_image_for_encoding(input_image, s1_h, s1_w, dtype=model_dtype))
                    mx.eval(stage1_image_latent)
                    input_image = load_image(image, height=s2_h, width=s2_w, dtype=model_dtype)
                    stage2_image_latent = vae_encoder(prepare_image_for_encoding(input_image, s2_h, s2_w, dtype=model_dtype))
                    mx.eval(stage2_image_latent)

                if has_end_image:
                    end_input = load_image(end_image, height=s1_h, width=s1_w, dtype=model_dtype)
                    stage1_end_image_latent = vae_encoder(prepare_image_for_encoding(end_input, s1_h, s1_w, dtype=model_dtype))
                    mx.eval(stage1_end_image_latent)
                    end_input = load_image(end_image, height=s2_h, width=s2_w, dtype=model_dtype)
                    stage2_end_image_latent = vae_encoder(prepare_image_for_encoding(end_input, s2_h, s2_w, dtype=model_dtype))
                    mx.eval(stage2_end_image_latent)

                del vae_encoder
                mx.clear_cache()
            console.print("[green]✓[/] VAE encoder loaded and image(s) encoded")

        # Stage 1
        console.print(
            f"\n[bold yellow]⚡ Stage 1:[/] Generating at {stage1_w*32}x{stage1_h*32} (8 steps)"
        )
        mx.random.seed(seed)

        positions = create_position_grid(1, latent_frames, stage1_h, stage1_w)
        mx.eval(positions)

        # Init audio latents/positions: use encoded A2V latents or random
        audio_positions = create_audio_position_grid(1, audio_frames)
        audio_latents = (
            a2v_audio_latents
            if is_a2v
            else mx.random.normal(
                (1, AUDIO_LATENT_CHANNELS, audio_frames, AUDIO_MEL_BINS)
            ).astype(model_dtype)
        )
        mx.eval(audio_positions, audio_latents)

        # Apply I2V conditioning (first frame via replace, end frame via guiding tokens)
        state1 = None
        if stage1_image_latent is not None:
            latent_shape = (1, 128, latent_frames, stage1_h, stage1_w)
            state1 = LatentState(
                latent=mx.zeros(latent_shape, dtype=model_dtype),
                clean_latent=mx.zeros(latent_shape, dtype=model_dtype),
                denoise_mask=mx.ones((1, 1, latent_frames, 1, 1), dtype=model_dtype),
            )
            first_frame_idx = 0 if has_end_image else image_frame_idx
            conditionings = _build_i2v_conditionings(
                stage1_image_latent, first_frame_idx, image_strength,
            )
            state1 = apply_conditioning(state1, conditionings)

            noise = mx.random.normal(latent_shape, dtype=model_dtype)
            noise_scale = mx.array(STAGE_1_SIGMAS[0], dtype=model_dtype)
            scaled_mask = state1.denoise_mask * noise_scale
            state1 = LatentState(
                latent=noise * scaled_mask
                + state1.latent * (mx.array(1.0, dtype=model_dtype) - scaled_mask),
                clean_latent=state1.clean_latent,
                denoise_mask=state1.denoise_mask,
            )
            latents = state1.latent
        else:
            latents = mx.random.normal(
                (1, 128, latent_frames, stage1_h, stage1_w), dtype=model_dtype
            )
        mx.eval(latents)

        s1_guiding_tokens, s1_guiding_positions = None, None
        if stage1_end_image_latent is not None:
            s1_guiding_tokens, s1_guiding_positions = _prepare_guiding_tokens(
                stage1_end_image_latent, positions, latent_frames, stage1_h, stage1_w
            )

        latents, audio_latents = denoise_distilled(
            latents,
            positions,
            text_embeddings,
            transformer,
            STAGE_1_SIGMAS,
            verbose=verbose,
            state=state1,
            audio_latents=audio_latents,
            audio_positions=audio_positions,
            audio_embeddings=audio_embeddings,
            audio_frozen=is_a2v,
            guiding_tokens=s1_guiding_tokens,
            guiding_positions=s1_guiding_positions,
            guiding_strength=end_image_strength,
        )

        # Upsample latents
        with console.status(
            f"[magenta]🔍 Upsampling latents {upscaler_scale}x...[/]", spinner="dots"
        ):
            if upscaler_path is None or not upscaler_path.exists():
                raise FileNotFoundError(f"No spatial upscaler found in {model_path}")
            upsampler, upscaler_scale = load_upsampler(str(upscaler_path))
            mx.eval(upsampler.parameters())

            vae_decoder = VideoDecoder.from_pretrained(
                str(model_path / "vae" / "decoder")
            )

            latents = upsample_latents(
                latents,
                upsampler,
                vae_decoder.per_channel_statistics.mean,
                vae_decoder.per_channel_statistics.std,
            )
            mx.eval(latents)

            del upsampler
            mx.clear_cache()
        console.print("[green]✓[/] Latents upsampled")

        # Stage 2
        console.print(
            f"\n[bold yellow]⚡ Stage 2:[/] Refining at {stage2_w*32}x{stage2_h*32} (3 steps)"
        )
        positions = create_position_grid(1, latent_frames, stage2_h, stage2_w)
        mx.eval(positions)

        state2 = None
        if stage2_image_latent is not None:
            state2 = LatentState(
                latent=latents,
                clean_latent=mx.zeros_like(latents),
                denoise_mask=mx.ones((1, 1, latent_frames, 1, 1), dtype=model_dtype),
            )
            first_frame_idx = 0 if has_end_image else image_frame_idx
            conditionings = _build_i2v_conditionings(
                stage2_image_latent, first_frame_idx, image_strength,
            )
            state2 = apply_conditioning(state2, conditionings)

            noise = mx.random.normal(latents.shape).astype(model_dtype)
            noise_scale = mx.array(STAGE_2_SIGMAS[0], dtype=model_dtype)
            scaled_mask = state2.denoise_mask * noise_scale
            state2 = LatentState(
                latent=noise * scaled_mask
                + state2.latent * (mx.array(1.0, dtype=model_dtype) - scaled_mask),
                clean_latent=state2.clean_latent,
                denoise_mask=state2.denoise_mask,
            )
            latents = state2.latent
            mx.eval(latents)
        else:
            noise_scale = mx.array(STAGE_2_SIGMAS[0], dtype=model_dtype)
            one_minus_scale = mx.array(1.0 - STAGE_2_SIGMAS[0], dtype=model_dtype)
            noise = mx.random.normal(latents.shape).astype(model_dtype)
            latents = noise * noise_scale + latents * one_minus_scale
            mx.eval(latents)

        s2_guiding_tokens, s2_guiding_positions = None, None
        if stage2_end_image_latent is not None:
            positions2 = create_position_grid(1, latent_frames, stage2_h, stage2_w)
            s2_guiding_tokens, s2_guiding_positions = _prepare_guiding_tokens(
                stage2_end_image_latent, positions2, latent_frames, stage2_h, stage2_w
            )

        # Re-noise audio at sigma=0.909375 for joint refinement (matches PyTorch)
        if audio_latents is not None and not is_a2v:
            audio_noise = mx.random.normal(audio_latents.shape, dtype=model_dtype)
            audio_noise_scale = mx.array(STAGE_2_SIGMAS[0], dtype=model_dtype)
            audio_latents = audio_noise * audio_noise_scale + audio_latents * (
                mx.array(1.0, dtype=model_dtype) - audio_noise_scale
            )
            mx.eval(audio_latents)

        # Joint video + audio refinement (no CFG, positive embeddings only)
        latents, audio_latents = denoise_distilled(
            latents,
            positions,
            text_embeddings,
            transformer,
            STAGE_2_SIGMAS,
            verbose=verbose,
            state=state2,
            audio_latents=audio_latents,
            audio_positions=audio_positions,
            audio_embeddings=audio_embeddings,
            audio_frozen=is_a2v,
            guiding_tokens=s2_guiding_tokens,
            guiding_positions=s2_guiding_positions,
            guiding_strength=end_image_strength,
        )

    elif pipeline == PipelineType.DEV:
        # ======================================================================
        # DEV PIPELINE: Single-stage with CFG
        # ======================================================================

        # Load VAE encoder for I2V
        image_latent = None
        end_image_latent = None
        if is_i2v:
            with console.status(
                "[blue]🖼️  Loading VAE encoder and encoding image(s)...[/]", spinner="dots"
            ):
                vae_encoder = VideoEncoder.from_pretrained(
                    model_path / "vae" / "encoder"
                )

                if image is not None:
                    input_image = load_image(image, height=height, width=width, dtype=model_dtype)
                    image_latent = vae_encoder(prepare_image_for_encoding(input_image, height, width, dtype=model_dtype))
                    mx.eval(image_latent)

                if has_end_image:
                    end_input = load_image(end_image, height=height, width=width, dtype=model_dtype)
                    end_image_latent = vae_encoder(prepare_image_for_encoding(end_input, height, width, dtype=model_dtype))
                    mx.eval(end_image_latent)

                del vae_encoder
                mx.clear_cache()
            console.print("[green]✓[/] VAE encoder loaded and image(s) encoded")

        # Generate sigma schedule with token-count-dependent shifting
        sigmas = ltx2_scheduler(steps=num_inference_steps)
        mx.eval(sigmas)
        console.print(
            f"[dim]Sigma schedule: {sigmas[0].item():.4f} → {sigmas[-2].item():.4f} → {sigmas[-1].item():.4f}[/]"
        )

        console.print(
            f"\n[bold yellow]⚡ Generating:[/] {width}x{height} ({num_inference_steps} steps, CFG={cfg_scale}, rescale={cfg_rescale})"
        )
        mx.random.seed(seed)

        video_positions = create_position_grid(1, latent_frames, latent_h, latent_w)
        mx.eval(video_positions)

        # Always init audio latents/positions - PyTorch unconditionally generates audio
        audio_positions = create_audio_position_grid(1, audio_frames)
        audio_latents = (
            a2v_audio_latents
            if is_a2v
            else mx.random.normal(
                (1, AUDIO_LATENT_CHANNELS, audio_frames, AUDIO_MEL_BINS),
                dtype=model_dtype,
            )
        )
        mx.eval(audio_positions, audio_latents)

        # Initialize latents with optional I2V conditioning
        video_state = None
        video_latent_shape = (1, 128, latent_frames, latent_h, latent_w)
        if is_i2v and image_latent is not None:
            video_state = LatentState(
                latent=mx.zeros(video_latent_shape, dtype=model_dtype),
                clean_latent=mx.zeros(video_latent_shape, dtype=model_dtype),
                denoise_mask=mx.ones((1, 1, latent_frames, 1, 1), dtype=model_dtype),
            )
            conditionings = _build_i2v_conditionings(
                image_latent, image_frame_idx, image_strength,
            )
            video_state = apply_conditioning(video_state, conditionings)

            noise = mx.random.normal(video_latent_shape, dtype=model_dtype)
            noise_scale = sigmas[0]
            scaled_mask = video_state.denoise_mask * noise_scale
            video_state = LatentState(
                latent=noise * scaled_mask
                + video_state.latent * (mx.array(1.0, dtype=model_dtype) - scaled_mask),
                clean_latent=video_state.clean_latent,
                denoise_mask=video_state.denoise_mask,
            )
            latents = video_state.latent
            mx.eval(latents)
        else:
            latents = mx.random.normal(video_latent_shape, dtype=model_dtype)
            mx.eval(latents)

        dev_guiding_tokens, dev_guiding_positions = None, None
        if end_image_latent is not None:
            dev_guiding_tokens, dev_guiding_positions = _prepare_guiding_tokens(
                end_image_latent, video_positions, latent_frames, latent_h, latent_w
            )

        # Always use A/V denoising - PyTorch always processes audio+video jointly
        latents, audio_latents = denoise_dev_av(
            latents,
            audio_latents,
            video_positions,
            audio_positions,
            video_embeddings_pos,
            video_embeddings_neg,
            audio_embeddings_pos,
            audio_embeddings_neg,
            transformer,
            sigmas,
            cfg_scale=cfg_scale,
            audio_cfg_scale=audio_cfg_scale,
            cfg_rescale=cfg_rescale,
            verbose=verbose,
            video_state=video_state,
            use_apg=use_apg,
            apg_eta=apg_eta,
            apg_norm_threshold=apg_norm_threshold,
            stg_scale=stg_scale,
            stg_video_blocks=stg_blocks,
            stg_audio_blocks=stg_blocks,
            modality_scale=modality_scale,
            audio_frozen=is_a2v,
            guiding_tokens=dev_guiding_tokens,
            guiding_positions=dev_guiding_positions,
            guiding_strength=end_image_strength,
        )

        # Load VAE decoder (for dev pipeline, loaded here instead of during upsampling)
        vae_decoder = VideoDecoder.from_pretrained(str(model_path / "vae" / "decoder"))

    elif pipeline == PipelineType.DEV_TWO_STAGE:
        # ======================================================================
        # DEV TWO-STAGE PIPELINE:
        #   Stage 1: Dev denoising at half resolution with CFG
        #   Upsample: 2x spatial via LatentUpsampler
        #   Stage 2: Distilled denoising at full resolution with LoRA, no CFG
        # ======================================================================

        # Load VAE encoder for I2V
        stage1_image_latent = None
        stage2_image_latent = None
        stage1_end_image_latent = None
        stage2_end_image_latent = None
        if is_i2v:
            with console.status(
                "[blue]🖼️  Loading VAE encoder and encoding image(s)...[/]", spinner="dots"
            ):
                vae_encoder = VideoEncoder.from_pretrained(
                    model_path / "vae" / "encoder"
                )

                s1_h, s1_w = stage1_h * 32, stage1_w * 32
                s2_h, s2_w = stage2_h * 32, stage2_w * 32

                if image is not None:
                    input_image = load_image(image, height=s1_h, width=s1_w, dtype=model_dtype)
                    stage1_image_latent = vae_encoder(prepare_image_for_encoding(input_image, s1_h, s1_w, dtype=model_dtype))
                    mx.eval(stage1_image_latent)
                    input_image = load_image(image, height=s2_h, width=s2_w, dtype=model_dtype)
                    stage2_image_latent = vae_encoder(prepare_image_for_encoding(input_image, s2_h, s2_w, dtype=model_dtype))
                    mx.eval(stage2_image_latent)

                if has_end_image:
                    end_input = load_image(end_image, height=s1_h, width=s1_w, dtype=model_dtype)
                    stage1_end_image_latent = vae_encoder(prepare_image_for_encoding(end_input, s1_h, s1_w, dtype=model_dtype))
                    mx.eval(stage1_end_image_latent)
                    end_input = load_image(end_image, height=s2_h, width=s2_w, dtype=model_dtype)
                    stage2_end_image_latent = vae_encoder(prepare_image_for_encoding(end_input, s2_h, s2_w, dtype=model_dtype))
                    mx.eval(stage2_end_image_latent)

                del vae_encoder
                mx.clear_cache()
            console.print("[green]✓[/] VAE encoder loaded and image(s) encoded")

        # Stage 1: Dev denoising at reduced resolution with CFG
        sigmas = ltx2_scheduler(steps=num_inference_steps)
        mx.eval(sigmas)
        console.print(
            f"[dim]Stage 1 sigma schedule: {sigmas[0].item():.4f} → {sigmas[-2].item():.4f} → {sigmas[-1].item():.4f}[/]"
        )

        console.print(
            f"\n[bold yellow]⚡ Stage 1:[/] Dev generating at {stage1_w*32}x{stage1_h*32} ({num_inference_steps} steps, CFG={cfg_scale}, rescale={cfg_rescale})"
        )
        mx.random.seed(seed)

        positions = create_position_grid(1, latent_frames, stage1_h, stage1_w)
        mx.eval(positions)

        # Always init audio latents/positions - PyTorch unconditionally generates audio
        audio_positions = create_audio_position_grid(1, audio_frames)
        audio_latents = (
            a2v_audio_latents
            if is_a2v
            else mx.random.normal(
                (1, AUDIO_LATENT_CHANNELS, audio_frames, AUDIO_MEL_BINS),
                dtype=model_dtype,
            )
        )
        mx.eval(audio_positions, audio_latents)

        # Apply I2V conditioning for stage 1
        state1 = None
        stage1_shape = (1, 128, latent_frames, stage1_h, stage1_w)
        if is_i2v and stage1_image_latent is not None:
            state1 = LatentState(
                latent=mx.zeros(stage1_shape, dtype=model_dtype),
                clean_latent=mx.zeros(stage1_shape, dtype=model_dtype),
                denoise_mask=mx.ones((1, 1, latent_frames, 1, 1), dtype=model_dtype),
            )
            conditionings = _build_i2v_conditionings(
                stage1_image_latent, image_frame_idx, image_strength,
            )
            state1 = apply_conditioning(state1, conditionings)

            noise = mx.random.normal(stage1_shape, dtype=model_dtype)
            noise_scale = sigmas[0]
            scaled_mask = state1.denoise_mask * noise_scale
            state1 = LatentState(
                latent=noise * scaled_mask
                + state1.latent * (mx.array(1.0, dtype=model_dtype) - scaled_mask),
                clean_latent=state1.clean_latent,
                denoise_mask=state1.denoise_mask,
            )
            latents = state1.latent
            mx.eval(latents)
        else:
            latents = mx.random.normal(stage1_shape, dtype=model_dtype)
            mx.eval(latents)

        s1_guiding_tokens, s1_guiding_positions = None, None
        if stage1_end_image_latent is not None:
            s1_guiding_tokens, s1_guiding_positions = _prepare_guiding_tokens(
                stage1_end_image_latent, positions, latent_frames, stage1_h, stage1_w
            )

        # Stage 1: Always use joint AV denoising (matches PyTorch)
        latents, audio_latents = denoise_dev_av(
            latents,
            audio_latents,
            positions,
            audio_positions,
            video_embeddings_pos,
            video_embeddings_neg,
            audio_embeddings_pos,
            audio_embeddings_neg,
            transformer,
            sigmas,
            cfg_scale=cfg_scale,
            audio_cfg_scale=audio_cfg_scale,
            cfg_rescale=cfg_rescale,
            verbose=verbose,
            video_state=state1,
            use_apg=use_apg,
            apg_eta=apg_eta,
            apg_norm_threshold=apg_norm_threshold,
            stg_scale=stg_scale,
            stg_video_blocks=stg_blocks,
            stg_audio_blocks=stg_blocks,
            modality_scale=modality_scale,
            audio_frozen=is_a2v,
            guiding_tokens=s1_guiding_tokens,
            guiding_positions=s1_guiding_positions,
            guiding_strength=end_image_strength,
        )

        mx.eval(audio_latents)

        # Upsample latents
        with console.status(
            f"[magenta]🔍 Upsampling latents {upscaler_scale}x...[/]", spinner="dots"
        ):
            if upscaler_path is None or not upscaler_path.exists():
                raise FileNotFoundError(f"No spatial upscaler found in {model_path}")
            upsampler, upscaler_scale = load_upsampler(str(upscaler_path))
            mx.eval(upsampler.parameters())

            vae_decoder = VideoDecoder.from_pretrained(
                str(model_path / "vae" / "decoder")
            )

            latents = upsample_latents(
                latents,
                upsampler,
                vae_decoder.per_channel_statistics.mean,
                vae_decoder.per_channel_statistics.std,
            )
            mx.eval(latents)

            del upsampler
            mx.clear_cache()
        console.print("[green]✓[/] Latents upsampled")

        # Merge LoRA weights for stage 2 (distilled refinement)
        if lora_path is None:
            # Auto-detect LoRA file in model directory
            lora_files = sorted(model_path.glob("*distilled-lora*.safetensors"))
            if lora_files:
                lora_path = str(lora_files[0])
                console.print(f"[dim]Auto-detected LoRA: {Path(lora_path).name}[/]")
            else:
                console.print(
                    "[yellow]⚠️  No LoRA file found. Stage 2 will use base weights.[/]"
                )

        if lora_path is not None:
            with console.status(
                "[blue]🔧 Merging distilled LoRA weights...[/]", spinner="dots"
            ):
                load_and_merge_lora(transformer, lora_path, strength=lora_strength)

        # Stage 2: Distilled refinement at full resolution (no CFG)
        # Matches PyTorch: re-noise audio at sigma=0.909375, then jointly refine
        # both video and audio through the distilled schedule using the LoRA-merged model.
        console.print(
            f"\n[bold yellow]⚡ Stage 2:[/] Distilled refining at {width}x{height} (3 steps, no CFG)"
        )
        positions = create_position_grid(1, latent_frames, stage2_h, stage2_w)
        mx.eval(positions)

        state2 = None
        if is_i2v and stage2_image_latent is not None:
            state2 = LatentState(
                latent=latents,
                clean_latent=mx.zeros_like(latents),
                denoise_mask=mx.ones((1, 1, latent_frames, 1, 1), dtype=model_dtype),
            )
            conditionings = _build_i2v_conditionings(
                stage2_image_latent, image_frame_idx, image_strength,
            )
            state2 = apply_conditioning(state2, conditionings)

            noise = mx.random.normal(latents.shape).astype(model_dtype)
            noise_scale = mx.array(STAGE_2_SIGMAS[0], dtype=model_dtype)
            scaled_mask = state2.denoise_mask * noise_scale
            state2 = LatentState(
                latent=noise * scaled_mask
                + state2.latent * (mx.array(1.0, dtype=model_dtype) - scaled_mask),
                clean_latent=state2.clean_latent,
                denoise_mask=state2.denoise_mask,
            )
            latents = state2.latent
            mx.eval(latents)
        else:
            noise_scale = mx.array(STAGE_2_SIGMAS[0], dtype=model_dtype)
            one_minus_scale = mx.array(1.0 - STAGE_2_SIGMAS[0], dtype=model_dtype)
            noise = mx.random.normal(latents.shape).astype(model_dtype)
            latents = noise * noise_scale + latents * one_minus_scale
            mx.eval(latents)

        # Re-noise audio at sigma=0.909375 for joint refinement (matches PyTorch)
        if audio_latents is not None and not is_a2v:
            audio_noise = mx.random.normal(audio_latents.shape, dtype=model_dtype)
            audio_noise_scale = mx.array(STAGE_2_SIGMAS[0], dtype=model_dtype)
            audio_latents = audio_noise * audio_noise_scale + audio_latents * (
                mx.array(1.0, dtype=model_dtype) - audio_noise_scale
            )
            mx.eval(audio_latents)

        s2_guiding_tokens, s2_guiding_positions = None, None
        if stage2_end_image_latent is not None:
            positions2 = create_position_grid(1, latent_frames, stage2_h, stage2_w)
            s2_guiding_tokens, s2_guiding_positions = _prepare_guiding_tokens(
                stage2_end_image_latent, positions2, latent_frames, stage2_h, stage2_w
            )

        # Joint video + audio refinement (no CFG, positive embeddings only)
        latents, audio_latents = denoise_distilled(
            latents,
            positions,
            text_embeddings,
            transformer,
            STAGE_2_SIGMAS,
            verbose=verbose,
            state=state2,
            audio_latents=audio_latents,
            audio_positions=audio_positions,
            audio_embeddings=audio_embeddings_pos,
            audio_frozen=is_a2v,
            guiding_tokens=s2_guiding_tokens,
            guiding_positions=s2_guiding_positions,
            guiding_strength=end_image_strength,
        )

    elif pipeline == PipelineType.DEV_TWO_STAGE_HQ:
        # ======================================================================
        # DEV TWO-STAGE HQ PIPELINE:
        #   Stage 1: res_2s denoising at half resolution with CFG + LoRA@0.25
        #   Upsample: 2x spatial via LatentUpsampler
        #   Stage 2: res_2s refinement at full resolution with LoRA@0.5, no CFG
        # ======================================================================

        # HQ defaults: STG disabled, lower rescale, fewer steps (PyTorch LTX_2_3_HQ_PARAMS)
        hq_lora_strength_s1 = (
            lora_strength_stage_1 if lora_strength_stage_1 is not None else 0.25
        )
        hq_lora_strength_s2 = (
            lora_strength_stage_2 if lora_strength_stage_2 is not None else 0.5
        )
        hq_cfg_rescale = (
            cfg_rescale if cfg_rescale != 0.7 else 0.45
        )  # Override default 0.7 → 0.45
        hq_steps = (
            num_inference_steps if num_inference_steps != 30 else 15
        )  # Override default 30 → 15
        hq_stg_scale = (
            stg_scale if stg_scale != 1.0 else 0.0
        )  # Override default 1.0 → 0.0

        # Load VAE encoder for I2V
        stage1_image_latent = None
        stage2_image_latent = None
        stage1_end_image_latent = None
        stage2_end_image_latent = None
        if is_i2v:
            with console.status(
                "[blue]Loading VAE encoder and encoding image(s)...[/]", spinner="dots"
            ):
                vae_encoder = VideoEncoder.from_pretrained(
                    model_path / "vae" / "encoder"
                )

                s1_h, s1_w = stage1_h * 32, stage1_w * 32
                s2_h, s2_w = stage2_h * 32, stage2_w * 32

                if image is not None:
                    input_image = load_image(image, height=s1_h, width=s1_w, dtype=model_dtype)
                    stage1_image_latent = vae_encoder(prepare_image_for_encoding(input_image, s1_h, s1_w, dtype=model_dtype))
                    mx.eval(stage1_image_latent)
                    input_image = load_image(image, height=s2_h, width=s2_w, dtype=model_dtype)
                    stage2_image_latent = vae_encoder(prepare_image_for_encoding(input_image, s2_h, s2_w, dtype=model_dtype))
                    mx.eval(stage2_image_latent)

                if has_end_image:
                    end_input = load_image(end_image, height=s1_h, width=s1_w, dtype=model_dtype)
                    stage1_end_image_latent = vae_encoder(prepare_image_for_encoding(end_input, s1_h, s1_w, dtype=model_dtype))
                    mx.eval(stage1_end_image_latent)
                    end_input = load_image(end_image, height=s2_h, width=s2_w, dtype=model_dtype)
                    stage2_end_image_latent = vae_encoder(prepare_image_for_encoding(end_input, s2_h, s2_w, dtype=model_dtype))
                    mx.eval(stage2_end_image_latent)

                del vae_encoder
                mx.clear_cache()
            console.print("[green]✓[/] VAE encoder loaded and image encoded")

        # Auto-detect and merge LoRA for stage 1 (strength 0.25)
        if lora_path is None:
            lora_files = sorted(model_path.glob("*distilled-lora*.safetensors"))
            if lora_files:
                lora_path = str(lora_files[0])
                console.print(f"[dim]Auto-detected LoRA: {Path(lora_path).name}[/]")
            else:
                console.print(
                    "[yellow]Warning: No LoRA file found. HQ pipeline works best with distilled LoRA.[/]"
                )

        if lora_path is not None:
            with console.status(
                f"[blue]Merging distilled LoRA (stage 1, strength={hq_lora_strength_s1})...[/]",
                spinner="dots",
            ):
                load_and_merge_lora(
                    transformer, lora_path, strength=hq_lora_strength_s1
                )

        # Stage 1: res_2s denoising at reduced resolution with CFG
        # HQ passes actual token count to scheduler (unlike regular dev-two-stage)
        num_tokens = latent_frames * stage1_h * stage1_w
        sigmas = ltx2_scheduler(steps=hq_steps, num_tokens=num_tokens)
        mx.eval(sigmas)
        console.print(
            f"[dim]Stage 1 sigma schedule: {sigmas[0].item():.4f} -> {sigmas[-2].item():.4f} -> {sigmas[-1].item():.4f} (tokens={num_tokens})[/]"
        )

        console.print(
            f"\n[bold yellow]Stage 1:[/] res_2s at {stage1_w*32}x{stage1_h*32} ({hq_steps} steps, CFG={cfg_scale}, rescale={hq_cfg_rescale})"
        )
        mx.random.seed(seed)

        positions = create_position_grid(1, latent_frames, stage1_h, stage1_w)
        mx.eval(positions)

        audio_positions = create_audio_position_grid(1, audio_frames)
        audio_latents = (
            a2v_audio_latents
            if is_a2v
            else mx.random.normal(
                (1, AUDIO_LATENT_CHANNELS, audio_frames, AUDIO_MEL_BINS),
                dtype=model_dtype,
            )
        )
        mx.eval(audio_positions, audio_latents)

        # Apply I2V conditioning for stage 1
        state1 = None
        stage1_shape = (1, 128, latent_frames, stage1_h, stage1_w)
        if is_i2v and stage1_image_latent is not None:
            state1 = LatentState(
                latent=mx.zeros(stage1_shape, dtype=model_dtype),
                clean_latent=mx.zeros(stage1_shape, dtype=model_dtype),
                denoise_mask=mx.ones((1, 1, latent_frames, 1, 1), dtype=model_dtype),
            )
            conditionings = _build_i2v_conditionings(
                stage1_image_latent, image_frame_idx, image_strength,
            )
            state1 = apply_conditioning(state1, conditionings)

            noise = mx.random.normal(stage1_shape, dtype=model_dtype)
            noise_scale = sigmas[0]
            scaled_mask = state1.denoise_mask * noise_scale
            state1 = LatentState(
                latent=noise * scaled_mask
                + state1.latent * (mx.array(1.0, dtype=model_dtype) - scaled_mask),
                clean_latent=state1.clean_latent,
                denoise_mask=state1.denoise_mask,
            )
            latents = state1.latent
            mx.eval(latents)
        else:
            latents = mx.random.normal(stage1_shape, dtype=model_dtype)
            mx.eval(latents)

        hq_s1_guiding_tokens, hq_s1_guiding_positions = None, None
        if stage1_end_image_latent is not None:
            hq_s1_guiding_tokens, hq_s1_guiding_positions = _prepare_guiding_tokens(
                stage1_end_image_latent, positions, latent_frames, stage1_h, stage1_w
            )

        # Stage 1: res_2s with CFG (STG disabled for HQ by default)
        latents, audio_latents = denoise_res2s_av(
            latents,
            audio_latents,
            positions,
            audio_positions,
            video_embeddings_pos,
            video_embeddings_neg,
            audio_embeddings_pos,
            audio_embeddings_neg,
            transformer,
            sigmas,
            cfg_scale=cfg_scale,
            audio_cfg_scale=audio_cfg_scale,
            cfg_rescale=hq_cfg_rescale,
            audio_cfg_rescale=1.0,
            verbose=verbose,
            video_state=state1,
            stg_scale=hq_stg_scale,
            stg_video_blocks=stg_blocks,
            stg_audio_blocks=stg_blocks,
            modality_scale=modality_scale,
            noise_seed=seed,
            audio_frozen=is_a2v,
            guiding_tokens=hq_s1_guiding_tokens,
            guiding_positions=hq_s1_guiding_positions,
            guiding_strength=end_image_strength,
        )

        mx.eval(audio_latents)

        # Upsample latents
        with console.status(
            f"[magenta]Upsampling latents {upscaler_scale}x...[/]", spinner="dots"
        ):
            if upscaler_path is None or not upscaler_path.exists():
                raise FileNotFoundError(f"No spatial upscaler found in {model_path}")
            upsampler, upscaler_scale = load_upsampler(str(upscaler_path))
            mx.eval(upsampler.parameters())

            vae_decoder = VideoDecoder.from_pretrained(
                str(model_path / "vae" / "decoder")
            )

            latents = upsample_latents(
                latents,
                upsampler,
                vae_decoder.per_channel_statistics.mean,
                vae_decoder.per_channel_statistics.std,
            )
            mx.eval(latents)

            del upsampler
            mx.clear_cache()
        console.print("[green]✓[/] Latents upsampled")

        # Merge additional LoRA for stage 2 (additive: 0.25 + 0.25 = 0.5 total)
        if lora_path is not None:
            additional_strength = hq_lora_strength_s2 - hq_lora_strength_s1
            if additional_strength > 0:
                with console.status(
                    f"[blue]Adjusting LoRA (stage 2, total={hq_lora_strength_s2})...[/]",
                    spinner="dots",
                ):
                    load_and_merge_lora(
                        transformer, lora_path, strength=additional_strength
                    )

        # Stage 2: res_2s refinement at full resolution (no CFG)
        console.print(
            f"\n[bold yellow]Stage 2:[/] res_2s refining at {stage2_w*32}x{stage2_h*32} (3 steps, no CFG)"
        )
        positions = create_position_grid(1, latent_frames, stage2_h, stage2_w)
        mx.eval(positions)

        state2 = None
        if is_i2v and stage2_image_latent is not None:
            state2 = LatentState(
                latent=latents,
                clean_latent=mx.zeros_like(latents),
                denoise_mask=mx.ones((1, 1, latent_frames, 1, 1), dtype=model_dtype),
            )
            conditionings = _build_i2v_conditionings(
                stage2_image_latent, image_frame_idx, image_strength,
            )
            state2 = apply_conditioning(state2, conditionings)

            noise = mx.random.normal(latents.shape).astype(model_dtype)
            noise_scale = mx.array(STAGE_2_SIGMAS[0], dtype=model_dtype)
            scaled_mask = state2.denoise_mask * noise_scale
            state2 = LatentState(
                latent=noise * scaled_mask
                + state2.latent * (mx.array(1.0, dtype=model_dtype) - scaled_mask),
                clean_latent=state2.clean_latent,
                denoise_mask=state2.denoise_mask,
            )
            latents = state2.latent
            mx.eval(latents)
        else:
            noise_scale = mx.array(STAGE_2_SIGMAS[0], dtype=model_dtype)
            one_minus_scale = mx.array(1.0 - STAGE_2_SIGMAS[0], dtype=model_dtype)
            noise = mx.random.normal(latents.shape).astype(model_dtype)
            latents = noise * noise_scale + latents * one_minus_scale
            mx.eval(latents)

        # Re-noise audio at sigma=0.909375 for joint refinement
        if audio_latents is not None and not is_a2v:
            audio_noise = mx.random.normal(audio_latents.shape, dtype=model_dtype)
            audio_noise_scale = mx.array(STAGE_2_SIGMAS[0], dtype=model_dtype)
            audio_latents = audio_noise * audio_noise_scale + audio_latents * (
                mx.array(1.0, dtype=model_dtype) - audio_noise_scale
            )
            mx.eval(audio_latents)

        hq_s2_guiding_tokens, hq_s2_guiding_positions = None, None
        if stage2_end_image_latent is not None:
            hq_positions2 = create_position_grid(1, latent_frames, stage2_h, stage2_w)
            hq_s2_guiding_tokens, hq_s2_guiding_positions = _prepare_guiding_tokens(
                stage2_end_image_latent, hq_positions2, latent_frames, stage2_h, stage2_w
            )

        # Stage 2: res_2s with no CFG (positive embeddings only)
        stage2_sigmas = mx.array(STAGE_2_SIGMAS, dtype=mx.float32)
        latents, audio_latents = denoise_res2s_av(
            latents,
            audio_latents,
            positions,
            audio_positions,
            video_embeddings_pos,
            video_embeddings_pos,  # both pos (no neg for stage 2)
            audio_embeddings_pos,
            audio_embeddings_pos,
            transformer,
            stage2_sigmas,
            cfg_scale=1.0,  # no CFG
            audio_cfg_scale=1.0,
            cfg_rescale=0.0,
            verbose=verbose,
            video_state=state2,
            noise_seed=seed + 1,
            audio_frozen=is_a2v,
            guiding_tokens=hq_s2_guiding_tokens,
            guiding_positions=hq_s2_guiding_positions,
            guiding_strength=end_image_strength,
        )

    del transformer
    mx.clear_cache()

    # ==========================================================================
    # Decode and save outputs (common to both pipelines)
    # ==========================================================================

    console.print("\n[blue]🎞️  Decoding video...[/]")

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
        console.print(f"[yellow]  Unknown tiling mode '{tiling}', using auto[/]")
        tiling_config = TilingConfig.auto(height, width, num_frames)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Stream mode
    video_writer = None
    stream_progress = None

    if stream and tiling_config is not None:
        import cv2

        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        stream_progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        )
        stream_progress.start()
        stream_task = stream_progress.add_task(
            "[cyan]Streaming frames[/]", total=num_frames
        )

        def on_frames_ready(frames: mx.array, _start_idx: int):
            frames = mx.squeeze(frames, axis=0)
            frames = mx.transpose(frames, (1, 2, 3, 0))
            frames = mx.clip((frames + 1.0) / 2.0, 0.0, 1.0)
            frames = (frames * 255).astype(mx.uint8)
            frames_np = np.array(frames)

            for frame in frames_np:
                video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                stream_progress.advance(stream_task)

    else:
        on_frames_ready = None

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
        console.print(
            f"[dim]  Tiling ({tiling}): spatial={spatial_info}, temporal={temporal_info}[/]"
        )
        video = vae_decoder.decode_tiled(
            latents,
            tiling_config=tiling_config,
            tiling_mode=tiling,
            debug=verbose,
            on_frames_ready=on_frames_ready,
        )
    else:
        console.print("[dim]  Tiling: disabled[/]")
        video = vae_decoder(latents)
    mx.eval(video)
    mx.clear_cache()

    # Close stream writer
    if video_writer is not None:
        video_writer.release()
        if stream_progress is not None:
            stream_progress.stop()
        console.print(f"[green]✅ Streamed video to[/] {output_path}")
        video = mx.squeeze(video, axis=0)
        video = mx.transpose(video, (1, 2, 3, 0))
        video = mx.clip((video + 1.0) / 2.0, 0.0, 1.0)
        video = (video * 255).astype(mx.uint8)
        video_np = np.array(video)
    else:
        video = mx.squeeze(video, axis=0)
        video = mx.transpose(video, (1, 2, 3, 0))
        video = mx.clip((video + 1.0) / 2.0, 0.0, 1.0)
        video = (video * 255).astype(mx.uint8)
        video_np = np.array(video)

        if audio:
            temp_video_path = output_path.with_suffix(".temp.mp4")
            save_path = temp_video_path
        else:
            save_path = output_path

        try:
            import cv2

            h, w = video_np.shape[1], video_np.shape[2]
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            out = cv2.VideoWriter(str(save_path), fourcc, fps, (w, h))
            for frame in video_np:
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            out.release()
            if not audio:
                console.print(f"[green]✅ Saved video to[/] {output_path}")
        except Exception as e:
            console.print(f"[red]❌ Could not save video: {e}[/]")

    # Decode and save audio if enabled
    audio_np = None
    vocoder_sample_rate = AUDIO_SAMPLE_RATE
    if audio and audio_latents is not None:
        if is_a2v and a2v_waveform is not None:
            # A2V: use original input audio waveform (no VAE decoding needed)
            audio_np = a2v_waveform
            if audio_np.ndim == 1:
                audio_np = audio_np[np.newaxis, :]
            vocoder_sample_rate = a2v_sr or AUDIO_LATENT_SAMPLE_RATE
            console.print("[green]✓[/] Using original input audio (A2V)")
        else:
            with console.status("[blue]Decoding audio...[/]", spinner="dots"):
                audio_decoder = load_audio_decoder(model_path, pipeline)
                vocoder = load_vocoder_model(model_path, pipeline)
                mx.eval(audio_decoder.parameters(), vocoder.parameters())

                mel_spectrogram = audio_decoder(audio_latents)
                mx.eval(mel_spectrogram)
                console.print(
                    f"[dim]  Mel spectrogram: shape={mel_spectrogram.shape}, std={mel_spectrogram.std().item():.4f}, mean={mel_spectrogram.mean().item():.4f}[/]"
                )

                audio_waveform = vocoder(mel_spectrogram)
                mx.eval(audio_waveform)

                audio_np = np.array(audio_waveform.astype(mx.float32))
                if audio_np.ndim == 3:
                    audio_np = audio_np[0]

                # Get sample rate from vocoder (dynamic: 24kHz for LTX-2, 48kHz for LTX-2.3 BWE)
                vocoder_sample_rate = getattr(
                    vocoder, "output_sampling_rate", AUDIO_SAMPLE_RATE
                )

                del audio_decoder, vocoder
                mx.clear_cache()
            console.print("[green]✓[/] Audio decoded")

        audio_path = (
            Path(output_audio_path)
            if output_audio_path
            else output_path.with_suffix(".wav")
        )
        save_audio(audio_np, audio_path, vocoder_sample_rate)
        console.print(f"[green]✅ Saved audio to[/] {audio_path}")

        with console.status("[blue]🎬 Combining video and audio...[/]", spinner="dots"):
            temp_video_path = output_path.with_suffix(".temp.mp4")
            success = mux_video_audio(temp_video_path, audio_path, output_path)
        if success:
            console.print(f"[green]✅ Saved video with audio to[/] {output_path}")
            temp_video_path.unlink()
        else:
            temp_video_path.rename(output_path)
            console.print(f"[yellow]⚠️  Saved video without audio to[/] {output_path}")

    del vae_decoder
    mx.clear_cache()

    if save_frames:
        frames_dir = output_path.parent / f"{output_path.stem}_frames"
        frames_dir.mkdir(exist_ok=True)
        for i, frame in enumerate(video_np):
            Image.fromarray(frame).save(frames_dir / f"frame_{i:04d}.png")
        console.print(f"[green]✅ Saved {len(video_np)} frames to {frames_dir}[/]")

    elapsed = time.time() - start_time
    minutes, seconds = divmod(elapsed, 60)
    time_str = f"{int(minutes)}m {seconds:.1f}s" if minutes >= 1 else f"{seconds:.1f}s"
    console.print(
        Panel(
            f"[bold green]🎉 Done![/] Generated in {time_str} ({elapsed/num_frames:.2f}s/frame)\n"
            f"[bold green]✨ Peak memory:[/] {mx.get_peak_memory() / (1024 ** 3):.2f}GB",
            expand=False,
        )
    )

    if audio:
        return video_np, audio_np
    return video_np


def main():
    parser = argparse.ArgumentParser(
        description="Generate videos with MLX LTX-2 (Distilled or Dev pipeline)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Distilled pipeline (two-stage, fast, no CFG)
  python -m mlx_video.generate --prompt "A cat walking on grass"
  python -m mlx_video.generate --prompt "Ocean waves" --pipeline distilled

  # Dev pipeline (single-stage, CFG, higher quality)
  python -m mlx_video.generate --prompt "A cat walking" --pipeline dev --cfg-scale 3.0
  python -m mlx_video.generate --prompt "Ocean waves" --pipeline dev --steps 40

  # Dev two-stage pipeline (dev + LoRA refinement)
  python -m mlx_video.generate --prompt "A cat walking" --pipeline dev-two-stage --cfg-scale 3.0

  # Image-to-Video (works with both pipelines)
  python -m mlx_video.generate --prompt "A person dancing" --image photo.jpg
  python -m mlx_video.generate --prompt "Waves crashing" --image beach.png --pipeline dev

  # With Audio (works with both pipelines)
  python -m mlx_video.generate --prompt "Ocean waves crashing" --audio
  python -m mlx_video.generate --prompt "A jazz band playing" --audio --pipeline dev
        """,
    )

    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        required=True,
        help="Text description of the video to generate",
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        default="distilled",
        choices=["distilled", "dev", "dev-two-stage", "dev-two-stage-hq"],
        help="Pipeline type: distilled (fast), dev (CFG), dev-two-stage (dev + LoRA), dev-two-stage-hq (res_2s + LoRA both stages)",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=DEFAULT_NEGATIVE_PROMPT,
        help="Negative prompt for CFG (dev pipeline only)",
    )
    parser.add_argument(
        "--height", "-H", type=int, default=512, help="Output video height"
    )
    parser.add_argument(
        "--width", "-W", type=int, default=512, help="Output video width"
    )
    parser.add_argument(
        "--num-frames", "-n", type=int, default=33, help="Number of frames"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Number of inference steps (dev pipeline only, default 30)",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=3.0,
        help="CFG guidance scale for video (dev pipeline only, default 3.0)",
    )
    parser.add_argument(
        "--audio-cfg-scale",
        type=float,
        default=7.0,
        help="CFG guidance scale for audio (default 7.0, PyTorch default)",
    )
    parser.add_argument(
        "--cfg-rescale",
        type=float,
        default=0.7,
        help="CFG rescale factor (0.0-1.0). Normalizes guided prediction variance to reduce artifacts (dev pipeline only, default 0.7)",
    )
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second")
    parser.add_argument(
        "--output-path", "-o", type=str, default="output.mp4", help="Output video path"
    )
    parser.add_argument(
        "--save-frames", action="store_true", help="Save individual frames as images"
    )
    parser.add_argument(
        "--model-repo", type=str, default="Lightricks/LTX-2", help="Model repository"
    )
    parser.add_argument(
        "--text-encoder-repo", type=str, default=None, help="Text encoder repository"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--enhance-prompt", action="store_true", help="Enhance the prompt using Gemma"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=512, help="Max tokens for prompt enhancement"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for prompt enhancement",
    )
    parser.add_argument(
        "--image",
        "-i",
        type=str,
        default=None,
        help="Path to conditioning image for I2V",
    )
    parser.add_argument(
        "--image-strength",
        type=float,
        default=1.0,
        help="Conditioning strength for I2V",
    )
    parser.add_argument(
        "--image-frame-idx",
        type=int,
        default=0,
        help="Frame index to condition for I2V (ignored when --end-image is set)",
    )
    parser.add_argument(
        "--end-image",
        type=str,
        default=None,
        help="Path to conditioning image for the last frame (I2V end-frame control)",
    )
    parser.add_argument(
        "--end-image-strength",
        type=float,
        default=None,
        help="Conditioning strength for end frame (defaults to --image-strength)",
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
        help="Tiling mode for VAE decoding",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream frames to output as they're decoded",
    )
    parser.add_argument(
        "--audio",
        "-a",
        action="store_true",
        help="Enable synchronized audio generation",
    )
    parser.add_argument(
        "--audio-file",
        type=str,
        default=None,
        help="Path to audio file for A2V (audio-to-video) conditioning",
    )
    parser.add_argument(
        "--audio-start-time",
        type=float,
        default=0.0,
        help="Start time in seconds for audio file (default: 0.0)",
    )
    parser.add_argument(
        "--output-audio", type=str, default=None, help="Output audio path"
    )
    parser.add_argument(
        "--apg",
        action="store_true",
        help="Use Adaptive Projected Guidance instead of CFG (more stable for I2V)",
    )
    parser.add_argument(
        "--apg-eta",
        type=float,
        default=1.0,
        help="APG parallel component weight (1.0 = keep full parallel)",
    )
    parser.add_argument(
        "--apg-norm-threshold",
        type=float,
        default=0.0,
        help="APG guidance norm clamp (0 = no clamping)",
    )
    parser.add_argument(
        "--stg-scale",
        type=float,
        default=1.0,
        help="STG (Spatiotemporal Guidance) scale (default 1.0, 0.0 = disabled)",
    )
    parser.add_argument(
        "--stg-blocks",
        type=int,
        nargs="+",
        default=None,
        help="Transformer block indices for STG perturbation (default: [29] for LTX-2, [28] for LTX-2.3)",
    )
    parser.add_argument(
        "--modality-scale",
        type=float,
        default=3.0,
        help="Cross-modal guidance scale (default 3.0, 1.0 = disabled)",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="Path to LoRA safetensors file (dev-two-stage pipeline)",
    )
    parser.add_argument(
        "--lora-strength",
        type=float,
        default=1.0,
        help="LoRA merge strength (dev-two-stage pipeline, default 1.0)",
    )
    parser.add_argument(
        "--lora-strength-stage-1",
        type=float,
        default=0.25,
        help="LoRA strength for HQ stage 1 (default 0.25)",
    )
    parser.add_argument(
        "--lora-strength-stage-2",
        type=float,
        default=0.5,
        help="LoRA strength for HQ stage 2 (default 0.5)",
    )
    parser.add_argument(
        "--spatial-upscaler",
        type=str,
        default=None,
        help="Spatial upscaler filename (e.g. ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors). "
        "Auto-detects x2 by default. Use this to select x1.5 or a specific version.",
    )
    args = parser.parse_args()

    pipeline_map = {
        "distilled": PipelineType.DISTILLED,
        "dev": PipelineType.DEV,
        "dev-two-stage": PipelineType.DEV_TWO_STAGE,
        "dev-two-stage-hq": PipelineType.DEV_TWO_STAGE_HQ,
    }
    pipeline = pipeline_map[args.pipeline]

    generate_video(
        model_repo=args.model_repo,
        text_encoder_repo=args.text_encoder_repo,
        prompt=args.prompt,
        pipeline=pipeline,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        cfg_scale=args.cfg_scale,
        audio_cfg_scale=args.audio_cfg_scale,
        cfg_rescale=args.cfg_rescale,
        seed=args.seed,
        fps=args.fps,
        output_path=args.output_path,
        save_frames=args.save_frames,
        verbose=args.verbose,
        enhance_prompt=args.enhance_prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        image=args.image,
        image_strength=args.image_strength,
        image_frame_idx=args.image_frame_idx,
        end_image=args.end_image,
        end_image_strength=args.end_image_strength,
        tiling=args.tiling,
        stream=args.stream,
        audio=args.audio,
        output_audio_path=args.output_audio,
        use_apg=args.apg,
        apg_eta=args.apg_eta,
        apg_norm_threshold=args.apg_norm_threshold,
        stg_scale=args.stg_scale,
        stg_blocks=args.stg_blocks,
        modality_scale=args.modality_scale,
        lora_path=args.lora_path,
        lora_strength=args.lora_strength,
        lora_strength_stage_1=args.lora_strength_stage_1,
        lora_strength_stage_2=args.lora_strength_stage_2,
        audio_file=args.audio_file,
        audio_start_time=args.audio_start_time,
        spatial_upscaler=args.spatial_upscaler,
    )


if __name__ == "__main__":
    main()
