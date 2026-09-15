"""Core layout and token helpers for LTX-2.5 Diffusion Fidelity Rendering."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import TYPE_CHECKING

import mlx.core as mx

from mlx_video.models.ltx_2.transformer import Modality

if TYPE_CHECKING:
    from mlx_video.models.ltx_2.ltx_2 import LTXModel


SEGMENT_CANDIDATES = (24, 32)


def padding_to_segment(content_frames: int, segment: int) -> int:
    """Return the frames needed to round content up to a segment boundary."""
    return (-content_frames) % segment


def choose_segment_length(content_frames: int) -> int:
    """Choose the DFR segment with the least padding, preferring the larger tie."""
    if content_frames < 1:
        raise ValueError(f"content_frames must be >= 1, got {content_frames}")
    return min(
        SEGMENT_CANDIDATES,
        key=lambda segment: (padding_to_segment(content_frames, segment), -segment),
    )


def resolve_dfr_canvas(num_frames: int) -> tuple[int, int, list[int]]:
    """Return padded frame count, segment size, and generated-keyframe positions."""
    if num_frames < 2:
        raise ValueError("DFR requires at least 2 pixel frames")
    if (num_frames - 1) % 8:
        raise ValueError(f"num_frames must be 1 + 8*k, got {num_frames}")
    content = num_frames - 1
    segment = choose_segment_length(content)
    padded = content + padding_to_segment(content, segment)
    positions = list(range(segment, padded + 1, segment))
    return padded + 1, segment, positions


def create_keyframe_slot_positions(
    pixel_frame_indices: Sequence[int],
    *,
    height: int,
    width: int,
    fps: float,
    spatial_scale: int = 32,
) -> mx.array:
    """Create RoPE bounds for single-pixel-frame generated keyframe slots."""
    if not pixel_frame_indices:
        raise ValueError("pixel_frame_indices must not be empty")
    if any(b <= a for a, b in itertools.pairwise(pixel_frame_indices)):
        raise ValueError("pixel_frame_indices must be strictly increasing")

    spatial_tokens = height * width
    h = mx.repeat(mx.arange(height), width)
    w = mx.tile(mx.arange(width), height)
    spatial = mx.stack(
        [
            mx.stack([h * spatial_scale, (h + 1) * spatial_scale], axis=-1),
            mx.stack([w * spatial_scale, (w + 1) * spatial_scale], axis=-1),
        ],
        axis=0,
    )
    planes = []
    for frame in pixel_frame_indices:
        temporal = mx.broadcast_to(
            mx.array([frame / fps, (frame + 1) / fps], dtype=mx.float32),
            (spatial_tokens, 2),
        )
        planes.append(mx.concatenate([temporal[None], spatial], axis=0))
    positions = mx.concatenate(planes, axis=1)[None]
    return positions.astype(mx.bfloat16).astype(mx.float32)


def flatten_video_latents(latents: mx.array) -> mx.array:
    """Convert BCHWT video latents to B-token-C transformer layout."""
    b, c, _, _, _ = latents.shape
    return mx.transpose(mx.reshape(latents, (b, c, -1)), (0, 2, 1))


def unflatten_video_tokens(
    tokens: mx.array, *, frames: int, height: int, width: int
) -> mx.array:
    """Convert B-token-C transformer output to BCHWT video latents."""
    b, _, c = tokens.shape
    return mx.reshape(mx.transpose(tokens, (0, 2, 1)), (b, c, frames, height, width))


def keyframe_token_mask(
    *, batch_size: int, video_tokens: int, slot_tokens: int, tokens_per_frame: int
) -> mx.array:
    """Mark the causal first frame and all generated slot tokens."""
    first = mx.ones((batch_size, tokens_per_frame), dtype=mx.float32)
    middle = mx.zeros((batch_size, video_tokens - tokens_per_frame), dtype=mx.float32)
    slots = mx.ones((batch_size, slot_tokens), dtype=mx.float32)
    return mx.concatenate([first, middle, slots], axis=1)


def _ancestral_step(
    sample: mx.array,
    denoised: mx.array,
    sigma: float,
    sigma_next: float,
    key: mx.array,
) -> tuple[mx.array, mx.array]:
    if sigma_next == 0:
        return denoised, key
    downstep_ratio = sigma_next / sigma
    sigma_down = sigma_next * downstep_ratio
    sigma_down_ratio = sigma_down / sigma
    result = sigma_down_ratio * sample + (1.0 - sigma_down_ratio) * denoised
    alpha_next = 1.0 - sigma_next
    alpha_down = 1.0 - sigma_down
    coefficient = (
        max(
            sigma_next**2 - sigma_down**2 * alpha_next**2 / alpha_down**2,
            0.0,
        )
        ** 0.5
    )
    next_key, step_key = mx.random.split(key)
    noise = mx.random.normal(sample.shape, dtype=mx.float32, key=step_key)
    return (alpha_next / alpha_down) * result + noise * coefficient, next_key


def denoise_dfr_tokens(
    video_latents: mx.array,
    slot_latents: mx.array,
    video_positions: mx.array,
    slot_positions: mx.array,
    text_embeddings: mx.array,
    transformer: LTXModel,
    sigmas: Sequence[float],
    *,
    audio_latents: mx.array,
    audio_positions: mx.array,
    audio_embeddings: mx.array,
    reference_latents: mx.array | None = None,
    reference_positions: mx.array | None = None,
    ancestral: bool = False,
    noise_seed: int = 42,
) -> tuple[mx.array, mx.array, mx.array]:
    """Denoise target video and generated slots with an optional clean IC reference."""
    dtype = video_latents.dtype
    b, _, frames, height, width = video_latents.shape
    slot_count = slot_latents.shape[2]
    video_count = frames * height * width
    slot_token_count = slot_count * height * width
    tokens_per_frame = height * width
    keyframes_mask = keyframe_token_mask(
        batch_size=b,
        video_tokens=video_count,
        slot_tokens=slot_token_count,
        tokens_per_frame=tokens_per_frame,
    )
    positions = mx.concatenate([video_positions, slot_positions], axis=2)

    reference_tokens = None
    if reference_latents is not None:
        if reference_positions is None:
            raise ValueError("reference_positions are required with reference_latents")
        reference_tokens = flatten_video_latents(reference_latents).astype(dtype)
        positions = mx.concatenate([positions, reference_positions], axis=2)
        keyframes_mask = mx.concatenate(
            [
                keyframes_mask,
                mx.zeros((b, reference_tokens.shape[1]), dtype=mx.float32),
            ],
            axis=1,
        )

    video_tokens = flatten_video_latents(video_latents).astype(mx.float32)
    slot_tokens = flatten_video_latents(slot_latents).astype(mx.float32)
    audio_latents = audio_latents.astype(mx.float32)
    noise_key = mx.random.key(noise_seed)

    for sigma, sigma_next in itertools.pairwise(sigmas):
        active_tokens = mx.concatenate([video_tokens, slot_tokens], axis=1)
        latent_tokens = active_tokens
        active_timesteps = mx.full(active_tokens.shape[:2], sigma, dtype=dtype)
        timesteps = active_timesteps
        if reference_tokens is not None:
            latent_tokens = mx.concatenate(
                [latent_tokens, reference_tokens.astype(mx.float32)], axis=1
            )
            timesteps = mx.concatenate(
                [
                    timesteps,
                    mx.zeros((b, reference_tokens.shape[1]), dtype=dtype),
                ],
                axis=1,
            )

        video_modality = Modality(
            latent=latent_tokens.astype(dtype),
            timesteps=timesteps,
            positions=positions,
            context=text_embeddings,
            sigma=mx.full((b,), sigma, dtype=dtype),
            keyframes_mask=keyframes_mask,
        )
        ab, ac, at, af = audio_latents.shape
        audio_flat = mx.reshape(
            mx.transpose(audio_latents, (0, 2, 1, 3)), (ab, at, ac * af)
        )
        audio_modality = Modality(
            latent=audio_flat.astype(dtype),
            timesteps=mx.full((ab, at), sigma, dtype=dtype),
            positions=audio_positions,
            context=audio_embeddings,
            sigma=mx.full((ab,), sigma, dtype=dtype),
        )
        velocity, audio_velocity = transformer(video_modality, audio_modality)
        sigma_value = mx.array(sigma, dtype=mx.float32)
        denoised = active_tokens - sigma_value * velocity[
            :, : active_tokens.shape[1]
        ].astype(mx.float32)
        audio_velocity = mx.transpose(
            mx.reshape(audio_velocity, (ab, at, ac, af)), (0, 2, 1, 3)
        )
        audio_denoised = audio_latents - sigma_value * audio_velocity.astype(mx.float32)

        if ancestral:
            active_tokens, noise_key = _ancestral_step(
                active_tokens, denoised, sigma, sigma_next, noise_key
            )
            audio_latents, noise_key = _ancestral_step(
                audio_latents, audio_denoised, sigma, sigma_next, noise_key
            )
        elif sigma_next > 0:
            next_value = mx.array(sigma_next, dtype=mx.float32)
            active_tokens = (
                denoised + next_value * (active_tokens - denoised) / sigma_value
            )
            audio_latents = (
                audio_denoised
                + next_value * (audio_latents - audio_denoised) / sigma_value
            )
        else:
            active_tokens = denoised
            audio_latents = audio_denoised
        video_tokens = active_tokens[:, :video_count]
        slot_tokens = active_tokens[:, video_count:]
        mx.eval(video_tokens, slot_tokens, audio_latents)

    return (
        unflatten_video_tokens(
            video_tokens.astype(dtype), frames=frames, height=height, width=width
        ),
        unflatten_video_tokens(
            slot_tokens.astype(dtype), frames=slot_count, height=height, width=width
        ),
        audio_latents.astype(dtype),
    )
