"""LTX-2.5 diffusion video VAE decoder for MLX."""

from __future__ import annotations

import json
import math
from pathlib import Path

import mlx.core as mx
from mlx import nn
from safetensors import safe_open

from mlx_video.models.ltx_2.diffusion_vae.layers import (
    LinearPixelShuffleUpsample,
    NABlock,
    NeighborhoodAttention3D,
    SwiGLU,
)


def timestep_embedding(timestep: mx.array, dimension: int = 256) -> mx.array:
    half = dimension // 2
    exponent = -math.log(10000) * mx.arange(half, dtype=mx.float32) / half
    angles = timestep[:, None].astype(mx.float32) * mx.exp(exponent)[None]
    return mx.concatenate([mx.cos(angles), mx.sin(angles)], axis=-1)


def patchify_pixels(x: mx.array, patch_size: int) -> mx.array:
    """BCTHW pixels to BTHW(C*r*q), matching the official c-r-q order."""
    batch, channels, frames, height, width = x.shape
    x = mx.reshape(
        x,
        (
            batch,
            channels,
            frames,
            height // patch_size,
            patch_size,
            width // patch_size,
            patch_size,
        ),
    )
    x = mx.transpose(x, (0, 2, 3, 5, 1, 6, 4))
    return mx.reshape(
        x,
        (
            batch,
            frames,
            height // patch_size,
            width // patch_size,
            channels * patch_size**2,
        ),
    )


def unpatchify_pixels(x: mx.array, patch_size: int, channels: int) -> mx.array:
    """BTHW(C*r*q) patches to BCTHW pixels."""
    batch, frames, height, width, _ = x.shape
    x = mx.reshape(
        x,
        (batch, frames, height, width, channels, patch_size, patch_size),
    )
    x = mx.transpose(x, (0, 4, 1, 2, 6, 3, 5))
    return mx.reshape(
        x,
        (
            batch,
            channels,
            frames,
            height * patch_size,
            width * patch_size,
        ),
    )


class TimestepEmbedder(nn.Module):
    def __init__(self, embedding_dim: int = 384) -> None:
        super().__init__()
        self.mlp = [
            nn.Linear(256, embedding_dim, bias=True),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim, bias=True),
        ]

    def __call__(self, timestep: mx.array) -> mx.array:
        hidden = timestep_embedding(timestep)
        for layer in self.mlp:
            hidden = layer(hidden)
        return hidden


class AdaLNZero(nn.Module):
    def __init__(self, dim: int, embedding_dim: int = 384) -> None:
        super().__init__()
        self.proj = nn.Linear(embedding_dim, 7 * dim, bias=True)

    def __call__(self, embedding: mx.array) -> tuple[mx.array, ...]:
        projected = self.proj(nn.silu(embedding))
        return tuple(
            part[:, None, None, None] for part in mx.split(projected, 7, axis=-1)
        )


class DiffusionNABlock(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: tuple[int, int, int],
        context_channels: int,
        head_dim: int = 64,
    ) -> None:
        super().__init__()
        self.context_proj = nn.Linear(context_channels, dim, bias=True)
        self.scale_shift_table = mx.zeros((7, dim))
        self.norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.attn = NeighborhoodAttention3D(dim, kernel_size, head_dim)
        self.norm2 = nn.RMSNorm(dim, eps=1e-6)
        self.mlp = SwiGLU(dim, 4 * dim)

    def __call__(
        self,
        context: mx.array,
        x: mx.array,
        modulation: tuple[mx.array, ...],
    ) -> mx.array:
        values = tuple(
            modulation[index] + self.scale_shift_table[index][None, None, None, None]
            for index in range(7)
        )
        scale_msa, shift_msa, _, scale_mlp, shift_mlp, _, _ = values
        x = x + self.context_proj(context)
        attention_input = self.norm1(x) * (1 + scale_msa) + shift_msa
        x = x + self.attn(attention_input)
        mlp_input = self.norm2(x) * (1 + scale_mlp) + shift_mlp
        return x + self.mlp(mlp_input)

    def forward_joint(
        self,
        context: mx.array,
        x: mx.array,
        keyframe_context: mx.array,
        keyframes: mx.array,
        modulation: tuple[mx.array, ...],
        keyframe_times: mx.array,
    ) -> tuple[mx.array, mx.array]:
        values = tuple(
            modulation[index] + self.scale_shift_table[index][None, None, None, None]
            for index in range(7)
        )
        scale_msa, shift_msa, _, scale_mlp, shift_mlp, _, _ = values
        x = x + self.context_proj(context)
        keyframes = keyframes + self.context_proj(keyframe_context)
        attention, keyframe_attention = self.attn.forward_joint(
            self.norm1(x) * (1 + scale_msa) + shift_msa,
            self.norm1(keyframes) * (1 + scale_msa) + shift_msa,
            keyframe_times,
        )
        x = x + attention
        keyframes = keyframes + keyframe_attention
        x = x + self.mlp(self.norm2(x) * (1 + scale_mlp) + shift_mlp)
        keyframes = keyframes + self.mlp(
            self.norm2(keyframes) * (1 + scale_mlp) + shift_mlp
        )
        return x, keyframes


class DiffusionVideoDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 3,
        patch_size: int = 4,
        head_dim: int = 64,
        stage_channels: tuple[int, ...] = (2048, 1024, 512, 512, 256),
        stage_depths: tuple[int, ...] = (4, 6, 4, 2, 8),
        stage_kernels: tuple[tuple[int, int, int], ...] = (
            (3, 7, 7),
            (3, 7, 7),
            (3, 5, 5),
            (3, 5, 5),
            (11, 11, 11),
        ),
        upsamples: tuple[tuple[tuple[int, int, int], int], ...] = (
            ((1, 2, 2), 2),
            ((2, 1, 1), 2),
            ((2, 2, 2), 1),
            ((2, 2, 2), 2),
        ),
        timestep_scale_multiplier: float = 1000.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.stage_channels = stage_channels
        self.stage_depths = stage_depths
        self.stage_kernels = stage_kernels
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.mean_of_means = mx.zeros((in_channels,))
        self.std_of_means = mx.ones((in_channels,))
        self.type_emb = mx.zeros((in_channels,))
        self.conv_in = nn.Linear(in_channels, stage_channels[0], bias=True)
        self.det_stages = []
        self.upsamples = []
        for index in range(4):
            self.det_stages.append(
                [
                    NABlock(stage_channels[index], stage_kernels[index], head_dim)
                    for _ in range(stage_depths[index])
                ]
            )
            stride, reduction = upsamples[index]
            self.upsamples.append(
                LinearPixelShuffleUpsample(stage_channels[index], stride, reduction)
            )

        stage5_channels = stage_channels[-1]
        pixel_channels = out_channels * patch_size**2
        self.conv_in_x_t = nn.Linear(pixel_channels, stage5_channels, bias=True)
        self.t_embedder = TimestepEmbedder()
        self.shared_adaln = AdaLNZero(stage5_channels)
        self.diff_blocks = [
            DiffusionNABlock(
                stage5_channels,
                stage_kernels[-1],
                stage5_channels,
                head_dim,
            )
            for _ in range(stage_depths[-1])
        ]
        self.norm_out = nn.RMSNorm(stage5_channels, eps=1e-6)
        self.conv_out = nn.Linear(stage5_channels, pixel_channels, bias=True)

    @staticmethod
    def _keyframe_times(
        pixel_frame_indices: mx.array, remaining_stride: int
    ) -> mx.array:
        frames = pixel_frame_indices.astype(mx.float32)
        offset = (remaining_stride - 1) / 2
        times = (frames + offset) / remaining_stride
        return mx.where(frames == 0, mx.zeros_like(times), times)

    @staticmethod
    def _upsample_keyframes(
        upsampler: LinearPixelShuffleUpsample, keyframes: mx.array
    ) -> mx.array:
        batch, planes, height, width, channels = keyframes.shape
        flat = mx.reshape(keyframes, (batch * planes, 1, height, width, channels))
        output = upsampler(flat, drop_leading_frame=True)
        return mx.reshape(
            output[:, 0],
            (batch, planes, output.shape[2], output.shape[3], output.shape[4]),
        )

    def _stages_1_to_3(self, latent: mx.array) -> mx.array:
        latent = mx.concatenate(
            [latent, mx.repeat(latent[:, :, -1:], 2, axis=2)], axis=2
        )
        mean = self.mean_of_means[None, :, None, None, None]
        std = self.std_of_means[None, :, None, None, None]
        hidden = latent * std + mean
        hidden = self.conv_in(mx.transpose(hidden, (0, 2, 3, 4, 1)))
        for stage_index, blocks in enumerate(self.det_stages[:3]):
            for block in blocks:
                hidden = block(hidden)
            hidden = self.upsamples[stage_index](hidden)
        return hidden

    def _stages_1_to_3_joint(
        self,
        latent: mx.array,
        keyframe_latents: mx.array,
        pixel_frame_indices: mx.array,
    ) -> tuple[mx.array, mx.array]:
        latent = mx.concatenate(
            [latent, mx.repeat(latent[:, :, -1:], 2, axis=2)], axis=2
        )
        mean = self.mean_of_means[None, :, None, None, None]
        std = self.std_of_means[None, :, None, None, None]
        hidden = self.conv_in(mx.transpose(latent * std + mean, (0, 2, 3, 4, 1)))
        keyframes = mx.transpose(keyframe_latents * std + mean, (0, 2, 3, 4, 1))
        keyframes = self.conv_in(keyframes + self.type_emb[None, None, None, None])
        remaining = (8, 8, 4, 2, 1)
        for stage_index, blocks in enumerate(self.det_stages[:3]):
            times = self._keyframe_times(pixel_frame_indices, remaining[stage_index])
            for block in blocks:
                hidden, keyframes = block.forward_joint(hidden, keyframes, times)
            hidden = self.upsamples[stage_index](hidden)
            keyframes = self._upsample_keyframes(self.upsamples[stage_index], keyframes)
        return hidden, keyframes

    def _stage_4(
        self,
        hidden: mx.array,
        original_frames: int,
        *,
        drop_leading_frame: bool = True,
    ) -> mx.array:
        for block in self.det_stages[3]:
            hidden = block(hidden)
        hidden = self.upsamples[3](hidden, drop_leading_frame=drop_leading_frame)
        return hidden[:, : max(original_frames, self.stage_kernels[-1][0])]

    def _stage_4_joint(
        self,
        hidden: mx.array,
        keyframes: mx.array,
        pixel_frame_indices: mx.array,
        original_frames: int,
        *,
        stage4_time_origin: int = 0,
        drop_leading_frame: bool = True,
    ) -> tuple[mx.array, mx.array]:
        times = self._keyframe_times(pixel_frame_indices, 2) - stage4_time_origin
        for block in self.det_stages[3]:
            hidden, keyframes = block.forward_joint(hidden, keyframes, times)
        hidden = self.upsamples[3](hidden, drop_leading_frame=drop_leading_frame)
        keyframes = self._upsample_keyframes(self.upsamples[3], keyframes)
        return (
            hidden[:, : max(original_frames, self.stage_kernels[-1][0])],
            keyframes,
        )

    def _diffuse(self, context: mx.array, pixels: mx.array) -> mx.array:
        batch = context.shape[0]
        hidden = self.conv_in_x_t(patchify_pixels(pixels, self.patch_size))
        timestep = mx.ones((batch,), dtype=mx.float32)
        embedding = self.t_embedder(timestep * self.timestep_scale_multiplier)
        modulation = self.shared_adaln(embedding)
        for block in self.diff_blocks:
            hidden = block(context, hidden, modulation)
        output = self.conv_out(self.norm_out(hidden))
        return unpatchify_pixels(output, self.patch_size, self.out_channels)

    def _diffuse_joint(
        self,
        context: mx.array,
        pixels: mx.array,
        keyframe_context: mx.array,
        keyframe_pixels: mx.array,
        keyframe_times: mx.array,
    ) -> mx.array:
        batch = context.shape[0]
        hidden = self.conv_in_x_t(patchify_pixels(pixels, self.patch_size))
        keyframes = self.conv_in_x_t(patchify_pixels(keyframe_pixels, self.patch_size))
        timestep = mx.ones((batch,), dtype=mx.float32)
        embedding = self.t_embedder(timestep * self.timestep_scale_multiplier)
        modulation = self.shared_adaln(embedding)
        for block in self.diff_blocks:
            hidden, keyframes = block.forward_joint(
                context,
                hidden,
                keyframe_context,
                keyframes,
                modulation,
                keyframe_times,
            )
        output = self.conv_out(self.norm_out(hidden))
        return unpatchify_pixels(output, self.patch_size, self.out_channels)

    @staticmethod
    def _tile_bounds(size: int, count: int) -> list[tuple[int, int]]:
        if count < 1 or count > size:
            raise ValueError(f"tile count must be in [1, {size}], got {count}")
        return [
            (index * size // count, (index + 1) * size // count)
            for index in range(count)
        ]

    def _spatial_halo(self) -> tuple[int, int]:
        stride_h, stride_w = self.upsamples[3].stride[1:]
        stage4 = self.stage_depths[3]
        stage5 = self.stage_depths[4]
        halo_h = stage4 * (self.stage_kernels[3][1] // 2) + math.ceil(
            stage5 * (self.stage_kernels[4][1] // 2) / stride_h
        )
        halo_w = stage4 * (self.stage_kernels[3][2] // 2) + math.ceil(
            stage5 * (self.stage_kernels[4][2] // 2) / stride_w
        )
        return halo_h, halo_w

    def _temporal_halo(self) -> int:
        stride = self.upsamples[3].stride[0]
        return self.stage_depths[3] * (self.stage_kernels[3][0] // 2) + math.ceil(
            self.stage_depths[4] * (self.stage_kernels[4][0] // 2) / stride
        )

    def _stage4_content_frames(self, latent_frames: int) -> int:
        frames = latent_frames
        for upsampler in self.upsamples[:3]:
            stride = upsampler.stride[0]
            frames *= stride
            if stride == 2:
                frames -= 1
        return frames

    def __call__(
        self,
        latent: mx.array,
        *,
        seed: int = 0,
        spatial_tiles: int = 1,
        temporal_tiles: int = 1,
        keyframe_latents: mx.array | None = None,
        keyframe_positions: list[int] | None = None,
    ) -> mx.array:
        output_frames = 8 * (latent.shape[2] - 1) + 1
        use_keyframes = keyframe_latents is not None
        if use_keyframes:
            if (
                not keyframe_positions
                or len(keyframe_positions) != keyframe_latents.shape[2]
            ):
                raise ValueError(
                    "keyframe positions must match the keyframe latent planes"
                )
            positions = mx.array(keyframe_positions, dtype=mx.int32)
            stage4_input, keyframe_input = self._stages_1_to_3_joint(
                latent, keyframe_latents, positions
            )
        else:
            positions = None
            keyframe_input = None
            stage4_input = self._stages_1_to_3(latent)
        canvas_frames = max(output_frames, self.stage_kernels[-1][0])
        pixel_scale_h = self.upsamples[3].stride[1] * self.patch_size
        pixel_scale_w = self.upsamples[3].stride[2] * self.patch_size
        mx.random.seed(seed)
        full_noise = mx.random.normal(
            (
                latent.shape[0],
                self.out_channels,
                canvas_frames,
                stage4_input.shape[2] * pixel_scale_h,
                stage4_input.shape[3] * pixel_scale_w,
            ),
            dtype=stage4_input.dtype,
        )
        full_keyframe_noise = None
        if use_keyframes:
            full_keyframe_noise = mx.random.normal(
                (
                    latent.shape[0],
                    self.out_channels,
                    keyframe_latents.shape[2],
                    stage4_input.shape[2] * pixel_scale_h,
                    stage4_input.shape[3] * pixel_scale_w,
                ),
                dtype=stage4_input.dtype,
            )
            mx.eval(stage4_input, keyframe_input, full_noise, full_keyframe_noise)
        else:
            mx.eval(stage4_input, full_noise)

        if temporal_tiles > 1:
            if spatial_tiles != 1:
                raise ValueError(
                    "spatial and temporal DiffVAE tiling cannot yet be combined"
                )
            content_frames = self._stage4_content_frames(latent.shape[2])
            halo = self._temporal_halo()
            stride = self.upsamples[3].stride[0]
            causal_offset = 1 if stride == 2 else 0
            parts = []
            for core_start, core_end in self._tile_bounds(
                content_frames, temporal_tiles
            ):
                input_start = max(0, core_start - halo)
                input_end = min(content_frames, core_end + halo)
                feature_end = (
                    stage4_input.shape[1] if input_end == content_frames else input_end
                )
                feature = stage4_input[:, input_start:feature_end]
                pixel_start = (
                    0 if input_start == 0 else input_start * stride - causal_offset
                )
                pixel_end = input_end * stride - causal_offset
                tile_frames = pixel_end - pixel_start
                noise = full_noise[:, :, pixel_start:pixel_end]
                is_origin = input_start == 0
                if use_keyframes:
                    context, keyframe_context = self._stage_4_joint(
                        feature,
                        keyframe_input,
                        positions,
                        tile_frames,
                        stage4_time_origin=input_start,
                        drop_leading_frame=is_origin,
                    )
                    decoded = self._diffuse_joint(
                        context,
                        noise,
                        keyframe_context,
                        full_keyframe_noise,
                        self._keyframe_times(positions, 1) - pixel_start,
                    )
                else:
                    context = self._stage_4(
                        feature,
                        tile_frames,
                        drop_leading_frame=is_origin,
                    )
                    decoded = self._diffuse(context, noise)
                mx.eval(decoded)
                mx.clear_cache()
                core_pixel_start = (
                    0 if core_start == 0 else core_start * stride - causal_offset
                )
                core_pixel_end = core_end * stride - causal_offset
                local_start = core_pixel_start - pixel_start
                local_end = core_pixel_end - pixel_start
                parts.append(decoded[:, :, local_start:local_end])
            return mx.concatenate(parts, axis=2)[:, :, :output_frames]

        if spatial_tiles == 1:
            if use_keyframes:
                context, keyframe_context = self._stage_4_joint(
                    stage4_input,
                    keyframe_input,
                    positions,
                    output_frames,
                )
                return self._diffuse_joint(
                    context,
                    full_noise,
                    keyframe_context,
                    full_keyframe_noise,
                    self._keyframe_times(positions, 1),
                )[:, :, :output_frames]
            context = self._stage_4(stage4_input, output_frames)
            return self._diffuse(context, full_noise)[:, :, :output_frames]

        halo_h, halo_w = self._spatial_halo()
        rows = []
        for core_h0, core_h1 in self._tile_bounds(stage4_input.shape[2], spatial_tiles):
            columns = []
            input_h0 = max(0, core_h0 - halo_h)
            input_h1 = min(stage4_input.shape[2], core_h1 + halo_h)
            for core_w0, core_w1 in self._tile_bounds(
                stage4_input.shape[3], spatial_tiles
            ):
                input_w0 = max(0, core_w0 - halo_w)
                input_w1 = min(stage4_input.shape[3], core_w1 + halo_w)
                feature = stage4_input[:, :, input_h0:input_h1, input_w0:input_w1]
                noise = full_noise[
                    :,
                    :,
                    :,
                    input_h0 * pixel_scale_h : input_h1 * pixel_scale_h,
                    input_w0 * pixel_scale_w : input_w1 * pixel_scale_w,
                ]
                if use_keyframes:
                    keyframe_feature = keyframe_input[
                        :, :, input_h0:input_h1, input_w0:input_w1
                    ]
                    context, keyframe_context = self._stage_4_joint(
                        feature,
                        keyframe_feature,
                        positions,
                        output_frames,
                    )
                    keyframe_noise = full_keyframe_noise[
                        :,
                        :,
                        :,
                        input_h0 * pixel_scale_h : input_h1 * pixel_scale_h,
                        input_w0 * pixel_scale_w : input_w1 * pixel_scale_w,
                    ]
                    decoded = self._diffuse_joint(
                        context,
                        noise,
                        keyframe_context,
                        keyframe_noise,
                        self._keyframe_times(positions, 1),
                    )[:, :, :output_frames]
                else:
                    context = self._stage_4(feature, output_frames)
                    decoded = self._diffuse(context, noise)[:, :, :output_frames]
                mx.eval(decoded)
                mx.clear_cache()
                local_h0 = (core_h0 - input_h0) * pixel_scale_h
                local_h1 = local_h0 + (core_h1 - core_h0) * pixel_scale_h
                local_w0 = (core_w0 - input_w0) * pixel_scale_w
                local_w1 = local_w0 + (core_w1 - core_w0) * pixel_scale_w
                columns.append(decoded[:, :, :, local_h0:local_h1, local_w0:local_w1])
            rows.append(mx.concatenate(columns, axis=4))
        return mx.concatenate(rows, axis=3)

    @staticmethod
    def sanitize(weights: dict[str, mx.array]) -> dict[str, mx.array]:
        sanitized: dict[str, mx.array] = {}
        for key, value in weights.items():
            if key == "per_channel_statistics.mean-of-means":
                sanitized["mean_of_means"] = value
                continue
            if key == "per_channel_statistics.std-of-means":
                sanitized["std_of_means"] = value
                continue
            if not key.startswith("decoder."):
                continue
            key = key.removeprefix("decoder.")
            if key.endswith(("attn.qkv.weight", "attn.qkv.bias")):
                prefix, suffix = key.rsplit("qkv", 1)
                query, key_value, value_value = mx.split(value, 3, axis=0)
                sanitized[f"{prefix}to_q{suffix}"] = query
                sanitized[f"{prefix}to_k{suffix}"] = key_value
                sanitized[f"{prefix}to_v{suffix}"] = value_value
            else:
                sanitized[key] = value
        return sanitized

    @classmethod
    def from_pretrained(cls, path: str | Path) -> DiffusionVideoDecoder:
        path = Path(path)
        with safe_open(path, framework="numpy") as file:
            metadata = file.metadata() or {}
        config = json.loads(metadata["config"])["vae"]["decoder"]
        model = cls(
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            patch_size=config["patch_size"],
            head_dim=config["head_dim"],
            stage_channels=tuple(config["stage_channels"]),
            stage_depths=tuple(config["stage_depths"]),
            stage_kernels=tuple(tuple(item) for item in config["stage_kernels"]),
            upsamples=tuple(
                (tuple(stride), reduction) for stride, reduction in config["upsamples"]
            ),
            timestep_scale_multiplier=config["timestep_scale_multiplier"],
        )
        weights = cls.sanitize(mx.load(str(path)))
        model.load_weights(list(weights.items()), strict=True)
        mx.eval(model.parameters())
        model.eval()
        return model
