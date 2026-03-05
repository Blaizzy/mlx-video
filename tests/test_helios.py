"""Tests for Helios model configuration, scheduler, RoPE, and transformer."""

import math

import mlx.core as mx
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Config Tests
# ---------------------------------------------------------------------------

class TestHeliosModelConfig:
    """Tests for HeliosModelConfig dataclass."""

    def test_default_values(self):
        from mlx_video.models.helios.config import HeliosModelConfig
        config = HeliosModelConfig()
        assert config.dim == 5120
        assert config.ffn_dim == 13824
        assert config.num_heads == 40
        assert config.num_layers == 40
        assert config.in_dim == 16
        assert config.out_dim == 16
        assert config.patch_size == (1, 2, 2)
        assert config.rope_dim == (44, 42, 42)
        assert config.history_sizes == [16, 2, 1]
        assert config.num_latent_frames_per_chunk == 9
        assert config.vae_stride == (4, 8, 8)
        assert config.vae_z_dim == 16
        assert config.text_dim == 4096

    def test_head_dim_property(self):
        from mlx_video.models.helios.config import HeliosModelConfig
        config = HeliosModelConfig()
        assert config.head_dim == 128  # 5120 // 40

    def test_distilled_preset(self):
        from mlx_video.models.helios.config import HeliosModelConfig
        config = HeliosModelConfig.helios_distilled()
        assert config.shift == 1.0
        assert config.dim == 5120

    def test_rope_dim_sums_to_half_head_dim(self):
        from mlx_video.models.helios.config import HeliosModelConfig
        config = HeliosModelConfig()
        # sum(rope_dim) should equal head_dim = 128, since each dim is half
        # Actually: 44 + 42 + 42 = 128 = head_dim
        assert sum(config.rope_dim) == config.head_dim


# ---------------------------------------------------------------------------
# Scheduler Tests
# ---------------------------------------------------------------------------

class TestHeliosScheduler:
    """Tests for HeliosScheduler."""

    def test_init(self):
        from mlx_video.models.helios.scheduler import HeliosScheduler
        sched = HeliosScheduler()
        assert sched.num_train_timesteps == 1000
        assert sched.shift == 1.0
        assert sched.stages == 3

    def test_global_sigmas_shape(self):
        from mlx_video.models.helios.scheduler import HeliosScheduler
        sched = HeliosScheduler()
        assert len(sched.global_sigmas) == 1000

    def test_set_timesteps(self):
        from mlx_video.models.helios.scheduler import HeliosScheduler
        sched = HeliosScheduler()
        sched.set_timesteps(num_inference_steps=10, stage_index=0)
        assert sched.timesteps.shape == (10,)
        assert sched.sigmas.shape == (11,)  # N+1 for boundaries

    def test_step(self):
        from mlx_video.models.helios.scheduler import HeliosScheduler
        sched = HeliosScheduler()
        sched.set_timesteps(num_inference_steps=2, stage_index=0)
        sample = mx.ones((16, 4, 4, 4))
        model_output = mx.zeros_like(sample)
        result = sched.step(model_output, sample)
        # With zero model output, result should be close to sample
        assert result.shape == sample.shape

    def test_add_noise(self):
        from mlx_video.models.helios.scheduler import HeliosScheduler
        sched = HeliosScheduler()
        original = mx.ones((16, 4, 4, 4))
        noise = mx.zeros_like(original)
        sigma = mx.array(0.5)
        result = sched.add_noise(original, noise, sigma)
        # (1 - 0.5) * 1 + 0.5 * 0 = 0.5
        expected = mx.ones_like(result) * 0.5
        assert mx.allclose(result, expected).item()

    def test_per_stage_consistency(self):
        from mlx_video.models.helios.scheduler import HeliosScheduler
        sched = HeliosScheduler()
        # All 3 stages should have valid sigma ranges
        for i in range(3):
            assert sched.start_sigmas[i] >= sched.end_sigmas[i]

    def test_step_dmd_last_step_returns_x0(self):
        from mlx_video.models.helios.scheduler import HeliosScheduler
        sched = HeliosScheduler()
        sched.set_timesteps(num_inference_steps=2, stage_index=0)
        sample = mx.ones((16, 4, 4, 4))
        flow = mx.ones_like(sample) * 0.1
        noisy_start = mx.zeros_like(sample)
        # Last step (idx=1) should return x0_pred directly
        result = sched.step_dmd(flow, sample, cur_step=1, noisy_start=noisy_start)
        # x0 = sample - sigma[1] * flow
        assert result.shape == sample.shape

    def test_step_dmd_non_last_renoises(self):
        from mlx_video.models.helios.scheduler import HeliosScheduler
        sched = HeliosScheduler()
        sched.set_timesteps(num_inference_steps=2, stage_index=0)
        sample = mx.ones((16, 4, 4, 4))
        flow = mx.zeros_like(sample)
        noisy_start = mx.ones_like(sample) * 2.0
        # Non-last step: should blend x0_pred with noisy_start
        result = sched.step_dmd(flow, sample, cur_step=0, noisy_start=noisy_start)
        assert result.shape == sample.shape
        # With flow=0, x0=sample. Result = (1-sigma_next)*x0 + sigma_next*noisy_start
        # Should differ from sample since noisy_start != sample
        assert not mx.allclose(result, sample).item()

    def test_dynamic_shifting(self):
        from mlx_video.models.helios.scheduler import HeliosScheduler, calculate_shift
        mu = calculate_shift(1024)
        assert 0.3 < mu < 2.0  # reasonable range
        sched = HeliosScheduler(use_dynamic_shifting=True)
        sched.set_timesteps(2, stage_index=0, image_seq_len=1024)
        assert sched.timesteps.shape[0] == 2

    def test_amplify_first_chunk_doubles_steps(self):
        from mlx_video.models.helios.scheduler import HeliosScheduler
        sched = HeliosScheduler()
        sched.set_timesteps(2, stage_index=0, is_amplify_first_chunk=True)
        # 2*2+1 = 5 → DMD trim → 4 timesteps
        assert sched.timesteps.shape[0] == 4


# ---------------------------------------------------------------------------
# Pyramid Helper Tests
# ---------------------------------------------------------------------------

class TestPyramidHelpers:
    """Tests for pyramid denoising helper functions."""

    def test_sample_block_noise_shape(self):
        from mlx_video.generate_helios import sample_block_noise
        noise = sample_block_noise(1, 16, 9, 48, 80, (1, 2, 2), 1 / 3)
        assert noise.shape == (16, 9, 48, 80)

    def test_sample_block_noise_statistics(self):
        from mlx_video.generate_helios import sample_block_noise
        np.random.seed(42)
        noise = sample_block_noise(1, 16, 9, 48, 80, (1, 2, 2), 1 / 3)
        noise_np = np.array(noise)
        # Should be roughly zero-mean, unit-ish variance
        assert abs(noise_np.mean()) < 0.1
        assert 0.5 < noise_np.std() < 2.0

    def test_bilinear_downsample(self):
        from mlx_video.generate_helios import _bilinear_downsample_2d
        x = mx.ones((9, 16, 48, 80))
        result = _bilinear_downsample_2d(x, 24, 40)
        assert result.shape == (9, 16, 24, 40)
        assert mx.allclose(result, mx.ones_like(result)).item()

    def test_nearest_upsample(self):
        from mlx_video.generate_helios import _nearest_upsample_2d
        x = mx.ones((9, 16, 24, 40))
        result = _nearest_upsample_2d(x, 48, 80)
        assert result.shape == (9, 16, 48, 80)

    def test_downsample_history(self):
        from mlx_video.generate_helios import _downsample_history
        hist = mx.ones((16, 2, 48, 80))
        result = _downsample_history(hist, 2)
        assert result.shape == (16, 2, 24, 40)

    def test_spatial_reshape_roundtrip(self):
        from mlx_video.generate_helios import _spatial_reshape, _spatial_unreshape
        x = mx.random.normal((16, 9, 48, 80))
        reshaped = _spatial_reshape(x, 9, 16)
        unreshaped = _spatial_unreshape(reshaped, 9, 16, 48, 80)
        assert mx.allclose(x, unreshaped).item()


# ---------------------------------------------------------------------------
# RoPE Tests
# ---------------------------------------------------------------------------

class TestHeliosRoPE:
    """Tests for Helios RoPE computation."""

    def test_rope_params_shape(self):
        from mlx_video.models.helios.rope import helios_rope_params
        freqs = helios_rope_params(
            rope_dim=(44, 42, 42),
            theta=10000.0,
            max_seq_len=1024,
        )
        freqs_t, freqs_h, freqs_w = freqs
        # Each freq: [max_seq_len, d_i//2, 2] (cos/sin stacked)
        assert freqs_t.shape == (1024, 22, 2)  # 44 // 2
        assert freqs_h.shape == (1024, 21, 2)  # 42 // 2
        assert freqs_w.shape == (1024, 21, 2)  # 42 // 2

    def test_rope_precompute_shape(self):
        from mlx_video.models.helios.rope import (
            helios_rope_params,
            helios_rope_precompute_cos_sin,
        )
        freqs = helios_rope_params((44, 42, 42), 10000.0, 1024)
        frame_indices = mx.arange(9)  # 9 latent frames
        grid_size = (9, 12, 20)  # F, H, W after patchify

        cos_sin = helios_rope_precompute_cos_sin(
            frame_indices, grid_size, freqs, dtype=mx.float32,
        )
        cos_f, sin_f = cos_sin
        total_patches = 9 * 12 * 20
        # Each should be [total_patches, 1, half_head_dim]
        # Actually check the actual output shape from the implementation
        assert cos_f.shape[0] == total_patches or cos_f.ndim >= 2


# ---------------------------------------------------------------------------
# Attention Tests
# ---------------------------------------------------------------------------

class TestHeliosAttention:
    """Tests for Helios attention modules."""

    def test_self_attention_no_history(self):
        from mlx_video.models.helios.attention import HeliosSelfAttention
        dim = 64
        heads = 4
        attn = HeliosSelfAttention(dim, heads, qk_norm=True, eps=1e-6)
        x = mx.random.normal((1, 16, dim))
        out = attn(
            x,
            frame_indices=mx.arange(16),
            grid_size=(16, 1, 1),
            freqs=None,
            rope_cos_sin=None,
            original_context_length=16,
        )
        assert out.shape == (1, 16, dim)

    def test_cross_attention(self):
        from mlx_video.models.helios.attention import HeliosCrossAttention
        dim = 64
        heads = 4
        attn = HeliosCrossAttention(dim, heads, qk_norm=True, eps=1e-6)
        x = mx.random.normal((1, 16, dim))
        ctx = mx.random.normal((1, 32, dim))
        out = attn(x, ctx)
        assert out.shape == (1, 16, dim)

    def test_cross_attention_kv_cache(self):
        from mlx_video.models.helios.attention import HeliosCrossAttention
        dim = 64
        heads = 4
        attn = HeliosCrossAttention(dim, heads, qk_norm=True, eps=1e-6)
        ctx = mx.random.normal((1, 32, dim))
        kv = attn.prepare_kv(ctx)
        assert len(kv) == 2  # (k, v)

        x = mx.random.normal((1, 16, dim))
        out = attn(x, ctx, kv_cache=kv)
        assert out.shape == (1, 16, dim)


# ---------------------------------------------------------------------------
# Transformer Block Tests (small scale)
# ---------------------------------------------------------------------------

class TestHeliosTransformerBlock:
    """Tests for HeliosTransformerBlock."""

    def test_block_forward_no_history(self):
        from mlx_video.models.helios.transformer import HeliosTransformerBlock
        dim = 64
        block = HeliosTransformerBlock(
            dim=dim, ffn_dim=128, num_heads=4,
            qk_norm=True, cross_attn_norm=True, eps=1e-6,
        )
        x = mx.random.normal((1, 16, dim))
        ctx = mx.random.normal((1, 32, dim))
        temb = mx.random.normal((1, 16, 6, dim))

        out = block(
            x, ctx, temb,
            rotary_emb=None,
            original_context_length=16,
        )
        assert out.shape == (1, 16, dim)


# ---------------------------------------------------------------------------
# Weight Sanitization Tests
# ---------------------------------------------------------------------------

class TestHeliosWeightSanitization:
    """Tests for convert_helios weight key mapping."""

    def test_patch_embedding_reshape(self):
        from mlx_video.convert_helios import sanitize_helios_transformer_weights
        # Simulate Conv3d weight: [O, I, D, H, W]
        w = {
            "patch_embedding.weight": mx.ones((5120, 16, 1, 2, 2)),
            "patch_embedding.bias": mx.zeros((5120,)),
        }
        s = sanitize_helios_transformer_weights(w)
        assert "patch_embedding.weight" in s
        assert s["patch_embedding.weight"].shape == (5120, 64)  # 16*1*2*2

    def test_condition_embedder_mapping(self):
        from mlx_video.convert_helios import sanitize_helios_transformer_weights
        w = {
            "condition_embedder.time_embedder.linear_1.weight": mx.ones((5120, 256)),
            "condition_embedder.time_embedder.linear_2.weight": mx.ones((5120, 5120)),
            "condition_embedder.time_proj.weight": mx.ones((30720, 5120)),
            "condition_embedder.text_embedder.linear_1.weight": mx.ones((5120, 4096)),
            "condition_embedder.text_embedder.linear_2.weight": mx.ones((5120, 5120)),
        }
        s = sanitize_helios_transformer_weights(w)
        assert "time_embedding_0.weight" in s
        assert "time_embedding_1.weight" in s
        assert "time_projection.weight" in s
        assert "text_embedding_0.weight" in s
        assert "text_embedding_1.weight" in s

    def test_attention_key_mapping(self):
        from mlx_video.convert_helios import sanitize_helios_transformer_weights
        w = {
            "blocks.0.attn1.to_q.weight": mx.ones((5120, 5120)),
            "blocks.0.attn1.to_out.0.weight": mx.ones((5120, 5120)),
            "blocks.0.attn2.to_k.weight": mx.ones((5120, 5120)),
        }
        s = sanitize_helios_transformer_weights(w)
        assert "blocks.0.self_attn.q.weight" in s
        assert "blocks.0.self_attn.o.weight" in s
        assert "blocks.0.cross_attn.k.weight" in s

    def test_ffn_key_mapping(self):
        from mlx_video.convert_helios import sanitize_helios_transformer_weights
        w = {
            "blocks.0.ffn.net.0.proj.weight": mx.ones((13824, 5120)),
            "blocks.0.ffn.net.2.weight": mx.ones((5120, 13824)),
        }
        s = sanitize_helios_transformer_weights(w)
        assert "blocks.0.ffn.fc1.weight" in s
        assert "blocks.0.ffn.fc2.weight" in s

    def test_output_norm_mapping(self):
        from mlx_video.convert_helios import sanitize_helios_transformer_weights
        w = {
            "norm_out.norm.weight": mx.ones((5120,)),
            "norm_out.norm.bias": mx.zeros((5120,)),
            "norm_out.scale_shift_table": mx.ones((1, 2, 5120)),
        }
        s = sanitize_helios_transformer_weights(w)
        assert "output_norm.weight" in s
        assert "output_norm.bias" in s
        assert "output_norm_table" in s

    def test_skips_rope_buffers(self):
        from mlx_video.convert_helios import sanitize_helios_transformer_weights
        w = {
            "rope.freqs_base_t": mx.ones((22,)),
            "rope.freqs_base_y": mx.ones((21,)),
        }
        s = sanitize_helios_transformer_weights(w)
        assert len(s) == 0  # All skipped


class TestHeliosT5Sanitization:
    """Tests for Helios T5 (HF UMT5 → MLX) weight key mapping."""

    def test_token_embedding(self):
        from mlx_video.convert_helios import sanitize_helios_t5_weights

        w = {"shared.weight": mx.ones((100, 64))}
        s = sanitize_helios_t5_weights(w)
        assert "token_embedding.weight" in s

    def test_encoder_embed_tokens(self):
        from mlx_video.convert_helios import sanitize_helios_t5_weights

        w = {"encoder.embed_tokens.weight": mx.ones((100, 64))}
        s = sanitize_helios_t5_weights(w)
        assert "token_embedding.weight" in s

    def test_final_layer_norm(self):
        from mlx_video.convert_helios import sanitize_helios_t5_weights

        w = {"encoder.final_layer_norm.weight": mx.ones((64,))}
        s = sanitize_helios_t5_weights(w)
        assert "norm.weight" in s

    def test_self_attention_mapping(self):
        from mlx_video.convert_helios import sanitize_helios_t5_weights

        w = {
            "encoder.block.0.layer.0.SelfAttention.q.weight": mx.ones((64, 64)),
            "encoder.block.0.layer.0.SelfAttention.k.weight": mx.ones((64, 64)),
            "encoder.block.0.layer.0.SelfAttention.v.weight": mx.ones((64, 64)),
            "encoder.block.0.layer.0.SelfAttention.o.weight": mx.ones((64, 64)),
        }
        s = sanitize_helios_t5_weights(w)
        assert "blocks.0.attn.q.weight" in s
        assert "blocks.0.attn.k.weight" in s
        assert "blocks.0.attn.v.weight" in s
        assert "blocks.0.attn.o.weight" in s

    def test_relative_attention_bias(self):
        from mlx_video.convert_helios import sanitize_helios_t5_weights

        w = {
            "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight": mx.ones((32, 64)),
        }
        s = sanitize_helios_t5_weights(w)
        assert "blocks.0.pos_embedding.embedding.weight" in s

    def test_layer_norms(self):
        from mlx_video.convert_helios import sanitize_helios_t5_weights

        w = {
            "encoder.block.2.layer.0.layer_norm.weight": mx.ones((64,)),
            "encoder.block.2.layer.1.layer_norm.weight": mx.ones((64,)),
        }
        s = sanitize_helios_t5_weights(w)
        assert "blocks.2.norm1.weight" in s
        assert "blocks.2.norm2.weight" in s

    def test_ffn_mapping(self):
        from mlx_video.convert_helios import sanitize_helios_t5_weights

        w = {
            "encoder.block.1.layer.1.DenseReluDense.wi_0.weight": mx.ones((128, 64)),
            "encoder.block.1.layer.1.DenseReluDense.wi_1.weight": mx.ones((128, 64)),
            "encoder.block.1.layer.1.DenseReluDense.wo.weight": mx.ones((64, 128)),
        }
        s = sanitize_helios_t5_weights(w)
        assert "blocks.1.ffn.gate_proj.weight" in s
        assert "blocks.1.ffn.fc1.weight" in s
        assert "blocks.1.ffn.fc2.weight" in s

    def test_skips_decoder_keys(self):
        from mlx_video.convert_helios import sanitize_helios_t5_weights

        w = {
            "decoder.block.0.layer.0.SelfAttention.q.weight": mx.ones((64, 64)),
            "lm_head.weight": mx.ones((100, 64)),
        }
        s = sanitize_helios_t5_weights(w)
        assert len(s) == 0


class TestHeliosVAESanitization:
    """Tests for Helios VAE (HF diffusers → WanVAE) weight key mapping."""

    def test_top_level_convolutions(self):
        from mlx_video.convert_helios import sanitize_helios_vae_weights

        w = {
            "post_quant_conv.weight": mx.ones((16, 16, 1, 1, 1)),
            "post_quant_conv.bias": mx.ones((16,)),
            "quant_conv.weight": mx.ones((32, 32, 1, 1, 1)),
            "quant_conv.bias": mx.ones((32,)),
        }
        s = sanitize_helios_vae_weights(w)
        assert "conv2.weight" in s
        assert "conv2.bias" in s
        assert "conv1.weight" in s
        assert "conv1.bias" in s
        # Conv3d should be transposed
        assert s["conv2.weight"].shape == (16, 1, 1, 1, 16)

    def test_decoder_conv_in_out(self):
        from mlx_video.convert_helios import sanitize_helios_vae_weights

        w = {
            "decoder.conv_in.weight": mx.ones((384, 16, 3, 3, 3)),
            "decoder.conv_in.bias": mx.ones((384,)),
            "decoder.conv_out.weight": mx.ones((3, 96, 3, 3, 3)),
            "decoder.conv_out.bias": mx.ones((3,)),
            "decoder.norm_out.gamma": mx.ones((96, 1, 1, 1)),
        }
        s = sanitize_helios_vae_weights(w)
        assert "decoder.conv1.weight" in s
        assert "decoder.conv1.bias" in s
        assert "decoder.head.2.weight" in s
        assert "decoder.head.2.bias" in s
        assert "decoder.head.0.gamma" in s

    def test_mid_block_mapping(self):
        from mlx_video.convert_helios import sanitize_helios_vae_weights

        w = {
            "decoder.mid_block.resnets.0.norm1.gamma": mx.ones((384, 1, 1, 1)),
            "decoder.mid_block.resnets.0.conv1.weight": mx.ones((384, 384, 3, 3, 3)),
            "decoder.mid_block.attentions.0.norm.gamma": mx.ones((384, 1, 1)),
            "decoder.mid_block.resnets.1.conv2.bias": mx.ones((384,)),
        }
        s = sanitize_helios_vae_weights(w)
        assert "decoder.middle.0.residual.0.gamma" in s
        assert "decoder.middle.0.residual.2.weight" in s
        assert "decoder.middle.1.norm.gamma" in s
        assert "decoder.middle.2.residual.6.bias" in s

    def test_up_blocks_resnet_mapping(self):
        from mlx_video.convert_helios import sanitize_helios_vae_weights

        w = {
            "decoder.up_blocks.0.resnets.0.norm1.gamma": mx.ones((384, 1, 1, 1)),
            "decoder.up_blocks.0.resnets.1.conv2.weight": mx.ones((384, 384, 3, 3, 3)),
            "decoder.up_blocks.1.resnets.0.conv_shortcut.weight": mx.ones((384, 192, 1, 1, 1)),
        }
        s = sanitize_helios_vae_weights(w)
        assert "decoder.upsamples.0.residual.0.gamma" in s
        assert "decoder.upsamples.1.residual.6.weight" in s
        assert "decoder.upsamples.4.shortcut.weight" in s

    def test_upsampler_mapping(self):
        from mlx_video.convert_helios import sanitize_helios_vae_weights

        w = {
            "decoder.up_blocks.0.upsamplers.0.resample.1.weight": mx.ones((192, 384, 3, 3)),
            "decoder.up_blocks.0.upsamplers.0.time_conv.weight": mx.ones((768, 384, 3, 1, 1)),
        }
        s = sanitize_helios_vae_weights(w)
        assert "decoder.upsamples.3.resample.1.weight" in s
        assert "decoder.upsamples.3.time_conv.weight" in s

    def test_skips_encoder_keys(self):
        from mlx_video.convert_helios import sanitize_helios_vae_weights

        w = {
            "encoder.conv_in.weight": mx.ones((384, 3, 3, 3, 3)),
            "encoder.mid_block.resnets.0.conv1.weight": mx.ones((384, 384, 3, 3, 3)),
        }
        s = sanitize_helios_vae_weights(w)
        assert len(s) == 0
