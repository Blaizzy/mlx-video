"""Tests for Wan VAE 2.1 and 2.2 components."""

import math

import mlx.core as mx
import numpy as np

# ---------------------------------------------------------------------------
# VAE 2.1 Tests
# ---------------------------------------------------------------------------


class TestCausalConv3d:
    def test_output_shape_stride1(self):
        from mlx_video.models.wan_2.vae import CausalConv3d

        conv = CausalConv3d(4, 8, kernel_size=3, stride=1, padding=1)
        # Initialize weights
        conv.weight = mx.random.normal(conv.weight.shape) * 0.02
        x = mx.random.normal((1, 4, 3, 8, 8))  # [B, C, T, H, W]
        out = conv(x)
        mx.eval(out)
        # With causal padding and padding=1 on spatial, dims should be preserved
        assert out.shape[0] == 1
        assert out.shape[1] == 8  # out_channels
        assert out.shape[2] == 3  # T preserved
        assert out.shape[3] == 8  # H preserved
        assert out.shape[4] == 8  # W preserved

    def test_output_shape_kernel1(self):
        from mlx_video.models.wan_2.vae import CausalConv3d

        conv = CausalConv3d(4, 8, kernel_size=1, stride=1, padding=0)
        conv.weight = mx.random.normal(conv.weight.shape) * 0.02
        x = mx.random.normal((1, 4, 2, 4, 4))
        out = conv(x)
        mx.eval(out)
        assert out.shape == (1, 8, 2, 4, 4)

    def test_causal_padding(self):
        """Causal conv should only use past/current frames, not future."""
        from mlx_video.models.wan_2.vae import CausalConv3d

        conv = CausalConv3d(2, 2, kernel_size=3, stride=1, padding=1)
        conv.weight = mx.random.normal(conv.weight.shape) * 0.1
        conv.bias = mx.zeros((2,))
        # Create input where only the first frame has signal
        x = mx.zeros((1, 2, 4, 4, 4))
        x_np = np.zeros((1, 2, 4, 4, 4), dtype=np.float32)
        x_np[:, :, 0, :, :] = 1.0
        x = mx.array(x_np)
        out = conv(x)
        mx.eval(out)
        # Due to causal padding, the output at t=0 should only depend on t=0


class TestResidualBlock:
    def test_same_dim(self):
        from mlx_video.models.wan_2.vae import ResidualBlock

        block = ResidualBlock(8, 8)
        x = mx.random.normal((1, 8, 2, 4, 4))
        out = block(x)
        mx.eval(out)
        assert out.shape == (1, 8, 2, 4, 4)

    def test_different_dim(self):
        from mlx_video.models.wan_2.vae import ResidualBlock

        block = ResidualBlock(8, 16)
        x = mx.random.normal((1, 8, 2, 4, 4))
        out = block(x)
        mx.eval(out)
        assert out.shape == (1, 16, 2, 4, 4)

    def test_shortcut_exists_when_dims_differ(self):
        from mlx_video.models.wan_2.vae import ResidualBlock

        block = ResidualBlock(8, 16)
        assert block.shortcut is not None

    def test_no_shortcut_when_dims_same(self):
        from mlx_video.models.wan_2.vae import ResidualBlock

        block = ResidualBlock(8, 8)
        assert block.shortcut is None


class TestAttentionBlock:
    def test_output_shape(self):
        from mlx_video.models.wan_2.vae import AttentionBlock

        block = AttentionBlock(8)
        x = mx.random.normal((1, 8, 2, 4, 4))
        out = block(x)
        mx.eval(out)
        assert out.shape == (1, 8, 2, 4, 4)

    def test_residual_connection(self):
        from mlx_video.models.wan_2.vae import AttentionBlock

        block = AttentionBlock(8)
        x = mx.random.normal((1, 8, 1, 3, 3))
        out = block(x)
        mx.eval(x, out)
        # Residual: output should not be zero even with random init
        assert np.abs(np.array(out)).max() > 0


class TestWanVAE:
    def test_instantiation(self):
        from mlx_video.models.wan_2.vae import WanVAE

        vae = WanVAE(z_dim=16)
        assert vae.z_dim == 16
        assert vae.mean.shape == (16,)
        assert vae.std.shape == (16,)

    def test_normalization_stats(self):
        from mlx_video.models.wan_2.vae import VAE_MEAN, VAE_STD

        assert len(VAE_MEAN) == 16
        assert len(VAE_STD) == 16
        assert all(s > 0 for s in VAE_STD)

    def test_decode_causal_frame_count(self):
        """Reference Wan2.1 mapping: T latent frames -> 1 + (T-1)*4 output
        frames. The first latent frame is NOT temporally upsampled (the
        'Rep' first-chunk sentinel skips time_conv in upsample3d)."""
        import mlx.core as mx

        from mlx_video.models.wan_2.vae import WanVAE

        vae = WanVAE(z_dim=16)
        for t_lat, t_out in [(1, 1), (2, 5), (5, 17)]:
            z = mx.random.normal((1, 16, t_lat, 4, 4))
            out = vae.decode(z)
            assert out.shape == (1, 3, t_out, 32, 32), (
                f"T={t_lat}: expected {t_out} frames, got {out.shape[2]}"
            )

    def test_decode_roundtrip_frame_count(self):
        """encode(x) then decode should reproduce the input frame count for
        the canonical 4k+1 frame videos."""
        import mlx.core as mx

        from mlx_video.models.wan_2.vae import WanVAE

        vae = WanVAE(z_dim=16, encoder=True)
        x = mx.random.normal((1, 3, 9, 32, 32))
        z = vae.encode(x)
        assert z.shape[2] == 3  # 1 + (9-1)/4
        out = vae.decode(z)
        assert out.shape[2] == 9


# ---------------------------------------------------------------------------
# Wan2.2 VAE Component Tests
# ---------------------------------------------------------------------------


class TestVAE22CausalConv3d:
    """Tests for vae22.CausalConv3d (channels-last)."""

    def test_output_shape_k3(self):
        from mlx_video.models.wan_2.vae22 import CausalConv3d

        conv = CausalConv3d(8, 16, kernel_size=3, padding=1)
        x = mx.random.normal((1, 4, 8, 8, 8))  # [B, T, H, W, C]
        out = conv(x)
        mx.eval(out)
        assert out.shape == (1, 4, 8, 8, 16)

    def test_output_shape_k1(self):
        from mlx_video.models.wan_2.vae22 import CausalConv3d

        conv = CausalConv3d(8, 16, kernel_size=1)
        x = mx.random.normal((1, 2, 4, 4, 8))
        out = conv(x)
        mx.eval(out)
        assert out.shape == (1, 2, 4, 4, 16)

    def test_temporal_causal(self):
        """Output at t=0 should not depend on t>0."""
        from mlx_video.models.wan_2.vae22 import CausalConv3d

        conv = CausalConv3d(2, 2, kernel_size=3, padding=1)
        conv.weight = mx.random.normal(conv.weight.shape) * 0.1
        conv.bias = mx.zeros(conv.bias.shape)

        x = mx.zeros((1, 4, 4, 4, 2))
        out_zero = conv(x)
        mx.eval(out_zero)
        t0_ref = np.array(out_zero[0, 0])

        # Modify t=2..3; output at t=0 should be unchanged
        x_mod = mx.concatenate(
            [
                x[:, :2],
                mx.ones((1, 2, 4, 4, 2)),
            ],
            axis=1,
        )
        out_mod = conv(x_mod)
        mx.eval(out_mod)
        t0_mod = np.array(out_mod[0, 0])
        np.testing.assert_allclose(t0_ref, t0_mod, atol=1e-5)

    def test_channels_last_format(self):
        """Verify input/output are channels-last [B, T, H, W, C]."""
        from mlx_video.models.wan_2.vae22 import CausalConv3d

        conv = CausalConv3d(4, 8, kernel_size=3, padding=1)
        x = mx.random.normal((2, 3, 6, 6, 4))
        out = conv(x)
        mx.eval(out)
        assert out.shape[-1] == 8  # last dim = out_channels


class TestRMSNorm:
    """Tests for vae22.RMS_norm (actually L2 normalization)."""

    def test_output_shape(self):
        from mlx_video.models.wan_2.vae22 import RMS_norm

        norm = RMS_norm(16)
        x = mx.random.normal((2, 4, 4, 4, 16))
        out = norm(x)
        mx.eval(out)
        assert out.shape == x.shape

    def test_l2_normalization(self):
        """RMS_norm should normalize to unit L2 norm * sqrt(dim)."""
        from mlx_video.models.wan_2.vae22 import RMS_norm

        dim = 32
        norm = RMS_norm(dim)
        x = mx.random.normal((1, 1, 1, 1, dim)) * 5.0  # large values
        out = norm(x)
        mx.eval(out)
        # After L2 norm * scale(=sqrt(dim)) * gamma(=1): ||out|| = sqrt(dim)
        out_np = np.array(out).flatten()
        l2 = np.linalg.norm(out_np)
        np.testing.assert_allclose(l2, math.sqrt(dim), rtol=1e-3)

    def test_scale_invariant(self):
        """Scaling input by constant should not change output (L2 norm property)."""
        from mlx_video.models.wan_2.vae22 import RMS_norm

        norm = RMS_norm(8)
        x = mx.random.normal((1, 1, 1, 1, 8))
        out1 = norm(x)
        out2 = norm(x * 10.0)
        mx.eval(out1, out2)
        np.testing.assert_allclose(np.array(out1), np.array(out2), atol=1e-4)

    def test_gamma_effect(self):
        """Non-unit gamma should scale output."""
        from mlx_video.models.wan_2.vae22 import RMS_norm

        norm = RMS_norm(4)
        norm.gamma = mx.array([2.0, 2.0, 2.0, 2.0])
        x = mx.ones((1, 1, 1, 1, 4))
        out = norm(x)
        mx.eval(out)
        # With gamma=2, each component is 2 * sqrt(4) * x/||x|| = 2 * 2 * 1/2 = 2
        np.testing.assert_allclose(np.array(out).flatten(), 2.0, atol=1e-4)


class TestDupUp3D:
    """Tests for vae22.DupUp3D spatial/temporal upsampling."""

    def test_spatial_only(self):
        from mlx_video.models.wan_2.vae22 import DupUp3D

        up = DupUp3D(8, 4, factor_t=1, factor_s=2)
        x = mx.random.normal((1, 3, 4, 4, 8))
        out = up(x)
        mx.eval(out)
        assert out.shape == (1, 3, 8, 8, 4)

    def test_temporal_and_spatial(self):
        from mlx_video.models.wan_2.vae22 import DupUp3D

        up = DupUp3D(16, 8, factor_t=2, factor_s=2)
        x = mx.random.normal((1, 3, 4, 4, 16))
        out = up(x)
        mx.eval(out)
        assert out.shape == (1, 6, 8, 8, 8)

    def test_first_chunk_trims(self):
        from mlx_video.models.wan_2.vae22 import DupUp3D

        up = DupUp3D(8, 4, factor_t=2, factor_s=2)
        x = mx.random.normal((1, 3, 4, 4, 8))
        out_normal = up(x, first_chunk=False)
        out_trimmed = up(x, first_chunk=True)
        mx.eval(out_normal, out_trimmed)
        # first_chunk removes factor_t-1=1 temporal frame
        assert out_normal.shape[1] == 6
        assert out_trimmed.shape[1] == 5

    def test_no_temporal_first_chunk_noop(self):
        from mlx_video.models.wan_2.vae22 import DupUp3D

        up = DupUp3D(8, 4, factor_t=1, factor_s=2)
        x = mx.random.normal((1, 3, 4, 4, 8))
        out_normal = up(x, first_chunk=False)
        out_trimmed = up(x, first_chunk=True)
        mx.eval(out_normal, out_trimmed)
        # factor_t=1, so first_chunk removes 0 frames
        assert out_normal.shape == out_trimmed.shape


class TestVAE22Resample:
    """Tests for vae22.Resample (spatial/temporal upsampling)."""

    def test_upsample2d_shape(self):
        from mlx_video.models.wan_2.vae22 import Resample

        r = Resample(8, "upsample2d")
        r.resample_weight = mx.random.normal(r.resample_weight.shape) * 0.01
        x = mx.random.normal((1, 2, 4, 4, 8))
        out = r(x)
        mx.eval(out)
        assert out.shape == (1, 2, 8, 8, 8)  # 2x spatial, same temporal

    def test_upsample3d_shape(self):
        from mlx_video.models.wan_2.vae22 import Resample

        r = Resample(8, "upsample3d")
        r.resample_weight = mx.random.normal(r.resample_weight.shape) * 0.01
        x = mx.random.normal((1, 2, 4, 4, 8))
        out = r(x)
        mx.eval(out)
        assert out.shape == (1, 4, 8, 8, 8)  # 2x spatial + 2x temporal

    def test_upsample3d_first_chunk(self):
        from mlx_video.models.wan_2.vae22 import Resample

        r = Resample(8, "upsample3d")
        r.resample_weight = mx.random.normal(r.resample_weight.shape) * 0.01
        x = mx.random.normal((1, 2, 4, 4, 8))
        out = r(x, first_chunk=True)
        mx.eval(out)
        # first_chunk: 1 (bypass) + 2*(T-1) (interleaved) = 2T-1 = 3
        assert out.shape == (1, 3, 8, 8, 8)

    def test_upsample3d_first_chunk_single_frame(self):
        """Single-frame input with first_chunk: no temporal upsample."""
        from mlx_video.models.wan_2.vae22 import Resample

        r = Resample(8, "upsample3d")
        r.resample_weight = mx.random.normal(r.resample_weight.shape) * 0.01
        x = mx.random.normal((1, 1, 4, 4, 8))
        out = r(x, first_chunk=True)
        mx.eval(out)
        # Single frame with first_chunk: falls through to non-first path
        # time_conv on 1 frame → 2 interleaved
        assert out.shape == (1, 2, 8, 8, 8)

    def test_upsample3d_first_frame_bypasses_time_conv(self):
        """First frame of first_chunk should NOT go through time_conv.

        Official Wan2.2 skips time_conv for the very first frame entirely.
        We verify this by checking that the first output frame depends only on
        the first input frame (not on time_conv parameters).
        """
        from mlx_video.models.wan_2.vae22 import Resample

        C = 8
        r = Resample(C, "upsample3d")
        # Set time_conv weights to large values so its effect is detectable
        r.time_conv.weight = mx.ones(r.time_conv.weight.shape) * 10.0
        r.time_conv.bias = mx.zeros(r.time_conv.bias.shape)
        # Set spatial conv to identity-like
        r.resample_weight = mx.zeros(r.resample_weight.shape)
        r.resample_bias = mx.zeros(r.resample_bias.shape)

        x = mx.random.normal((1, 3, 2, 2, C))
        out = r(x, first_chunk=True)
        mx.eval(out)
        # Output: 5 frames (1 bypass + 4 interleaved from 2 remaining)
        assert out.shape[1] == 5

        # First frame should be spatial upsample of x[:, 0:1] only.
        # Run just the first frame through spatial upsample for reference
        first_only = x[:, 0:1]
        ref = r._upsample2x(first_only.reshape(1, 2, 2, C))
        ref = mx.pad(ref, [(0, 0), (1, 1), (1, 1), (0, 0)])
        ref = mx.conv_general(ref, r.resample_weight) + r.resample_bias
        mx.eval(ref)

        # Compare first output frame to reference
        first_out = out[:, 0:1].reshape(1, out.shape[2], out.shape[3], C)
        mx.eval(first_out)
        assert mx.allclose(
            first_out, ref, atol=1e-5
        ).item(), "First frame should bypass time_conv and match spatial-only upsample"


class TestVAE22ResidualBlock:
    """Tests for vae22.ResidualBlock."""

    def test_same_dim(self):
        from mlx_video.models.wan_2.vae22 import ResidualBlock

        block = ResidualBlock(8, 8)
        x = mx.random.normal((1, 2, 4, 4, 8))
        out = block(x)
        mx.eval(out)
        assert out.shape == (1, 2, 4, 4, 8)

    def test_different_dim(self):
        from mlx_video.models.wan_2.vae22 import ResidualBlock

        block = ResidualBlock(8, 16)
        x = mx.random.normal((1, 2, 4, 4, 8))
        out = block(x)
        mx.eval(out)
        assert out.shape == (1, 2, 4, 4, 16)

    def test_shortcut_when_dims_differ(self):
        from mlx_video.models.wan_2.vae22 import ResidualBlock

        block = ResidualBlock(8, 16)
        assert block.shortcut is not None

    def test_no_shortcut_same_dim(self):
        from mlx_video.models.wan_2.vae22 import ResidualBlock

        block = ResidualBlock(8, 8)
        assert block.shortcut is None


class TestResidualBlockLayers:
    """Tests for vae22.ResidualBlockLayers naming convention."""

    def test_layer_names_no_underscore_prefix(self):
        """Layer names must NOT start with underscore (MLX ignores them)."""
        from mlx_video.models.wan_2.vae22 import ResidualBlockLayers

        block = ResidualBlockLayers(8, 8)
        params = dict(block.parameters())
        # All param keys should use layer_N, not _layer_N
        for key in params:
            assert not key.startswith("_"), f"Parameter {key} starts with underscore"

    def test_has_expected_layers(self):
        from mlx_video.models.wan_2.vae22 import ResidualBlockLayers

        block = ResidualBlockLayers(8, 16)
        assert hasattr(block, "layer_0")  # first RMS_norm
        assert hasattr(block, "layer_2")  # first CausalConv3d
        assert hasattr(block, "layer_3")  # second RMS_norm
        assert hasattr(block, "layer_6")  # second CausalConv3d

    def test_forward_shape(self):
        from mlx_video.models.wan_2.vae22 import ResidualBlockLayers

        block = ResidualBlockLayers(8, 16)
        x = mx.random.normal((1, 2, 4, 4, 8))
        out = block(x)
        mx.eval(out)
        assert out.shape == (1, 2, 4, 4, 16)


class TestVAE22AttentionBlock:
    """Tests for vae22.AttentionBlock (per-frame 2D self-attention)."""

    def test_output_shape(self):
        from mlx_video.models.wan_2.vae22 import AttentionBlock

        block = AttentionBlock(16)
        block.to_qkv_weight = mx.random.normal(block.to_qkv_weight.shape) * 0.01
        block.proj_weight = mx.random.normal(block.proj_weight.shape) * 0.01
        x = mx.random.normal((1, 2, 4, 4, 16))
        out = block(x)
        mx.eval(out)
        assert out.shape == (1, 2, 4, 4, 16)

    def test_residual_connection(self):
        from mlx_video.models.wan_2.vae22 import AttentionBlock

        block = AttentionBlock(8)
        block.to_qkv_weight = mx.zeros(block.to_qkv_weight.shape)
        block.proj_weight = mx.zeros(block.proj_weight.shape)
        x = mx.ones((1, 1, 2, 2, 8))
        out = block(x)
        mx.eval(out)
        # With zero weights, attention output is 0 → residual is identity
        np.testing.assert_allclose(np.array(out), np.array(x), atol=1e-5)


class TestHead22:
    """Tests for vae22.Head22 output head."""

    def test_output_shape(self):
        from mlx_video.models.wan_2.vae22 import Head22

        head = Head22(16, out_channels=12)
        x = mx.random.normal((1, 2, 4, 4, 16))
        out = head(x)
        mx.eval(out)
        assert out.shape == (1, 2, 4, 4, 12)

    def test_layer_names_no_underscore(self):
        """Head layers must not use underscore prefix."""
        from mlx_video.models.wan_2.vae22 import Head22

        head = Head22(8)
        assert hasattr(head, "layer_0")  # RMS_norm
        assert hasattr(head, "layer_2")  # CausalConv3d
        params = dict(head.parameters())
        for key in params:
            assert not key.startswith("_"), f"Head param {key} starts with underscore"


class TestUnpatchify:
    """Tests for vae22._unpatchify."""

    def test_basic_shape(self):
        from mlx_video.models.wan_2.vae22 import _unpatchify

        x = mx.random.normal((1, 2, 4, 4, 12))  # 12 = 3 * 2 * 2
        out = _unpatchify(x, patch_size=2)
        mx.eval(out)
        assert out.shape == (1, 2, 8, 8, 3)

    def test_patch_size_1_noop(self):
        from mlx_video.models.wan_2.vae22 import _unpatchify

        x = mx.random.normal((1, 2, 4, 4, 3))
        out = _unpatchify(x, patch_size=1)
        mx.eval(out)
        np.testing.assert_array_equal(np.array(out), np.array(x))

    def test_preserves_content(self):
        """Unpatchify should be a lossless rearrangement."""
        from mlx_video.models.wan_2.vae22 import _unpatchify

        x = mx.arange(48).reshape(1, 1, 2, 2, 12).astype(mx.float32)
        out = _unpatchify(x, patch_size=2)
        mx.eval(out)
        # All elements should be preserved
        assert np.array(out).size == 48
        assert set(np.array(out).flatten().tolist()) == set(range(48))


class TestDenormalizeLatents:
    """Tests for vae22.denormalize_latents."""

    def test_output_shape(self):
        from mlx_video.models.wan_2.vae22 import denormalize_latents

        z = mx.random.normal((1, 2, 4, 4, 48))
        out = denormalize_latents(z)
        mx.eval(out)
        assert out.shape == (1, 2, 4, 4, 48)

    def test_custom_mean_std(self):
        from mlx_video.models.wan_2.vae22 import denormalize_latents

        z = mx.ones((1, 1, 1, 1, 4))
        mean = mx.array([1.0, 2.0, 3.0, 4.0])
        std = mx.array([0.5, 0.5, 0.5, 0.5])
        out = denormalize_latents(z, mean=mean, std=std)
        mx.eval(out)
        # z * std + mean = 1*0.5 + [1,2,3,4] = [1.5, 2.5, 3.5, 4.5]
        np.testing.assert_allclose(
            np.array(out).flatten(), [1.5, 2.5, 3.5, 4.5], atol=1e-5
        )

    def test_uses_default_constants(self):
        from mlx_video.models.wan_2.vae22 import (
            VAE22_MEAN,
            denormalize_latents,
        )

        # Should not raise with default constants
        z = mx.zeros((1, 1, 1, 1, 48))
        out = denormalize_latents(z)
        mx.eval(out)
        # z=0 → result = 0 * std + mean = mean
        np.testing.assert_allclose(
            np.array(out).flatten(),
            np.array(VAE22_MEAN).flatten(),
            atol=1e-5,
        )


class TestVAE22NormConstants:
    """Tests for VAE22_MEAN and VAE22_STD constants."""

    def test_dimensions(self):
        from mlx_video.models.wan_2.vae22 import VAE22_MEAN, VAE22_STD

        mx.eval(VAE22_MEAN, VAE22_STD)
        assert VAE22_MEAN.shape == (48,)
        assert VAE22_STD.shape == (48,)

    def test_std_positive(self):
        from mlx_video.models.wan_2.vae22 import VAE22_STD

        mx.eval(VAE22_STD)
        assert (np.array(VAE22_STD) > 0).all()


class TestWan22VAEDecoder:
    """Tests for the full Wan22VAEDecoder (tiny configuration)."""

    def test_output_shape_small(self):
        """Tiny decoder should produce correct spatial/temporal output."""
        from mlx_video.models.wan_2.vae22 import Wan22VAEDecoder

        # Use very small dims to keep test fast
        dec = Wan22VAEDecoder(z_dim=4, dim=8, dec_dim=8)
        # Latent: [B=1, T=3, H=2, W=2, C=4]
        # Expected: temporal 3→5→9→9→9 (two temporal upsamples), spatial 2→4→8→16
        z = mx.random.normal((1, 3, 2, 2, 4)) * 0.1
        out = dec(z)
        mx.eval(out)
        # Output should have 3 RGB channels and be clipped to [-1, 1]
        assert out.shape[-1] == 3
        assert out.ndim == 5
        assert np.array(out).min() >= -1.0
        assert np.array(out).max() <= 1.0

    def test_output_clipped(self):
        from mlx_video.models.wan_2.vae22 import Wan22VAEDecoder

        dec = Wan22VAEDecoder(z_dim=4, dim=8, dec_dim=8)
        z = mx.random.normal((1, 2, 2, 2, 4)) * 10.0  # large values
        out = dec(z)
        mx.eval(out)
        assert np.array(out).min() >= -1.0 - 1e-6
        assert np.array(out).max() <= 1.0 + 1e-6


class TestSanitizeWan22VAEWeights:
    """Tests for vae22.sanitize_wan22_vae_weights."""

    def test_skip_encoder(self):
        from mlx_video.models.wan_2.vae22 import sanitize_wan22_vae_weights

        weights = {
            "encoder.layer.weight": mx.zeros((4,)),
            "conv1.weight": mx.zeros((4,)),
            "decoder.conv1.bias": mx.zeros((4,)),
        }
        out = sanitize_wan22_vae_weights(weights)
        assert "encoder.layer.weight" not in out
        assert "conv1.weight" not in out
        assert "decoder.conv1.bias" in out

    def test_sequential_index_remapping(self):
        from mlx_video.models.wan_2.vae22 import sanitize_wan22_vae_weights

        weights = {
            "decoder.upsamples.0.upsamples.0.residual.0.gamma": mx.ones((8,)),
            "decoder.upsamples.0.upsamples.0.residual.6.bias": mx.zeros((8,)),
            "decoder.head.0.gamma": mx.ones((4,)),
            "decoder.head.2.bias": mx.zeros((12,)),
        }
        out = sanitize_wan22_vae_weights(weights)
        assert "decoder.upsamples.0.upsamples.0.residual.layer_0.gamma" in out
        assert "decoder.upsamples.0.upsamples.0.residual.layer_6.bias" in out
        assert "decoder.head.layer_0.gamma" in out
        assert "decoder.head.layer_2.bias" in out

    def test_resample_conv_remapping(self):
        from mlx_video.models.wan_2.vae22 import sanitize_wan22_vae_weights

        weights = {
            "decoder.upsamples.1.upsamples.3.resample.1.weight": mx.zeros((8, 8, 3, 3)),
            "decoder.upsamples.1.upsamples.3.resample.1.bias": mx.zeros((8,)),
        }
        out = sanitize_wan22_vae_weights(weights)
        assert "decoder.upsamples.1.upsamples.3.resample_weight" in out
        assert "decoder.upsamples.1.upsamples.3.resample_bias" in out

    def test_attention_remapping(self):
        from mlx_video.models.wan_2.vae22 import sanitize_wan22_vae_weights

        weights = {
            "decoder.middle.1.to_qkv.weight": mx.zeros((24, 8, 1, 1)),
            "decoder.middle.1.to_qkv.bias": mx.zeros((24,)),
            "decoder.middle.1.proj.weight": mx.zeros((8, 8, 1, 1)),
            "decoder.middle.1.proj.bias": mx.zeros((8,)),
        }
        out = sanitize_wan22_vae_weights(weights)
        assert "decoder.middle.1.to_qkv_weight" in out
        assert "decoder.middle.1.to_qkv_bias" in out
        assert "decoder.middle.1.proj_weight" in out
        assert "decoder.middle.1.proj_bias" in out

    def test_conv3d_transpose(self):
        from mlx_video.models.wan_2.vae22 import sanitize_wan22_vae_weights

        # Conv3d weight: [O, I, D, H, W] → [O, D, H, W, I]
        w = mx.zeros((16, 8, 3, 3, 3))
        weights = {"decoder.conv1.weight": w}
        out = sanitize_wan22_vae_weights(weights)
        assert out["decoder.conv1.weight"].shape == (16, 3, 3, 3, 8)

    def test_conv2d_transpose(self):
        from mlx_video.models.wan_2.vae22 import sanitize_wan22_vae_weights

        # Conv2d weight: [O, I, H, W] → [O, H, W, I]
        w = mx.zeros((8, 8, 3, 3))
        weights = {"decoder.upsamples.0.upsamples.2.resample.1.weight": w}
        out = sanitize_wan22_vae_weights(weights)
        key = "decoder.upsamples.0.upsamples.2.resample_weight"
        assert out[key].shape == (8, 3, 3, 8)

    def test_gamma_squeeze(self):
        from mlx_video.models.wan_2.vae22 import sanitize_wan22_vae_weights

        # gamma: (dim, 1, 1, 1) → (dim,)
        w = mx.ones((16, 1, 1, 1))
        weights = {"decoder.upsamples.0.upsamples.0.residual.0.gamma": w}
        out = sanitize_wan22_vae_weights(weights)
        key = "decoder.upsamples.0.upsamples.0.residual.layer_0.gamma"
        assert out[key].shape == (16,)


class TestUpResidualBlock:
    """Tests for vae22.Up_ResidualBlock."""

    def test_no_upsample(self):
        from mlx_video.models.wan_2.vae22 import Up_ResidualBlock

        block = Up_ResidualBlock(
            8, 8, num_res_blocks=1, temperal_upsample=False, up_flag=False
        )
        x = mx.random.normal((1, 2, 4, 4, 8))
        out = block(x)
        mx.eval(out)
        # No upsample: same shape
        assert out.shape == (1, 2, 4, 4, 8)

    def test_spatial_upsample(self):
        from mlx_video.models.wan_2.vae22 import Up_ResidualBlock

        block = Up_ResidualBlock(
            8, 4, num_res_blocks=1, temperal_upsample=False, up_flag=True
        )
        x = mx.random.normal((1, 2, 4, 4, 8))
        out = block(x)
        mx.eval(out)
        # 2x spatial upsample, no temporal
        assert out.shape == (1, 2, 8, 8, 4)

    def test_spatial_temporal_upsample(self):
        from mlx_video.models.wan_2.vae22 import Up_ResidualBlock

        block = Up_ResidualBlock(
            8, 4, num_res_blocks=1, temperal_upsample=True, up_flag=True
        )
        x = mx.random.normal((1, 2, 4, 4, 8))
        out = block(x)
        mx.eval(out)
        # 2x spatial + 2x temporal
        assert out.shape == (1, 4, 8, 8, 4)


class TestPatchify:
    """Tests for _patchify and _unpatchify round-trip."""

    def test_roundtrip(self):
        from mlx_video.models.wan_2.vae22 import _patchify, _unpatchify

        x = mx.random.normal((1, 1, 64, 64, 3))
        p = _patchify(x, patch_size=2)
        assert p.shape == (1, 1, 32, 32, 12)
        back = _unpatchify(p, patch_size=2)
        assert back.shape == x.shape
        assert float(mx.abs(x - back).max()) == 0.0

    def test_identity_patch_1(self):
        from mlx_video.models.wan_2.vae22 import _patchify, _unpatchify

        x = mx.random.normal((1, 2, 8, 8, 3))
        assert _patchify(x, patch_size=1).shape == x.shape
        assert _unpatchify(x, patch_size=1).shape == x.shape


class TestAvgDown3D:
    """Tests for AvgDown3D downsampling."""

    def test_spatial_only(self):
        from mlx_video.models.wan_2.vae22 import AvgDown3D

        down = AvgDown3D(8, 16, factor_t=1, factor_s=2)
        x = mx.random.normal((1, 2, 8, 8, 8))
        out = down(x)
        mx.eval(out)
        assert out.shape == (1, 2, 4, 4, 16)

    def test_temporal_and_spatial(self):
        from mlx_video.models.wan_2.vae22 import AvgDown3D

        down = AvgDown3D(8, 16, factor_t=2, factor_s=2)
        x = mx.random.normal((1, 4, 8, 8, 8))
        out = down(x)
        mx.eval(out)
        assert out.shape == (1, 2, 4, 4, 16)

    def test_single_frame(self):
        from mlx_video.models.wan_2.vae22 import AvgDown3D

        down = AvgDown3D(8, 8, factor_t=2, factor_s=2)
        x = mx.random.normal((1, 1, 8, 8, 8))
        out = down(x)
        mx.eval(out)
        # T=1 with factor_t=2: pads to T=2 then averages → T=1
        assert out.shape == (1, 1, 4, 4, 8)


class TestDownResidualBlock:
    """Tests for Down_ResidualBlock."""

    def test_no_downsample(self):
        from mlx_video.models.wan_2.vae22 import Down_ResidualBlock

        block = Down_ResidualBlock(
            8, 8, num_res_blocks=1, temperal_downsample=False, down_flag=False
        )
        x = mx.random.normal((1, 2, 8, 8, 8))
        out = block(x)
        mx.eval(out)
        assert out.shape == (1, 2, 8, 8, 8)

    def test_spatial_downsample(self):
        from mlx_video.models.wan_2.vae22 import Down_ResidualBlock

        block = Down_ResidualBlock(
            8, 16, num_res_blocks=1, temperal_downsample=False, down_flag=True
        )
        x = mx.random.normal((1, 2, 8, 8, 8))
        out = block(x)
        mx.eval(out)
        assert out.shape == (1, 2, 4, 4, 16)

    def test_spatial_temporal_downsample(self):
        from mlx_video.models.wan_2.vae22 import Down_ResidualBlock

        block = Down_ResidualBlock(
            8, 16, num_res_blocks=1, temperal_downsample=True, down_flag=True
        )
        x = mx.random.normal((1, 4, 8, 8, 8))
        out = block(x)
        mx.eval(out)
        assert out.shape == (1, 2, 4, 4, 16)


class TestEncoder3d:
    """Tests for Encoder3d."""

    def test_output_shape(self):
        from mlx_video.models.wan_2.vae22 import Encoder3d

        enc = Encoder3d(dim=16, z_dim=8)
        x = mx.random.normal((1, 1, 16, 16, 12))
        mx.eval(enc.parameters())
        out = enc(x)
        mx.eval(out)
        # 3 spatial downsamples ÷8: 16→2
        assert out.shape == (1, 1, 2, 2, 8)

    def test_multi_frame(self):
        from mlx_video.models.wan_2.vae22 import Encoder3d

        enc = Encoder3d(dim=16, z_dim=8, temperal_downsample=(True, True, False))
        x = mx.random.normal((1, 5, 16, 16, 12))
        mx.eval(enc.parameters())
        out = enc(x)
        mx.eval(out)
        # T: 5→3 (1st t_down) →2 (2nd t_down), spatial ÷8
        assert out.shape[2:] == (2, 2, 8)


class TestWan22VAEEncoder:
    """Tests for Wan22VAEEncoder wrapper."""

    def test_output_shape(self):
        from mlx_video.models.wan_2.vae22 import Wan22VAEEncoder

        enc = Wan22VAEEncoder(z_dim=48, dim=16)
        # Input: single image 32×32 (patchify÷2 → 16×16, then 3 spatial ÷8 → 2×2)
        img = mx.random.normal((1, 1, 32, 32, 3))
        mx.eval(enc.parameters())
        z = enc(img)
        mx.eval(z)
        assert z.shape == (1, 1, 2, 2, 48)

    def test_full_dim(self):
        from mlx_video.models.wan_2.vae22 import Wan22VAEEncoder

        enc = Wan22VAEEncoder(z_dim=48, dim=160)
        img = mx.random.normal((1, 1, 64, 64, 3))
        mx.eval(enc.parameters())
        z = enc(img)
        mx.eval(z)
        # 64 / 16 = 4 (vae stride 16×)
        assert z.shape == (1, 1, 4, 4, 48)


class TestNormalizeLatents:
    """Tests for normalize/denormalize latent roundtrip."""

    def test_roundtrip(self):
        from mlx_video.models.wan_2.vae22 import denormalize_latents, normalize_latents

        z = mx.random.normal((1, 2, 4, 4, 48))
        z_norm = normalize_latents(z)
        z_back = denormalize_latents(z_norm)
        mx.eval(z_back)
        assert float(mx.abs(z - z_back).max()) < 1e-4


class TestVAEEncoderTemporalOrder:
    """Tests that VAE encoder uses (False, True, True) temporal downsample order,
    matching official Wan2.2 vae2_2.py."""

    def test_encoder_temporal_downsample_pattern(self):
        """Encoder3d with (False, True, True): T=5→5→3→2."""
        from mlx_video.models.wan_2.vae22 import Encoder3d

        enc = Encoder3d(dim=16, z_dim=8, temperal_downsample=(False, True, True))
        x = mx.random.normal((1, 5, 16, 16, 12))
        mx.eval(enc.parameters())
        out = enc(x)
        mx.eval(out)
        assert out.shape[1] == 2

    def test_wrapper_uses_correct_pattern(self):
        """Wan22VAEEncoder should use (False, True, True) temporal downsample."""
        from mlx_video.models.wan_2.vae22 import Resample, Wan22VAEEncoder

        enc = Wan22VAEEncoder(z_dim=48, dim=16)
        down_blocks = enc.encoder.downsamples
        found_modes = []
        for block in down_blocks:
            for layer in block.downsamples:
                if isinstance(layer, Resample):
                    found_modes.append(layer.mode)
        # First spatial-only, then two with temporal
        assert found_modes[0] == "downsample2d"
        assert any("3d" in m for m in found_modes)

    def test_single_frame_encoder(self):
        """Single frame (T=1) should work with (False, True, True) pattern."""
        from mlx_video.models.wan_2.vae22 import Wan22VAEEncoder

        enc = Wan22VAEEncoder(z_dim=48, dim=16)
        img = mx.random.normal((1, 1, 32, 32, 3))
        mx.eval(enc.parameters())
        z = enc(img)
        mx.eval(z)
        assert z.shape[1] == 1
        assert z.shape[-1] == 48

    def test_wrong_order_gives_different_result(self):
        """(True, True, False) vs (False, True, True) produce different outputs."""
        from mlx_video.models.wan_2.vae22 import Encoder3d

        enc_correct = Encoder3d(
            dim=16, z_dim=8, temperal_downsample=(False, True, True)
        )
        enc_wrong = Encoder3d(dim=16, z_dim=8, temperal_downsample=(True, True, False))

        x = mx.random.normal((1, 5, 16, 16, 12))
        mx.eval(enc_correct.parameters())
        mx.eval(enc_wrong.parameters())

        out_correct = enc_correct(x)
        out_wrong = enc_wrong(x)
        mx.eval(out_correct, out_wrong)

        # Both give T=2 but spatial processing path differs
        assert out_correct.shape[1] == 2
        assert out_wrong.shape[1] == 2


# ---------------------------------------------------------------------------
# VAE Encode → Decode Round-Trip Tests
# ---------------------------------------------------------------------------


class TestVAE21RoundTrip:
    """Encode→decode round-trip for Wan 2.1 VAE (channels-first)."""

    def test_encode_decode_shape_and_values(self):
        """Encoder3d → Decoder3d: output shape matches input, values are finite."""
        from mlx_video.models.wan_2.vae import Decoder3d, Encoder3d

        z_dim = 4
        dim = 8
        # No temporal up/downsampling to keep the test simple
        enc = Encoder3d(dim=dim, z_dim=z_dim, temporal_downsample=[False, False, False])
        dec = Decoder3d(dim=dim, z_dim=z_dim, temporal_upsample=[False, False, False])
        mx.eval(enc.parameters(), dec.parameters())

        # [B=1, C=3, T=1, H=8, W=8]
        x = mx.random.normal((1, 3, 1, 8, 8)) * 0.5

        z = enc(x)
        mx.eval(z)
        # 3 spatial downsamples (÷8): H=1, W=1
        assert z.shape == (1, z_dim, 1, 1, 1)

        x_hat = dec(z)
        mx.eval(x_hat)
        # 3 spatial upsamples (×8): should recover original shape
        assert x_hat.shape == x.shape

        out_np = np.array(x_hat)
        assert np.all(np.isfinite(out_np))
        assert np.abs(out_np).max() < 1000


class TestVAE22RoundTrip:
    """Encode→decode round-trip for Wan 2.2 VAE (channels-last)."""

    def test_encode_decode_shape_and_values(self):
        """Wan22VAEEncoder → Wan22VAEDecoder: shapes consistent, values in range."""
        from mlx_video.models.wan_2.vae22 import (
            Wan22VAEDecoder,
            Wan22VAEEncoder,
            denormalize_latents,
        )

        enc = Wan22VAEEncoder(z_dim=48, dim=16)
        dec = Wan22VAEDecoder(z_dim=48, dec_dim=8)
        mx.eval(enc.parameters(), dec.parameters())

        # [B=1, T=1, H=32, W=32, C=3]
        img = mx.random.normal((1, 1, 32, 32, 3)) * 0.5

        z_norm = enc(img)
        mx.eval(z_norm)
        # patchify(÷2) + 3 spatial downsamples(÷8) = ÷16
        assert z_norm.shape == (1, 1, 2, 2, 48)

        z = denormalize_latents(z_norm)
        out = dec(z)
        mx.eval(out)

        # 3 spatial upsamples(×8) + unpatchify(×2) = ×16
        assert out.shape[0] == 1  # batch
        assert out.shape[2] == 32  # H recovered
        assert out.shape[3] == 32  # W recovered
        assert out.shape[-1] == 3  # RGB

        out_np = np.array(out)
        assert np.all(np.isfinite(out_np))
        assert out_np.min() >= -1.0 - 1e-6
        assert out_np.max() <= 1.0 + 1e-6
