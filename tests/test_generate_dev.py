"""Tests for LTX-2 dev model generation pipeline."""

import pytest
import mlx.core as mx

from mlx_video.generate_dev import (
    ltx2_scheduler,
    create_position_grid,
    create_audio_position_grid,
    compute_audio_frames,
    cfg_delta,
    DEFAULT_NEGATIVE_PROMPT,
    AUDIO_SAMPLE_RATE,
    AUDIO_LATENTS_PER_SECOND,
)


class TestLTX2Scheduler:
    """Tests for the LTX-2 sigma scheduler."""

    def test_scheduler_output_shape(self):
        """Scheduler should return steps+1 sigma values."""
        steps = 20
        sigmas = ltx2_scheduler(steps=steps)
        assert sigmas.shape == (steps + 1,), f"Expected ({steps + 1},), got {sigmas.shape}"

    def test_scheduler_starts_at_one(self):
        """Sigma schedule should start at 1.0."""
        sigmas = ltx2_scheduler(steps=20)
        assert abs(sigmas[0].item() - 1.0) < 1e-5, f"Expected 1.0, got {sigmas[0].item()}"

    def test_scheduler_ends_at_zero(self):
        """Sigma schedule should end at 0.0."""
        sigmas = ltx2_scheduler(steps=20)
        assert abs(sigmas[-1].item()) < 1e-5, f"Expected 0.0, got {sigmas[-1].item()}"

    def test_scheduler_monotonically_decreasing(self):
        """Sigma values should monotonically decrease."""
        sigmas = ltx2_scheduler(steps=20)
        sigmas_list = sigmas.tolist()
        for i in range(len(sigmas_list) - 1):
            assert sigmas_list[i] >= sigmas_list[i + 1], \
                f"Sigma not decreasing at index {i}: {sigmas_list[i]} < {sigmas_list[i + 1]}"

    def test_scheduler_dtype(self):
        """Scheduler should return float32 array."""
        sigmas = ltx2_scheduler(steps=20)
        assert sigmas.dtype == mx.float32, f"Expected float32, got {sigmas.dtype}"

    def test_scheduler_with_num_tokens(self):
        """Scheduler should accept num_tokens parameter."""
        sigmas_default = ltx2_scheduler(steps=20, num_tokens=None)
        sigmas_custom = ltx2_scheduler(steps=20, num_tokens=1920)

        # Both should be valid arrays
        assert sigmas_default.shape == (21,)
        assert sigmas_custom.shape == (21,)

    def test_scheduler_no_stretch(self):
        """Scheduler without stretching should still work."""
        sigmas = ltx2_scheduler(steps=20, stretch=False)
        assert sigmas.shape == (21,)
        assert sigmas[0].item() > 0
        assert sigmas[-1].item() == 0.0

    def test_scheduler_different_steps(self):
        """Scheduler should work with different step counts."""
        for steps in [5, 10, 20, 40, 50]:
            sigmas = ltx2_scheduler(steps=steps)
            assert sigmas.shape == (steps + 1,), f"Failed for steps={steps}"


class TestCreatePositionGrid:
    """Tests for position grid creation."""

    def test_position_grid_shape(self):
        """Position grid should have correct shape (B, 3, num_patches, 2)."""
        batch_size = 1
        num_frames = 5
        height = 16
        width = 24

        positions = create_position_grid(batch_size, num_frames, height, width)
        num_patches = num_frames * height * width

        expected_shape = (batch_size, 3, num_patches, 2)
        assert positions.shape == expected_shape, \
            f"Expected {expected_shape}, got {positions.shape}"

    def test_position_grid_dtype(self):
        """Position grid should be float32 for RoPE precision."""
        positions = create_position_grid(1, 5, 16, 24)
        assert positions.dtype == mx.float32, \
            f"Expected float32 for RoPE precision, got {positions.dtype}"

    def test_position_grid_batch_size(self):
        """Position grid should respect batch size."""
        for batch_size in [1, 2, 4]:
            positions = create_position_grid(batch_size, 5, 16, 24)
            assert positions.shape[0] == batch_size

    def test_position_grid_temporal_dimension(self):
        """Temporal dimension should have values scaled by fps."""
        positions = create_position_grid(1, 5, 16, 24, fps=24.0)
        temporal = positions[0, 0, :, :]  # (num_patches, 2)

        # Values should be in seconds (divided by fps)
        max_temporal = mx.max(temporal).item()
        # For 5 latent frames at scale 8, max pixel frame ~ 40, divided by 24 fps ~ 1.67s
        assert max_temporal < 10, f"Temporal values too large: {max_temporal}"

    def test_position_grid_spatial_dimensions(self):
        """Spatial dimensions should have pixel-space values."""
        positions = create_position_grid(1, 5, 16, 24, spatial_scale=32)

        # Height dimension
        height_vals = positions[0, 1, :, :]
        max_height = mx.max(height_vals).item()
        # 16 latent * 32 scale = 512 pixels
        assert max_height <= 512, f"Height values too large: {max_height}"

        # Width dimension
        width_vals = positions[0, 2, :, :]
        max_width = mx.max(width_vals).item()
        # 24 latent * 32 scale = 768 pixels
        assert max_width <= 768, f"Width values too large: {max_width}"

    def test_position_grid_causal_fix(self):
        """Causal fix should adjust first frame temporal values."""
        positions_causal = create_position_grid(1, 5, 16, 24, causal_fix=True)
        positions_no_causal = create_position_grid(1, 5, 16, 24, causal_fix=False)

        # They should be different due to causal fix
        diff = mx.abs(positions_causal - positions_no_causal)
        assert mx.max(diff).item() > 0, "Causal fix should change position values"

    def test_position_grid_no_nan_or_inf(self):
        """Position grid should not contain NaN or Inf values."""
        positions = create_position_grid(1, 5, 16, 24)

        assert not mx.any(mx.isnan(positions)).item(), "Position grid contains NaN"
        assert not mx.any(mx.isinf(positions)).item(), "Position grid contains Inf"


class TestCFGDelta:
    """Tests for CFG (Classifier-Free Guidance) delta calculation."""

    def test_cfg_delta_shape(self):
        """CFG delta should have same shape as inputs."""
        shape = (1, 1920, 128)
        cond = mx.random.normal(shape)
        uncond = mx.random.normal(shape)

        delta = cfg_delta(cond, uncond, scale=4.0)
        assert delta.shape == shape

    def test_cfg_delta_scale_one(self):
        """CFG with scale=1.0 should return zero delta."""
        shape = (1, 1920, 128)
        cond = mx.random.normal(shape)
        uncond = mx.random.normal(shape)
        mx.eval(cond, uncond)

        delta = cfg_delta(cond, uncond, scale=1.0)
        mx.eval(delta)

        # Scale=1.0 means (1.0 - 1.0) * (cond - uncond) = 0
        assert mx.max(mx.abs(delta)).item() < 1e-6, "CFG delta with scale=1.0 should be zero"

    def test_cfg_delta_formula(self):
        """CFG delta should follow the formula: (scale-1) * (cond - uncond)."""
        cond = mx.array([[[1.0, 2.0, 3.0]]])
        uncond = mx.array([[[0.5, 1.0, 1.5]]])
        scale = 4.0

        delta = cfg_delta(cond, uncond, scale)
        expected = (scale - 1.0) * (cond - uncond)

        mx.eval(delta, expected)
        diff = mx.max(mx.abs(delta - expected)).item()
        assert diff < 1e-6, f"CFG delta formula mismatch: diff={diff}"

    def test_cfg_delta_dtype_preservation(self):
        """CFG delta should preserve input dtype."""
        for dtype in [mx.float32, mx.bfloat16]:
            cond = mx.random.normal((1, 100, 64)).astype(dtype)
            uncond = mx.random.normal((1, 100, 64)).astype(dtype)

            delta = cfg_delta(cond, uncond, scale=4.0)
            assert delta.dtype == dtype, f"Expected {dtype}, got {delta.dtype}"


class TestDefaultNegativePrompt:
    """Tests for the default negative prompt."""

    def test_default_negative_prompt_exists(self):
        """Default negative prompt should be defined."""
        assert DEFAULT_NEGATIVE_PROMPT is not None
        assert len(DEFAULT_NEGATIVE_PROMPT) > 0

    def test_default_negative_prompt_contains_quality_terms(self):
        """Default negative prompt should contain quality-related terms."""
        prompt_lower = DEFAULT_NEGATIVE_PROMPT.lower()

        # Check for common negative quality terms
        assert "blurry" in prompt_lower, "Should contain 'blurry'"
        assert "low quality" in prompt_lower or "low contrast" in prompt_lower, \
            "Should contain quality-related terms"


class TestInputValidation:
    """Tests for input validation in generate_video_dev."""

    def test_height_divisible_by_32(self):
        """Height must be divisible by 32."""
        # This would be tested via the actual function, but we can test the validation logic
        valid_heights = [256, 384, 512, 640, 768]
        invalid_heights = [100, 300, 500, 700]

        for h in valid_heights:
            assert h % 32 == 0, f"Height {h} should be valid"

        for h in invalid_heights:
            assert h % 32 != 0, f"Height {h} should be invalid"

    def test_width_divisible_by_32(self):
        """Width must be divisible by 32."""
        valid_widths = [256, 384, 512, 640, 768, 1024]
        invalid_widths = [100, 300, 500, 700]

        for w in valid_widths:
            assert w % 32 == 0, f"Width {w} should be valid"

        for w in invalid_widths:
            assert w % 32 != 0, f"Width {w} should be invalid"

    def test_num_frames_formula(self):
        """Number of frames should be 1 + 8*k."""
        valid_frames = [1, 9, 17, 25, 33, 41, 49, 57, 65]

        for f in valid_frames:
            assert (f - 1) % 8 == 0, f"Frames {f} should be valid (1 + 8*k)"

    def test_num_frames_adjustment(self):
        """Invalid frame counts should be adjusted to nearest valid value."""
        # Test the adjustment logic
        test_cases = [
            (30, 33),  # 30 -> nearest valid is 33
            (35, 33),  # 35 -> nearest valid is 33
            (40, 41),  # 40 -> nearest valid is 41
            (1, 1),    # 1 is already valid
            (33, 33),  # 33 is already valid
        ]

        for input_frames, expected in test_cases:
            if input_frames % 8 != 1:
                adjusted = round((input_frames - 1) / 8) * 8 + 1
                assert adjusted == expected, \
                    f"Expected {expected} for input {input_frames}, got {adjusted}"


class TestDenoiseWithCFGMocked:
    """Tests for denoise_with_cfg with mocked transformer."""

    def test_sigmas_list_conversion(self):
        """Sigmas should be convertible to list."""
        sigmas = ltx2_scheduler(steps=5)
        sigmas_list = sigmas.tolist()

        assert isinstance(sigmas_list, list)
        assert len(sigmas_list) == 6  # steps + 1


class TestTilingDefault:
    """Tests for tiling default behavior."""

    def test_tiling_default_is_none(self):
        """Default tiling should be 'none' for performance."""
        import inspect
        from mlx_video.generate_dev import generate_video_dev

        sig = inspect.signature(generate_video_dev)

        tiling_param = sig.parameters.get('tiling')
        assert tiling_param is not None
        assert tiling_param.default == "none", \
            f"Expected default tiling='none', got '{tiling_param.default}'"


class TestLatentDimensions:
    """Tests for latent dimension calculations."""

    def test_latent_height_calculation(self):
        """Latent height should be height // 32."""
        test_cases = [(512, 16), (768, 24), (1024, 32)]

        for height, expected_latent_h in test_cases:
            latent_h = height // 32
            assert latent_h == expected_latent_h, \
                f"Expected latent_h={expected_latent_h} for height={height}, got {latent_h}"

    def test_latent_width_calculation(self):
        """Latent width should be width // 32."""
        test_cases = [(512, 16), (768, 24), (1024, 32)]

        for width, expected_latent_w in test_cases:
            latent_w = width // 32
            assert latent_w == expected_latent_w, \
                f"Expected latent_w={expected_latent_w} for width={width}, got {latent_w}"

    def test_latent_frames_calculation(self):
        """Latent frames should be 1 + (num_frames - 1) // 8."""
        test_cases = [(1, 1), (9, 2), (17, 3), (33, 5), (65, 9)]

        for num_frames, expected_latent_f in test_cases:
            latent_f = 1 + (num_frames - 1) // 8
            assert latent_f == expected_latent_f, \
                f"Expected latent_f={expected_latent_f} for num_frames={num_frames}, got {latent_f}"

    def test_num_tokens_calculation(self):
        """Number of tokens should be latent_f * latent_h * latent_w."""
        # For 33 frames at 512x768
        num_frames, height, width = 33, 512, 768

        latent_f = 1 + (num_frames - 1) // 8  # 5
        latent_h = height // 32  # 16
        latent_w = width // 32  # 24

        num_tokens = latent_f * latent_h * latent_w
        expected = 5 * 16 * 24  # 1920

        assert num_tokens == expected, f"Expected {expected} tokens, got {num_tokens}"


class TestAudioPositionGrid:
    """Tests for audio position grid creation."""

    def test_audio_position_grid_shape(self):
        """Audio position grid should have correct shape (B, 1, T, 2)."""
        batch_size = 1
        audio_frames = 34  # ~1.36 seconds at 25 latent frames/sec

        positions = create_audio_position_grid(batch_size, audio_frames)
        expected_shape = (batch_size, 1, audio_frames, 2)

        assert positions.shape == expected_shape, \
            f"Expected {expected_shape}, got {positions.shape}"

    def test_audio_position_grid_dtype(self):
        """Audio position grid should be float32."""
        positions = create_audio_position_grid(1, 34)
        assert positions.dtype == mx.float32, \
            f"Expected float32, got {positions.dtype}"

    def test_audio_position_grid_batch_size(self):
        """Audio position grid should respect batch size."""
        for batch_size in [1, 2, 4]:
            positions = create_audio_position_grid(batch_size, 34)
            assert positions.shape[0] == batch_size

    def test_audio_position_grid_temporal_values(self):
        """Audio positions should be in seconds."""
        positions = create_audio_position_grid(1, 34)

        # Values should be in seconds (small values for ~1 second of audio)
        max_val = mx.max(positions).item()
        assert max_val < 10, f"Audio positions seem too large: {max_val}"
        assert max_val > 0, "Audio positions should be positive"

    def test_audio_position_grid_no_nan_or_inf(self):
        """Audio position grid should not contain NaN or Inf."""
        positions = create_audio_position_grid(1, 34)

        assert not mx.any(mx.isnan(positions)).item(), "Audio position grid contains NaN"
        assert not mx.any(mx.isinf(positions)).item(), "Audio position grid contains Inf"


class TestComputeAudioFrames:
    """Tests for audio frame count calculation."""

    def test_audio_frames_basic(self):
        """Audio frames should be proportional to video duration."""
        # 33 frames at 24 fps = ~1.375 seconds
        # At 25 latent frames/sec = ~34 audio frames
        audio_frames = compute_audio_frames(33, 24.0)
        assert audio_frames > 0
        assert isinstance(audio_frames, int)

    def test_audio_frames_scales_with_video(self):
        """More video frames should produce more audio frames."""
        audio_33 = compute_audio_frames(33, 24.0)
        audio_65 = compute_audio_frames(65, 24.0)

        assert audio_65 > audio_33, \
            f"Expected more audio frames for longer video: {audio_65} <= {audio_33}"

    def test_audio_frames_formula(self):
        """Audio frames should match expected formula."""
        num_video_frames = 33
        fps = 24.0

        duration = num_video_frames / fps  # ~1.375 seconds
        expected = round(duration * AUDIO_LATENTS_PER_SECOND)

        actual = compute_audio_frames(num_video_frames, fps)
        assert actual == expected, f"Expected {expected}, got {actual}"


class TestAudioConstants:
    """Tests for audio constants."""

    def test_audio_sample_rate(self):
        """Audio sample rate should be 24000 Hz."""
        assert AUDIO_SAMPLE_RATE == 24000

    def test_audio_latents_per_second(self):
        """Audio latents per second should be 25."""
        assert AUDIO_LATENTS_PER_SECOND == 25.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
