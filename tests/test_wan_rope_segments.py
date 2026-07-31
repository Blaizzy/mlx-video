"""Tests for rope_precompute_cos_sin_segments — S2V ref/motion RoPE support.

Verifies that placing a ref token at t=-1 (or any other position) and video
tokens at t=0..F-1 produces cos/sin values matching the manual
theta**(-2i/dim) computation, and that the concatenated freqs work correctly
in rope_apply.
"""

import mlx.core as mx
import numpy as np
import pytest


class TestRopeAtPositions:
    """Verify rope_cos_sin_at_positions works for arbitrary positions."""

    def test_positive_positions_match_rope_params(self):
        """For non-negative int positions, at-positions should match the
        precomputed rope_params table row-by-row."""
        from mlx_video.models.wan_2.rope import (
            rope_cos_sin_at_positions,
            rope_params,
        )

        dim = 44  # temporal dim for head_dim=128 (22 pairs)
        theta = 10000.0
        table = rope_params(1024, dim, theta=theta)  # (1024, 22, 2)
        mx.eval(table)

        positions = mx.array([0.0, 1.0, 5.0, 30.0])
        cos, sin = rope_cos_sin_at_positions(positions, dim, theta=theta)
        mx.eval(cos, sin)

        for i, t in enumerate([0, 1, 5, 30]):
            np.testing.assert_allclose(
                np.array(cos[i]),
                np.array(table[t, :, 0]),
                atol=1e-5,
                err_msg=f"cos mismatch at t={t}",
            )
            np.testing.assert_allclose(
                np.array(sin[i]),
                np.array(table[t, :, 1]),
                atol=1e-5,
                err_msg=f"sin mismatch at t={t}",
            )

    def test_negative_positions_reflect(self):
        """cos(-t) = cos(t) and sin(-t) = -sin(t)."""
        from mlx_video.models.wan_2.rope import rope_cos_sin_at_positions

        dim = 44
        theta = 10000.0

        pos_pos = mx.array([1.0, 3.0, 19.0])
        pos_neg = mx.array([-1.0, -3.0, -19.0])

        cos_p, sin_p = rope_cos_sin_at_positions(pos_pos, dim, theta=theta)
        cos_n, sin_n = rope_cos_sin_at_positions(pos_neg, dim, theta=theta)
        mx.eval(cos_p, sin_p, cos_n, sin_n)

        np.testing.assert_allclose(np.array(cos_p), np.array(cos_n), atol=1e-5)
        np.testing.assert_allclose(np.array(sin_p), -np.array(sin_n), atol=1e-5)

    def test_manual_theta_computation(self):
        """Directly verify cos/sin = cos/sin(pos * theta**(-2i/dim))."""
        from mlx_video.models.wan_2.rope import rope_cos_sin_at_positions

        dim = 22  # small even
        theta = 10000.0
        pos = -1.0
        cos, sin = rope_cos_sin_at_positions(
            mx.array([pos]), dim, theta=theta, dtype=mx.float32
        )
        mx.eval(cos, sin)

        # Manual reference: for i=0..dim/2-1, freq = pos * theta^(-2i/dim)
        i = np.arange(0, dim, 2, dtype=np.float64)
        inv_freq = 1.0 / np.power(theta, i / dim)
        angles = pos * inv_freq
        ref_cos = np.cos(angles).astype(np.float32)
        ref_sin = np.sin(angles).astype(np.float32)

        np.testing.assert_allclose(np.array(cos[0]), ref_cos, atol=1e-5)
        np.testing.assert_allclose(np.array(sin[0]), ref_sin, atol=1e-5)


class TestRopePrecomputeCosSinSegments:
    """Verify rope_precompute_cos_sin_segments produces correct multi-segment freqs."""

    def _freqs_128(self):
        """Build the head_dim=128 freqs table used by 14B."""
        from mlx_video.models.wan_2.rope import rope_params

        d = 128
        return mx.concatenate(
            [
                rope_params(1024, d - 4 * (d // 6)),  # temporal 44
                rope_params(1024, 2 * (d // 6)),  # height 42
                rope_params(1024, 2 * (d // 6)),  # width 42
            ],
            axis=1,
        )

    def test_single_video_segment_matches_precompute_cos_sin(self):
        """A single [0..F-1] x [0..H-1] x [0..W-1] segment must equal the
        existing rope_precompute_cos_sin path."""
        from mlx_video.models.wan_2.rope import (
            rope_precompute_cos_sin,
            rope_precompute_cos_sin_segments,
        )

        freqs = self._freqs_128()
        F, H, W = 5, 4, 4

        # Legacy path
        cos_a, sin_a = rope_precompute_cos_sin([(F, H, W)], freqs)
        # New path (single video segment)
        cos_b, sin_b = rope_precompute_cos_sin_segments(
            [{"t_indices": list(range(F)), "h": H, "w": W}], freqs
        )
        mx.eval(cos_a, sin_a, cos_b, sin_b)

        np.testing.assert_allclose(
            np.array(cos_a), np.array(cos_b), atol=1e-5, err_msg="cos mismatch"
        )
        np.testing.assert_allclose(
            np.array(sin_a), np.array(sin_b), atol=1e-5, err_msg="sin mismatch"
        )

    def test_ref_at_t_neg1_matches_manual_computation(self):
        """Ref at t=-1, video at t=0..F-1: cos/sin match manual theta**(-2i/d) formula."""
        from mlx_video.models.wan_2.rope import (
            rope_precompute_cos_sin_segments,
        )

        freqs = self._freqs_128()
        F, H, W = 3, 2, 2

        segments = [
            {"t_indices": list(range(F)), "h": H, "w": W},
            {"t_indices": [-1], "h": H, "w": W},
        ]
        cos_f, sin_f = rope_precompute_cos_sin_segments(segments, freqs)
        mx.eval(cos_f, sin_f)

        # Slice out the ref segment (last H*W = 4 rows)
        ref_len = 1 * H * W
        cos_ref = np.array(cos_f[-ref_len:, 0, :])
        sin_ref = np.array(sin_f[-ref_len:, 0, :])

        # Manually compute expected cos/sin at (t=-1, h=0..H-1, w=0..W-1)
        half_d = 64
        d_t = half_d - 2 * (half_d // 3)  # 22
        d_h = half_d // 3  # 21
        d_w = half_d // 3  # 21
        theta = 10000.0

        def _angles(pos, dim):
            i = np.arange(0, dim, 2, dtype=np.float64)
            return pos * (1.0 / np.power(theta, i / dim))

        # Temporal (t=-1): cos same as t=1, sin negated
        cos_t = np.cos(_angles(-1.0, d_t * 2)).astype(np.float32)
        sin_t = np.sin(_angles(-1.0, d_t * 2)).astype(np.float32)

        for hi in range(H):
            for wi in range(W):
                idx = hi * W + wi
                cos_h = np.cos(_angles(float(hi), d_h * 2)).astype(np.float32)
                sin_h = np.sin(_angles(float(hi), d_h * 2)).astype(np.float32)
                cos_w = np.cos(_angles(float(wi), d_w * 2)).astype(np.float32)
                sin_w = np.sin(_angles(float(wi), d_w * 2)).astype(np.float32)
                expected_cos = np.concatenate([cos_t, cos_h, cos_w])
                expected_sin = np.concatenate([sin_t, sin_h, sin_w])
                np.testing.assert_allclose(
                    cos_ref[idx], expected_cos, atol=1e-5,
                    err_msg=f"cos mismatch at ref h={hi}, w={wi}",
                )
                np.testing.assert_allclose(
                    sin_ref[idx], expected_sin, atol=1e-5,
                    err_msg=f"sin mismatch at ref h={hi}, w={wi}",
                )

    def test_motion_negative_indices(self):
        """Motion post/2x/4x buckets with t=-1, t=-3, t=-19..-16 produce
        expected cos/sin (magnitude 1, sign flip on sin)."""
        from mlx_video.models.wan_2.rope import (
            rope_precompute_cos_sin_segments,
        )

        freqs = self._freqs_128()
        H, W = 4, 4

        # Motion buckets in kijai concat order: fine(-1), medium(-3), coarse(-19..-16)
        segments = [
            {"t_indices": [-1], "h": H, "w": W},
            {"t_indices": [-3], "h": H // 2, "w": W // 2},
            {"t_indices": [-19, -18, -17, -16], "h": H // 4, "w": W // 4},
        ]
        # H//4 = 1, W//4 = 1: single spatial cell per coarse frame
        cos_f, sin_f = rope_precompute_cos_sin_segments(segments, freqs)
        mx.eval(cos_f, sin_f)

        # Total length
        expected_len = 1 * H * W + 1 * (H // 2) * (W // 2) + 4 * (H // 4) * (W // 4)
        assert cos_f.shape[0] == expected_len
        assert sin_f.shape[0] == expected_len

        # cos^2 + sin^2 == 1 for all elements (rotations are unit-magnitude)
        m = np.array(cos_f[:, 0, :]) ** 2 + np.array(sin_f[:, 0, :]) ** 2
        np.testing.assert_allclose(m, 1.0, atol=1e-5)

    def test_ref_at_positive_t_start(self):
        """Ref at t=max(30, F+9) — the actual kijai formula."""
        from mlx_video.models.wan_2.rope import (
            rope_cos_sin_at_positions,
            rope_precompute_cos_sin_segments,
        )

        freqs = self._freqs_128()
        F, H, W = 21, 4, 4  # 81-frame clip → F=21 latent frames → t_ref=30
        t_ref = max(30, F + 9)
        assert t_ref == 30

        segments = [
            {"t_indices": list(range(F)), "h": H, "w": W},
            {"t_indices": [t_ref], "h": H, "w": W},
        ]
        cos_f, sin_f = rope_precompute_cos_sin_segments(segments, freqs)
        mx.eval(cos_f, sin_f)

        # Verify ref rows match rope_cos_sin_at_positions(t=30, ...) on the
        # temporal-freq portion.
        ref_len = H * W
        cos_ref = np.array(cos_f[-ref_len:, 0, :])

        half_d = 64
        d_t = half_d - 2 * (half_d // 3)
        cos_t30, _ = rope_cos_sin_at_positions(
            mx.array([float(t_ref)]), d_t * 2, theta=10000.0
        )
        mx.eval(cos_t30)
        cos_t30_np = np.array(cos_t30[0])

        # All ref-segment rows share the same temporal freq (since only t=30 given).
        for row in cos_ref:
            np.testing.assert_allclose(row[:d_t], cos_t30_np, atol=1e-5)

    def test_rope_apply_accepts_multi_segment_cos_sin(self):
        """rope_apply with precomputed_cos_sin from segments must handle
        seq_len > grid_sizes[0]'s F*H*W."""
        from mlx_video.models.wan_2.rope import (
            rope_apply,
            rope_precompute_cos_sin_segments,
        )

        freqs = self._freqs_128()
        F, H, W = 3, 2, 2
        noise_len = F * H * W  # 12
        ref_len = H * W  # 4
        total_len = noise_len + ref_len

        segments = [
            {"t_indices": list(range(F)), "h": H, "w": W},
            {"t_indices": [30], "h": H, "w": W},
        ]
        cos_f, sin_f = rope_precompute_cos_sin_segments(segments, freqs)
        assert cos_f.shape[0] == total_len

        B, N = 1, 4
        d = 128
        x = mx.random.normal((B, total_len, N, d))
        # grid_sizes provided for API compatibility (noise portion only)
        out = rope_apply(x, [(F, H, W)], freqs, precomputed_cos_sin=(cos_f, sin_f))
        mx.eval(out)
        assert out.shape == (B, total_len, N, d)
