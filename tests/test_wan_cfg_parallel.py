"""Tests for --cfg-parallel: validation and CFG split math.

The 2-process exchange itself (mx.distributed.all_gather on the CPU stream) is
verified manually on two Macs with mlx.launch; these tests cover the argument
validation and the math of splitting cond/uncond across ranks.
"""

import mlx.core as mx
import numpy as np
import pytest
from wan_test_helpers import _make_tiny_config


class TestValidateCfgParallel:
    def test_rejects_wrong_world_size(self):
        from mlx_video.models.wan_2.generate import _validate_cfg_parallel

        with pytest.raises(ValueError, match="exactly 2 processes"):
            _validate_cfg_parallel(1, False, 42, False, False)

    def test_rejects_cfg_disabled(self):
        from mlx_video.models.wan_2.generate import _validate_cfg_parallel

        with pytest.raises(ValueError, match="classifier-free guidance"):
            _validate_cfg_parallel(2, True, 42, False, False)

    def test_rejects_unset_seed(self):
        from mlx_video.models.wan_2.generate import _validate_cfg_parallel

        with pytest.raises(ValueError, match="fixed --seed"):
            _validate_cfg_parallel(2, False, -1, False, False)

    def test_rejects_dual_model(self):
        from mlx_video.models.wan_2.generate import _validate_cfg_parallel

        with pytest.raises(ValueError, match="single-model text-to-video"):
            _validate_cfg_parallel(2, False, 42, True, False)

    def test_rejects_i2v(self):
        from mlx_video.models.wan_2.generate import _validate_cfg_parallel

        with pytest.raises(ValueError, match="single-model text-to-video"):
            _validate_cfg_parallel(2, False, 42, False, True)

    def test_accepts_valid_settings(self):
        from mlx_video.models.wan_2.generate import _validate_cfg_parallel

        _validate_cfg_parallel(2, False, 42, False, False)


class TestCfgSplitEquivalence:
    """Splitting the batch of 2 into two single-item predictions is exact."""

    def _setup(self):
        from mlx_video.models.wan_2.wan_2 import WanModel
        from mlx_video.models.wan_2.scheduler import FlowMatchEulerScheduler

        mx.random.seed(0)
        config = _make_tiny_config()
        model = WanModel(config)

        C, F, H, W = config.in_dim, 1, 4, 4
        pt, ph, pw = config.patch_size
        seq_len = (F // pt) * (H // ph) * (W // pw)

        latents = mx.random.normal((C, F, H, W))
        ctx_cond = mx.random.normal((4, config.text_dim))
        ctx_uncond = mx.random.normal((4, config.text_dim))

        sched = FlowMatchEulerScheduler()
        sched.set_timesteps(5, shift=3.0)
        t = sched.timesteps[0].item()
        return model, latents, ctx_cond, ctx_uncond, t, seq_len

    def test_split_matches_batched(self):
        model, latents, ctx_cond, ctx_uncond, t, seq_len = self._setup()

        batched = model(
            [latents, latents],
            mx.array([t, t]),
            [ctx_cond, ctx_uncond],
            seq_len,
        )
        cond = model([latents], mx.array([t]), [ctx_cond], seq_len)[0]
        uncond = model([latents], mx.array([t]), [ctx_uncond], seq_len)[0]
        mx.eval(batched, cond, uncond)

        np.testing.assert_allclose(
            np.array(cond), np.array(batched[0]), atol=1e-4, rtol=1e-4
        )
        np.testing.assert_allclose(
            np.array(uncond), np.array(batched[1]), atol=1e-4, rtol=1e-4
        )

    def test_guided_prediction_matches(self):
        model, latents, ctx_cond, ctx_uncond, t, seq_len = self._setup()

        batched = model(
            [latents, latents],
            mx.array([t, t]),
            [ctx_cond, ctx_uncond],
            seq_len,
        )
        cond = model([latents], mx.array([t]), [ctx_cond], seq_len)[0]
        uncond = model([latents], mx.array([t]), [ctx_uncond], seq_len)[0]

        # all_gather returns rank 0 first, i.e. [cond, uncond].
        gathered = mx.stack([cond, uncond])
        gs = 5.0
        guided_split = gathered[1] + gs * (gathered[0] - gathered[1])
        guided_batched = batched[1] + gs * (batched[0] - batched[1])
        mx.eval(guided_split, guided_batched)

        np.testing.assert_allclose(
            np.array(guided_split),
            np.array(guided_batched),
            atol=1e-4,
            rtol=1e-4,
        )
