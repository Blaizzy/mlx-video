"""Phase-1 acceptance test for Wan 2.2 S2V-14B port.

Validates the weight-loading skeleton:

  1. ``WanS2VModel`` instantiates from a tiny S2V config without error.
  2. A synthetic PyTorch-style state dict (with all key patterns that appear
     in the released ``Wan-AI/Wan2.2-S2V-14B`` checkpoint) round-trips through
     ``sanitize_wan_s2v_weights`` and loads into the MLX model with
     ``strict=True``.
  3. Every synthetic key is consumed. No MLX parameter is left un-loaded.

Phase-1 goal: model construction + weight mapping only. Forward passes are
still ``NotImplementedError`` — that is intentional and is covered by test
``test_forward_raises_not_implemented``.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.utils
import pytest


def _make_tiny_s2v_config():
    """Small S2V config for a fast in-process load test.

    Dims chosen so that all size relationships in the reference are preserved
    (e.g. ``hidden_dim // 4`` is a positive integer, ``num_heads`` divides
    ``dim``). Actual released config uses dim=5120, num_layers=40; this test
    uses dim=64, num_layers=4 for speed.
    """
    from mlx_video.models.wan_2.config import WanModelConfig

    cfg = WanModelConfig.wan22_s2v_14b()
    cfg.dim = 64
    cfg.ffn_dim = 128
    cfg.num_heads = 4
    cfg.num_layers = 4
    cfg.audio_dim = 32
    cfg.num_audio_token = 4
    cfg.audio_inject_layers = (0, 3)  # 2 injectors instead of 12
    cfg.freq_dim = 32
    cfg.text_dim = 32
    cfg.text_len = 8
    cfg.cond_dim = 4
    cfg.in_dim = 4
    cfg.out_dim = 4
    cfg.vae_z_dim = 4  # framepack in_channels
    return cfg


def _synthesize_s2v_state_dict(cfg) -> dict:
    """Build a PyTorch-shaped state dict for every key in a real S2V checkpoint.

    Uses zero-valued tensors of the exact shapes the released S2V-14B
    ``diffusion_pytorch_model.safetensors.index.json`` reports (scaled down
    per ``cfg``). Key names use the *pre-sanitizer* PyTorch convention (e.g.
    ``text_embedding.0.weight`` not ``text_embedding_0.weight``).
    """
    dim = cfg.dim
    heads = cfg.num_heads
    n_layers = cfg.num_layers
    audio_dim = cfg.audio_dim
    num_token = cfg.num_audio_token
    in_dim = cfg.in_dim
    out_dim = cfg.out_dim
    ffn_dim = cfg.ffn_dim
    cond_dim = cfg.cond_dim
    fp_in = cfg.vae_z_dim
    text_dim = cfg.text_dim

    sd = {}

    # ---- base transformer (identical to T2V shape conventions) ----
    sd["patch_embedding.weight"] = mx.zeros((dim, in_dim, 1, 2, 2))
    sd["patch_embedding.bias"] = mx.zeros((dim,))
    sd["text_embedding.0.weight"] = mx.zeros((dim, text_dim))
    sd["text_embedding.0.bias"] = mx.zeros((dim,))
    sd["text_embedding.2.weight"] = mx.zeros((dim, dim))
    sd["text_embedding.2.bias"] = mx.zeros((dim,))
    sd["time_embedding.0.weight"] = mx.zeros((dim, cfg.freq_dim))
    sd["time_embedding.0.bias"] = mx.zeros((dim,))
    sd["time_embedding.2.weight"] = mx.zeros((dim, dim))
    sd["time_embedding.2.bias"] = mx.zeros((dim,))
    sd["time_projection.1.weight"] = mx.zeros((dim * 6, dim))
    sd["time_projection.1.bias"] = mx.zeros((dim * 6,))
    sd["head.head.weight"] = mx.zeros((math.prod(cfg.patch_size) * out_dim, dim))
    sd["head.head.bias"] = mx.zeros((math.prod(cfg.patch_size) * out_dim,))
    sd["head.modulation"] = mx.zeros((1, 2, dim))

    for i in range(n_layers):
        b = f"blocks.{i}"
        sd[f"{b}.modulation"] = mx.zeros((1, 6, dim))
        # WanLayerNorm without elementwise_affine has no params in MLX — but
        # PyTorch's WanLayerNorm still has weight (norm3 has cross_attn_norm
        # elementwise_affine=True in S2V per released config).
        sd[f"{b}.norm3.weight"] = mx.zeros((dim,))
        sd[f"{b}.norm3.bias"] = mx.zeros((dim,))
        for attn in ("self_attn", "cross_attn"):
            for w in ("q", "k", "v", "o"):
                sd[f"{b}.{attn}.{w}.weight"] = mx.zeros((dim, dim))
                sd[f"{b}.{attn}.{w}.bias"] = mx.zeros((dim,))
            sd[f"{b}.{attn}.norm_q.weight"] = mx.zeros((dim,))
            sd[f"{b}.{attn}.norm_k.weight"] = mx.zeros((dim,))
        sd[f"{b}.ffn.0.weight"] = mx.zeros((ffn_dim, dim))
        sd[f"{b}.ffn.0.bias"] = mx.zeros((ffn_dim,))
        sd[f"{b}.ffn.2.weight"] = mx.zeros((dim, ffn_dim))
        sd[f"{b}.ffn.2.bias"] = mx.zeros((dim,))

    # ---- S2V-specific ----
    for k, inj_layer in enumerate(cfg.audio_inject_layers):
        p = f"audio_injector.injector.{k}"
        for w in ("q", "k", "v", "o"):
            sd[f"{p}.{w}.weight"] = mx.zeros((dim, dim))
            sd[f"{p}.{w}.bias"] = mx.zeros((dim,))
        sd[f"{p}.norm_q.weight"] = mx.zeros((dim,))
        sd[f"{p}.norm_k.weight"] = mx.zeros((dim,))
        # AdaLayerNorm: linear(embedding_dim=dim, 2*dim)
        pa = f"audio_injector.injector_adain_layers.{k}"
        sd[f"{pa}.linear.weight"] = mx.zeros((2 * dim, dim))
        sd[f"{pa}.linear.bias"] = mx.zeros((2 * dim,))

    # casual_audio_encoder
    sd["casual_audio_encoder.weights"] = mx.zeros((1, 25, 1, 1))
    ae = "casual_audio_encoder.encoder"
    sd[f"{ae}.conv1_local.conv.weight"] = mx.zeros(
        (dim // 4 * num_token, audio_dim, 3)
    )
    sd[f"{ae}.conv1_local.conv.bias"] = mx.zeros((dim // 4 * num_token,))
    sd[f"{ae}.conv1_global.conv.weight"] = mx.zeros((dim // 4, audio_dim, 3))
    sd[f"{ae}.conv1_global.conv.bias"] = mx.zeros((dim // 4,))
    sd[f"{ae}.conv2.conv.weight"] = mx.zeros((dim // 2, dim // 4, 3))
    sd[f"{ae}.conv2.conv.bias"] = mx.zeros((dim // 2,))
    sd[f"{ae}.conv3.conv.weight"] = mx.zeros((dim, dim // 2, 3))
    sd[f"{ae}.conv3.conv.bias"] = mx.zeros((dim,))
    sd[f"{ae}.final_linear.weight"] = mx.zeros((dim, dim))
    sd[f"{ae}.final_linear.bias"] = mx.zeros((dim,))
    sd[f"{ae}.padding_tokens"] = mx.zeros((1, 1, 1, dim))

    # frame_packer (Conv3d in raw PyTorch layout)
    sd["frame_packer.proj.weight"] = mx.zeros((dim, fp_in, 1, 2, 2))
    sd["frame_packer.proj.bias"] = mx.zeros((dim,))
    sd["frame_packer.proj_2x.weight"] = mx.zeros((dim, fp_in, 2, 4, 4))
    sd["frame_packer.proj_2x.bias"] = mx.zeros((dim,))
    sd["frame_packer.proj_4x.weight"] = mx.zeros((dim, fp_in, 4, 8, 8))
    sd["frame_packer.proj_4x.bias"] = mx.zeros((dim,))

    # cond_encoder (Conv3d, raw layout, flat key)
    sd["cond_encoder.weight"] = mx.zeros((dim, cond_dim, 1, 2, 2))
    sd["cond_encoder.bias"] = mx.zeros((dim,))

    # trainable_cond_mask (nn.Embedding(3, dim))
    sd["trainable_cond_mask.weight"] = mx.zeros((3, dim))

    return sd


class TestWanS2VConstruction:
    def test_config_factory(self):
        from mlx_video.models.wan_2.config import WanModelConfig

        cfg = WanModelConfig.wan22_s2v_14b()
        assert cfg.model_type == "s2v"
        assert cfg.audio_dim == 1024
        assert cfg.num_audio_token == 4
        assert len(cfg.audio_inject_layers) == 12
        assert cfg.enable_adain is True
        assert cfg.enable_framepack is True
        assert cfg.enable_motioner is False

    def test_model_instantiates(self):
        from mlx_video.models.wan_2.wan_2 import WanS2VModel

        cfg = _make_tiny_s2v_config()
        model = WanS2VModel(cfg)
        assert model is not None
        # Sanity: expected top-level attributes exist
        assert hasattr(model, "casual_audio_encoder")
        assert hasattr(model, "audio_injector")
        assert hasattr(model, "frame_packer")
        assert hasattr(model, "cond_encoder")
        assert hasattr(model, "trainable_cond_mask")
        assert len(model.blocks) == cfg.num_layers
        assert len(model.audio_injector.injector) == len(cfg.audio_inject_layers)

    def test_forward_shape_synthetic(self):
        """Phase-2 smoke test: forward runs end-to-end on a tiny synthetic input.

        Tiny config (dim=64, layers=4, heads=4, audio_inject at (0, 3),
        num_audio_token=4). Uses random synthetic inputs and only checks that
        the output has the expected shape [C_out, F, H, W]. Numerical
        correctness is NOT validated here — we only need shape consistency and
        no NaNs.
        """
        import numpy as np

        from mlx_video.models.wan_2.wan_2 import WanS2VModel

        cfg = _make_tiny_s2v_config()
        # Force divisibility for the framepack Conv3d kernels (2,4,4) & (4,8,8).
        # Skip framepack for the smoke test — pass motion_history_latent=None.
        model = WanS2VModel(cfg)
        # Initialise all zero params in place so RMSNorm doesn't divide by zero.
        # Instead of tinkering with parameters, just call with a small clip and
        # trust load_weights=zeros → deterministic zero output.
        # NOTE: fp/bf16 zero divide protection via RMSNorm eps=1e-5.

        # Shape choices: F_video = 5 (aligned), H_lat = 8, W_lat = 8.
        # Patchify grid = (5, 4, 4) → 80 noise tokens per batch.
        C, F_vid, H, W = cfg.vae_z_dim, 5, 8, 8
        latent = mx.zeros((C, F_vid, H, W), dtype=mx.float32)
        t = mx.array([500], dtype=mx.float32)
        # Context: [B, text_len, text_dim] pre-embedded (skip text MLP for speed).
        context = mx.zeros((1, cfg.text_len, cfg.dim), dtype=mx.float32)
        # audio_input: (B, 25, audio_dim, T_audio = num_audio_token * F_video)
        T_audio = cfg.num_audio_token * F_vid
        audio = mx.zeros((1, 25, cfg.audio_dim, T_audio), dtype=mx.float32)

        seq_len_hint = F_vid * (H // cfg.patch_size[1]) * (W // cfg.patch_size[2])
        out = model(
            [latent],
            t=t,
            context=context,
            seq_len=seq_len_hint,
            audio_input=audio,
            ref_image_latent=None,
            motion_history_latent=None,
        )
        assert isinstance(out, list) and len(out) == 1
        y = out[0]
        # Expected shape: (C_out, F_vid, H, W) — patchify is (1,2,2) so H,W
        # preserved after unpatchify.
        assert y.shape == (cfg.out_dim, F_vid, H, W), (
            f"Unexpected output shape: {y.shape}, expected "
            f"{(cfg.out_dim, F_vid, H, W)}"
        )
        assert not np.any(np.isnan(np.array(y)))


class TestSanitizerRoundtrip:
    def test_sanitizer_produces_all_mlx_keys(self):
        """Every MLX parameter must have a corresponding sanitized key."""
        from mlx_video.models.wan_2.convert import sanitize_wan_s2v_weights
        from mlx_video.models.wan_2.wan_2 import WanS2VModel

        cfg = _make_tiny_s2v_config()
        model = WanS2VModel(cfg)
        mlx_params = dict(mlx.utils.tree_flatten(model.parameters()))
        # `freqs` is a RoPE buffer computed at model init, not a loaded
        # checkpoint parameter — the sanitizer intentionally skips it
        # (see convert.py sanitize_wan_transformer_weights lines 150-151).
        expected_keys = {k for k in mlx_params.keys() if k != "freqs"}

        pt_state_dict = _synthesize_s2v_state_dict(cfg)
        sanitized = sanitize_wan_s2v_weights(pt_state_dict)
        sanitized_keys = set(sanitized.keys())

        missing = expected_keys - sanitized_keys
        assert not missing, (
            f"Sanitizer failed to produce {len(missing)} MLX-expected keys: "
            f"{sorted(missing)[:10]}"
        )

    def test_no_unused_pytorch_keys(self):
        """Every PyTorch key in the synthetic dict must land in sanitized output."""
        from mlx_video.models.wan_2.convert import sanitize_wan_s2v_weights

        cfg = _make_tiny_s2v_config()
        pt = _synthesize_s2v_state_dict(cfg)
        out = sanitize_wan_s2v_weights(pt)
        # Count check: base sanitizer drops nothing, S2V sanitizer drops
        # nothing → output size == input size (up to key renames, still 1:1).
        assert len(out) == len(pt), (
            f"Sanitizer dropped keys: in={len(pt)}, out={len(out)}"
        )

    def test_shapes_match_after_sanitize(self):
        """Reshaped Conv3d weight for patch_embedding must land in the right shape."""
        from mlx_video.models.wan_2.convert import sanitize_wan_s2v_weights

        cfg = _make_tiny_s2v_config()
        pt = _synthesize_s2v_state_dict(cfg)
        out = sanitize_wan_s2v_weights(pt)
        # patch_embedding weight: Conv3d [dim, in, 1, 2, 2] -> Linear [dim, in*4]
        w = out["patch_embedding_proj.weight"]
        assert w.shape == (cfg.dim, cfg.in_dim * 1 * 2 * 2), w.shape
        # S2V-only keys pass through unchanged
        w = out["casual_audio_encoder.weights"]
        assert w.shape == (1, 25, 1, 1), w.shape
        w = out["frame_packer.proj_2x.weight"]
        assert w.shape == (cfg.dim, cfg.vae_z_dim, 2, 4, 4), w.shape


# Buffers that live on the model as raw ``mx.array`` attributes but are
# *not* loaded from any checkpoint (they're pre-computed constants):
_MODEL_BUFFERS_NOT_IN_CHECKPOINT = {"freqs", "_inv_freq"}


class TestFullLoadCycle:
    def test_load_synthetic_s2v_weights(self):
        """End-to-end: synth PT state dict -> sanitize -> load.

        Uses strict=False because :class:`WanModel` carries pre-computed RoPE
        constants (``freqs``, ``_inv_freq``) as raw ``mx.array`` attributes.
        Instead of strict=True we verify explicitly:

          * Every non-buffer MLX parameter is covered by the sanitized dict.
          * Every sanitized-dict key maps to a real MLX parameter.
        """
        from mlx_video.models.wan_2.convert import sanitize_wan_s2v_weights
        from mlx_video.models.wan_2.wan_2 import WanS2VModel

        cfg = _make_tiny_s2v_config()
        model = WanS2VModel(cfg)
        pt = _synthesize_s2v_state_dict(cfg)
        sanitized = sanitize_wan_s2v_weights(pt)

        # This performs the actual assignment; will raise on shape mismatch.
        model.load_weights(list(sanitized.items()), strict=False)
        mx.eval(model.parameters())

        # Coverage check: every parameter except the known runtime buffers
        # must be present in the sanitized dict.
        mlx_params = dict(mlx.utils.tree_flatten(model.parameters()))
        expected = set(mlx_params) - _MODEL_BUFFERS_NOT_IN_CHECKPOINT
        missing = expected - set(sanitized)
        assert not missing, (
            f"{len(missing)} params not covered by checkpoint: "
            f"{sorted(missing)[:10]}"
        )

        # Unused-key check: no sanitized-dict key should be foreign to the model.
        foreign = set(sanitized) - set(mlx_params)
        assert not foreign, (
            f"{len(foreign)} sanitized keys have no MLX param: "
            f"{sorted(foreign)[:10]}"
        )

    def test_no_unused_audio_keys(self):
        """Every audio-related tensor in the checkpoint gets consumed."""
        from mlx_video.models.wan_2.convert import sanitize_wan_s2v_weights
        from mlx_video.models.wan_2.wan_2 import WanS2VModel

        cfg = _make_tiny_s2v_config()
        model = WanS2VModel(cfg)
        pt = _synthesize_s2v_state_dict(cfg)

        audio_pt_keys = {
            k for k in pt
            if any(t in k for t in ("audio", "casual_audio", "wav"))
        }
        sanitized = sanitize_wan_s2v_weights(pt)
        mlx_params = dict(mlx.utils.tree_flatten(model.parameters()))

        for pt_key in audio_pt_keys:
            # The sanitizer identity-passes audio keys, so the key survives.
            assert pt_key in sanitized, f"Audio key dropped: {pt_key}"
            # And matches an MLX parameter.
            assert pt_key in mlx_params, (
                f"Audio key not present in MLX model params: {pt_key}"
            )
