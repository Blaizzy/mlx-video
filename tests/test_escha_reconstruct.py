"""Escha AQLM reconstruct + PackedScaledExpertLinear wiring tests.

These use a SYNTHETIC codebook — real codebook extraction is out-of-band
(see mlx_video/models/qwen3_5_moe_escha/codebooks/extract_codebooks.py and
docs/ESCHA_PORT_FEASIBILITY.md §5). The point of the tests is to validate
the plumbing (shape math, dtype, index widening from int16→uint16, cache
behaviour, forward path) so that when real codebooks land we only debug
numerics, not wiring.
"""

from __future__ import annotations

import numpy as np
import pytest
import mlx.core as mx


@pytest.fixture
def synth_codebooks(tmp_path, monkeypatch):
    """Write a deterministic (65536, 16) fp16 codebook and point the module at it.

    Codebook value: cb[i, j] = (i * 16 + j) / 32768 - 1  (in [-1, 1))
    Both K=2 and K=3 share the same layout so any test that fixes indices can
    check reconstruction analytically.
    """
    cb = np.arange(65536 * 16, dtype=np.float32).reshape(65536, 16)
    cb = (cb / 32768.0 - 1.0).astype(np.float16)
    path = tmp_path / "escha_codebooks_v1.npz"
    np.savez_compressed(path, cb_A_K2=cb, cb_A_K3=cb)

    import mlx_video.models.qwen3_5_moe_escha.eschamoe as em
    monkeypatch.setattr(em, "_CB_PATH", path)
    em._CB_CACHE.clear()
    yield cb
    em._CB_CACHE.clear()


def test_reconstruct_shape_and_dtype(synth_codebooks):
    from mlx_video.models.qwen3_5_moe_escha.eschamoe import escham_reconstruct

    # Smallest possible: 16x16 block, K=2, single (bi=0, bj=0).
    code = mx.zeros((1, 1, 32), dtype=mx.int16)
    w = escham_reconstruct(code, in_features=16, out_features=16, K=2)
    assert w.shape == (16, 16)
    assert w.dtype == mx.float16


def test_reconstruct_matches_codebook_lookup(synth_codebooks):
    """For row r, code index i in slot k=0: block[r] should equal cb[i] + cb[0] (K-1 zeros).

    With K=2 and all indices zero except code[0, 0, r]=i, the block row r
    reconstruction = cb[i] + cb[0]. Verify a few rows.
    """
    from mlx_video.models.qwen3_5_moe_escha.eschamoe import escham_reconstruct

    K = 2
    code = np.zeros((1, 1, 16 * K), dtype=np.int16)
    # Set row 0 slot 0 to idx 123; row 5 slot 0 to idx 999.
    code[0, 0, 0] = 123
    code[0, 0, 5] = 999
    w = escham_reconstruct(mx.array(code), in_features=16, out_features=16, K=K)
    w = np.array(w)
    cb = synth_codebooks
    # Row 0: cb[123] (slot k=0) + cb[0] (slot k=1). Other slots also index 0.
    expected_row0 = (cb[123].astype(np.float32) + cb[0].astype(np.float32) * (K - 1)).astype(np.float16)
    expected_row5 = (cb[999].astype(np.float32) + cb[0].astype(np.float32) * (K - 1)).astype(np.float16)
    # Non-set rows: all-zero code → K * cb[0]
    expected_row1 = (cb[0].astype(np.float32) * K).astype(np.float16)
    np.testing.assert_allclose(w[0], expected_row0, atol=1e-3)
    np.testing.assert_allclose(w[5], expected_row5, atol=1e-3)
    np.testing.assert_allclose(w[1], expected_row1, atol=1e-3)


def test_reconstruct_int16_signbit_treated_as_uint16(synth_codebooks):
    """Codes stored as int16 must be interpreted as uint16 for the lookup.

    idx=32768 stored as int16 is -32768. Make sure we still fetch cb[32768].
    """
    from mlx_video.models.qwen3_5_moe_escha.eschamoe import escham_reconstruct

    code = np.zeros((1, 1, 32), dtype=np.int16)
    code[0, 0, 0] = -32768  # bit pattern for uint16 32768
    w = np.array(escham_reconstruct(mx.array(code), 16, 16, K=2))
    cb = synth_codebooks
    expected = (cb[32768].astype(np.float32) + cb[0].astype(np.float32)).astype(np.float16)
    np.testing.assert_allclose(w[0], expected, atol=1e-3)


def test_reconstruct_K3(synth_codebooks):
    from mlx_video.models.qwen3_5_moe_escha.eschamoe import escham_reconstruct

    code = mx.zeros((1, 1, 48), dtype=mx.int16)  # 16*3
    w = escham_reconstruct(code, in_features=16, out_features=16, K=3)
    assert w.shape == (16, 16)


def test_reconstruct_multi_block(synth_codebooks):
    """32x32 weight = 2x2 grid of 16x16 blocks. Verify assembly order (bi=row-block, bj=col-block)."""
    from mlx_video.models.qwen3_5_moe_escha.eschamoe import escham_reconstruct

    K = 2
    code = np.zeros((2, 2, 16 * K), dtype=np.int16)
    # Put a distinct index in each block's row 0.
    code[0, 0, 0] = 100
    code[0, 1, 0] = 200
    code[1, 0, 0] = 300
    code[1, 1, 0] = 400
    w = np.array(escham_reconstruct(mx.array(code), 32, 32, K=K))
    cb = synth_codebooks
    # Row 0 (bi=0, block-row=0), col 0..15 (bj=0): cb[100] + cb[0]
    expected_bi0_bj0 = (cb[100].astype(np.float32) + cb[0].astype(np.float32)).astype(np.float16)
    expected_bi0_bj1 = (cb[200].astype(np.float32) + cb[0].astype(np.float32)).astype(np.float16)
    expected_bi1_bj0 = (cb[300].astype(np.float32) + cb[0].astype(np.float32)).astype(np.float16)
    np.testing.assert_allclose(w[0, 0:16], expected_bi0_bj0, atol=1e-3)
    np.testing.assert_allclose(w[0, 16:32], expected_bi0_bj1, atol=1e-3)
    np.testing.assert_allclose(w[16, 0:16], expected_bi1_bj0, atol=1e-3)


def test_reconstruct_shape_validation(synth_codebooks):
    from mlx_video.models.qwen3_5_moe_escha.eschamoe import escham_reconstruct

    with pytest.raises(ValueError, match="in_features"):
        escham_reconstruct(mx.zeros((1, 1, 32), dtype=mx.int16), in_features=32, out_features=16, K=2)
    with pytest.raises(ValueError, match="16\\*K"):
        escham_reconstruct(mx.zeros((1, 1, 33), dtype=mx.int16), in_features=16, out_features=16, K=2)
    with pytest.raises(NotImplementedError, match="cb_id"):
        escham_reconstruct(mx.zeros((1, 1, 32), dtype=mx.int16), 16, 16, K=2, cb_id=2)


def test_packed_scaled_expert_forward(synth_codebooks):
    """PackedScaledExpertLinear: full path with T128 pre/post and rin/rout."""
    from mlx_video.models.qwen3_5_moe_escha.eschamoe import PackedScaledExpertLinear

    in_f = out_f = in_p = out_p = 128   # smallest multiple of 128 (T128 block)
    K = 2
    # in_p/16 = 8, out_p/16 = 8, 16*K = 32
    code = mx.zeros((in_p // 16, out_p // 16, 16 * K), dtype=mx.int16)
    rin = mx.ones((in_p,), dtype=mx.float16)
    rout = mx.ones((out_p,), dtype=mx.float16)

    layer = PackedScaledExpertLinear(
        code=code, rin=rin, rout=rout,
        in_f=in_f, out_f=out_f, in_p=in_p, out_p=out_p, K=K,
    )
    x = mx.random.normal((3, in_f)).astype(mx.bfloat16)
    y = layer(x)
    assert y.shape == (3, out_f)
    assert y.dtype == mx.bfloat16


def test_missing_codebook_raises_with_helpful_message(tmp_path, monkeypatch):
    import mlx_video.models.qwen3_5_moe_escha.eschamoe as em
    monkeypatch.setattr(em, "_CB_PATH", tmp_path / "nope.npz")
    em._CB_CACHE.clear()
    with pytest.raises(FileNotFoundError, match="codebook"):
        em._load_codebook(2)
    em._CB_CACHE.clear()
