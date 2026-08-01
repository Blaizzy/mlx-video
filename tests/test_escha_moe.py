"""Router + shared-expert unit tests. Do NOT depend on codebooks."""

from __future__ import annotations

import numpy as np
import mlx.core as mx


def test_router_topk_sums_to_one():
    from mlx_video.models.qwen3_5_moe_escha.moe import Qwen35MoeRouter

    mx.random.seed(0)
    num_experts, hidden, top_k = 256, 2048, 8
    gw = mx.random.normal((num_experts, hidden)).astype(mx.float16) * 0.1
    router = Qwen35MoeRouter(gw, top_k=top_k)
    x = mx.random.normal((16, hidden)).astype(mx.bfloat16)
    idx, gates = router(x)
    assert idx.shape == (16, top_k)
    assert gates.shape == (16, top_k)
    assert idx.dtype == mx.int32
    # Gates renormalized per token.
    sums = np.array(gates.astype(mx.float32).sum(axis=-1))
    # bf16 gates lose ~0.2 % on sum→1; that's fine (matches the reference).
    np.testing.assert_allclose(sums, np.ones(16), atol=3e-3)


def test_router_deterministic_on_identity_gate():
    """Identity gate: gate weights are I → top-k should match top-k logits of x."""
    from mlx_video.models.qwen3_5_moe_escha.moe import Qwen35MoeRouter

    num_experts, hidden, top_k = 8, 8, 3
    gw = mx.eye(num_experts).astype(mx.float16)
    router = Qwen35MoeRouter(gw, top_k=top_k)
    x = mx.array([[5.0, 1.0, 0.0, 3.0, 2.0, 0.5, 4.0, 0.0]]).astype(mx.bfloat16)
    idx, _ = router(x)
    top3 = sorted(np.array(idx[0]).tolist())
    assert top3 == [0, 3, 6]  # values 5, 3, 4 → indices 0, 3, 6
