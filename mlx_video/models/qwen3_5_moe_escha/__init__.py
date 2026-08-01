"""MLX port of EschaLabs' Qwen3.6-35B-A3B-Escha-W2 — a 2-bit AQLM-quantized MoE.

World-first Mac port. See docs/ESCHA_PORT_FEASIBILITY.md for the full design audit.

Storage recap (per MoE expert projection):
- escha_code:  int16 [E, in/16, out/16, 16*K]  — AQLM residual codes (K=2 gate_up, K=3 down)
- escha_rin:   fp16  [E, in]                    — pre-block-Hadamard per-in-channel scale
- escha_rout:  fp16  [E, out]                   — post-block-Hadamard per-out-channel scale
- escha_s_in/s_out: fp32 all-ones (folded into rin/rout at export — dropped at load)
- escha_config: int32[9] = [block=16, K, V=2, cb_id=1, E, in_f, out_f, in_p, out_p] (informational)

Forward per projection:
    xp  = x                                              # s_in folded → identity
    xh  = T128(xp * rin)                                 # 128-block Walsh-Hadamard on last dim
    W   = escham_reconstruct(code, in, out, K, cb_id)    # 2/3-bit codes → fp16 dense weight
    y   = xh @ W
    y   = T128(y) * rout                                 # post-Hadamard, post-scale
    y   = y[..., :out_f]                                 # truncate padded cols

Dense/attention/lm_head/embed = per-row int8 (weight * scale[i]) — decoded to bf16 at load.

Status: Phase 1 skeleton. See PORT_STATUS.md next to this file.
"""

from .model import load_model, Qwen3_5MoeEschaForConditionalGeneration
from .eschamoe import escham_reconstruct, PackedScaledExpertLinear
from .transform import t128, hadamard_128
from .gated_deltanet import GatedDeltaNet, GDNCache, gated_delta_update

__all__ = [
    "load_model",
    "Qwen3_5MoeEschaForConditionalGeneration",
    "escham_reconstruct",
    "PackedScaledExpertLinear",
    "t128",
    "hadamard_128",
    "GatedDeltaNet",
    "GDNCache",
    "gated_delta_update",
]
