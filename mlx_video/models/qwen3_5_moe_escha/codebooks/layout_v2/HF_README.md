---
license: apache-2.0
tags:
  - reverse-engineering
  - aqlm
  - quantization
  - mlx
  - escha
base_model: EschaLabs/Qwen3.6-35B-A3B-Escha-W2
---

# Escha-W2 packed AQLM codebook — reverse-engineered reference dump

This repository contains the **first public extraction** of the `escham_reconstruct`
codebook lattice used by EschaLabs' 2-bit AQLM+Hadamard quantized checkpoint
[`EschaLabs/Qwen3.6-35B-A3B-Escha-W2`](https://huggingface.co/EschaLabs/Qwen3.6-35B-A3B-Escha-W2).

The Escha packed format stores each MoE expert's `gate_up_proj` (K=2) and
`down_proj` (K=3) projections as (in/16, out/16, 16·K) int16 codes plus per-row
and per-column scales. Decoding those codes into a dense fp16 weight matrix
requires two fixed codebook tables that ship **inside the CUDA `.so`** rather
than in the safetensors — they were previously inaccessible outside a Linux
GPU environment running the reference `escha` wheel.

This repo makes the codebook portable.

## Contents

| File | Size | Description |
|---|---|---|
| `compact.pkl` | 120 MB | The extracted codebook in sparse-compact form (fp16). See "Format" below. |
| `OP_SIGNATURE_AUDIT.md` | 15 KB | Full Modal-side introspection of `escha._C` — every operator, its schema, and the linearity / (bi, bj)-invariance proofs. |
| `LAYOUT_NOTES.md` | 5 KB | Notes on the structural regularities of the k_slot layout, the residual "baseline" question, and known limitations. |
| `modal_op_audit.py` | 12 KB | Reproducible Modal script (~1 min A10G) that produces the audit report. |
| `modal_smart_probe.py` | 11 KB | Reproducible Modal script (~2 min A10G) that produces `compact.pkl`. |

## Format

```python
import pickle
d = pickle.load(open("compact.pkl", "rb"))
# For each K in {2, 3}:
for K in (2, 3):
    positions = d[f"K{K}_positions"]  # list of (n_nz, 2) int8 (row, col) positions
    values    = d[f"K{K}_values"]     # list of (65536, n_nz) fp16 codebook values
    # Reconstruct dense (k_max, 65536, 16, 16) fp16:
    import numpy as np
    k_max = len(positions)
    dense = np.zeros((k_max, 65536, 16, 16), dtype=np.float16)
    for k, (pos, val) in enumerate(zip(positions, values)):
        for i, (r, c) in enumerate(pos):
            dense[k, :, r, c] = val[:, i]
```

To decode a packed expert weight tile back into fp16, sum the per-slot codebook
lookups placed at each (bi, bj) block:

```python
# code: int16 (in_f/16, out_f/16, 16*K)
in_f = 2048  # or 512 for down_proj
out_f = 1024  # or 2048 for down_proj
K = 2  # or 3 for down_proj
w = np.zeros((in_f, out_f), dtype=np.float32)
bi_max, bj_max = in_f // 16, out_f // 16
for k in range(16 * K):
    idx = code[:, :, k].astype(np.int32) & 0xFFFF  # int16 -> uint16
    blocks = dense[k, idx]  # (bi_max, bj_max, 16, 16)
    w += blocks.transpose(0, 2, 1, 3).reshape(in_f, out_f)
```

**IMPORTANT — known limitation.** For a real expert whose codes activate all
262 K slots simultaneously, the above summation matches the CUDA op only up to
an unresolved additive term (per-projection norm ~4e3). This term is _not_
captured in the codebook (which stores deltas from `op(all-zeros code)`) and
could not be extracted in this session — the Modal workspace hit its spend
limit after the codebook extraction completed. See `LAYOUT_NOTES.md` for
the 30-second follow-up probe that would resolve it.

For a working end-to-end port that skips `escham_reconstruct` entirely (pre-
dequantized to fp16 on Modal, no runtime decode needed), see
[`KaedeTai/Qwen3.6-35B-A3B-Escha-W2-MLX`](https://huggingface.co/KaedeTai/Qwen3.6-35B-A3B-Escha-W2-MLX).

## Verified properties

- **Linearity** (superposition): `op(A+B) - op(0) = (op(A) - op(0)) + (op(B) - op(0))`
  holds exactly for up to 100 random slot activations.
- **(bi, bj)-invariance**: same code at any tile position produces the same
  16x16 delta (offset by (bi*16, bj*16)). Tested for corners including
  (bi=127, bj=63) vs (bi=0, bj=0).
- **Structural regularity**: the per-k_slot (row, col) support cycles with
  period 4 in k_slot (K=2).
- **Op signature**: `escham_reconstruct(Tensor packed, int in_f, int out_f,
  int K, bool cbA, bool mul1) -> Tensor` — one default overload, accepts
  leading batch dims on `packed`.

## Reproducing the extraction

Requires a Modal account and the `EschaLabs/escha-runtime-qwen3moe` wheel on
Hugging Face (public):

```bash
modal run modal_op_audit.py          # ~1 min A10G, produces OP_SIGNATURE_AUDIT.md
modal run modal_smart_probe.py       # ~2 min A10G, produces compact.pkl
```

The smart probe uses **~1024 op calls total** across both K values —
compared to the naive one-code-at-a-time sweep which would require
**~328 million op calls (91 h A10G, ~$100)**. The speedup comes from three
observations, each verified by the audit:

1. Op is exactly linear in codes -> many perturbations can be superposed
   in a single op call.
2. The codebook is (bi, bj)-invariant -> each of the 8192 tile-blocks in
   the (128, 64, 32) code tensor is a free "test bed" for a different
   codebook entry.
3. Different k_slots at the same (bi, bj) overlap in output positions ->
   we must use different (bi, bj) for different (k, v) probes, but that's
   fine since we have 8192 of them.

Net: 65,536 codes x 32 k_slots / 8192 blocks-per-op = 256 op calls for K=2.

## Credits

- **EschaLabs** — for open-weight Qwen3.6-35B-A3B-Escha-W2 and the reference
  runtime.
- **AQLM** (Egiazarian et al., 2024) — the residual-codebook quantization
  scheme that Escha builds on.

## License

Apache 2.0. Same as the base model.
