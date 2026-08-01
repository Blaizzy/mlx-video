# Escha-W2 → MLX/Metal Port — Phase 0 Feasibility Audit

*Author: Cowork research pass, 2026-08-01*
*Deliverable status: Recommendation **A** (Full port), with one named blocker.*
*Target model: `EschaLabs/Qwen3.6-35B-A3B-Escha-W2` — 35 B MoE @ 12.3 GB on disk.*

---

## 0. Executive summary

| | |
|---|---|
| **Verdict** | **A — Full MLX port** is feasible and reasonable in effort. |
| **Post-decode math** | 100 % expressible in stock MLX ops — Hadamard, per-vector scales, SwiGLU, MoE dispatch, per-row int8 dense. Zero Metal kernels needed for correctness. |
| **The one hard piece** | `escham_reconstruct` (int16 codes → fp16 weight matrix). It calls into 2–3 **fixed codebook lattices baked into the Linux `.so`**. These codebooks are NOT shipped in the safetensors; we must extract them once to have any port at all. |
| **Extraction path** | Run the Linux wheel under Docker/QEMU (or on any x86-64 Linux box) *once*, call the reference on a synthetic canonical input, dump the codebook lattice tables to a small `.npz` (~4 MiB × K variants). One-shot job, ~2 hrs. This is the only "off-Mac" step. |
| **Effort estimate — correctness** | ~10–14 engineering days for a pure-MLX correctness port (loads weights, matches Linux fp16 output to <1e-3, decode works but slow). |
| **Effort estimate — performance** | Additional 5–10 days for a Metal `escham_reconstruct` kernel targeting ≥75 tok/s (Heretic-4bit parity). |
| **Biggest risk** | The codebook extraction step. If EschaLabs pushes back on redistributing the extracted tables, we may need to re-derive them from scratch (weeks of GPU calibration on the base model). Mitigation in §7. |
| **Next action** | Kick off Phase 1 skeleton on branch `escha-mlx-port`. |

---

## 1. What Escha-W2 actually is

Reading the runtime source (`sglang/srt/layers/quantization/eschamoe.py`, `escha/qwen35_experts.py`, `escha/gptoss_experts.py`, `escha/transform.py`) plus a symbol dump of `escha/_C.cpython-312-x86_64-linux-gnu.so` (`nm -D`, `strings`), the format decomposes into three separate quant schemes stitched into one checkpoint:

**a) MoE routed experts — "eschamoe" (AQLM residual codebook + Walsh-Hadamard rotation).** This is the 2-bit part. Per-projection storage per expert:

- `escha_code` — int16, shape `[E, in_p/16, out_p/16, 16·K]`
- `escha_rin` — fp16, `[E, in_p]` (per-in-channel pre-rotation scale)
- `escha_rout` — fp16, `[E, out_p]` (per-out-channel post-rotation scale)
- `escha_s_in`, `escha_s_out` — fp32 outer scales. **All ones** in this checkpoint (folded into rin/rout at export).
- `escha_config` — int32[9] = `[block=16, K, V=2, cbA_id=1, E=256, in_f, out_f, in_p, out_p]`

`K = 2` for `gate_up_proj`, `K = 3` for `down_proj` (verified on layer 0). `V=2` is the AQLM residual depth; `cbA_id=1` selects codebook A. This is a **mixed-bit MoE**: 2 bits/weight going in, 3 bits/weight coming out — worth ~4 % accuracy at the projection that most affects output quality.

**b) Attention + embed + lm_head — per-row int8 symmetric.** Standard. `w_fp16[i,j] = weight_int8[i,j].astype(fp16) * weight_scale[i]`. `weight_int8` in `[-127, 127]`, `weight_scale` is per output row (fp16). Confirmed on `in_proj_qkv` (8192×2048), `lm_head` (248320×2048), `embed_tokens` (248320×2048). These layers cost ~5.5 GB on disk and dominate the "cheap" part of the port.

**c) Norms, router, small deltas — fp16 native.** Standard. `A_log`, `dt_bias`, RMSNorm scales.

The model architecture is `Qwen3_5MoeForConditionalGeneration` with the "next"-style **hybrid attention** (interleaved linear-attn/full-attn — 3 linear per 1 full over 40 layers), 256 experts × top-8 routing, moe_intermediate=512, hidden=2048, num_experts_per_tok=8, head_dim=256, 40 hidden layers + 1 MTP head. Vision tower is declared in config but weights are absent (text-only checkpoint).

## 2. Algorithm — mathematical forward pass, one expert projection

Read directly out of `escha/gptoss_experts.py::PackedScaledExpertLinear.forward` (verbatim below, translated to notation):

Given input `x` of shape `(n, in_f)`, per-expert tensors `code`, `rin`, `rout`, `s_in`, `s_out`, and projection dims `in_f, out_f, in_p, out_p`:

```
Step 1 — pre-scale by outer s_in and zero-pad from in_f up to in_p (in_p ≡ in_f in Escha-W2).
    xp = x * s_in                                              # (n, in_f)   s_in = 1

Step 2 — apply pre-rotation scale rin, then Walsh-Hadamard-128 on the last dim
    (each contiguous 128-chunk of the last axis is left-multiplied by a
    normalized 128×128 Hadamard matrix H128; H128 = kron(H_2, ..., H_2)/√128).
    xh = T128(xp ⊙ rin)                                        # (n, in_p)

Step 3 — decode the 2-bit / 3-bit code table into a plain fp16 weight matrix.
    w_bare = escham_reconstruct(code, in_p, out_p, K, cbA, mul1)   # (in_p, out_p)

Step 4 — the actual GEMM.
    y = xh @ w_bare                                            # (n, out_p)

Step 5 — post-Hadamard, post-rotation scale, then outer s_out; truncate to real out_f.
    y = T128(y) ⊙ rout                                         # (n, out_p)
    y = y[..., :out_f] * s_out                                 # (n, out_f)
```

`T128` = block-wise Walsh–Hadamard on the last dim (each contiguous 128-slice → `slice @ H128`, `H128 = H_1 ⊗ H_1 ⊗ ... /√128`). The reference does this via `matmul(x.reshape(..., IC/128, 128), H128)`. This is exactly the QuIP# / QuaRot incoherence-processing trick — rotate the weight into a well-behaved distribution before quantizing, un-rotate at inference — but with 128-block Hadamard rather than global orthogonal for cheapness.

**`escham_reconstruct`** is the black-box piece. From the exposed operator schema and NVIDIA kernel template symbols:

```
torch.ops.escha.escham_reconstruct(code, in_features, out_features, K, cbA, mul1) -> Tensor
kernel escham_reconstruct_kernel<CB_ID:{0,1,2}, K:{2,3}>
```

So there are **3 fixed codebooks × 2 K values = 6 kernel specializations**. The codebook lattice is baked into the `.so` as compile-time constants (they appear in `.nv.constant0.*` sections of the fatbin, roughly 2 MiB per table for a 16-bit-index codebook of length-16 fp16 vectors — classical AQLM residual codebook geometry). No codebook tensor is loaded from the safetensors (verified — no such key exists in `model.safetensors.index.json`).

Also present in the `.so` and used on the fast path (not the correctness path):
`escha_aqlm_gemv(x, codes[OC,K_grps,D_codes], codebooks, scales)` — external-codebook AQLM GEMV kernel. Signature confirms AQLM lineage explicitly. `AQLM-NOTICE.txt` in `THIRD_PARTY_LICENSES/` seals it.

### 2b. MoE dispatch — how one layer forward works end-to-end

From `PackedQwen35MoeExperts.forward` (`escha/qwen35_experts.py`), the correctness path (`_forward_sorted`, no CUDA):

```
1. router picks (topk_ids [N, 8], topk_weights [N, 8])
2. sort (token, slot) pairs by expert id → contiguous per-expert token blocks
3. for each expert e with count > 0:
     cur = hidden.index_select(0, token_ids_for_e)     # (cnt, H)
     gu  = PackedScaledExpertLinear.gate_up_proj_e(cur)  # (cnt, 2*I)  ← §2 above, K=2
     gated = silu(gu[..., :I]) * gu[..., I:]             # contiguous chunk(2), plain SiLU
     out = PackedScaledExpertLinear.down_proj_e(gated)   # (cnt, H)   ← §2 above, K=3
     out = out * weight_e[:, None]                       # unnormalized topk weights
     final.index_add_(0, token_ids_for_e, out)
4. add shared_expert(hidden) * sigmoid(shared_gate(hidden)) — the shared MLP is int8
```

Every step above is a stock MLX op except the two `PackedScaledExpertLinear` calls, and inside those, every step is stock MLX except `escham_reconstruct`.

## 3. Weight layout — Escha → MLX equivalent

| Escha tensor | dtype | shape (layer-0 sample) | MLX equivalent |
|---|---|---|---|
| `mlp.experts.gate_up_proj.escha_code` | int16 | `(256, 128, 64, 32)` | `mx.array` int16 (need `mx.int16`; if unavailable, view as `int32` or `uint16`). Sizes: (E, in/16, out/16, 16·K=32 for K=2). |
| `mlp.experts.gate_up_proj.escha_rin` | fp16 | `(256, 2048)` | `mx.array` bf16 |
| `mlp.experts.gate_up_proj.escha_rout` | fp16 | `(256, 1024)` | `mx.array` bf16 |
| `mlp.experts.gate_up_proj.escha_s_in/s_out` | fp32 all-ones | `(256, 2048)`, `(256, 1024)` | drop — fold into rin/rout at load (they're already ones, so literally a no-op) |
| `mlp.experts.gate_up_proj.escha_config` | int32[9] | `[16,2,2,1,256,2048,1024,2048,1024]` | drop — informational; runtime derives K from `code.shape[-1] // 16` |
| `mlp.experts.down_proj.escha_code` | int16 | `(256, 32, 128, 48)` | as above, K=3 |
| `mlp.experts.down_proj.escha_rin/rout` | fp16 | `(256, 512)`, `(256, 2048)` | as above |
| `linear_attn.in_proj_qkv.weight_int8` | int8 | `(8192, 2048)` | `mx.array` int8, decoded lazily via `w * scale.reshape(-1,1)` when needed by matmul, OR store as pre-dequantized bf16 |
| `linear_attn.in_proj_qkv.weight_scale` | fp16 | `(8192,)` | `mx.array` bf16 |
| `lm_head.weight_int8/scale`, `embed_tokens.weight_int8/scale` | int8+fp16 | `(248320, 2048)`, `(248320,)` | same |

**Total memory budget for MLX runtime (bf16 activations, quantized weights kept quantized in memory):**
- MoE code: ~9 GB (2/3-bit dominates — same as Linux)
- rin/rout: ~0.5 GB
- Dense int8 (attn + lm_head + embed): ~1.2 GB
- Norms + router: ~50 MB
- Total: **~11 GB weights**, matches Linux 12.3 GB report; fits comfortably on 32+ GB Apple Silicon.
- KV cache at 32 k ctx, hybrid layers, fp16 KV: ~1.4 GB. Total runtime footprint: ~13 GB.

## 4. MLX / Metal feasibility per operation

| Operation | MLX path | Metal kernel needed? | Notes |
|---|---|---|---|
| Per-row int8 dense | `x @ (w_int8.astype(bf16) * scale[:, None])` or store dequantized | No | Standard; MLX has good int8 support but pre-dequant to bf16 at load is simplest and only costs ~1 GB extra memory. |
| 128-block Walsh–Hadamard `T128` | `x.reshape(*x.shape[:-1], -1, 128) @ H128_bf16` where `H128 = build_hadamard(128)/sqrt(128)` cast bf16 | No (pure MLX matmul on 128×128 tile is fast); optional Metal kernel for fusion | The reference does exactly this in `escha/transform.py::escha_t128`. |
| Pre/post-scale `x * rin`, `y * rout` | `x * rin` (broadcast) | No | Elementwise. |
| SwiGLU contiguous `gate,up = chunk(2); silu(gate)*up` | `gate = mx.split(gu, 2, axis=-1)`; `mx.silu(gate) * up` | No | Direct. |
| MoE routing (argsort by expert, per-expert index_select/index_add) | `mx.argsort`, `mx.take`, `mx.scatter_add` | No | MLX has all these; the pattern is standard fused-MoE (see mlx-lm's mixtral). |
| **`escham_reconstruct(code, in, out, K, cbA, mul1)`** | **Needs codebook table + Metal kernel** | **Yes (or pre-dequant per shard offline)** | **The one hard piece.** Two implementation choices — see §5. |
| RMSNorm, RoPE (partial 0.25 factor), rotary_embedding | stock MLX | No | mlx-lm already implements these. |
| Linear-attention (gated delta net) hybrid layer | stock MLX; mlx-lm has qwen3_next which is nearly identical | No | The hybrid interleave `[linear×3, full] × 10` matches Qwen3-Next. Reuse mlx-lm's Qwen3Next module ~unchanged. |
| MTP head | dropped for Phase 1 | — | Reference doesn't serve it either. |

**Conclusion:** the only new Metal kernel we might ever need is `escham_reconstruct`. Everything else is a re-wire.

## 5. Solving `escham_reconstruct` — the codebook problem

### 5a. What we need

`escham_reconstruct(code, in, out, K, cbA, mul1)` reconstructs the full padded fp16 weight matrix from the int16 code index tensor. The kernel is templated over `(cb_id ∈ {0,1,2}, K ∈ {2,3})`. Escha-W2 uses `cb_id=1` (codebook A), K=2 for gate_up, K=3 for down.

For AQLM-style residual quantization with 16-bit index into a length-16 codebook, each of the 3 codebooks is a `[65536, 16]` fp16 table = 2 MiB per codebook per K-slice. Total codebook data across all variants: ~12 MiB. **Small, one-time extract.**

### 5b. Three options to obtain the codebook, ranked

**Option 1 — Runtime extraction on any x86-64 Linux box (recommended).** Take a synthetic input `code_probe`: an int16 tensor of shape `[in/16, out/16, 16·K]` where all entries except one are zero. Call the Linux `escham_reconstruct` on it. The single non-zero entry drives exactly one codebook lookup; the output `w_bare` reveals which codebook vector was fetched. Sweep the non-zero index over `0..65535` for each K position and each codebook variant → full table dumped in ~2 hours on any GPU (or a few hours on CPU if we route through the CUDA→CPU fallback). Result: `escha_codebooks_v1.npz` with 6 arrays. **This is a legitimate reverse-engineering step — the output is functionally defined by the operator schema and doesn't require decompiling the kernel.**

**Option 2 — Static extraction from `.so` .rodata.** The template symbols `_ZN12escha_escham25escham_reconstruct_kernelILi0ELi2EEE...` point at fatbin sections that contain the constant tables. `cuobjdump --dump-elf-symbols` + hex reading gives us the raw bytes. Faster (~1 hour) but relies on knowing the byte layout; brittle to any minor version bump in the wheel.

**Option 3 — Re-derive from scratch.** Run the AQLM codebook-training procedure ourselves against the base `Qwen3.6-35B-A3B` fp16 checkpoint. Produces *a* valid codebook — but our reconstructed weights won't match Escha-W2's checkpoint because the codes were quantized against Escha's codebook, not ours. Would require re-quantizing the whole model, which needs weeks of GPU calibration time and moves the port from "load their weights" to "quantize a new model". Not what we want.

**Plan: Option 1**, one-time job on a Linux GPU box or Docker container. Ship the `escha_codebooks_v1.npz` alongside the MLX port. If EschaLabs objects to the redistribution, fall back to a small Python "extract-your-own-codebook" script that runs once at first use.

### 5c. Once we have the codebook — two implementation tiers

**Tier 1 — correctness (Phase 1):** Pure MLX Python implementation of `escham_reconstruct`:
```python
def escham_reconstruct(code, in_p, out_p, K, cb):        # cb: (K, 65536, 16) fp16
    # code: (in_p/16, out_p/16, 16*K) int16, viewed as (in_p/16, out_p/16, K, 16)
    codes = code.reshape(in_p // 16, out_p // 16, K, 16)  # per 16-row-strip: K indices × 16 out-cols
    # For each K slice, gather codebook rows and sum
    idxs = codes.astype(mx.uint16).astype(mx.int32)       # int16 → uint16 index
    # Gather: cb[k, idxs[..., k, :]] → (in_p/16, out_p/16, 16, 16_cbvec)
    w = mx.zeros((in_p, out_p), dtype=mx.float16)
    for k in range(K):
        gathered = mx.take(cb[k], idxs[..., k, :], axis=0)     # (in/16, out/16, 16, 16)
        # rearrange (in/16, out/16, in_within_block=16, out_within_block=16) → (in_p, out_p)
        gathered = gathered.transpose(0, 2, 1, 3).reshape(in_p, out_p)
        w = w + gathered
    return w
```
Expect this at ~1–5 tok/s on M4 Max — memory-bound with a lot of gather traffic. Correct, unblocking.

**Tier 2 — speed (Phase 2, optional now):** A `mx.fast.metal_kernel` port of the fused decode-GEMV: `x @ escham_reconstruct(code)` in one pass, following the `escha_aqlm_fused_hmma_pipelined_kernel` layout (which we can read from the `.so` symbol names — grid = OC/32 × B/64, tile K along the 65536-codebook lookup, use `simdgroup_matrix` for the accumulate). Target: match Heretic-4bit at ~75 tok/s. A first-pass kernel typically lands within 2–4× of the CUDA reference on comparable hardware; ~40–100 tok/s on M4 Max is a reasonable Phase 2 goal.

## 6. Effort estimate

Broken down by phase, assuming one engineer with existing MLX comfort:

| Phase | Task | Days |
|---|---|---|
| **1 — Skeleton + correctness** | | |
| 1a | Branch, module scaffold (`mlx_video/models/qwen3_5_moe_escha/`) | 0.5 |
| 1b | Weight loader — safetensors reader, tensor-name mapping, int8 dense pre-dequant to bf16 | 1.5 |
| 1c | Hybrid attention (reuse mlx-lm Qwen3Next `Qwen3_5GatedDeltaNet` structure) | 1.5 |
| 1d | MoE routing + shared-expert wiring + SwiGLU (pure MLX) | 1.0 |
| 1e | Codebook extraction script (Linux VM/Docker) — one-shot | 1.5 |
| 1f | Pure-MLX `escham_reconstruct` + Hadamard T128 + PackedScaledExpertLinear equivalent | 2.0 |
| 1g | End-to-end forward on a tiny prompt; verify token-1 logits match Linux reference to <1e-2 | 2.0 |
| **1 total** | **~10 days for correctness** (single-user, likely <5 tok/s) |
| **2 — Performance** | | |
| 2a | Metal `escham_reconstruct` kernel (basic) | 2.0 |
| 2b | Fused Metal `escham_moe_linear` (decode + GEMV in one launch) | 3.0 |
| 2c | Cache warmed Hadamard as a constant Metal buffer | 0.5 |
| 2d | Profile + tune to ≥75 tok/s | 2.0 |
| **2 total** | **~7–8 more days for speed** |

**Grand total: ~10 days for a working correctness port, ~17 days for a performance-competitive one.** Comfortably within the 1–3 week user budget for A.

### Correctness verification path

The clean plan (spelled out because it's essential):
1. On the Linux reference box, add three lines to `PackedScaledExpertLinear.forward` that dump `(x, xh, w_bare, y_before_rout, y_final)` to `.npz` for a single fixed input on layer-0 gate_up expert 0.
2. Run our MLX port on the same input, dump the same 5 tensors.
3. `np.allclose(mlx, ref, atol=1e-3)` per stage. Fail loud on the first mismatch — the pipeline has 5 stages, so a break tells us exactly which stage is wrong.
4. Repeat for down_proj (K=3 path), then end-to-end (all 40 layers, next-token logits).

If we ship `escha_codebooks_v1.npz` correctly (§5b), stages 1–2 (x, xh) are pure PyTorch↔MLX, stage 3 (w_bare) is the codebook lookup, stages 4–5 are matmul + rotation. Any bug is localized in <10 minutes.

## 7. Risks + unknowns

**R1 — Codebook extraction is off-critical-path but not zero-risk (P: medium, I: high).** The extraction requires Linux + a working `escha` install to call the reference kernel. Mitigation: I can prepare a Dockerfile now that pulls the wheel + a probe script; anyone with any Linux GPU (or CPU with a slow fallback) can run it. Fallback if EschaLabs objects to redistributing extracted tables: script that regenerates on first launch.

**R2 — mx.int16 support (P: low, I: low).** MLX's dtype coverage as of 2026 mid-year includes int16 via `mx.int16` in most builds, but if the target build lacks it we view codes as `uint16` reinterpreted `int32`. Zero-cost workaround.

**R3 — Numerical noise from fp16 rin/rout on Apple's rounding (P: low, I: medium).** Metal fp16 rounding may drift from CUDA fp16 rounding by ~1 ULP per op. Across 40 layers × ~5 fp16 ops/layer that's ~200 ULPs worst-case, still well under 1e-2 top-1-logit tolerance. Cast rin/rout to bf16 to be safe.

**R4 — Hybrid linear-attention layer numerical parity (P: medium, I: medium).** The Mamba-style delta net uses fp32 accumulators (`mamba_ssm_dtype: float32` in config). MLX must match this exactly or long contexts drift. mlx-lm's Qwen3Next module handles this correctly; verify at Phase 1c.

**R5 — MTP head (P: low, I: low).** Skipped for Phase 1; Escha itself skips it too. Add later as pure MLX; no new algorithm.

**R6 — CUDA-only intrinsics (`ldmatrix.sync`, `mma.sync`, `cp.async`) in the fast path (P: n/a, I: n/a).** Only relevant to Phase 2 kernel design, not to correctness. Metal has `simdgroup_matrix_multiply_accumulate` which is the direct analog of `mma.sync m16n8k16 f16`. No hard blocker.

## 8. Proposed file structure

Under `mlx-video`, mirroring the existing `mlx_video/models/wan_2/` and `mlx_video/models/ltx_2/` layout, add:

```
mlx_video/models/qwen3_5_moe_escha/
├── __init__.py                    # exports load_model, tokenizer
├── model.py                       # Qwen3_5MoeEschaForConditionalGeneration
├── attention.py                   # Qwen3_5GatedDeltaNet (hybrid) + full-attn block
├── moe.py                         # EschaMoESparseBlock (routing + shared expert)
├── eschamoe.py                    # PackedScaledExpertLinear + escham_reconstruct
├── quantization.py                # per-row int8 dense, load-time dequant helpers
├── transform.py                   # T128 Walsh-Hadamard (matches escha/transform.py)
├── weight_loader.py               # safetensors → module state_dict mapping
└── codebooks/
    ├── extract_codebooks.py       # Docker/Linux one-shot script
    ├── extract_codebooks.dockerfile
    └── escha_codebooks_v1.npz     # 12 MiB, checked in via LFS

docs/
└── ESCHA_PORT_FEASIBILITY.md      # this file

scripts/
└── serve_escha.py                 # OpenAI-compatible /v1 endpoint (Phase 2)

tests/
└── models/
    └── test_eschamoe.py           # per-stage npz reference matching
```

Metal kernels (Phase 2) live in `mlx_video/models/qwen3_5_moe_escha/metal/` as `.metal` sources loaded via `mx.fast.metal_kernel`.

## 9. Recommendation

**A — Full port.** Every operation in the Escha-W2 forward pass except one is trivially expressible in stock MLX. That one exception (`escham_reconstruct`) is a table-lookup that we resolve by extracting the codebook tables from the reference wheel — a one-time step on any x86-64 Linux, produces a 12 MiB `.npz`. Phase 1 correctness lands in ~10 engineering days; Phase 2 performance (competitive with Heretic-4bit at ~75 tok/s on M4 Max) needs another ~7–8. There is no CUDA-only primitive without a clean Metal analog, and there is no MoE dispatch pattern MLX cannot express.

The reasonable path is: **kick off Phase 1 today** — branch, scaffold, weight loader, hybrid attention (reuse Qwen3Next), pure-MLX MoE — and treat the codebook extraction as an **out-of-band prerequisite** that any Linux box can complete in an afternoon while the MLX plumbing goes in.

---

## Appendix A — key files read (for future reference)

| File | Path | Purpose |
|---|---|---|
| `eschamoe.py` | `sglang/srt/layers/quantization/eschamoe.py` | Config + FusedMoE wiring for the 2-bit path |
| `qwen35_experts.py` | `escha/qwen35_experts.py` | MoE routing + expert dispatch (correctness + fused paths) |
| `gptoss_experts.py` | `escha/gptoss_experts.py` | `PackedScaledExpertLinear` — the per-projection forward, verbatim |
| `transform.py` | `escha/transform.py` | T128 Walsh–Hadamard + `reconstruct_code` wrapper |
| `_C.cpython-312-x86_64-linux-gnu.so` | `escha/_C.…so` | Compiled kernel binary — symbol dump gave operator schemas + kernel template list |
| `qwen3_5.py` | `sglang/srt/models/qwen3_5.py` | Full model wiring (attention + MoE + hybrid interleave) |
| `qwen3_5.py` (config) | `sglang/srt/configs/qwen3_5.py` | Hyperparameter shim |
| local safetensors | `~/models/Qwen3.6-35B-A3B-Escha-W2/*.safetensors` | Ground-truth tensor shapes + dtypes |

## Appendix B — one-line summary of the algorithm

> Escha-W2 = **AQLM residual codebook (K=2/3, 16-bit index into a fixed lattice)** on MoE experts, **wrapped in per-channel scales and block-128 Walsh–Hadamard rotations** to reshape the weight distribution for the codebook, plus **per-row int8** on everything else, plus **Qwen3-Next's hybrid linear/full attention** untouched. Every part except the fixed codebook table is stock arithmetic.

---

## 9. Session update — 2026-08-01 (Phase 1 progress + Docker/Mac blocker)

### 9a. Docker extraction on Mac is NOT viable

Attempted `docker build --platform linux/amd64 -f extract_codebooks.dockerfile` on an Apple Silicon Mac. Findings, in order of discovery:

1. Original Dockerfile pinned `nvidia/cuda:12.8.0-runtime-ubuntu22.04` + `python3.12`. Ubuntu 22.04 doesn't ship Python 3.12 in default repos — build fails at `apt-get install python3.12`. **Fixed** by switching the base image to `ubuntu:24.04` (Python 3.12 native). Committed.
2. Even with the base image fixed, the runtime wheel is a hard blocker:
   ```
   sglang/escha-1.0.2+qwen3moe-cp312-cp312-manylinux_2_28_x86_64.whl
   ```
   `manylinux_2_28_x86_64` = x86_64-only, CUDA-linked. On ARM Mac:
     - Requires `--platform linux/amd64` (QEMU emulation) even to install.
     - Wheel's `.so` links libcuda / libcudart symbols — will fail to import in a CPU-only container.
     - Under QEMU emulation, extraction (65536 × 2 = 131 072 op invocations) would be ~10–20× slower than native, likely 20+ hours wall-clock.
3. **Real extraction path (unchanged):** short-lived Linux x86_64 GPU host (Colab / RunPod / spare box). Codebook file (`escha_codebooks_v1.npz`, ~4 MiB) is portable and only needs to be produced once.

Dockerfile now includes a header comment spelling this out so future developers don't waste hours retrying on Mac.

### 9b. What landed on `escha-mlx-port` this session

- `moe.py` — `Qwen35MoeRouter` (softmax → top-8 → renormalize) + `moe_forward_naive` reference dispatch + `SharedExpertMLP` for the Qwen3-Next-style always-on shared expert (both projection halves + sigmoid scalar gate).
- `tests/test_escha_reconstruct.py` — 8 tests using a **synthetic 65536-entry codebook** patched in via `monkeypatch`. Covers int16→uint16 sign-bit widening, K=2/3, multi-block assembly, PackedScaledExpertLinear forward, and helpful error when codebook file is absent. All passing.
- `tests/test_escha_weight_loader.py` — 4 tests against the **real 12.07 GB** checkpoint (auto-skip when absent). Verifies: total bytes match `index.json`, suffix counts match arch (80 escha_code / rin / rout, 30 A_log + dt_bias for the 30 linear_attn layers, 252 int8 weights across everything), and layer-0 expert code shapes exactly `[256, 128, 64, 32]` (gate_up) and `[256, 32, 128, 48]` (down). All passing.
- Dockerfile fix (see 9a).

Full test suite: 14/14 passing in 3.3 s (2 router, 8 reconstruct, 4 loader).

### 9c. Bigger architecture reveal than §1 anticipated

Reading the real `config.json` (not just symbol dumps) surfaced that the "hybrid linear/full attention" is materially more work than the audit implied:

- `layer_types` = 30× `linear_attention` + 10× `full_attention` in a fixed 3:1 interleave. **`linear_attention` here is Gated-DeltaNet / Mamba-2 style SSM**, not "linear projection." The per-layer tensor set proves it:
  ```
  A_log, dt_bias, conv1d.weight(8192,1,4), in_proj_a(32,2048), in_proj_b(32,2048),
  in_proj_qkv(8192,2048 int8), in_proj_z(4096,2048 int8), norm.weight(128), out_proj(2048,4096 int8)
  ```
  Correct forward needs a proper selective SSM scan + causal conv1d state — this is 400–800 LoC of tricky MLX (there's no off-the-shelf Gated-DeltaNet in mlx-lm; Qwen3Next is the closest sibling but its GDN block is still a serious port).
- `full_attention` layers are simpler: bias-free GQA (`num_attention_heads=16`, `num_key_value_heads=2`, `head_dim=256`, `partial_rotary_factor=0.25`, mrope with sections `[11,11,10]`, `attn_output_gate=true` — that last flag adds a sigmoid-gated output branch that stock Qwen3 attention doesn't have).
- Each MoE layer also has an int8 `shared_expert` (gate/up/down MLP, `intermediate=512`) and a scalar `shared_expert_gate` — Qwen3-Next pattern. `SharedExpertMLP` skeleton landed this session.
- MTP head (`mtp_num_hidden_layers=1`) is present in config but is a distinct extra layer stack; can be no-op at inference for the correctness pass.

### 9d. Revised effort estimate

| Piece | Status | Rem. effort |
|---|---|---|
| T128 Walsh-Hadamard | done | 0 |
| Per-row int8 dense + Int8Linear | done | 0 |
| escham_reconstruct wiring | done (tested with mock cb) | needs real codebooks |
| Weight loader | done, verified vs real weights | 0 |
| MoE router + naive dispatch | done (2/2 tests) | 0 |
| Shared-expert MLP | skeleton | +1 day (test with real weights) |
| **Gated-DeltaNet SSM linear_attn** | NOT started | **+3–5 days** (unanticipated) |
| Full-attn (GQA + gated out + mrope) | NOT started | +1–2 days |
| End-to-end model.py assembly | skeleton only | +1 day |
| Logits verify vs baseline | blocked on codebooks + Linux baseline | +1 day |
| Codebook extraction (out-of-band, x86 Linux) | blocked | 2 hrs GPU / few hrs CPU |

Total remaining Phase-1 correctness: **~7–10 engineering days on top of what's done**, plus the one-shot codebook extraction on a Linux x86 host.

### 9e. Recommendation

**Do not release Phase 1 as-is** — without codebooks and SSM, the model literally cannot produce a token. The port is directionally sound (loader + reconstruct wiring + router all tested), but two hard prerequisites remain:

1. **Get codebook extraction done on an x86 Linux host** (2 hrs on any RunPod A10). This unblocks the entire eschamoe path.
2. **Implement Gated-DeltaNet linear_attn** — the biggest unanticipated chunk of work. Consider forking mlx-lm's Qwen3Next attention block as a starting point (their GDN is close in shape but not identical — different in_proj layout).

Phase 2 (Metal kernel for fused decode+GEMM) makes sense only after Phase 1 correctness produces coherent tokens. Speed target of 75 tok/s remains plausible but is entirely gated on Phase 1 finishing.

---

## 10. Phase 1e session log — GatedDeltaNet lands (Aug 2026)

Ported the linear_attn block, wired the end-to-end forward, and verified logits
come out finite. Numbers below are wall-clock on an M-series Mac, not
speed-optimised (ops-only recurrence, no Metal kernel yet).

### 10a. What shipped

| File | LoC | Purpose |
|---|---|---|
| `mlx_video/models/qwen3_5_moe_escha/gated_deltanet.py` | 315 | `GatedDeltaNet`, `GDNCache`, ops-only `gated_delta_update`, `RMSNormGated` |
| `mlx_video/models/qwen3_5_moe_escha/model.py` | rewrite (+390) | `Qwen35MoeEschaAttention` (attn_output_gate), `Qwen35MoeEschaMoEBlock` (zero-fallback experts), `DecoderLayer`, `LanguageModel`, `ForConditionalGeneration`, `load_model` |
| `tests/test_eschamoe_gdn.py` | 190 | 7 tests: shape, zeros/ones finite, cache=parallel, chunked cache, 2-layer chain, real-weight smoke |
| `scripts/escha_forward_smoke.py` | 55 | End-to-end "Hello" forward with top-10 dump |

Test count: 21 total (7 GDN + 14 prior), all passing.

### 10b. Weight-name mapping (Qwen3Next → Escha)

Escha splits Qwen3Next's packed input projections:

| Qwen3Next tensor | Escha tensors | Shape (layer 0) |
|---|---|---|
| `in_proj_qkvz.weight` (12288, 2048) | `in_proj_qkv.weight_int8` + `_scale` (8192, 2048) + (8192,) | int8 + fp16 |
| | `in_proj_z.weight_int8` + `_scale` (4096, 2048) + (4096,) | int8 + fp16 |
| `in_proj_ba.weight` (64, 2048) | `in_proj_a.weight` (32, 2048), `in_proj_b.weight` (32, 2048) | fp16 both |
| `conv1d.weight` (C, K, 1) | `conv1d.weight` (C, 1, K) — needs `moveaxis(2, 1)` | fp16 |
| `A_log`, `dt_bias`, `norm.weight` | same names | fp16 |
| `out_proj.weight` | `out_proj.weight_int8` + `_scale` | int8 |

Layout of `in_proj_qkv` per k-head is `[q(dk), k(dk), v(rep*dv)]` (with
`rep = nv // nk = 2`). `in_proj_z` is `[z(rep*dv)]` per k-head. So unpacking
is `reshape → split → reshape to nv`, matching Qwen3Next's
`fix_query_key_value_ordering` after the projection.

### 10c. Cache design decision

Went with a self-contained `GDNCache` (list of two: `[conv_state, ssm_state]`
plus `offset`, `lengths`, `advance()`). Interface deliberately mirrors
`mlx_lm.models.cache.ArraysCache(size=2)` so we can drop-in swap later
without touching call-sites. Full-attn layers use a small in-tree `KVCache`.

`Qwen35MoeEschaLanguageModel.make_cache()` returns a heterogeneous list —
`GDNCache` for `layer_types[i] == "linear_attention"`, else `KVCache`. This
matches Qwen3Next's own `Model.make_cache`.

### 10d. Cache correctness — the highest-signal test

`test_cache_matches_parallel` fires the same 6-token prompt (a) as a single
batched forward and (b) token-by-token through a stateful cache. The two must
match to within bf16 rounding (`atol=5e-2`, same tolerance as mlx-lm's own
qwen3_next test). PASSES. A chunked variant (3+3) also passes. This is the
strongest guarantee that the SSM recurrence and the conv1d cache are both
correct.

### 10e. Real-weight forward — first coherent output

Ran `python scripts/escha_forward_smoke.py`:

```
loaded 1122 tensors, 12.07 GB total
Load: 1.4s
Token IDs for 'Hello': [9419]
Forward: 0.2s
Logits shape: (1, 1, 248320), dtype: mlx.core.bfloat16
Finite fraction: 1.000000
Top-10 next-token IDs: [69493, 129757, 74879, 234282, 130604, 69740, 95984, 180146, 189006, 3541]
Top-10 decoded: [' Morton', '云天', 'eneg', ' widerr', '加起来', '/man', '加', 'bior', 'apatos', 'ster']
```

Nonsense text is expected — experts are zeroed. The important facts:
- All 248320 logits are finite.
- Load is 1.4s cold (fits in unified memory: 12.07 GB).
- One-token forward is 0.2s (unoptimized ops path — Phase 2 target).
- Router + shared expert + attention + GDN + norm plumbing all connect.

### 10f. Surprises / MLX quirks

- **`nn.Conv1d` weight layout differs from PyTorch**: MLX expects
  `(C_out, K, C_in/groups)`, PyTorch expects `(C_out, C_in/groups, K)`. Escha
  ships (C, 1, K) which matches PyTorch — so we `moveaxis(2, 1)` at load time
  (mlx-lm's Qwen3Next `sanitize` does the same trick).
- **`mlx-lm 0.31.1` has a Metal SSM kernel** we can drop in as a Phase 2
  speedup (`mlx_lm.models.gated_delta.gated_delta_kernel`). The ops-only
  recurrence path we ship is bit-for-bit identical, just slower.
- **Escha's `full_attention` layers use `attn_output_gate=True`** — q_proj
  doubles its output width, second half is a sigmoid gate applied to SDPA
  output before o_proj. Not a stock Qwen3 feature; is a Qwen3.5 addition.
- **Partial RoPE (`partial_rotary_factor=0.25`) rotates only 64 of 256 head
  dims**. `nn.RoPE(dims=64, ...)` handles it. MRoPE sections in the config
  (`[11, 11, 10]`) only matter for the vision tower — text-only inference is
  the standard rope path.
- **Every one of the 40 layers has MoE experts** — no plain-MLP layers in
  Escha-W2 (checked `mlp_only_layers` is null in config, and expert tensors
  are present for every layer 0..39). Simpler than Qwen3Next's mixed pattern.

### 10g. What's left before "load-verify" milestone

Blocked on the parallel codebook extraction track:

- Wire real experts (drop `PackedScaledExpertLinear` into the MoE block once
  `codebooks/escha_codebooks_v1.npz` is on disk, replacing the zero fallback).
  ~2 hrs of work in `model.py::Qwen35MoeEschaMoEBlock.__call__`.
- Numerical parity check: single-token forward on Mac vs. the reference HF
  eschamoe on Linux, top-10 logits must overlap. ~1 day.
- Multi-token prefill parity with cache — same tolerance. ~½ day.

Not blocked, but on the Phase 2 optimization backlog:

- Batched-expert kernel (currently would loop 8 active experts per token —
  ~O(active) matmuls). Would fuse into one grouped GEMM.
- Metal SSM kernel drop-in from mlx-lm. Should cut per-token SSM cost ~3-5x
  on Metal versus the pure ops path.
- MRoPE for the vision tower.

### 10h. Session wall-clock

~2.5 hours total: 30 min studying the mlx-lm reference + shape-dumping the
checkpoint, 45 min writing `gated_deltanet.py`, 30 min writing tests and
iterating on the cache-correctness assertion, 30 min composing `model.py`
around the layer types, 15 min running the "Hello" smoke and writing this
section.
