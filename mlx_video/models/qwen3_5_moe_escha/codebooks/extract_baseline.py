"""Extract the shape-dependent baseline w0 (= op(all-zeros code)) from Option-B
M matrices, by inverting the escha pipeline.

Pipeline:
    M = A @ w_bare @ B
where A = t128(diag(rin))     (block-diag 128-block-of-rows applied on rows)
      B = ...                 (t128 on cols, then multiply by rout on cols)

Inversion:
    w_bare = A^{-1} @ M @ B^{-1}
    A^{-1} @ X = t128 on rows of X, then divide each row by rin[row]
    X @ B^{-1} = divide each col of X by rout[col], then t128 on cols of X

Since w_bare = w0 + sum_{k} cb[K, k, code[bi, bj, k]] (from smart_probe),
and cb is expert-invariant, we compute w0 = w_bare_ref - delta_mlx once per shape.

Verify w0 is expert-invariant by extracting it from multiple experts and
comparing. If invariant (up to fp16 rounding), save as codebooks/baseline_v2.npz.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors import safe_open

from mlx_video.models.qwen3_5_moe_escha.eschamoe import escham_reconstruct
from mlx_video.models.qwen3_5_moe_escha.transform import t128


ORIG_DIR = Path("/Users/kaede/models/Qwen3.6-35B-A3B-Escha-W2")
DEQUANT_DIR = Path("/Users/kaede/models/Qwen3.6-35B-A3B-Escha-W2-MLX-dequant/dequant_v1")


def _load_expert(layer: int, expert: int, proj: str) -> dict:
    idx = json.loads((ORIG_DIR / "model.safetensors.index.json").read_text())
    wm = idx["weight_map"]
    prefix = f"model.language_model.layers.{layer}.mlp.experts.{proj}"
    out = {}
    for suf in ("escha_code", "escha_rin", "escha_rout"):
        key = f"{prefix}.{suf}"
        shard = wm[key]
        with safe_open(ORIG_DIR / shard, framework="numpy") as f:
            out[suf] = mx.array(f.get_tensor(key)[expert])
    return out


def _load_M(layer: int, expert: int, proj: str) -> mx.array:
    key = f"layer_{layer}.expert_{expert}.{proj}.weight"
    return mx.load(str(DEQUANT_DIR / f"layer_{layer:02d}.safetensors"))[key]


def invert_pipeline(M: mx.array, rin: mx.array, rout: mx.array) -> mx.array:
    """Recover w_bare from M = t128(t128(I, pre=rin) @ w_bare, post=rout).

    Forward-pass algebra (with I = eye(in_f) fed through):
        M[i, k] = rin[i] * rout[k] * (t128(t128(w_bare, cols), rows))[i, k]
    i.e.  M = diag(rin) * H_row(H_col(w_bare)) * diag(rout)  (elementwise scalings)

    Inversion (t128 is self-inverse in either axis):
        h = M / rin[:, None] / rout[None, :]
        g = t128(h)                    # apply H128 on cols (undoes H_col in the pipeline)
        w_bare = t128(g.T).T           # apply H128 on rows (undoes H_row)
    """
    M = M.astype(mx.bfloat16)
    rin = rin.astype(mx.bfloat16)
    rout = rout.astype(mx.bfloat16)
    h = M / rin[:, None] / rout[None, :]
    g = t128(h)                  # H128 on cols
    w_bare = t128(g.T).T         # H128 on rows via transpose
    return w_bare.astype(mx.float16)


def extract_w0_for_shape(proj: str, in_f: int, out_f: int, K: int,
                         experts_to_test: list[tuple[int, int]]) -> mx.array:
    """Extract w0 for a given (in_f, out_f, K) shape from multiple experts.
    Returns the mean w0. Also asserts variance across experts is small.
    """
    print(f"\n=== extracting w0 for {proj} (in={in_f}, out={out_f}, K={K}) ===")
    w0_samples = []
    for layer, expert in experts_to_test:
        ex = _load_expert(layer, expert, proj)
        code = ex["escha_code"]
        rin = ex["escha_rin"]
        rout = ex["escha_rout"]
        M = _load_M(layer, expert, proj).T                # (in_f, out_f)

        w_bare_ref = invert_pipeline(M, rin, rout)        # (in_f, out_f)
        w_bare_mlx = escham_reconstruct(code, in_f, out_f, K, cb_id=1, mul1=False)

        mx.eval(w_bare_ref, w_bare_mlx)
        w0_est = w_bare_ref.astype(mx.float32) - w_bare_mlx.astype(mx.float32)
        norm_est = float(mx.linalg.norm(w0_est))
        norm_ref = float(mx.linalg.norm(w_bare_ref.astype(mx.float32)))
        norm_mlx = float(mx.linalg.norm(w_bare_mlx.astype(mx.float32)))
        print(f"  L{layer} E{expert}: |w_bare_ref|={norm_ref:.3e} "
              f"|w_bare_mlx|={norm_mlx:.3e} |w0_est|={norm_est:.3e}")
        w0_samples.append(w0_est)

    # Check consistency
    stacked = mx.stack(w0_samples, axis=0)                  # (n, in_f, out_f)
    mean_w0 = mx.mean(stacked, axis=0)
    std_w0 = mx.std(stacked, axis=0)
    max_std = float(mx.max(std_w0))
    mean_norm = float(mx.linalg.norm(mean_w0))
    print(f"  MEAN |w0|={mean_norm:.3e}   MAX per-element std={max_std:.3e}")
    # If w0 is truly shape-only (expert-independent), std should be ~fp16 noise (~1e-3).
    return mean_w0


def main():
    experts_to_test = [(0, 0), (0, 1), (0, 5), (5, 3), (20, 10)]
    w0_gu = extract_w0_for_shape("gate_up_proj", 2048, 1024, 2, experts_to_test)
    w0_dn = extract_w0_for_shape("down_proj", 512, 2048, 3, experts_to_test)

    out_path = Path("/Users/kaede/mlx-video/mlx_video/models/qwen3_5_moe_escha/codebooks/layout_v2/baseline_v2.npz")
    np.savez(out_path,
             w0_K2=np.array(w0_gu.astype(mx.float16)),
             w0_K3=np.array(w0_dn.astype(mx.float16)))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
