"""Test whether different k-slots at the same (bi,bj) sum additively.

The extraction assumes:
  block(bi,bj) of op(C) - w0 = sum_k cb_pattern[k, C[bi,bj,k]]

For this to hold, k-slots at the same block must be additive.
Test: at fixed (bi=0, bj=0), set p[0,0,k=0]=v0 alone → delta_A.
                                    set p[0,0,k=1]=v1 alone → delta_B.
                                    set BOTH               → delta_AB.
If additive: delta_AB == delta_A + delta_B.

Also test full-density additivity across all k for one (bi, bj):
random code at (0,0,:) → single-block prediction vs actual.
"""

from __future__ import annotations

import modal


WHEEL_REVISION = "1.0.2+qwen3moe"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04",
        add_python="3.12",
    )
    .apt_install("curl", "binutils", "git", "ca-certificates")
    .pip_install("wheel", "pip", "setuptools")
    .pip_install(
        "torch==2.9.*",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install("numpy", "safetensors", "huggingface_hub[cli]", "hf_transfer")
    .run_commands(
        f"echo escha wheel revision: {WHEEL_REVISION}",
        "mkdir -p /escha",
        "hf download EschaLabs/escha-runtime-qwen3moe --include 'sglang/*' --local-dir /escha",
        "pip install --no-deps /escha/sglang/escha-*.whl",
    )
)

app = modal.App("escha-kslot-interact", image=image)


@app.function(gpu="A10G", timeout=600, memory=16 * 1024)
def check() -> dict:
    import numpy as np
    import torch
    import escha  # noqa: F401

    op = torch.ops.escha.escham_reconstruct
    device = "cuda"
    in_f, out_f, K = 2048, 1024, 2
    cshape = (128, 64, 32)
    bi_max, bj_max, k_max = cshape

    p0 = torch.zeros(cshape, dtype=torch.int16, device=device)
    w0 = op(p0, in_f, out_f, K, True, False).detach().cpu().numpy().astype(np.float32)

    def block(w, bi, bj):
        return w[bi*16:(bi+1)*16, bj*16:(bj+1)*16]

    # Test 1: pairwise k-slot additivity at (0,0).
    bi, bj = 0, 0
    for (k_a, v_a), (k_b, v_b) in [((0, 7), (1, 13)), ((0, 100), (15, 200)), ((5, 500), (16, 5000))]:
        pa = torch.zeros(cshape, dtype=torch.int16, device=device); pa[bi, bj, k_a] = v_a
        pb = torch.zeros(cshape, dtype=torch.int16, device=device); pb[bi, bj, k_b] = v_b
        pab = torch.zeros(cshape, dtype=torch.int16, device=device); pab[bi, bj, k_a] = v_a; pab[bi, bj, k_b] = v_b
        wa = op(pa, in_f, out_f, K, True, False).detach().cpu().numpy().astype(np.float32)
        wb = op(pb, in_f, out_f, K, True, False).detach().cpu().numpy().astype(np.float32)
        wab = op(pab, in_f, out_f, K, True, False).detach().cpu().numpy().astype(np.float32)
        da = block(wa - w0, bi, bj)
        db = block(wb - w0, bi, bj)
        dab = block(wab - w0, bi, bj)
        pred = da + db
        diff = float(np.abs(dab - pred).max())
        print(
            f"[kadd] (k_a={k_a},v_a={v_a})+(k_b={k_b},v_b={v_b}): "
            f"|dab|_inf={float(np.abs(dab).max()):.3f} "
            f"|dab - (da+db)|_inf={diff:.3e}",
            flush=True,
        )

    # Test 2: full-code density at ONE (bi, bj).
    # Set random values at (0,0,:) across all k, compute op(this)-w0 block(0,0),
    # and compare with sum of per-k single-slot deltas.
    rng = np.random.default_rng(42)
    code_at_00 = rng.integers(0, 65536, size=k_max, dtype=np.int32)
    pfull = torch.zeros(cshape, dtype=torch.int16, device=device)
    for k in range(k_max):
        v = int(code_at_00[k])
        pfull[0, 0, k] = np.int16(np.uint16(v & 0xFFFF)) if v < 32768 else np.int16(v - 65536)
    wfull = op(pfull, in_f, out_f, K, True, False).detach().cpu().numpy().astype(np.float32)
    dfull = block(wfull - w0, 0, 0)

    # Per-k single-slot deltas.
    pred_sum = np.zeros((16, 16), dtype=np.float32)
    for k in range(k_max):
        v = int(code_at_00[k])
        p = torch.zeros(cshape, dtype=torch.int16, device=device)
        p[0, 0, k] = np.int16(np.uint16(v & 0xFFFF)) if v < 32768 else np.int16(v - 65536)
        w = op(p, in_f, out_f, K, True, False).detach().cpu().numpy().astype(np.float32)
        pred_sum += block(w - w0, 0, 0)
    diff2 = float(np.abs(dfull - pred_sum).max())
    print(f"\n[density] random-code (k=0..{k_max-1}) at (0,0): |dfull|_inf={float(np.abs(dfull).max()):.3f}", flush=True)
    print(f"[density]                          |dfull - sum_k pred_k|_inf={diff2:.3e}", flush=True)
    print(f"[density]                          |sum_k pred_k|_inf={float(np.abs(pred_sum).max()):.3f}", flush=True)

    return {
        "full_diff": float(diff2),
        "full_block_max": float(np.abs(dfull).max()),
        "pred_block_max": float(np.abs(pred_sum).max()),
    }


@app.local_entrypoint()
def main() -> None:
    result = check.remote()
    print(f"\n[local] {result}")
