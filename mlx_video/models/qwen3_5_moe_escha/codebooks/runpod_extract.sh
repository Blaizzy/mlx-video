#!/usr/bin/env bash
# One-shot Escha-W2 codebook extractor for a RunPod (or any Linux x86_64 + NVIDIA)
# SSH terminal. Idempotent: re-running skips download / install / extraction
# steps that already succeeded.
#
# Recommended pod image: `runpod/pytorch:2.9.0-py3.12-cuda12.8.0-devel-ubuntu24.04`
# (Ubuntu 24.04 for Python 3.12; ~$0.39/hr on A10, ~$0.79/hr on A100).
# Anything Linux x86_64 with an NVIDIA GPU, Python 3.12, and CUDA drivers works.
#
# Usage on the pod:
#     curl -O https://raw.githubusercontent.com/<fork>/escha-mlx-port/mlx_video/models/qwen3_5_moe_escha/codebooks/runpod_extract.sh
#     bash runpod_extract.sh
# ...or just paste the file contents into a shell.
#
# End state: prints a `transfer.sh` URL you can `curl` from your Mac to fetch
# the resulting `escha_codebooks_v1.npz` (~4-6 MiB).

set -euo pipefail

WORK=${WORK:-/workspace/escha-extract}
OUT_NPZ="$WORK/escha_codebooks_v1.npz"
OUT_ST="$WORK/escha_codebooks_v1.safetensors"
VENV="$WORK/.venv"

mkdir -p "$WORK"
cd "$WORK"

# ------------------------------------------------------------
# 1. Sanity — GPU + Python 3.12
# ------------------------------------------------------------
echo "=== Environment ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || {
    echo "!!! nvidia-smi failed. Pick a GPU pod." >&2
    exit 1
}

if ! command -v python3.12 >/dev/null; then
    echo "Installing Python 3.12 (needed by the escha wheel)..."
    apt-get update -qq
    apt-get install -y -qq python3.12 python3.12-venv python3-pip curl ca-certificates
fi
python3.12 --version

# ------------------------------------------------------------
# 2. Virtualenv + deps (torch 2.9 CUDA12.8 to match escha ABI)
# ------------------------------------------------------------
if [ ! -d "$VENV" ]; then
    python3.12 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "=== Installing deps ==="
pip install -q -U pip wheel
pip install -q "torch==2.9.*" --index-url https://download.pytorch.org/whl/cu128
pip install -q "huggingface_hub[cli]" numpy safetensors

python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('torch', torch.__version__, '| device:', torch.cuda.get_device_name(0))"

# ------------------------------------------------------------
# 3. Download escha runtime wheel (~150 MB, cached)
# ------------------------------------------------------------
if ! ls sglang/escha-*.whl >/dev/null 2>&1; then
    echo "=== Downloading escha wheel ==="
    hf download EschaLabs/escha-runtime-qwen3moe --include "sglang/*" --local-dir "$WORK"
fi
pip install -q sglang/escha-*.whl

# ------------------------------------------------------------
# 4. Extract codebooks (~10-15 min on A10, ~5 min on A100)
# ------------------------------------------------------------
if [ -f "$OUT_NPZ" ] && [ -f "$OUT_ST" ]; then
    echo "=== Codebooks already extracted, skipping. Delete $OUT_NPZ to re-run. ==="
else
    echo "=== Extracting codebooks ==="
    python - <<'PY'
import time, numpy as np, torch, escha  # noqa
from safetensors.numpy import save_file
import os

WORK = os.environ.get('WORK', '/workspace/escha-extract')
op = torch.ops.escha.escham_reconstruct

def extract(K: int) -> np.ndarray:
    cb = np.zeros((65536, 16), dtype=np.float16)
    baseline = None
    t0 = time.time()
    for i in range(65536):
        probe = torch.zeros((1, 8, 16 * K), dtype=torch.int16, device='cuda')
        probe[0, 0, 0] = i
        w = op(probe, 128, 16, K, True, False)
        row = w[0].detach().cpu().numpy()
        if i == 0:
            baseline = row.copy()
            cb[0] = baseline
        else:
            cb[i] = row - baseline
        if i and i % 4096 == 0:
            el = time.time() - t0
            eta = el * (65536 - i) / i
            print(f'  K={K}: {i}/65536 ({100*i/65536:.1f}%)  elapsed={el:.0f}s  ETA={eta:.0f}s', flush=True)
    print(f'  K={K}: done in {time.time()-t0:.0f}s')
    return cb

print('K=2...'); cb2 = extract(2)
print('K=3...'); cb3 = extract(3)

np.savez_compressed(f'{WORK}/escha_codebooks_v1.npz', cb_A_K2=cb2, cb_A_K3=cb3)
save_file({'cb_A_K2': cb2, 'cb_A_K3': cb3}, f'{WORK}/escha_codebooks_v1.safetensors')
print('Wrote', f'{WORK}/escha_codebooks_v1.npz')
print('Wrote', f'{WORK}/escha_codebooks_v1.safetensors')
PY
fi

ls -lh "$OUT_NPZ" "$OUT_ST"

# ------------------------------------------------------------
# 5. Upload to transfer.sh → print URL for user to curl
# ------------------------------------------------------------
echo ""
echo "=== Uploading to transfer.sh ==="
NPZ_URL=$(curl -sS --upload-file "$OUT_NPZ" "https://transfer.sh/escha_codebooks_v1.npz")
ST_URL=$(curl -sS --upload-file "$OUT_ST"  "https://transfer.sh/escha_codebooks_v1.safetensors")

cat <<EOF

============================================================
  DONE. Download from your Mac with:

      curl -O '$NPZ_URL'
      curl -O '$ST_URL'

  Files expire in 14 days. Send the .npz to the person who
  gave you this script.
============================================================
EOF
