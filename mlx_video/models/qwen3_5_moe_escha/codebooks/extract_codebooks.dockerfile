# One-shot codebook extractor for Escha-W2 → MLX port.
#
# Recommended host: Linux x86_64 with NVIDIA GPU + nvidia-container-toolkit.
#     docker build -f extract_codebooks.dockerfile -t escha-extract .
#     docker run --rm --gpus all -v $(pwd):/out escha-extract
#
# Apple Silicon Mac: the EschaLabs wheel is manylinux_2_28_x86_64 (CUDA-linked).
# This image will only build under --platform linux/amd64 (QEMU emulation),
# and even then the wheel's .so likely fails to import without a CUDA runtime.
# Do NOT attempt CPU-only extraction on ARM Mac — the wheel isn't cross-arch.
# Use a Linux GPU host, a cloud x86 VM, or a temporary Colab/RunPod session.
#
# CPU fallback (Linux x86_64 only): drop --gpus all and pass --cpu.

FROM ubuntu:24.04

# Ubuntu 24.04 ships Python 3.12 natively (22.04 does not, breaking older builds).
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3-pip curl ca-certificates git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /work
RUN python3.12 -m venv /venv
ENV PATH=/venv/bin:$PATH

RUN pip install -U pip wheel && \
    pip install "torch==2.9.*" --index-url https://download.pytorch.org/whl/cu128 && \
    pip install "huggingface_hub[cli]" numpy

# Fetch the runtime wheel + install it.
RUN hf download EschaLabs/escha-runtime-qwen3moe --include "sglang/*" --local-dir /work && \
    pip install /work/sglang/escha-*.whl

COPY extract_codebooks.py /work/extract_codebooks.py

ENTRYPOINT ["python", "/work/extract_codebooks.py", "--out", "/out/escha_codebooks_v1.npz"]
