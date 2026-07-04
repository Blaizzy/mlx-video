"""Tests for Wan2.2 dual-expert relay-shedding (memory_mode).

The core correctness contract — memory_mode="relay" and memory_mode="parallel"
produce BIT-IDENTICAL latents for the same seed — needs real A14B weights, so
the full contract test is opt-in: point WAN22_A14B_DIR at a converted dual
model dir and it will run two short generations and compare the dumped
pre-VAE latents byte-for-byte. (Validated during development on an M5 Pro
48GB against both the 4-bit and bf16 conversions: MD5-identical latents;
bf16 A14B peaked at 36.8GB in relay mode at 832x480x81f.)

The unit tests below run without weights.
"""
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

A14B_DIR = os.environ.get("WAN22_A14B_DIR", "")


def test_memory_mode_validated():
    """A typo'd memory_mode must fail loudly, not silently load both experts."""
    from mlx_video.models.wan_2.generate import generate_video

    with pytest.raises(ValueError, match="memory_mode"):
        generate_video(
            model_dir="/nonexistent",
            prompt="x",
            memory_mode="relai",
        )


def test_cli_exposes_memory_mode():
    out = subprocess.run(
        [sys.executable, "-m", "mlx_video.models.wan_2.generate", "--help"],
        capture_output=True,
        text=True,
    )
    assert "--memory-mode" in out.stdout
    assert "--dump-latents" in out.stdout


@pytest.mark.skipif(not A14B_DIR, reason="set WAN22_A14B_DIR to a converted dual-model dir")
def test_relay_parallel_bit_identical():
    """Same seed, relay vs parallel: pre-VAE latents must match byte-for-byte."""
    from mlx_video.models.wan_2.generate import generate_video

    digests = {}
    with tempfile.TemporaryDirectory() as td:
        for mode in ("parallel", "relay"):
            lat = Path(td) / f"{mode}.npy"
            generate_video(
                model_dir=A14B_DIR,
                prompt="a red cube on a wooden table, studio light",
                width=448,
                height=256,
                num_frames=9,
                steps=4,
                guide_scale="1",
                seed=42,
                output_path=str(Path(td) / f"{mode}.mp4"),
                tiling="aggressive",
                memory_mode=mode,
                dump_latents=str(lat),
            )
            digests[mode] = hashlib.md5(lat.read_bytes()).hexdigest()
    assert digests["parallel"] == digests["relay"]
