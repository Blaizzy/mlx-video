"""Regression matrix for the wan22-relay-shedding branch (pre-PR gate).

Small/fast runs (17f, few steps): we're hunting crashes, black clips and
path regressions — quality evidence comes from the full-size probes in
research/videogen. Every case runs the FORK code (this repo on sys.path).

Cases:
  1. 5B single-model, CFG            (single path was reorganized)
  2. 5B T2V (no image), CFG          (t2v branch never exercised before)
  3. Q4 dual + Lightning, parallel   (the app's historical path, LoRA merge)
  4. Q4 dual + CFG, relay            (relay x quantized x CFG, 17f = alive zone)
  5. bf16 dual relay, no_compile     (debug path)
  6. Q4 dual + Lightning + trim_first_frames=1 (trim path)
Then: pytest tests/test_wan_relay.py with WAN22_A14B_DIR=Q4 (bit-identity
contract) and the weight-free suite.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

FORK = Path(__file__).resolve().parent
VB = Path.home() / "Documents/videoboom/local"
PY = str(VB / ".venv/bin/python")
MODELS = VB / "models"
OUT = Path.home() / "Documents/research/videogen/artifacts/regression"
KEY = Path.home() / "Documents/videoboom/.vbdata-test/E78FC29DE7B442EABC257FDD97/keyframes"
LIGHT = (VB / ".lightning-dir").read_text().strip()

PROMPT = "the man turns toward the camera in the rainy street, cinematic"
IMG = str(KEY / "scene_8.png")

CASES = [
    dict(name="single_5b_cfg", model="Wan2.2-TI2V-5B-MLX", image=IMG,
         frames=17, steps=6, guide="5.0"),
    dict(name="t2v_5b_cfg", model="Wan2.2-TI2V-5B-MLX", image=None,
         frames=17, steps=6, guide="5.0"),
    dict(name="q4_lightning_parallel", model="Wan2.2-I2V-A14B-MLX-Q4", image=IMG,
         frames=17, steps=4, guide="1", lora=True, memory_mode="parallel"),
    dict(name="q4_cfg_relay", model="Wan2.2-I2V-A14B-MLX-Q4", image=IMG,
         frames=17, steps=6, guide="3.5,3.5", memory_mode="relay"),
    dict(name="bf16_relay_nocompile", model="Wan2.2-I2V-A14B-MLX-bf16", image=IMG,
         frames=9, steps=2, guide="1", memory_mode="relay", no_compile=True),
    dict(name="q4_lightning_trim", model="Wan2.2-I2V-A14B-MLX-Q4", image=IMG,
         frames=13, steps=4, guide="1", lora=True, memory_mode="parallel",
         trim=1),
]

RUNNER = r"""
import json, sys, time
sys.path.insert(0, "{fork}")
import mlx.core as mx
from mlx_video.models.wan_2.generate import generate_video
c = json.loads(sys.argv[1])
t0 = time.time()
generate_video(
    model_dir=c["model_dir"], prompt=c["prompt"], image=c.get("image"),
    width=832, height=480, num_frames=c["frames"], steps=c["steps"],
    guide_scale=c["guide"], seed=42, output_path=c["out"],
    tiling="aggressive", loras_high=c.get("lh"), loras_low=c.get("ll"),
    memory_mode=c.get("memory_mode", "auto"), no_compile=c.get("no_compile", False),
    trim_first_frames=c.get("trim", 0),
)
print("RUN_OK", round(time.time()-t0, 1), round(mx.get_peak_memory()/1e9, 1))
"""


def luma(clip):
    tmp = str(clip) + ".png"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", "0.3",
                    "-i", str(clip), "-frames:v", "1", tmp], timeout=60)
    p = subprocess.run([PY, "-c",
                        "import numpy as np, sys; from PIL import Image; "
                        f"print(float(np.asarray(Image.open('{tmp}').convert('L')).mean()))"],
                       capture_output=True, text=True, timeout=60)
    return float(p.stdout.strip())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runner = OUT / "_runner.py"
    runner.write_text(RUNNER.format(fork=FORK))
    results = []
    for c in CASES:
        clip = OUT / f"{c['name']}.mp4"
        cfg = dict(model_dir=str(MODELS / c["model"]), prompt=PROMPT,
                   image=c.get("image"), frames=c["frames"], steps=c["steps"],
                   guide=c["guide"], out=str(clip),
                   memory_mode=c.get("memory_mode", "auto"),
                   no_compile=c.get("no_compile", False), trim=c.get("trim", 0))
        if c.get("lora"):
            cfg["lh"] = [[f"{LIGHT}/high_noise_model.safetensors", 1.0]]
            cfg["ll"] = [[f"{LIGHT}/low_noise_model.safetensors", 1.0]]
        print(f"[case] {c['name']}", flush=True)
        t0 = time.time()
        p = subprocess.run([PY, str(runner), json.dumps(cfg)],
                           capture_output=True, text=True, timeout=40 * 60)
        ok = clip.exists() and "RUN_OK" in p.stdout
        rec = dict(name=c["name"], ok=ok, wall=round(time.time() - t0, 1))
        if ok:
            rec["luma"] = round(luma(clip), 1)
            rec["black"] = rec["luma"] < 8.0
        else:
            rec["tail"] = (p.stdout + p.stderr).splitlines()[-6:]
        results.append(rec)
        print(f"[done] {json.dumps(rec)}", flush=True)

    # pytest: contract (with weights) + weight-free suite
    env_q4 = str(MODELS / "Wan2.2-I2V-A14B-MLX-Q4")
    p = subprocess.run(
        [PY, "-m", "pytest", "tests/test_wan_relay.py", "tests/test_rope.py",
         "tests/test_vae_streaming.py", "-q"],
        capture_output=True, text=True, cwd=str(FORK), timeout=40 * 60,
        env={**__import__("os").environ, "WAN22_A14B_DIR": env_q4,
             "PYTHONPATH": str(FORK)},
    )
    pytest_tail = p.stdout.splitlines()[-3:]
    print("[pytest]", pytest_tail, flush=True)

    (OUT / "matrix_results.json").write_text(json.dumps(
        dict(cases=results, pytest=pytest_tail), indent=1))
    n_bad = sum(1 for r in results if not r["ok"] or r.get("black"))
    print(f"MATRIX COMPLETE bad={n_bad}/{len(results)}", flush=True)


if __name__ == "__main__":
    main()
