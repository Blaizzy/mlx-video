"""Extract all frames from both videos, compute pixel-wise diffs."""
import subprocess
import sys
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

REAL = Path.home() / "movie/wang_wenchin/output/wan_s2v_verify_real_audio.mp4"
SIL  = Path.home() / "movie/wang_wenchin/output/wan_s2v_verify_silence.mp4"
OUT  = Path("/tmp/wan_verify/frames")
OUT.mkdir(exist_ok=True, parents=True)

def extract(mp4, tag):
    d = OUT / tag
    d.mkdir(exist_ok=True)
    # clean existing
    for f in d.glob("*.png"): f.unlink()
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4),
        str(d / "f%04d.png")
    ], check=True)
    return sorted(d.glob("*.png"))

def md5f(p):
    return hashlib.md5(p.read_bytes()).hexdigest()[:12]

def main():
    print(f"REAL = {REAL} (exists={REAL.exists()})")
    print(f"SIL  = {SIL}  (exists={SIL.exists()})")
    if not REAL.exists() or not SIL.exists():
        print("Missing input(s); abort."); sys.exit(1)

    print("\nExtracting frames...")
    fr_real = extract(REAL, "real")
    fr_sil  = extract(SIL,  "sil")
    print(f"real: {len(fr_real)} frames   sil: {len(fr_sil)} frames")

    n = min(len(fr_real), len(fr_sil))
    print(f"\n{'idx':>3} {'md5_real':>13} {'md5_sil':>13} {'match':>5} "
          f"{'mean|Δ|':>10} {'max|Δ|':>7} {'diff_frac':>10}")
    print("-" * 68)

    per_frame = []
    identical_count = 0
    for i in range(n):
        r_path, s_path = fr_real[i], fr_sil[i]
        r_md5, s_md5 = md5f(r_path), md5f(s_path)
        match = (r_md5 == s_md5)
        if match: identical_count += 1
        r_arr = np.asarray(Image.open(r_path).convert("RGB"), dtype=np.int16)
        s_arr = np.asarray(Image.open(s_path).convert("RGB"), dtype=np.int16)
        d = np.abs(r_arr - s_arr)
        mean_d = float(d.mean())
        max_d = int(d.max())
        # Fraction of pixels that differ by any amount.
        diff_frac = float((d.sum(axis=-1) > 0).mean())
        per_frame.append((i, mean_d, max_d, diff_frac, match))
        print(f"{i:>3} {r_md5:>13} {s_md5:>13} {str(match):>5} "
              f"{mean_d:>10.3f} {max_d:>7d} {diff_frac:>10.3%}")

    print("-" * 68)
    total_mean = float(np.mean([p[1] for p in per_frame]))
    total_max = int(np.max([p[2] for p in per_frame]))
    total_frac = float(np.mean([p[3] for p in per_frame]))
    print(f"\nSUMMARY over {n} frames:")
    print(f"  identical (md5)      = {identical_count}/{n}")
    print(f"  mean pixel diff avg  = {total_mean:.3f}  (on 0..255 scale)")
    print(f"  max pixel diff       = {total_max}")
    print(f"  mean diffing-frac    = {total_frac:.3%}  (fraction of pixels changed)")

    # Side-by-side of frame 24.
    mid = min(24, n - 1)
    r_arr = np.asarray(Image.open(fr_real[mid]).convert("RGB"))
    s_arr = np.asarray(Image.open(fr_sil[mid]).convert("RGB"))
    h, w, _ = r_arr.shape
    sep = np.full((h, 6, 3), 255, dtype=np.uint8)
    side = np.concatenate([r_arr, sep, s_arr], axis=1)
    Image.fromarray(side).save(OUT / f"sidebyside_f{mid}.png")
    print(f"\nSide-by-side of frame {mid} saved to {OUT}/sidebyside_f{mid}.png")

    # Verdict.
    print("\n" + "=" * 60)
    if identical_count == n:
        print("VERDICT: ALL frames byte-identical.  AUDIO CONDITIONING IS BROKEN.")
    elif total_mean < 0.5:
        print("VERDICT: Frames differ but only by tiny noise-level amounts. "
              "Audio may be reaching model but making negligible impact.")
    else:
        print("VERDICT: Frames differ substantially -> audio conditioning IS working. "
              "User's observation was misled by post-mux naming.")

if __name__ == "__main__":
    main()
