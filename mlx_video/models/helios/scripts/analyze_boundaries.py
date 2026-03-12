#!/usr/bin/env python3
"""Analyze chunk boundary quality in Helios-generated videos.

Measures brightness, contrast, color shifts, spatial distribution, and
frame-to-frame differences at chunk boundaries. Compares multiple videos
side-by-side when given multiple paths.

This was the primary diagnostic tool used to identify and fix:
- 40% contrast drops from pixel cross-fade (→ disabled cross-fade)
- 7% contrast drops from VAE causal padding warmup (→ contrast correction)
- Per-channel color shifts at boundaries (→ per-channel matching)
- Spatial brightness redistribution (→ low-frequency spatial correction)

Usage:
    # Analyze a single video
    python mlx_video/models/helios/scripts/analyze_boundaries.py /tmp/helios_output.mp4

    # Compare multiple videos
    python mlx_video/models/helios/scripts/analyze_boundaries.py \
        /tmp/helios_before.mp4 /tmp/helios_after.mp4

    # Custom chunk size (default: 32 frames per chunk)
    python mlx_video/models/helios/scripts/analyze_boundaries.py --chunk-size 33 /tmp/ref.mp4
"""

import argparse
import sys

import cv2
import numpy as np


def analyze_video(path, chunk_size=32):
    """Analyze boundary quality metrics for a video."""
    vid = cv2.VideoCapture(path)
    if not vid.isOpened():
        print(f"Error: cannot open {path}", file=sys.stderr)
        return None

    frames = []
    while True:
        ret, f = vid.read()
        if not ret:
            break
        frames.append(f)
    vid.release()

    n = len(frames)
    if n == 0:
        print(f"Error: no frames in {path}", file=sys.stderr)
        return None

    # Compute per-frame statistics
    means = np.zeros(n)
    stds = np.zeros(n)
    ch_means = np.zeros((n, 3))
    diffs = np.zeros(n - 1)

    for i, f in enumerate(frames):
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float64)
        means[i] = gray.mean()
        stds[i] = gray.std()
        ch_means[i] = [f[:, :, c].mean() for c in range(3)]
        if i > 0:
            prev_gray = cv2.cvtColor(frames[i - 1], cv2.COLOR_BGR2GRAY).astype(np.float64)
            diffs[i - 1] = np.abs(gray - prev_gray).mean()

    # Find chunk boundaries
    boundaries = []
    b = chunk_size - 1
    while b < n - 1:
        boundaries.append(b)
        b += chunk_size

    results = {
        "path": path,
        "num_frames": n,
        "chunk_size": chunk_size,
        "boundaries": [],
    }

    for b in boundaries:
        if b >= n - 1:
            break

        # Contrast
        pre_std = stds[max(0, b - 2) : b + 1]
        post_std = stds[b + 1 : min(n, b + 4)]
        contrast_jump = post_std[0] - pre_std[-1]
        contrast_pct = contrast_jump / max(pre_std[-1], 1e-6) * 100

        # Brightness
        bright_jump = means[b + 1] - means[b]
        bright_pct = bright_jump / max(means[b], 1e-6) * 100

        # Per-channel color shift
        ch_shifts = ch_means[b + 1] - ch_means[b]  # B, G, R

        # Frame diff ratio
        boundary_diff = diffs[b]
        window = 3
        nearby_indices = list(range(max(0, b - window), b)) + list(
            range(b + 1, min(len(diffs), b + 1 + window))
        )
        nearby_avg = np.mean(diffs[nearby_indices]) if nearby_indices else 1.0
        diff_ratio = boundary_diff / max(nearby_avg, 1e-6)

        # Spatial analysis
        f_pre = frames[b].astype(np.float64)
        f_post = frames[b + 1].astype(np.float64)
        gray_diff = cv2.cvtColor(frames[b + 1], cv2.COLOR_BGR2GRAY).astype(
            np.float64
        ) - cv2.cvtColor(frames[b], cv2.COLOR_BGR2GRAY).astype(np.float64)
        h, w = gray_diff.shape
        ch, cw = h // 4, w // 4
        center_shift = gray_diff[ch : 3 * ch, cw : 3 * cw].mean()
        periph_mask = np.ones_like(gray_diff, dtype=bool)
        periph_mask[ch : 3 * ch, cw : 3 * cw] = False
        periph_shift = gray_diff[periph_mask].mean()

        results["boundaries"].append(
            {
                "frame": b,
                "contrast_pct": contrast_pct,
                "bright_pct": bright_pct,
                "ch_shifts_bgr": ch_shifts.tolist(),
                "diff_ratio": diff_ratio,
                "center_shift": center_shift,
                "periph_shift": periph_shift,
                "boundary_diff": boundary_diff,
                "nearby_diff": nearby_avg,
            }
        )

    # Per-chunk stats
    chunk_stats = []
    for c in range(0, n, chunk_size):
        end = min(c + chunk_size, n)
        chunk_stats.append(
            {
                "frames": f"{c}-{end - 1}",
                "mean_bright": means[c:end].mean(),
                "mean_contrast": stds[c:end].mean(),
                "first_contrast": stds[c],
                "last_contrast": stds[end - 1],
            }
        )
    results["chunk_stats"] = chunk_stats

    return results


def print_results(results):
    """Pretty-print analysis results."""
    print(f"\n{'=' * 70}")
    print(f"  {results['path']}")
    print(f"  {results['num_frames']} frames, chunk size = {results['chunk_size']}")
    print(f"{'=' * 70}")

    for bd in results["boundaries"]:
        b = bd["frame"]
        print(f"\n  Boundary {b}→{b + 1}:")
        print(f"    Contrast jump:    {bd['contrast_pct']:+.1f}%")
        print(f"    Brightness jump:  {bd['bright_pct']:+.1f}%")
        print(
            f"    Color shift B/G/R: {bd['ch_shifts_bgr'][0]:+.1f} / "
            f"{bd['ch_shifts_bgr'][1]:+.1f} / {bd['ch_shifts_bgr'][2]:+.1f}"
        )
        print(
            f"    Frame diff:       {bd['boundary_diff']:.1f} vs nearby "
            f"{bd['nearby_diff']:.1f} ({bd['diff_ratio']:.1f}×)"
        )
        print(
            f"    Spatial:          center {bd['center_shift']:+.2f}, "
            f"periphery {bd['periph_shift']:+.2f}"
        )

    print(f"\n  Per-chunk summary:")
    for cs in results["chunk_stats"]:
        print(
            f"    Frames {cs['frames']:>7s}: brightness={cs['mean_bright']:.1f}, "
            f"contrast={cs['mean_contrast']:.1f} "
            f"(first={cs['first_contrast']:.1f}, last={cs['last_contrast']:.1f})"
        )


def print_comparison(all_results):
    """Print side-by-side comparison table."""
    if len(all_results) < 2:
        return

    print(f"\n{'=' * 70}")
    print("  COMPARISON SUMMARY")
    print(f"{'=' * 70}")

    # Header
    labels = [r["path"].split("/")[-1] for r in all_results]
    header = f"{'Metric':<25s}"
    for label in labels:
        header += f"  {label:>18s}"
    print(f"\n{header}")
    print("-" * (25 + 20 * len(labels)))

    # For each boundary index
    max_boundaries = max(len(r["boundaries"]) for r in all_results)
    for bi in range(max_boundaries):
        print(f"\n  Boundary {bi + 1}:")
        for metric, key, fmt in [
            ("Contrast jump", "contrast_pct", "{:+.1f}%"),
            ("Brightness jump", "bright_pct", "{:+.1f}%"),
            ("Frame diff ratio", "diff_ratio", "{:.1f}×"),
            ("Center shift", "center_shift", "{:+.2f}"),
            ("Periphery shift", "periph_shift", "{:+.2f}"),
        ]:
            row = f"    {metric:<23s}"
            for r in all_results:
                if bi < len(r["boundaries"]):
                    val = r["boundaries"][bi][key]
                    row += f"  {fmt.format(val):>18s}"
                else:
                    row += f"  {'N/A':>18s}"
            print(row)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze chunk boundary quality in Helios videos"
    )
    parser.add_argument("videos", nargs="+", help="Video file paths to analyze")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=32,
        help="Frames per chunk (default: 32, use 33 for reference pipeline)",
    )
    args = parser.parse_args()

    all_results = []
    for path in args.videos:
        results = analyze_video(path, args.chunk_size)
        if results is not None:
            print_results(results)
            all_results.append(results)

    if len(all_results) > 1:
        print_comparison(all_results)


if __name__ == "__main__":
    main()
