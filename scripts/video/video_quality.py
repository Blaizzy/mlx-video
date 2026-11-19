#!/usr/bin/env python3
"""Analyze quality of a single generated video.

Reports sharpness, temporal stability, color distribution, motion smoothness,
chunk boundary artifacts, and common generation defects. Useful for quick
quality checks during model porting and debugging.

Usage:
    # Basic analysis
    python scripts/video/video_quality.py output.mp4

    # With chunk boundary analysis (e.g., 32 frames/chunk)
    python scripts/video/video_quality.py output.mp4 --chunk-size 32

    # Detailed per-frame CSV export
    python scripts/video/video_quality.py output.mp4 --csv metrics.csv

    # Analyze specific frame range
    python scripts/video/video_quality.py output.mp4 --start 0 --end 64
"""

import argparse
import sys

import cv2
import numpy as np


def load_video(path, start=0, end=None):
    """Load video frames as float32 numpy arrays (0-255 range)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Error: cannot open {path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    frames = []
    idx = start
    while end is None or idx < end:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame.astype(np.float32))
        idx += 1
    cap.release()
    return frames, fps, total


def sharpness_laplacian(frame):
    """Laplacian variance — higher means sharper."""
    gray = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def sharpness_gradient(frame):
    """Mean gradient magnitude — higher means more edges/detail."""
    gray = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.mean(np.sqrt(gx**2 + gy**2))


def color_stats(frame):
    """Per-channel mean and std in BGR order."""
    means = [np.mean(frame[:, :, c]) for c in range(3)]
    stds = [np.std(frame[:, :, c]) for c in range(3)]
    return means, stds


def detect_uniform_color(frame, std_threshold=15.0):
    """Detect if frame is near-uniform (common failure mode)."""
    return np.std(frame) < std_threshold


def detect_noise(frame, threshold=200.0):
    """High Laplacian variance with low gradient can indicate noise."""
    lap = sharpness_laplacian(frame)
    grad = sharpness_gradient(frame)
    # Noise has high variance but less coherent edges
    return lap > threshold and grad < 5.0


def frame_difference(a, b):
    """Mean absolute pixel difference between frames."""
    return np.mean(np.abs(a - b))


def optical_flow_magnitude(prev, curr):
    """Mean optical flow magnitude (Farneback method)."""
    prev_gray = cv2.cvtColor(prev.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    return np.mean(mag), np.max(mag)


def analyze_video(frames, chunk_size=None, compute_flow=False):
    """Compute per-frame and aggregate quality metrics."""
    n = len(frames)

    metrics = {
        "sharpness_lap": [],
        "sharpness_grad": [],
        "brightness": [],
        "contrast": [],
        "color_mean_b": [],
        "color_mean_g": [],
        "color_mean_r": [],
        "frame_diff": [],
        "is_uniform": [],
        "is_noisy": [],
    }
    if compute_flow:
        metrics["flow_mean"] = []
        metrics["flow_max"] = []

    for i in range(n):
        f = frames[i]
        metrics["sharpness_lap"].append(sharpness_laplacian(f))
        metrics["sharpness_grad"].append(sharpness_gradient(f))
        metrics["brightness"].append(np.mean(f))
        metrics["contrast"].append(np.std(f))
        means, _ = color_stats(f)
        metrics["color_mean_b"].append(means[0])
        metrics["color_mean_g"].append(means[1])
        metrics["color_mean_r"].append(means[2])
        metrics["is_uniform"].append(detect_uniform_color(f))
        metrics["is_noisy"].append(detect_noise(f))

        if i > 0:
            metrics["frame_diff"].append(frame_difference(frames[i - 1], f))
            if compute_flow:
                fm, fmx = optical_flow_magnitude(frames[i - 1], f)
                metrics["flow_mean"].append(fm)
                metrics["flow_max"].append(fmx)
        else:
            metrics["frame_diff"].append(0.0)
            if compute_flow:
                metrics["flow_mean"].append(0.0)
                metrics["flow_max"].append(0.0)

    # Convert to arrays
    for k in metrics:
        metrics[k] = np.array(metrics[k])

    # Chunk boundary analysis
    if chunk_size and n > chunk_size:
        boundaries = list(range(chunk_size, n, chunk_size))
        boundary_metrics = []
        for b in boundaries:
            if b < n and b > 0:
                pre = (
                    metrics["frame_diff"][b - 1] if b > 1 else metrics["frame_diff"][1]
                )
                at = metrics["frame_diff"][b]
                ratio = at / (pre + 1e-10)
                brightness_jump = (
                    metrics["brightness"][b] - metrics["brightness"][b - 1]
                )
                contrast_jump = (
                    (metrics["contrast"][b] - metrics["contrast"][b - 1])
                    / (metrics["contrast"][b - 1] + 1e-10)
                    * 100
                )
                sharpness_jump = (
                    (metrics["sharpness_lap"][b] - metrics["sharpness_lap"][b - 1])
                    / (metrics["sharpness_lap"][b - 1] + 1e-10)
                    * 100
                )
                boundary_metrics.append(
                    {
                        "frame": b,
                        "diff_ratio": ratio,
                        "brightness_jump": brightness_jump,
                        "contrast_jump_pct": contrast_jump,
                        "sharpness_jump_pct": sharpness_jump,
                    }
                )
        metrics["boundaries"] = boundary_metrics

    return metrics


def print_report(metrics, path, fps, total_frames, frames_analyzed):
    """Print a formatted quality report."""
    sl = metrics["sharpness_lap"]
    sg = metrics["sharpness_grad"]
    br = metrics["brightness"]
    ct = metrics["contrast"]
    fd = metrics["frame_diff"]

    print("=" * 72)
    print("VIDEO QUALITY REPORT")
    print("=" * 72)
    print(f"  File: {path}")
    print(
        f"  Total frames: {total_frames}  Analyzed: {frames_analyzed}  FPS: {fps:.1f}"
    )
    duration = total_frames / fps if fps > 0 else 0
    print(f"  Duration: {duration:.1f}s")
    print()

    # Defect detection
    n_uniform = int(np.sum(metrics["is_uniform"]))
    n_noisy = int(np.sum(metrics["is_noisy"]))
    if n_uniform > 0 or n_noisy > 0:
        print("⚠  DEFECTS DETECTED")
        print("-" * 40)
        if n_uniform:
            frames_list = np.where(metrics["is_uniform"])[0][:10]
            print(
                f"  Uniform/blank frames: {n_uniform} — frames {list(frames_list)}{'...' if n_uniform > 10 else ''}"
            )
        if n_noisy:
            frames_list = np.where(metrics["is_noisy"])[0][:10]
            print(
                f"  Noisy frames: {n_noisy} — frames {list(frames_list)}{'...' if n_noisy > 10 else ''}"
            )
        print()

    print("SHARPNESS")
    print("-" * 40)
    print(
        f"  Laplacian var:  mean={np.mean(sl):8.1f}  min={np.min(sl):8.1f}  max={np.max(sl):8.1f}  std={np.std(sl):.1f}"
    )
    print(
        f"  Gradient mag:   mean={np.mean(sg):8.2f}  min={np.min(sg):8.2f}  max={np.max(sg):8.2f}  std={np.std(sg):.2f}"
    )
    if np.std(sl) / (np.mean(sl) + 1e-10) > 0.3:
        print("  ⚠  High sharpness variation — possible blur artifacts")
    print()

    print("BRIGHTNESS & CONTRAST")
    print("-" * 40)
    print(
        f"  Brightness:     mean={np.mean(br):6.1f}  min={np.min(br):6.1f}  max={np.max(br):6.1f}  std={np.std(br):.2f}"
    )
    print(
        f"  Contrast (std): mean={np.mean(ct):6.1f}  min={np.min(ct):6.1f}  max={np.max(ct):6.1f}  std={np.std(ct):.2f}"
    )
    if np.std(br) > 3.0:
        print("  ⚠  Brightness instability — may indicate chunk boundary artifacts")
    print()

    print("COLOR DISTRIBUTION (BGR)")
    print("-" * 40)
    print(
        f"  Blue:  mean={np.mean(metrics['color_mean_b']):6.1f}  std={np.std(metrics['color_mean_b']):.2f}"
    )
    print(
        f"  Green: mean={np.mean(metrics['color_mean_g']):6.1f}  std={np.std(metrics['color_mean_g']):.2f}"
    )
    print(
        f"  Red:   mean={np.mean(metrics['color_mean_r']):6.1f}  std={np.std(metrics['color_mean_r']):.2f}"
    )
    print()

    print("TEMPORAL STABILITY")
    print("-" * 40)
    fd_nz = fd[1:]  # skip first frame (always 0)
    if len(fd_nz) > 0:
        print(
            f"  Frame diff:     mean={np.mean(fd_nz):6.2f}  min={np.min(fd_nz):6.2f}  max={np.max(fd_nz):6.2f}  std={np.std(fd_nz):.2f}"
        )
        if np.std(fd_nz) / (np.mean(fd_nz) + 1e-10) > 0.5:
            print("  ⚠  High diff variance — jitter or discontinuities")
    if "flow_mean" in metrics:
        fm = metrics["flow_mean"][1:]
        print(
            f"  Optical flow:   mean={np.mean(fm):6.2f}  max_frame={np.max(metrics['flow_max'][1:]):.1f}"
        )
    print()

    # Chunk boundaries
    if "boundaries" in metrics and metrics["boundaries"]:
        print("CHUNK BOUNDARIES")
        print("-" * 40)
        print(
            f"  {'Frame':>6}  {'Diff ratio':>10}  {'Brightness':>10}  {'Contrast %':>10}  {'Sharpness %':>11}"
        )
        for bm in metrics["boundaries"]:
            print(
                f"  {bm['frame']:6d}"
                f"  {bm['diff_ratio']:10.2f}x"
                f"  {bm['brightness_jump']:+10.1f}"
                f"  {bm['contrast_jump_pct']:+10.1f}%"
                f"  {bm['sharpness_jump_pct']:+11.1f}%"
            )
        avg_ratio = np.mean([b["diff_ratio"] for b in metrics["boundaries"]])
        if avg_ratio > 2.0:
            print(
                f"  ⚠  Boundary diff ratio {avg_ratio:.1f}x — visible chunk transitions"
            )
        print()

    # Overall grade
    print("OVERALL ASSESSMENT")
    print("-" * 40)
    issues = []
    if n_uniform > 0:
        issues.append("uniform/blank frames")
    if n_noisy > 0:
        issues.append("noisy frames")
    if np.std(br) > 3.0:
        issues.append("brightness flicker")
    if np.std(sl) / (np.mean(sl) + 1e-10) > 0.3:
        issues.append("sharpness variation")
    if "boundaries" in metrics and metrics["boundaries"]:
        avg_ratio = np.mean([b["diff_ratio"] for b in metrics["boundaries"]])
        if avg_ratio > 2.0:
            issues.append("chunk boundary artifacts")
    if issues:
        print(f"  Issues found: {', '.join(issues)}")
    else:
        print("  ✓ No significant quality issues detected")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze quality of a single generated video"
    )
    parser.add_argument("video", help="Path to video file")
    parser.add_argument(
        "--chunk-size",
        type=int,
        help="Frames per chunk for boundary analysis (e.g., 32)",
    )
    parser.add_argument("--start", type=int, default=0, help="Start frame (default: 0)")
    parser.add_argument("--end", type=int, help="End frame (default: all)")
    parser.add_argument(
        "--flow",
        action="store_true",
        help="Compute optical flow (slower but more detailed)",
    )
    parser.add_argument("--csv", help="Export per-frame metrics to CSV")
    args = parser.parse_args()

    print(f"Loading: {args.video}")
    frames, fps, total = load_video(args.video, args.start, args.end)
    if not frames:
        parser.error("no video frames in the requested range")
    h, w = frames[0].shape[:2]
    print(f"  → {len(frames)} frames, {fps:.1f} fps, {w}x{h}")

    print("Analyzing...")
    metrics = analyze_video(frames, args.chunk_size, args.flow)
    print()
    print_report(metrics, args.video, fps, total, len(frames))

    if args.csv:
        import csv

        keys = [
            "sharpness_lap",
            "sharpness_grad",
            "brightness",
            "contrast",
            "color_mean_b",
            "color_mean_g",
            "color_mean_r",
            "frame_diff",
        ]
        if args.flow:
            keys += ["flow_mean", "flow_max"]

        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame"] + keys)
            for i in range(len(frames)):
                row = [i] + [f"{metrics[k][i]:.4f}" for k in keys]
                writer.writerow(row)
        print(f"Per-frame metrics saved to {args.csv}")


if __name__ == "__main__":
    main()
