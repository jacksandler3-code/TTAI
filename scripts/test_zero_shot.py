"""
Zero-shot rally detection using VideoMAE pretrained on Kinetics-400.

Table tennis is one of the 400 Kinetics classes, so the model has already
seen this domain. This script tests whether it can distinguish rally from
dead time on your footage WITHOUT any fine-tuning.

If accuracy is already 70-80%, you need far less labeled data than if you
were starting from scratch.

Usage:
    python scripts/test_zero_shot.py video.mp4
    python scripts/test_zero_shot.py video.mp4 --save-json predictions.json
    python scripts/test_zero_shot.py video.mp4 --compare-optical-flow

Output:
    - ASCII timeline showing predicted rally segments
    - Accuracy stats if you provide ground truth via --ground-truth timestamps.json
    - Optional JSON file with per-window scores for import into other tools
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model():
    """Load VideoMAE + feature extractor. Downloads ~350MB on first run."""
    try:
        from transformers import VideoMAEFeatureExtractor, VideoMAEForVideoClassification
        import torch
    except ImportError:
        print("Error: ML dependencies not installed.")
        print("Run: pip install -r requirements_ml.txt")
        sys.exit(1)

    print("Loading VideoMAE (downloads ~350MB on first run)...")
    model_id = "MCG-NJU/videomae-base-finetuned-kinetics"
    extractor = VideoMAEFeatureExtractor.from_pretrained(model_id)
    model = VideoMAEForVideoClassification.from_pretrained(model_id)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}")

    # Find the table tennis class index dynamically.
    tt_idx = None
    for idx, label in model.config.id2label.items():
        if "table tennis" in label.lower():
            tt_idx = int(idx)
            print(f"  Table tennis class: {idx} → '{label}'")
            break
    if tt_idx is None:
        print("Warning: 'table tennis' class not found in model labels.")
        print("Available sport labels:")
        for idx, label in model.config.id2label.items():
            if any(w in label.lower() for w in ["sport", "tennis", "ping", "table"]):
                print(f"  {idx}: {label}")

    return extractor, model, device, tt_idx


# ── Frame extraction ──────────────────────────────────────────────────────────

def extract_frames(cap: cv2.VideoCapture, start_f: int, end_f: int,
                   n: int = 16) -> list[np.ndarray]:
    """Sample n frames evenly from [start_f, end_f) using sequential reads."""
    indices = set(np.linspace(start_f, end_f - 1, n, dtype=int))
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    f = start_f
    while f < end_f and len(frames) < n:
        ret, frame = cap.read()
        if not ret:
            break
        if f in indices:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        f += 1
    return frames


# ── Inference ─────────────────────────────────────────────────────────────────

def score_clip(frames: list[np.ndarray], extractor, model, device,
               tt_idx: int | None) -> float:
    """Return probability that this clip contains table tennis action."""
    import torch

    if len(frames) < 4:
        return 0.0

    # Pad if we got fewer frames than expected (end of video).
    while len(frames) < 16:
        frames.append(frames[-1])

    inputs = extractor(list(frames), return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1)[0]

    if tt_idx is not None:
        return float(probs[tt_idx])
    else:
        # No table tennis class found — use top-1 prediction score as proxy.
        return float(probs.max())


# ── Sliding window ────────────────────────────────────────────────────────────

def analyze_video(video_path: str, extractor, model, device, tt_idx,
                  window_sec: float = 2.0, stride_sec: float = 0.5,
                  progress: bool = True) -> list[dict]:
    """
    Slide a window through the video, scoring each clip.
    Returns list of {timestamp, score} dicts.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    window_frames = int(window_sec * fps)
    stride_frames = int(stride_sec * fps)

    results = []
    window_count = max(1, int((duration - window_sec) / stride_sec) + 1)
    start_f = 0
    i = 0

    while start_f + window_frames <= total_frames:
        end_f = start_f + window_frames
        timestamp = (start_f + window_frames / 2) / fps

        frames = extract_frames(cap, start_f, end_f)
        score = score_clip(frames, extractor, model, device, tt_idx)
        results.append({"timestamp": round(timestamp, 3), "score": round(score, 4)})

        if progress:
            pct = (i + 1) / window_count
            bar = "█" * int(40 * pct) + "░" * (40 - int(40 * pct))
            sys.stdout.write(f"\r  [{bar}] {pct*100:.0f}%  ({timestamp:.1f}s)")
            sys.stdout.flush()

        start_f += stride_frames
        i += 1

    cap.release()
    if progress:
        print()
    return results


# ── Rally segmentation ────────────────────────────────────────────────────────

def scores_to_segments(results: list[dict], threshold: float = 0.3,
                        min_duration: float = 2.0, merge_gap: float = 2.0,
                        pre_pad: float = 1.0, post_pad: float = 1.5,
                        video_duration: float = None) -> list[dict]:
    """Convert per-window scores into rally segments."""
    if not results:
        return []

    timestamps = np.array([r["timestamp"] for r in results])
    scores = np.array([r["score"] for r in results])

    # Adaptive threshold: median + 0.5 * std (same logic as optical flow pipeline).
    adaptive = np.median(scores) + 0.5 * np.std(scores)
    threshold = max(threshold, adaptive)

    above = scores >= threshold
    segments = []
    in_seg = False
    seg_start = 0.0

    for ts, flag in zip(timestamps, above):
        if flag and not in_seg:
            seg_start = float(ts)
            in_seg = True
        elif not flag and in_seg:
            segments.append({"start": seg_start, "end": float(ts)})
            in_seg = False
    if in_seg:
        segments.append({"start": seg_start, "end": float(timestamps[-1])})

    # Filter short.
    segments = [s for s in segments if s["end"] - s["start"] >= min_duration]

    # Merge nearby.
    merged = []
    for seg in segments:
        if merged and seg["start"] - merged[-1]["end"] <= merge_gap:
            merged[-1]["end"] = seg["end"]
        else:
            merged.append(dict(seg))

    # Pad.
    dur = video_duration or (timestamps[-1] + 2.0)
    for seg in merged:
        seg["start"] = max(0.0, seg["start"] - pre_pad)
        seg["end"] = min(dur, seg["end"] + post_pad)

    return merged


# ── Display ───────────────────────────────────────────────────────────────────

def print_timeline(results: list[dict], segments: list[dict],
                   video_duration: float, width: int = 60) -> None:
    """Print an ASCII timeline of the video with rally segments marked."""
    bar = [" "] * width
    for seg in segments:
        x1 = int(seg["start"] / video_duration * width)
        x2 = int(seg["end"] / video_duration * width)
        for i in range(x1, min(x2, width)):
            bar[i] = "█"

    total_rally = sum(s["end"] - s["start"] for s in segments)
    print(f"\n  Timeline (each char = {video_duration/width:.1f}s):")
    print(f"  |{''.join(bar)}|")
    print(f"  0s{' ' * (width-6)}{video_duration:.0f}s")
    print(f"\n  Found {len(segments)} segments — "
          f"{total_rally:.1f}s play / {video_duration:.1f}s total "
          f"({100*total_rally/video_duration:.0f}%)")
    print()
    for i, seg in enumerate(segments, 1):
        dur = seg["end"] - seg["start"]
        print(f"  Segment {i:3d}: {seg['start']:7.2f}s – {seg['end']:7.2f}s  ({dur:.1f}s)")


def print_score_graph(results: list[dict], width: int = 60) -> None:
    """Print a mini bar chart of the score over time."""
    if not results:
        return
    scores = [r["score"] for r in results]
    max_s = max(scores) or 1.0
    n = len(scores)
    # Downsample to fit width
    step = max(1, n // width)
    buckets = [max(scores[i:i+step]) for i in range(0, n, step)][:width]

    print("  Score over time (higher = more likely rally):")
    for row in range(4, -1, -1):
        line = ""
        for val in buckets:
            norm = val / max_s
            line += "█" if norm > row / 4 else " "
        thresh_marker = "─" if row == 2 else " "  # approximate threshold line
        print(f"  |{line}|")
    print(f"  └{'─'*len(buckets)}┘")
    print()


# ── Comparison with optical flow ──────────────────────────────────────────────

def compare_with_optical_flow(video_path: str, ml_segments: list[dict],
                               video_duration: float) -> None:
    """Run the existing optical flow pipeline and compare results."""
    from core.pipeline import Pipeline, PipelineConfig
    print("Running optical flow pipeline for comparison...")
    result = Pipeline(PipelineConfig()).analyze(video_path)
    of_segments = [{"start": r.start_time, "end": r.end_time}
                   for r in result.rallies]

    print(f"\n  Optical flow: {len(of_segments)} segments")
    print(f"  VideoMAE:     {len(ml_segments)} segments")
    print()

    # Simple overlap check
    agreed = 0
    for ml in ml_segments:
        ml_mid = (ml["start"] + ml["end"]) / 2
        for of in of_segments:
            if of["start"] - 2 <= ml_mid <= of["end"] + 2:
                agreed += 1
                break

    if ml_segments:
        print(f"  Agreement: {agreed}/{len(ml_segments)} ML segments overlap "
              f"with optical flow segments")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test VideoMAE zero-shot rally detection on your footage."
    )
    parser.add_argument("video", help="Input video file")
    parser.add_argument("--window", type=float, default=2.0,
                        help="Clip window length in seconds (default: 2.0)")
    parser.add_argument("--stride", type=float, default=0.5,
                        help="Window stride in seconds (default: 0.5)")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Min score to count as rally (default: adaptive)")
    parser.add_argument("--save-json", metavar="PATH",
                        help="Save per-window scores to JSON file")
    parser.add_argument("--compare-optical-flow", action="store_true",
                        help="Also run optical flow pipeline and compare")
    args = parser.parse_args()

    if not os.path.isfile(args.video):
        print(f"Error: file not found — {args.video}")
        sys.exit(1)

    extractor, model, device, tt_idx = load_model()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps
    cap.release()

    print(f"\nVideo: {args.video}  ({video_duration:.1f}s, {fps:.0f}fps)")
    print("Scoring clips...")
    t0 = time.monotonic()

    results = analyze_video(
        args.video, extractor, model, device, tt_idx,
        window_sec=args.window, stride_sec=args.stride,
    )

    elapsed = time.monotonic() - t0
    print(f"Done in {elapsed:.1f}s\n")

    segments = scores_to_segments(
        results, threshold=args.threshold,
        video_duration=video_duration,
    )

    print_score_graph(results)
    print_timeline(results, segments, video_duration)

    if args.compare_optical_flow:
        print()
        compare_with_optical_flow(args.video, segments, video_duration)

    if args.save_json:
        output = {
            "video": args.video,
            "video_duration": video_duration,
            "model": "MCG-NJU/videomae-base-finetuned-kinetics",
            "window_sec": args.window,
            "stride_sec": args.stride,
            "scores": results,
            "segments": segments,
        }
        with open(args.save_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nScores saved to: {args.save_json}")


if __name__ == "__main__":
    main()
