#!/usr/bin/env python3
"""
Table Tennis Rally Detector — CLI entry point.

Usage:
    python main.py training.mp4
    python main.py training.mp4 --sensitivity 0.3
    python main.py training.mp4 --export individual
    python main.py training.mp4 --analyze-only

The default behaviour is fully hands-free: drop a video in, get a highlights
reel out next to the original file.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from core.pipeline import Pipeline, PipelineConfig
from core.video_exporter import FFmpegNotFoundError


# ── Progress bar ────────────────────────────────────────────────────────────

def _progress_bar(p: float, width: int = 40) -> None:
    filled = int(width * min(p, 1.0))
    bar = "█" * filled + "░" * (width - filled)
    sys.stdout.write(f"\r  [{bar}] {p * 100:5.1f}%")
    sys.stdout.flush()


def _clear_line() -> None:
    sys.stdout.write("\r" + " " * 60 + "\r")
    sys.stdout.flush()


# ── Argument parsing ─────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rally-detector",
        description="Extract table tennis rallies from training footage.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("video", help="Input video file path")
    p.add_argument(
        "-o", "--output",
        help="Output path (file or directory depending on --export mode). "
             "Default: <input>_highlights.mp4 / <input>_clips/ / <input>_speedup.mp4",
    )
    p.add_argument(
        "--export",
        choices=["highlights", "individual", "speedup"],
        default="highlights",
        help="Export mode.",
    )
    p.add_argument(
        "--sensitivity", type=float, default=0.5, metavar="N",
        help="Detection threshold multiplier (0.1 = very sensitive, 2.0 = strict).",
    )
    p.add_argument(
        "--min-duration", type=float, default=2.0, metavar="S",
        help="Minimum rally duration to keep (seconds).",
    )
    p.add_argument(
        "--merge-gap", type=float, default=2.0, metavar="S",
        help="Merge rallies separated by less than this gap (seconds).",
    )
    p.add_argument(
        "--sample-rate", type=int, default=3, metavar="N",
        help="Process every Nth frame (higher = faster, less precise).",
    )
    p.add_argument(
        "--speedup-factor", type=float, default=8.0, metavar="X",
        help="Dead-time speed multiplier for --export speedup.",
    )
    p.add_argument(
        "--analyze-only", action="store_true",
        help="Print detected rallies without exporting video.",
    )
    return p


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if not os.path.isfile(args.video):
        print(f"Error: file not found — {args.video}", file=sys.stderr)
        sys.exit(1)

    # Fail fast on missing FFmpeg (unless the user only wants analysis).
    if not args.analyze_only:
        try:
            from core.video_exporter import _check_ffmpeg
            _check_ffmpeg()
        except FFmpegNotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

    config = PipelineConfig(
        sample_rate=args.sample_rate,
        sensitivity=args.sensitivity,
        min_rally_duration=args.min_duration,
        merge_gap=args.merge_gap,
        speedup_factor=args.speedup_factor,
    )
    pipeline = Pipeline(config)

    # ── Analysis phase ────────────────────────────────────────────────────
    print(f"\nInput : {args.video}")
    print("Phase : analysis")
    t0 = time.monotonic()

    try:
        result = pipeline.analyze(args.video, progress_callback=_progress_bar)
    except ValueError as exc:
        _clear_line()
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)

    _clear_line()
    elapsed = time.monotonic() - t0
    print(f"Done  : analysis in {elapsed:.1f} s\n")
    print(result.summary())

    if not result.rallies:
        print(
            "\nNo rallies detected. Try lowering --sensitivity "
            f"(current: {args.sensitivity})."
        )
        sys.exit(0)

    print()
    for i, rally in enumerate(result.rallies, start=1):
        print(
            f"  Rally {i:3d} : {_fmt(rally.start_time)} – {_fmt(rally.end_time)}"
            f"  ({rally.duration:.1f} s)"
        )

    if args.analyze_only:
        sys.exit(0)

    # ── Export phase ──────────────────────────────────────────────────────
    print(f"\nPhase : export ({args.export})")
    t1 = time.monotonic()

    try:
        pipeline.run(
            video_path=args.video,
            output_path=args.output,
            export_mode=args.export,
            progress_callback=_progress_bar,
        )
    except (ValueError, RuntimeError) as exc:
        _clear_line()
        print(f"\nExport error: {exc}", file=sys.stderr)
        sys.exit(1)

    _clear_line()
    elapsed_export = time.monotonic() - t1

    # Determine what was written for the confirmation message.
    base, _ = os.path.splitext(args.video)
    if args.export == "highlights":
        written = args.output or f"{base}_highlights.mp4"
        print(f"Done  : {written}  ({elapsed_export:.1f} s)")
    elif args.export == "individual":
        written = args.output or f"{base}_clips"
        print(
            f"Done  : {len(result.rallies)} clips in {written}/  ({elapsed_export:.1f} s)"
        )
    elif args.export == "speedup":
        written = args.output or f"{base}_speedup.mp4"
        print(f"Done  : {written}  ({elapsed_export:.1f} s)")

    print()


def _fmt(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"


if __name__ == "__main__":
    main()
