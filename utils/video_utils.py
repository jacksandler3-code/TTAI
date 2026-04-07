"""
Lightweight video metadata helpers.

These are thin wrappers so the rest of the codebase never imports cv2 or calls
subprocess for metadata — only these helpers do.  That makes it straightforward
to swap the implementation (e.g. use ffprobe exclusively) without touching
callers.
"""

from __future__ import annotations

import json
import shutil
import subprocess

import cv2


def get_video_duration(video_path: str) -> float:
    """
    Return video duration in seconds.

    Tries ffprobe first (more accurate for VFR content); falls back to OpenCV.
    """
    if shutil.which("ffprobe"):
        try:
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                video_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                return float(data["format"]["duration"])
        except Exception:
            pass  # fall through to OpenCV

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return float(frames) / fps


def get_video_info(video_path: str) -> dict:
    """Return a dict with fps, total_frames, width, height, duration."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    info = {
        "fps": fps,
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    info["duration"] = info["total_frames"] / info["fps"]
    return info


def check_video_readable(video_path: str) -> None:
    """Raise ValueError if the file cannot be opened or yields no frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Cannot open video file: {video_path}")
    ret, _ = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Cannot read frames from: {video_path}")
