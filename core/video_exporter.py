"""
FFmpeg-based video export.

Three export modes:
  highlights  — all rallies concatenated into one file.
  individual  — each rally as a separate numbered file.
  speedup     — full video with non-rally sections sped up (dead time removed
                visually without discarding it entirely).

All segment extraction uses re-encoding (libx264 + aac) so cuts are
frame-accurate regardless of the input keyframe structure.  The final concat
step uses stream copy (-c copy) because all segments have the same codec and
parameters at that point, making it fast.

Frame-accurate cutting note
---------------------------
With stream copy you can only cut at keyframes, which can be 2-4 s apart in
standard H.264. Re-encoding at cut points costs a few extra seconds of CPU but
means every output starts/ends exactly where requested — important for rally
clips that may be as short as 2 s.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

from core.rally_detector import RallySegment


class FFmpegNotFoundError(RuntimeError):
    pass


def _check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise FFmpegNotFoundError(
            "FFmpeg not found in PATH.\n"
            "Install it from https://ffmpeg.org/download.html and make sure it\n"
            "is accessible as 'ffmpeg' on the command line."
        )


class VideoExporter:
    """
    All methods accept an optional progress_callback(float) where the float
    is in [0, 1].  This makes it straightforward to wire up a web-socket or
    GUI progress bar without changing any export logic.
    """

    def __init__(self) -> None:
        _check_ffmpeg()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def export_highlights(
        self,
        input_path: str,
        segments: list[RallySegment],
        output_path: str,
        progress_callback=None,
    ) -> None:
        """Concatenate all rally segments into a single output file."""
        if not segments:
            raise ValueError("No segments to export.")

        with tempfile.TemporaryDirectory() as tmpdir:
            seg_files = self._extract_segments(
                input_path, segments, tmpdir, progress_callback, weight=0.9
            )
            self._concat(seg_files, output_path)

        if progress_callback:
            progress_callback(1.0)

    def export_individual(
        self,
        input_path: str,
        segments: list[RallySegment],
        output_dir: str,
        progress_callback=None,
    ) -> list[str]:
        """Export each rally as rally_001.mp4, rally_002.mp4, …"""
        if not segments:
            raise ValueError("No segments to export.")

        os.makedirs(output_dir, exist_ok=True)
        output_paths: list[str] = []

        for i, seg in enumerate(segments):
            out = os.path.join(output_dir, f"rally_{i + 1:03d}.mp4")
            self._extract_segment(input_path, seg.start_time, seg.end_time, out)
            output_paths.append(out)
            if progress_callback:
                progress_callback((i + 1) / len(segments))

        return output_paths

    def export_speedup(
        self,
        input_path: str,
        segments: list[RallySegment],
        output_path: str,
        video_duration: float,
        speedup_factor: float = 8.0,
        progress_callback=None,
    ) -> None:
        """
        Full video with dead-time sections sped up by speedup_factor.
        Rally sections play at normal speed.
        """
        if not segments:
            raise ValueError("No segments provided.")

        with tempfile.TemporaryDirectory() as tmpdir:
            parts: list[str] = []
            prev_end = 0.0
            total_parts = len(segments) * 2 + 1  # rough count for progress

            part_idx = 0
            for i, seg in enumerate(segments):
                # Dead time before this rally
                if seg.start_time > prev_end + 0.1:
                    dead_out = os.path.join(tmpdir, f"dead_{i:04d}.mp4")
                    self._extract_segment_speedup(
                        input_path, prev_end, seg.start_time, dead_out, speedup_factor
                    )
                    parts.append(dead_out)
                    part_idx += 1

                # Rally at normal speed
                rally_out = os.path.join(tmpdir, f"rally_{i:04d}.mp4")
                self._extract_segment(
                    input_path, seg.start_time, seg.end_time, rally_out
                )
                parts.append(rally_out)
                part_idx += 1
                prev_end = seg.end_time

                if progress_callback:
                    progress_callback(part_idx / max(total_parts, 1) * 0.9)

            # Remaining dead time after last rally
            if prev_end < video_duration - 0.1:
                dead_out = os.path.join(tmpdir, "dead_final.mp4")
                self._extract_segment_speedup(
                    input_path, prev_end, video_duration, dead_out, speedup_factor
                )
                parts.append(dead_out)

            self._concat(parts, output_path)

        if progress_callback:
            progress_callback(1.0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_segments(
        self,
        input_path: str,
        segments: list[RallySegment],
        tmpdir: str,
        progress_callback,
        weight: float = 1.0,
    ) -> list[str]:
        seg_files: list[str] = []
        for i, seg in enumerate(segments):
            out = os.path.join(tmpdir, f"seg_{i:04d}.mp4")
            self._extract_segment(input_path, seg.start_time, seg.end_time, out)
            seg_files.append(out)
            if progress_callback:
                progress_callback((i + 1) / len(segments) * weight)
        return seg_files

    def _extract_segment(
        self, input_path: str, start: float, end: float, output_path: str
    ) -> None:
        duration = max(end - start, 0.0)
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.6f}",
            "-i", input_path,
            "-t", f"{duration:.6f}",
            # Re-encode for frame-accurate cuts.
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-c:a", "aac", "-b:a", "192k",
            "-avoid_negative_ts", "1",
            "-movflags", "+faststart",
            output_path,
        ]
        self._run(cmd)

    def _extract_segment_speedup(
        self,
        input_path: str,
        start: float,
        end: float,
        output_path: str,
        factor: float,
    ) -> None:
        duration = max(end - start, 0.0)
        if duration < 0.1:
            return  # skip negligible dead-time gaps

        # atempo filter is capped at 2.0× per stage; chain stages for higher factors.
        tempo_chain = _build_atempo_chain(factor)

        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.6f}",
            "-i", input_path,
            "-t", f"{duration:.6f}",
            "-filter_complex",
            f"[0:v]setpts=PTS/{factor:.4f}[v];[0:a]{tempo_chain}[a]",
            "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-crf", "22", "-preset", "fast",
            "-c:a", "aac", "-b:a", "128k",
            "-avoid_negative_ts", "1",
            output_path,
        ]
        self._run(cmd)

    def _concat(self, segment_files: list[str], output_path: str) -> None:
        """Write a concat list and merge all segments with stream copy."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            concat_path = f.name
            for sf in segment_files:
                # Escape single quotes in paths.
                escaped = sf.replace("'", "'\\''")
                f.write(f"file '{escaped}'\n")

        try:
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", concat_path,
                "-c", "copy",
                output_path,
            ]
            self._run(cmd)
        finally:
            os.unlink(concat_path)

    @staticmethod
    def _run(cmd: list[str]) -> None:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg command failed (exit {result.returncode}):\n"
                f"  {' '.join(cmd)}\n\n"
                f"stderr:\n{result.stderr[-2000:]}"
            )


def _build_atempo_chain(factor: float) -> str:
    """
    Build a chain of atempo filters.  Each stage is capped at 2.0×; e.g.
    8× speed = atempo=2.0,atempo=2.0,atempo=2.0.
    """
    stages: list[str] = []
    remaining = factor
    while remaining > 2.0 + 1e-9:
        stages.append("atempo=2.0")
        remaining /= 2.0
    stages.append(f"atempo={remaining:.6f}")
    return ",".join(stages)
