"""
Pipeline — the single entry point for the processing logic.

Designed to be called identically from:
  • the CLI (main.py)
  • a future web API handler (e.g. FastAPI background task)
  • a desktop GUI worker thread

The only interface it requires from its caller is:
  • A local file path to the input video.
  • An optional progress_callback(float) where float ∈ [0, 1].
  • A local file path for the output (or None for the default).

This isolation means the web service only needs to:
  1. Download the uploaded video to a temp file.
  2. Call Pipeline(...).run(tmp_path, output_path, progress_callback=ws_send).
  3. Upload the output file to cloud storage.
  4. Delete the temp files.
No changes to Pipeline itself are required.

Example FastAPI sketch (not implemented here):

    @app.post("/process")
    async def process_video(file: UploadFile, config: PipelineConfig = Body(...)):
        tmp_in  = await save_upload(file)
        tmp_out = tmp_in.with_suffix("_highlights.mp4")
        result  = Pipeline(config).run(str(tmp_in), str(tmp_out),
                                       progress_callback=lambda p: ws.send(p))
        url = await upload_to_s3(tmp_out)
        return {"url": url, **result.to_dict()}
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable

from core.ball_tracker import BallTracker
from core.rally_detector import RallyDetector, RallySegment
from core.video_exporter import VideoExporter
from utils.video_utils import get_video_duration, check_video_readable


@dataclass
class PipelineConfig:
    # Sampling
    sample_rate: int = 3           # process every Nth frame (1 = every frame)

    # Detection
    sensitivity: float = 0.5      # adaptive threshold multiplier (lower → more sensitive)
    min_rally_duration: float = 2.0
    merge_gap: float = 2.0
    pre_padding: float = 1.0
    post_padding: float = 1.5
    smooth_window_seconds: float = 1.0

    # ROI — only count activity inside this central region of the frame.
    # Values are normalised [0, 1]; default strips ~15 % from every edge.
    # Tighten (e.g. 0.2, 0.2, 0.8, 0.8) if background objects keep triggering.
    center_roi: tuple[float, float, float, float] = (0.15, 0.15, 0.85, 0.85)

    # Export
    speedup_factor: float = 8.0


@dataclass
class PipelineResult:
    rallies: list[RallySegment]
    video_duration: float
    total_rally_duration: float

    def summary(self) -> str:
        pct = (
            100 * self.total_rally_duration / self.video_duration
            if self.video_duration > 0
            else 0
        )
        return (
            f"Found {len(self.rallies)} rallies — "
            f"{_fmt_duration(self.total_rally_duration)} play time "
            f"out of {_fmt_duration(self.video_duration)} total ({pct:.0f}%)"
        )

    def to_dict(self) -> dict:
        return {
            "rally_count": len(self.rallies),
            "video_duration": round(self.video_duration, 3),
            "total_rally_duration": round(self.total_rally_duration, 3),
            "rallies": [r.to_dict() for r in self.rallies],
        }


class Pipeline:
    """
    Orchestrates detection and optional export.

    progress_callback receives values in [0, 1]:
      0.00 – 0.90  analysis phase
      0.90 – 1.00  export phase (when run() is used)
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        video_path: str,
        progress_callback: Callable[[float], None] | None = None,
    ) -> PipelineResult:
        """Detect rallies.  Does not write any output files."""
        check_video_readable(video_path)
        cfg = self.config

        def _track_cb(p: float) -> None:
            if progress_callback:
                progress_callback(p * 0.90)

        tracker = BallTracker(
            sample_rate=cfg.sample_rate,
            center_roi=cfg.center_roi,
        )
        # Note: no warmup period needed — optical flow works from the first
        # frame pair, unlike the previous MOG2 background subtractor.
        samples = tracker.analyze(video_path, progress_callback=_track_cb)

        duration = get_video_duration(video_path)

        detector = RallyDetector(
            sensitivity=cfg.sensitivity,
            min_rally_duration=cfg.min_rally_duration,
            merge_gap=cfg.merge_gap,
            pre_padding=cfg.pre_padding,
            post_padding=cfg.post_padding,
            smooth_window_seconds=cfg.smooth_window_seconds,
        )
        rallies = detector.detect(samples, duration)

        total_rally = sum(r.duration for r in rallies)

        if progress_callback:
            progress_callback(0.90)

        return PipelineResult(
            rallies=rallies,
            video_duration=duration,
            total_rally_duration=total_rally,
        )

    def run(
        self,
        video_path: str,
        output_path: str | None = None,
        export_mode: str = "highlights",
        progress_callback: Callable[[float], None] | None = None,
    ) -> PipelineResult:
        """
        Analyze and export.

        export_mode: "highlights" | "individual" | "speedup"
        Returns the PipelineResult so callers can inspect what was found.
        """
        result = self.analyze(video_path, progress_callback=progress_callback)

        if not result.rallies:
            return result

        def _export_cb(p: float) -> None:
            if progress_callback:
                # map [0,1] export progress onto [0.90, 1.0]
                progress_callback(0.90 + p * 0.10)

        exporter = VideoExporter()
        base, _ = os.path.splitext(video_path)

        if export_mode == "highlights":
            out = output_path or f"{base}_highlights.mp4"
            exporter.export_highlights(
                video_path, result.rallies, out, progress_callback=_export_cb
            )

        elif export_mode == "individual":
            out_dir = output_path or f"{base}_clips"
            exporter.export_individual(
                video_path, result.rallies, out_dir, progress_callback=_export_cb
            )

        elif export_mode == "speedup":
            out = output_path or f"{base}_speedup.mp4"
            exporter.export_speedup(
                video_path,
                result.rallies,
                out,
                video_duration=result.video_duration,
                speedup_factor=self.config.speedup_factor,
                progress_callback=_export_cb,
            )

        else:
            raise ValueError(f"Unknown export_mode: {export_mode!r}")

        if progress_callback:
            progress_callback(1.0)

        return result


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _fmt_duration(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"
