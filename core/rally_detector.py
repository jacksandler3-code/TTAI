"""
Rally detection via signal processing on ball-activity time series.

Pipeline:
  1. Smooth the raw signal (rolling average) to reduce single-frame noise.
  2. Adaptive threshold: median + sensitivity * std.  This self-calibrates to
     each video so you don't need per-video tuning.
  3. Find contiguous above-threshold regions.
  4. Discard very short segments (noise / single person moving through frame).
  5. Merge segments separated by a short gap (let / brief pause mid-rally).
  6. Pad each segment slightly so the serve windup and post-point reaction are
     included.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.ball_tracker import FrameSample


@dataclass
class RallySegment:
    start_time: float   # seconds
    end_time: float     # seconds

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def to_dict(self) -> dict:
        return {
            "start_time": round(self.start_time, 3),
            "end_time": round(self.end_time, 3),
            "duration": round(self.duration, 3),
        }

    def __repr__(self) -> str:
        return f"RallySegment({self.start_time:.2f}s – {self.end_time:.2f}s, {self.duration:.1f}s)"


class RallyDetector:
    def __init__(
        self,
        sensitivity: float = 0.5,
        min_rally_duration: float = 2.0,
        merge_gap: float = 2.0,
        pre_padding: float = 1.0,
        post_padding: float = 1.5,
        smooth_window_seconds: float = 1.0,
    ):
        """
        sensitivity:
            Threshold multiplier. Lower → more sensitive (more segments, more
            false positives). Higher → stricter.  Range roughly 0.1 – 2.0.
        min_rally_duration:
            Segments shorter than this (seconds) are dropped.
        merge_gap:
            Adjacent segments separated by less than this (seconds) are merged.
        pre_padding / post_padding:
            Buffer added before/after each segment (seconds).
        smooth_window_seconds:
            Width of the rolling-average smoothing window in seconds.
        """
        self.sensitivity = sensitivity
        self.min_rally_duration = min_rally_duration
        self.merge_gap = merge_gap
        self.pre_padding = pre_padding
        self.post_padding = post_padding
        self.smooth_window_seconds = smooth_window_seconds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self, samples: list[FrameSample], video_duration: float
    ) -> list[RallySegment]:
        if not samples:
            return []

        timestamps = np.array([s.timestamp for s in samples])
        activity = np.array([s.ball_activity for s in samples])

        smoothed = self._smooth(activity, timestamps)
        is_rally = self._threshold(smoothed)
        segments = self._find_segments(timestamps, is_rally)
        segments = self._filter_short(segments)
        segments = self._merge_gaps(segments)
        segments = self._pad(segments, video_duration)
        return segments

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _smooth(self, activity: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        if len(timestamps) < 2:
            return activity.copy()
        avg_interval = float(np.median(np.diff(timestamps)))
        window = max(1, round(self.smooth_window_seconds / avg_interval))
        kernel = np.ones(window) / window
        return np.convolve(activity, kernel, mode="same")

    def _threshold(self, smoothed: np.ndarray) -> np.ndarray:
        median = np.median(smoothed)
        std = np.std(smoothed)
        threshold = median + self.sensitivity * std
        return smoothed > threshold

    def _find_segments(
        self, timestamps: np.ndarray, is_rally: np.ndarray
    ) -> list[RallySegment]:
        segments: list[RallySegment] = []
        in_rally = False
        start_ts = 0.0

        for ts, active in zip(timestamps, is_rally):
            if active and not in_rally:
                start_ts = float(ts)
                in_rally = True
            elif not active and in_rally:
                segments.append(RallySegment(start_ts, float(ts)))
                in_rally = False

        if in_rally:
            segments.append(RallySegment(start_ts, float(timestamps[-1])))

        return segments

    def _filter_short(self, segments: list[RallySegment]) -> list[RallySegment]:
        return [s for s in segments if s.duration >= self.min_rally_duration]

    def _merge_gaps(self, segments: list[RallySegment]) -> list[RallySegment]:
        if not segments:
            return []
        merged = [RallySegment(segments[0].start_time, segments[0].end_time)]
        for seg in segments[1:]:
            if seg.start_time - merged[-1].end_time <= self.merge_gap:
                merged[-1] = RallySegment(merged[-1].start_time, seg.end_time)
            else:
                merged.append(RallySegment(seg.start_time, seg.end_time))
        return merged

    def _pad(
        self, segments: list[RallySegment], video_duration: float
    ) -> list[RallySegment]:
        padded: list[RallySegment] = []
        for seg in segments:
            start = max(0.0, seg.start_time - self.pre_padding)
            end = min(video_duration, seg.end_time + self.post_padding)
            padded.append(RallySegment(start, end))
        return padded
