"""
Ball/activity tracker using MOG2 background subtraction + blob size filtering.

Key insight: people walking create large foreground blobs; the ball and paddle
tips create small, fast-moving blobs. By only counting small-blob activity we
naturally discard walkers without knowing where the table is.

Blob area thresholds are expressed in pixels². At typical filming distances
(3-8m) a 40mm ball projects to roughly 50-800 px² depending on resolution and
distance. Paddles are larger but their moving tip/edge also falls in this range
during a swing. Human body parts (hands, arms) are generally > 5,000 px².
"""

import cv2
import numpy as np
from dataclasses import dataclass


# Blob area thresholds (pixels²). Tune if the camera is very close or far.
BALL_AREA_MIN = 30
BALL_AREA_MAX = 2_500
PERSON_AREA_MIN = 6_000   # blobs larger than this are treated as people/background


@dataclass
class FrameSample:
    timestamp: float      # seconds from start of video
    ball_activity: float  # normalised small-blob area (0..1 relative to frame)
    person_motion: float  # normalised large-blob area — useful for diagnostics


class BallTracker:
    """
    Analyses every Nth frame and returns a time series of ball-activity scores.

    The MOG2 subtractor needs a short warmup period (~5 s) before its background
    model is stable, so results from the warmup window are discarded.
    """

    def __init__(self, sample_rate: int = 3, warmup_seconds: float = 6.0):
        self.sample_rate = sample_rate
        self.warmup_seconds = warmup_seconds

    def analyze(
        self, video_path: str, progress_callback=None
    ) -> list[FrameSample]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_area = (
            cap.get(cv2.CAP_PROP_FRAME_WIDTH) * cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        ) or 1.0

        # Use ~20 s of history so players standing still between rallies fade
        # into the background model, making them visible again when they move.
        history = int(fps * 20)
        bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=40,
            detectShadows=False,
        )

        warmup_frames = int(self.warmup_seconds * fps)
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        samples: list[FrameSample] = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Always feed every frame to the background model so it learns
            # continuously, but only record samples at the chosen rate and
            # after the warmup window.
            fg = bg_sub.apply(frame)

            if frame_idx >= warmup_frames and frame_idx % self.sample_rate == 0:
                sample = self._score_mask(fg, morph_kernel, frame_idx / fps, frame_area)
                samples.append(sample)

            if progress_callback and frame_idx % 60 == 0:
                progress_callback(frame_idx / max(total_frames, 1))

            frame_idx += 1

        cap.release()
        return samples

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_mask(
        self,
        fg: np.ndarray,
        kernel: np.ndarray,
        timestamp: float,
        frame_area: float,
    ) -> FrameSample:
        # Remove single-pixel noise before finding contours.
        cleaned = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        ball_px = 0
        person_px = 0

        for c in contours:
            area = cv2.contourArea(c)
            if BALL_AREA_MIN <= area <= BALL_AREA_MAX:
                ball_px += area
            elif area >= PERSON_AREA_MIN:
                person_px += area
            # blobs between BALL_AREA_MAX and PERSON_AREA_MIN are ambiguous
            # (could be a hand close-up, shadow artefact, etc.) — skip them.

        return FrameSample(
            timestamp=timestamp,
            ball_activity=ball_px / frame_area,
            person_motion=person_px / frame_area,
        )
