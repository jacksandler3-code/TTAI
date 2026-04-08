"""
Motion analyzer using Farneback dense optical flow within the central ROI.

Why optical flow instead of the previous MOG2 + ball-blob approach:
  - The ball is often invisible in real footage (motion blur, overexposure,
    small size at distance). Trying to detect it directly produces a signal
    that's too noisy to threshold reliably.
  - Player and paddle motion is the strong, consistent signal. During a rally
    both players are swinging continuously. Between rallies they are largely
    still. Optical flow captures this directly.

ROI crop + resize pipeline:
  1. Crop each frame to center_roi (excludes background clutter at edges).
  2. Resize to FLOW_SIZE (320×240) regardless of input resolution. This
     makes computation fast and consistent across 720p / 1080p / 4K input.
  3. Compute Farneback flow between consecutive sampled frames.
  4. Record mean flow magnitude as the motion_energy score.

The resulting signal is smooth, well-behaved, and clearly separable between
active play (sustained high energy) and dead time (near-zero energy).
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass

# Process at this resolution regardless of input. Low enough to be fast on
# CPU, high enough to capture player/paddle motion accurately.
FLOW_SIZE = (320, 240)

FLOW_PARAMS = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0,
)


@dataclass
class FrameSample:
    timestamp: float       # seconds from start of video
    motion_energy: float   # mean optical flow magnitude in ROI (pixels/frame)


class BallTracker:
    """
    Samples every Nth frame, computes optical flow within the central ROI,
    and returns a time series of motion_energy scores.

    The name BallTracker is kept for import compatibility; internally this
    measures player/paddle motion, which is a far more reliable rally signal
    than attempting to locate the ball itself.
    """

    def __init__(
        self,
        sample_rate: int = 3,
        center_roi: tuple[float, float, float, float] = (0.15, 0.15, 0.85, 0.85),
    ):
        self.sample_rate = sample_rate
        self.center_roi = center_roi

    def analyze(
        self, video_path: str, progress_callback=None
    ) -> list[FrameSample]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Convert normalised ROI to pixel coordinates once.
        x1 = int(self.center_roi[0] * fw)
        y1 = int(self.center_roi[1] * fh)
        x2 = int(self.center_roi[2] * fw)
        y2 = int(self.center_roi[3] * fh)

        samples: list[FrameSample] = []
        prev_gray: np.ndarray | None = None
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % self.sample_rate == 0:
                gray = self._preprocess(frame, x1, y1, x2, y2)

                if prev_gray is not None:
                    energy = self._flow_energy(prev_gray, gray)
                    samples.append(
                        FrameSample(
                            timestamp=frame_idx / fps,
                            motion_energy=energy,
                        )
                    )

                prev_gray = gray

            if progress_callback and frame_idx % 60 == 0:
                progress_callback(frame_idx / max(total_frames, 1))

            frame_idx += 1

        cap.release()
        return samples

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _preprocess(
        frame: np.ndarray, x1: int, y1: int, x2: int, y2: int
    ) -> np.ndarray:
        """Crop ROI, resize to FLOW_SIZE, convert to grayscale."""
        crop = frame[y1:y2, x1:x2]
        small = cv2.resize(crop, FLOW_SIZE, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _flow_energy(prev: np.ndarray, curr: np.ndarray) -> float:
        """Return mean optical flow magnitude between two grayscale frames."""
        flow = cv2.calcOpticalFlowFarneback(prev, curr, None, **FLOW_PARAMS)
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        return float(np.mean(mag))
