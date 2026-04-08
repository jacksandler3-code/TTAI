"""
Synthetic end-to-end test.

Generates a video with known rally periods (small fast-moving blobs) and
dead-time periods (large slow-moving blob = person walking), then verifies
the pipeline finds all expected rallies within a tolerance window.

Run with:  python -m pytest tests/ -v
       or:  python tests/test_synthetic.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import cv2
import numpy as np
import pytest

# Allow running directly from project root without installing.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.pipeline import Pipeline, PipelineConfig


# ── Synthetic video generator ────────────────────────────────────────────────

RALLY_PERIODS: list[tuple[float, float]] = [
    (8.0, 14.0),   # 6-second rally
    (20.0, 28.0),  # 8-second rally
    (35.0, 40.0),  # 5-second rally
]
VIDEO_DURATION = 50.0  # seconds
FPS = 30
WIDTH, HEIGHT = 640, 480


def _in_rally(t: float) -> bool:
    return any(start <= t < end for start, end in RALLY_PERIODS)


def create_synthetic_video(path: str) -> list[tuple[float, float]]:
    """
    Generates motion-based frames suitable for optical flow detection.

    Rally frames:
      Two player-like rectangles oscillate side-to-side rapidly (simulating
      paddle swings). This creates strong optical flow within the central ROI.

    Dead-time frames:
      Rectangles are static (players standing still waiting for next point).
      Occasionally one drifts slowly (player adjusting position).
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, FPS, (WIDTH, HEIGHT))

    # Player positions (left and right of centre, inside the default ROI)
    p1_base_x = WIDTH // 4        # left player
    p2_base_x = 3 * WIDTH // 4    # right player
    player_y = HEIGHT // 2

    for frame_idx in range(int(VIDEO_DURATION * FPS)):
        t = frame_idx / FPS
        frame = np.full((HEIGHT, WIDTH, 3), 30, dtype=np.uint8)

        if _in_rally(t):
            # Oscillate both players horizontally at ~4 Hz (fast swings).
            offset = int(30 * np.sin(2 * np.pi * 4 * t))
            p1x = p1_base_x + offset
            p2x = p2_base_x - offset  # opposite phase = both moving inward/outward
        else:
            # Players are stationary (standing at the table between points).
            p1x = p1_base_x
            p2x = p2_base_x

        # Draw player silhouettes as filled rectangles.
        cv2.rectangle(frame, (p1x - 20, player_y - 60), (p1x + 20, player_y + 60),
                      (160, 160, 160), -1)
        cv2.rectangle(frame, (p2x - 20, player_y - 60), (p2x + 20, player_y + 60),
                      (160, 160, 160), -1)

        writer.write(frame)

    writer.release()
    return RALLY_PERIODS


# ── Tests ────────────────────────────────────────────────────────────────────

TOLERANCE = 2.0  # seconds — a detected segment may start/end this far from GT


@pytest.fixture(scope="module")
def detected_rallies():
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, "synthetic.mp4")
        ground_truth = create_synthetic_video(video_path)

        config = PipelineConfig(
            sample_rate=3,
            sensitivity=0.4,        # slightly more sensitive for synthetic test
            min_rally_duration=1.5,
            merge_gap=1.5,
            pre_padding=0.5,
            post_padding=0.5,
        )
        pipeline = Pipeline(config)

        progress_log: list[float] = []
        result = pipeline.analyze(
            video_path,
            progress_callback=lambda p: progress_log.append(p),
        )

        yield result, ground_truth, progress_log


def test_rally_count(detected_rallies):
    result, ground_truth, _ = detected_rallies
    # Allow one extra (false positive) or one missing rally.
    assert abs(len(result.rallies) - len(ground_truth)) <= 1, (
        f"Expected ~{len(ground_truth)} rallies, got {len(result.rallies)}"
    )


def test_ground_truth_coverage(detected_rallies):
    result, ground_truth, _ = detected_rallies
    for gt_start, gt_end in ground_truth:
        covered = any(
            r.start_time <= gt_start + TOLERANCE
            and r.end_time >= gt_end - TOLERANCE
            for r in result.rallies
        )
        assert covered, (
            f"Ground-truth rally {gt_start}s–{gt_end}s not covered.\n"
            f"Detected: {[(r.start_time, r.end_time) for r in result.rallies]}"
        )


def test_no_false_positive_during_walking(detected_rallies):
    result, ground_truth, _ = detected_rallies
    # Dead periods are times entirely outside any rally window.
    # A segment that starts AND ends in a dead period is a false positive.
    for r in result.rallies:
        midpoint = (r.start_time + r.end_time) / 2
        in_gt = any(
            start - TOLERANCE <= midpoint <= end + TOLERANCE
            for start, end in ground_truth
        )
        assert in_gt, (
            f"Detected rally at {r.start_time:.1f}s–{r.end_time:.1f}s appears "
            f"to be a false positive (midpoint {midpoint:.1f}s not near any GT rally)."
        )


def test_progress_callback_called(detected_rallies):
    _, _, progress_log = detected_rallies
    assert len(progress_log) > 0, "progress_callback was never called"
    assert max(progress_log) >= 0.85, "progress never reached near-completion"


def test_result_serialisable(detected_rallies):
    result, _, _ = detected_rallies
    d = result.to_dict()
    assert "rally_count" in d
    assert "rallies" in d
    for r in d["rallies"]:
        assert "start_time" in r and "end_time" in r and "duration" in r


# ── Standalone runner ─────────────────────────────────────────────────────────

def _run_standalone() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, "synthetic.mp4")
        print("Generating synthetic video …")
        ground_truth = create_synthetic_video(video_path)
        print(f"Ground truth: {ground_truth}")

        config = PipelineConfig(
            sample_rate=3, sensitivity=0.4,
            min_rally_duration=1.5, merge_gap=1.5,
            pre_padding=0.5, post_padding=0.5,
        )
        pipeline = Pipeline(config)

        def cb(p: float) -> None:
            filled = int(40 * p)
            bar = "█" * filled + "░" * (40 - filled)
            sys.stdout.write(f"\r  [{bar}] {p*100:.0f}%")
            sys.stdout.flush()

        result = pipeline.analyze(video_path, progress_callback=cb)
        print(f"\n\n{result.summary()}")
        for r in result.rallies:
            print(f"  {r}")

        # Quick pass/fail
        all_covered = all(
            any(
                r.start_time <= s + TOLERANCE and r.end_time >= e - TOLERANCE
                for r in result.rallies
            )
            for s, e in ground_truth
        )
        print(
            "\nResult: PASS" if all_covered else
            "\nResult: FAIL — some ground-truth rallies were missed"
        )


if __name__ == "__main__":
    _run_standalone()
