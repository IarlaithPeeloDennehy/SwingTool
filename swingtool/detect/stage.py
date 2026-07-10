"""Detection stage orchestration: video + keypoints -> detections.json,
scoped to the swing window."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2
from tqdm import tqdm

from swingtool.detect.detector import PROMPT, ClubBallDetector
from swingtool.detect.schema import (
    DetectionSource,
    DetectionsResult,
    FrameDetections,
)
from swingtool.ingest import VideoReader
from swingtool.metrics.events import detect_events
from swingtool.metrics.signals import (
    body_scale,
    confident_hands,
    frame_kp_dicts,
    interp_low_conf,
    smooth,
    timestamps,
)
from swingtool.schema import AnalysisResult

DETECTOR_MODEL_ID = "IDEA-Research/grounding-dino-tiny"


def swing_window(result: AnalysisResult, pre_s: float = 0.4, post_s: float = 0.9
                 ) -> tuple[int, int]:
    """Frame-index window [start, end] covering the swing, from the Phase-2
    event detector, with lead-in/lead-out margins. Falls back to the whole
    clip if no swing is detected."""
    frames = result.frames
    kd = frame_kp_dicts(frames)
    t = timestamps(frames)
    hx, hy, hconf = confident_hands(kd, t)
    ev = detect_events(t, smooth(interp_low_conf(hx, hconf, t)),
                       smooth(interp_low_conf(hy, hconf, t)), hconf, body_scale(kd))
    lo_idx, hi_idx = 0, len(frames) - 1
    if ev is not None:
        fps = result.video.fps or 30.0
        a = ev["address"]["index"]
        i = ev["impact"]["index"]
        lo_idx = max(0, a - int(pre_s * fps))
        hi_idx = min(len(frames) - 1, i + int(post_s * fps))
    return frames[lo_idx].frame_index, frames[hi_idx].frame_index


def run_detect_stage(
    video_path: Path,
    keypoints_path: Path,
    output_dir: Path,
    device: str = "cpu",
    box_threshold: float = 0.25,
    text_threshold: float = 0.20,
    frame_stride: int = 1,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
) -> DetectionsResult:
    keypoints_path = Path(keypoints_path)
    if not keypoints_path.exists():
        raise FileNotFoundError(f"Keypoints file not found: {keypoints_path}")
    result = AnalysisResult.model_validate(json.loads(keypoints_path.read_text(encoding="utf-8")))

    win_start, win_end = swing_window(result)
    if start_frame is not None:
        win_start = start_frame
    if end_frame is not None:
        win_end = end_frame

    detector = ClubBallDetector(DETECTOR_MODEL_ID, device=device)
    frames_out: list[FrameDetections] = []
    n_expected = max(1, (win_end - win_start) // frame_stride + 1)
    try:
        with VideoReader(video_path, frame_stride=1) as reader:
            progress = tqdm(reader.frames(), total=reader.info.total_frames,
                            unit="frame", desc="Detecting")
            for frame in progress:
                if frame.index < win_start or frame.index > win_end:
                    continue
                if (frame.index - win_start) % frame_stride != 0:
                    continue
                rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
                club, ball = detector.detect(rgb, box_threshold, text_threshold)
                frames_out.append(FrameDetections(
                    frame_index=frame.index, timestamp_s=frame.timestamp_s,
                    club=club, ball=ball))
                if frame.index >= win_end:
                    break
    finally:
        detector.unload()

    detections = DetectionsResult(
        source=DetectionSource(
            video=str(video_path), keypoints_path=str(keypoints_path),
            detector_model=DETECTOR_MODEL_ID, device=device, prompt=PROMPT,
            box_threshold=box_threshold, text_threshold=text_threshold,
            window_start=win_start, window_end=win_end, frame_stride=frame_stride),
        frames=frames_out,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "detections.json").write_text(
        json.dumps(detections.model_dump(), indent=2), encoding="utf-8")
    return detections
