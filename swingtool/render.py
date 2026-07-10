"""Skeleton overlay rendering.

Draws on original-resolution frames using original-resolution coordinates,
and streams annotated frames straight to disk via cv2.VideoWriter - no
frame accumulation in memory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from swingtool.ingest import VideoReader
from swingtool.schema import COCO_SKELETON, AnalysisResult, FramePose

_EDGE_COLOR = (80, 200, 80)     # BGR green
_POINT_COLOR = (60, 80, 230)    # BGR red-ish
_BOX_COLOR = (200, 160, 40)     # BGR teal-blue
_CLUB_COLOR = (0, 255, 255)     # BGR yellow  - detected club trace
_CLUB_INTERP = (0, 150, 150)    # BGR dim yellow - interpolated (short-gap) trace
_BALL_COLOR = (255, 0, 255)     # BGR magenta - struck ball


def draw_skeleton(image: np.ndarray, pose: FramePose, min_score: float = 0.3) -> np.ndarray:
    """Draw the skeleton in place and return the image. Keypoints below
    min_score are hidden (e.g. face points when filmed from behind)."""
    h, w = image.shape[:2]
    # Scale line thickness with resolution so portrait 1080x1920 looks right.
    thickness = max(2, round(min(w, h) / 300))
    radius = thickness + 2

    x1, y1, x2, y2 = (round(v) for v in pose.box)
    cv2.rectangle(image, (x1, y1), (x2, y2), _BOX_COLOR, max(1, thickness - 1))

    kps = pose.keypoints
    for a, b in COCO_SKELETON:
        if kps[a].score >= min_score and kps[b].score >= min_score:
            cv2.line(image,
                     (round(kps[a].x), round(kps[a].y)),
                     (round(kps[b].x), round(kps[b].y)),
                     _EDGE_COLOR, thickness, cv2.LINE_AA)
    for kp in kps:
        if kp.score >= min_score:
            cv2.circle(image, (round(kp.x), round(kp.y)), radius,
                       _POINT_COLOR, -1, cv2.LINE_AA)
    return image


def draw_club_trace(image: np.ndarray, line_upto: list[dict], detected_frames: set[int],
                    ball: Optional[dict], thickness: int) -> np.ndarray:
    """Draw the smooth, continuous club-head path up to the current frame as one
    unbroken polyline, with small dim dots marking where real detections backed
    it, a highlighted current head position, and the struck ball."""
    if len(line_upto) >= 2:
        pts = np.array([[round(p["x"]), round(p["y"])] for p in line_upto], dtype=np.int32)
        cv2.polylines(image, [pts], False, _CLUB_COLOR, thickness, cv2.LINE_AA)
    for p in line_upto:
        if p["frame_index"] in detected_frames:
            cv2.circle(image, (round(p["x"]), round(p["y"])), max(1, thickness - 1),
                       _CLUB_INTERP, -1, cv2.LINE_AA)
    if line_upto:
        last = line_upto[-1]
        cv2.circle(image, (round(last["x"]), round(last["y"])), thickness + 4,
                   (255, 255, 255), 2, cv2.LINE_AA)
    if ball is not None and ball.get("x") is not None:
        cv2.circle(image, (round(ball["x"]), round(ball["y"])), thickness + 6,
                   _BALL_COLOR, 2, cv2.LINE_AA)
    return image


def render_club_overlay(video_path: Path, analysis_path: Path, output_path: Path,
                        keypoints_path: Optional[Path] = None) -> Path:
    """Draw the growing, smoothed club-head path and the struck ball onto the
    video. The line is a confidence-weighted smoothing of the honest per-frame
    detections (which keep their gaps/flags in analysis.json). Streams to disk."""
    from swingtool.analysis.clubpath import fit_clean_path

    analysis = json.loads(Path(analysis_path).read_text(encoding="utf-8"))
    club = analysis["club_path"]
    ball = analysis["ball"]["address"]
    detected_frames = {p["frame_index"] for p in club if p.get("detected")}

    present = [p["frame_index"] for p in club if p["x"] is not None]
    line = fit_clean_path(club, (min(present), max(present)), bandwidth=3.0) if present else []

    poses_by_frame: dict[int, FramePose] = {}
    if keypoints_path is not None and Path(keypoints_path).exists():
        kp = AnalysisResult.model_validate(json.loads(Path(keypoints_path).read_text(encoding="utf-8")))
        poses_by_frame = {f.frame_index: f for f in kp.frames}

    with VideoReader(video_path, frame_stride=1) as reader:
        info = reader.info
        thickness = max(2, round(min(info.width, info.height) / 300))
        with OverlayWriter(output_path, info.fps, info.width, info.height) as writer:
            for frame in reader.frames():
                img = frame.image
                pose = poses_by_frame.get(frame.index)
                if pose is not None:
                    draw_skeleton(img, pose, min_score=0.3)
                line_upto = [p for p in line if p["frame_index"] <= frame.index]
                draw_club_trace(img, line_upto, detected_frames, ball, thickness)
                writer.write(img)
    return output_path


class OverlayWriter:
    """Streams annotated frames to an mp4. FPS should be the effective rate
    (source fps / frame_stride) so wall-clock duration is preserved."""

    def __init__(self, path: Path, fps: float, width: int, height: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), max(fps, 1.0), (width, height)
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {path}")

    def write(self, image: np.ndarray) -> None:
        self._writer.write(image)

    def close(self) -> None:
        self._writer.release()

    def __enter__(self) -> "OverlayWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
