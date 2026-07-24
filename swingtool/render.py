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
_FLIGHT_OBS = (255, 0, 255)     # BGR magenta - REAL post-impact ball flight (solid)
_FLIGHT_PRED = (60, 160, 255)   # BGR orange  - MODELLED predicted flight (dashed)

_SHAPE_COLOR = {                 # BGR, for the end-card headline
    "draw": (120, 220, 120), "fade": (120, 220, 120),
    "hook": (80, 120, 240), "slice": (80, 120, 240),
    "straight": (230, 230, 230),
}


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


def _dashed_polyline(image: np.ndarray, pts: np.ndarray, color, thickness: int,
                     dash: int = 14, gap: int = 10) -> None:
    """Draw a polyline as dashes - marks the MODELLED (predicted) flight so it
    reads as an estimate, visually distinct from the solid observed tracer."""
    if len(pts) < 2:
        return
    carry = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        seg = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        if seg < 1e-6:
            continue
        d = 0.0
        while d < seg:
            phase = (carry + d) % (dash + gap)
            drawing = phase < dash
            step = (dash - phase) if drawing else (dash + gap - phase)
            step = max(1.0, min(step, seg - d))
            if drawing:
                p0 = a + (b - a) * (d / seg)
                p1 = a + (b - a) * (min(d + step, seg) / seg)
                cv2.line(image, (round(p0[0]), round(p0[1])),
                         (round(p1[0]), round(p1[1])), color, thickness, cv2.LINE_AA)
            d += step
        carry += seg


def draw_ball_flight(image: np.ndarray, observed: list[dict], predicted: list[dict],
                     frame_index: int, impact_frame: Optional[int], thickness: int,
                     reveal_span: int = 15) -> np.ndarray:
    """Draw the real (solid) post-impact ball points up to the current frame and
    progressively reveal the modelled (dashed) predicted arc after impact."""
    seen = [p for p in observed if p["frame_index"] <= frame_index]
    if len(seen) >= 2:
        pts = np.array([[round(p["x"]), round(p["y"])] for p in seen], dtype=np.int32)
        cv2.polylines(image, [pts], False, _FLIGHT_OBS, thickness + 1, cv2.LINE_AA)
    for p in seen:
        cv2.circle(image, (round(p["x"]), round(p["y"])), thickness + 2, _FLIGHT_OBS, -1, cv2.LINE_AA)

    if predicted and impact_frame is not None and frame_index >= impact_frame:
        frac = min(1.0, (frame_index - impact_frame) / max(reveal_span, 1))
        k = max(2, int(len(predicted) * frac))
        arc = np.array([[p["x"], p["y"]] for p in predicted[:k]], dtype=np.float64)
        _dashed_polyline(image, arc, _FLIGHT_PRED, thickness)
    return image


def _put_centered(image: np.ndarray, text: str, cy: int, scale: float, color,
                  thickness: int, font=cv2.FONT_HERSHEY_SIMPLEX) -> None:
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = (image.shape[1] - tw) // 2
    cv2.putText(image, text, (x, cy + th // 2), font, scale, color, thickness, cv2.LINE_AA)


def make_endcard(width: int, height: int, shot_shape: dict) -> np.ndarray:
    """A dark end-card announcing the predicted shot shape, with the honesty
    caveat baked into the frame (it's a model estimate, not a launch monitor)."""
    card = np.full((height, width, 3), 22, dtype=np.uint8)
    base = min(width, height) / 900.0
    shape = shot_shape.get("shape")
    color = _SHAPE_COLOR.get(shape, (230, 230, 230))

    _put_centered(card, "PREDICTED SHOT", int(height * 0.30), 1.1 * base, (160, 160, 160), max(1, round(2 * base)))
    headline = (shape.upper() if shape else "UNDETERMINED")
    _put_centered(card, headline, int(height * 0.45), 3.4 * base, color, max(2, round(6 * base)))

    conf = shot_shape.get("confidence", 0.0)
    _put_centered(card, f"model estimate - confidence {conf:.2f}", int(height * 0.57),
                  0.9 * base, (170, 170, 170), max(1, round(2 * base)))

    f2p = (shot_shape.get("face_to_path") or {}).get("value")
    start = (shot_shape.get("start_direction") or {}).get("value")
    bits = []
    if start is not None:
        bits.append(f"start {start:+.0f} deg")
    if f2p is not None:
        bits.append(f"face-to-path {f2p:+.0f} deg")
    if bits:
        _put_centered(card, "  |  ".join(bits), int(height * 0.64), 0.8 * base, (150, 150, 150), max(1, round(2 * base)))

    for i, line in enumerate((
        "Curvature and spin are NOT measured.",
        "Confirm shot shape with a launch monitor.",
    )):
        _put_centered(card, line, int(height * (0.74 + 0.05 * i)), 0.7 * base, (120, 120, 120), max(1, round(1 * base)))
    return card


def render_club_overlay(video_path: Path, analysis_path: Path, output_path: Path,
                        keypoints_path: Optional[Path] = None,
                        endcard_seconds: float = 2.5) -> Path:
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

    flight = analysis.get("ball_flight") or {}
    observed = flight.get("observed", [])
    predicted = flight.get("predicted", [])
    impact_frame = flight.get("impact_frame")
    shot_shape = flight.get("shot_shape")

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
                draw_ball_flight(img, observed, predicted, frame.index, impact_frame, thickness)
                writer.write(img)

            # End-card: hold the predicted shot shape for a couple of seconds.
            if shot_shape is not None and endcard_seconds > 0:
                card = make_endcard(info.width, info.height, shot_shape)
                for _ in range(max(1, round(info.fps * endcard_seconds))):
                    writer.write(card.copy())
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
