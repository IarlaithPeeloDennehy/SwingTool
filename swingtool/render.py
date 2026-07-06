"""Skeleton overlay rendering.

Draws on original-resolution frames using original-resolution coordinates,
and streams annotated frames straight to disk via cv2.VideoWriter - no
frame accumulation in memory.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from swingtool.schema import COCO_SKELETON, FramePose

_EDGE_COLOR = (80, 200, 80)     # BGR green
_POINT_COLOR = (60, 80, 230)    # BGR red-ish
_BOX_COLOR = (200, 160, 40)     # BGR teal-blue


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
