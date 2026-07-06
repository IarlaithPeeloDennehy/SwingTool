"""Streaming video ingest.

Frames are yielded one at a time - a whole clip is never decoded into memory
(the 8GB-RAM constraint). Rotation metadata (portrait phone clips) is applied
by OpenCV; the first decoded frame's shape is treated as the authoritative
resolution because CAP_PROP_FRAME_WIDTH/HEIGHT may report pre-rotation values
on some backends.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class Frame:
    index: int          # index in the original video
    timestamp_s: float
    image: np.ndarray   # BGR, original resolution (post-rotation)


@dataclass
class VideoInfo:
    path: Path
    fps: float
    width: int
    height: int
    total_frames: int   # backend estimate; use for progress only


class VideoIngestError(Exception):
    pass


def compute_scale(width: int, height: int, max_dim: int | None) -> float:
    """Scale factor that brings max(width, height) down to max_dim.
    Returns 1.0 when no downscaling is needed. Works for portrait (height
    is the long side) and landscape alike."""
    if max_dim is None:
        return 1.0
    longest = max(width, height)
    if longest <= max_dim:
        return 1.0
    return max_dim / longest


def downscale(image: np.ndarray, scale: float) -> np.ndarray:
    if scale >= 1.0:
        return image
    h, w = image.shape[:2]
    return cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))),
                      interpolation=cv2.INTER_AREA)


def select_indices(total: int, stride: int) -> list[int]:
    """Original-video indices processed at a given stride (0, stride, 2*stride...)."""
    if stride < 1:
        raise ValueError(f"frame_stride must be >= 1, got {stride}")
    return list(range(0, total, stride))


class VideoReader:
    """Context manager streaming frames from a video file."""

    def __init__(self, path: Path | str, frame_stride: int = 1):
        self.path = Path(path)
        if frame_stride < 1:
            raise ValueError(f"frame_stride must be >= 1, got {frame_stride}")
        self.frame_stride = frame_stride
        self._cap: cv2.VideoCapture | None = None
        self._first_frame: np.ndarray | None = None
        self.info: VideoInfo | None = None

    def __enter__(self) -> "VideoReader":
        if not self.path.exists():
            raise VideoIngestError(f"Video not found: {self.path}")
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise VideoIngestError(
                f"Could not open video (unsupported codec or corrupt file): {self.path}"
            )
        # Apply rotation metadata so portrait phone clips decode as portrait.
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)

        ok, first = cap.read()
        if not ok or first is None:
            cap.release()
            raise VideoIngestError(f"Video contains no decodable frames: {self.path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or np.isnan(fps):
            fps = 30.0  # metadata missing; assume a sane default
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        h, w = first.shape[:2]  # authoritative, post-rotation
        self._cap = cap
        self._first_frame = first
        self.info = VideoInfo(path=self.path, fps=fps, width=w, height=h,
                              total_frames=max(total, 1))
        return self

    def __exit__(self, *exc) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def frames(self) -> Iterator[Frame]:
        """Yield frames at the configured stride, one at a time."""
        assert self._cap is not None and self.info is not None, "use as context manager"
        fps = self.info.fps
        index = 0
        image = self._first_frame
        self._first_frame = None
        while image is not None:
            if index % self.frame_stride == 0:
                yield Frame(index=index, timestamp_s=index / fps, image=image)
            ok, image = self._cap.read()
            if not ok:
                image = None
            index += 1
