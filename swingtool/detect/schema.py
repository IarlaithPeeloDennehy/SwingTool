"""Raw detection contract (detections.json).

Deliberately stores the top-K candidate boxes per class per frame rather than
a single pick - the geometry stage does confidence/continuity selection, so
this file stays an honest record of what the model actually saw (including
frames where it saw nothing: empty lists = a genuine gap, never fabricated).
"""

from __future__ import annotations

from pydantic import BaseModel

DETECTIONS_SCHEMA_VERSION = "1.0"


class BoxDet(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    score: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2


class FrameDetections(BaseModel):
    frame_index: int
    timestamp_s: float
    club: list[BoxDet]   # candidate club boxes, highest score first (may be empty)
    ball: list[BoxDet]   # candidate ball boxes, highest score first (may be empty)


class DetectionSource(BaseModel):
    video: str
    keypoints_path: str
    detector_model: str
    device: str
    prompt: str
    box_threshold: float
    text_threshold: float
    window_start: int
    window_end: int
    frame_stride: int


class DetectionsResult(BaseModel):
    schema_version: str = DETECTIONS_SCHEMA_VERSION
    source: DetectionSource
    frames: list[FrameDetections]
