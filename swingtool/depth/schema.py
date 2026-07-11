"""Depth sampling contract (depth_samples.json).

Stores RELATIVE depth sampled at the pose keypoints and the club-head point per
frame, plus per-frame normalisation stats (median + scale over the person box)
so downstream can compare across frames. Values are unitless and scale-free -
they are NOT distances.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

DEPTH_SCHEMA_VERSION = "1.0"


class KeypointDepth(BaseModel):
    name: str
    z: float          # relative (inverse) depth, higher = closer; NOT metric


class FrameDepth(BaseModel):
    frame_index: int
    timestamp_s: float
    frame_median: float   # median relative depth over the person box
    frame_scale: float    # robust spread (MAD-based) over the person box
    keypoints: list[KeypointDepth]
    club_z: Optional[float]   # relative depth at the club-head point (None if gap)


class DepthSource(BaseModel):
    video: str
    keypoints_path: str
    detections_path: str
    depth_model: str
    device: str
    window_start: int
    window_end: int
    patch_radius: int


class DepthResult(BaseModel):
    schema_version: str = DEPTH_SCHEMA_VERSION
    source: DepthSource
    frames: list[FrameDepth]
