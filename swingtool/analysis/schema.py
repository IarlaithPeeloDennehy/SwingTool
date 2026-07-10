"""Output contract for analysis.json (Phase 3A).

Same self-documenting style as metrics.json: every derived value carries
units and a quality/confidence flag. Quality vocabulary is extended for the
honesty constraints of this phase - nothing here claims a physical scale.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

ANALYSIS_SCHEMA_VERSION = "1.0"

# reliable_2d                 - trustworthy as a 2D pixel measurement
# relative_only               - normalised/relative units; NOT a physical scale
# coarse                      - severely undersampled (e.g. ~7-frame downswing)
# depth_assisted_approximate  - lifted with relative depth; not true 3D (Phase 3B)
# not_detected                - the model could not find it; deliberately empty
# low_confidence              - present but weakly supported
Quality = Literal[
    "reliable_2d", "relative_only", "coarse",
    "depth_assisted_approximate", "not_detected", "low_confidence",
]


class MetricValue(BaseModel):
    value: Optional[float]
    unit: str
    quality: Quality
    confidence: float = Field(ge=0.0, le=1.0)
    notes: str = ""


class ClubPathPoint(BaseModel):
    frame_index: int
    timestamp_s: float
    x: Optional[float]          # null on a genuine gap - never fabricated
    y: Optional[float]
    confidence: float = Field(ge=0.0, le=1.0)
    detected: bool              # True if a real detection backed this point
    interpolated: bool          # True if filled across a SHORT gap (<= max_gap)


class BallAtAddress(BaseModel):
    frame_index: Optional[int]
    x: Optional[float]
    y: Optional[float]
    confidence: float = Field(ge=0.0, le=1.0)
    quality: Quality
    notes: str = ""


class BallInfo(BaseModel):
    address: BallAtAddress
    launch_direction: MetricValue


class RelativeClubSpeed(BaseModel):
    window_frames: Optional[tuple[int, int]]
    peak: MetricValue
    profile: list[MetricValue]   # per-step relative speed across the window


class AnalysisSource(BaseModel):
    video: str
    keypoints_path: str
    detections_path: str
    detector_model: str
    handed: str
    body_scale_px: Optional[float]


class SwingAnalysis(BaseModel):
    schema_version: str = ANALYSIS_SCHEMA_VERSION
    source: AnalysisSource
    club_path: list[ClubPathPoint]
    relative_club_speed: RelativeClubSpeed
    ball: BallInfo
