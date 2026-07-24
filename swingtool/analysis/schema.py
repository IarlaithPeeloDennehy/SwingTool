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
# model_estimate              - a MODELLED prediction from weak proxies, NOT a
#                               measurement (e.g. shot shape / predicted flight);
#                               always low confidence, confirm with a launch monitor
# not_detected                - the model could not find it; deliberately empty
# low_confidence              - present but weakly supported
Quality = Literal[
    "reliable_2d", "relative_only", "coarse",
    "depth_assisted_approximate", "model_estimate",
    "not_detected", "low_confidence",
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


class DepthAssisted(BaseModel):
    """Approximate 3D-ish metrics from relative depth. Not true 3D."""
    swing_plane_tilt: MetricValue     # how far the swing plane comes out of the image
    xfactor: MetricValue              # hip/shoulder separation at the top


class ObservedBallPoint(BaseModel):
    """A REAL post-impact ball detection (magenta solid tracer). May be empty -
    the ball usually leaves frame in 1-2 blurred frames."""
    frame_index: int
    x: float
    y: float
    confidence: float = Field(ge=0.0, le=1.0)


class PredictedPoint(BaseModel):
    """A point on the MODELLED flight arc (image pixels). Visualisation only -
    not a tracked ball position and carries no physical scale."""
    x: float
    y: float


class ShotShape(BaseModel):
    """MODEL ESTIMATE of shot shape. Shape is driven by face-to-path angle and
    spin, which we do NOT measure; this is inferred from weak 2D proxies and is
    always low confidence. `shape` is one of draw/fade/hook/slice/straight."""
    shape: Optional[str]
    handed: str
    start_direction: MetricValue      # deg, + = started right of target (~face)
    club_path_direction: MetricValue  # deg, + = club moving right through impact
    face_to_path: MetricValue         # deg, drives the curvature
    confidence: float = Field(ge=0.0, le=1.0)
    quality: Quality
    notes: str = ""


class BallFlight(BaseModel):
    """Post-impact ball flight: the honestly-detected points, a modelled
    predicted arc, and the shot-shape estimate. Everything predicted is flagged
    `model_estimate` - confirm shape/curve with a launch monitor."""
    impact_frame: Optional[int]
    observed: list[ObservedBallPoint]
    predicted: list[PredictedPoint]
    shot_shape: ShotShape
    notes: str = ""


class AnalysisSource(BaseModel):
    video: str
    keypoints_path: str
    detections_path: str
    detector_model: str
    depth_model: Optional[str] = None
    handed: str
    body_scale_px: Optional[float]


class SwingAnalysis(BaseModel):
    schema_version: str = ANALYSIS_SCHEMA_VERSION
    source: AnalysisSource
    club_path: list[ClubPathPoint]
    relative_club_speed: RelativeClubSpeed
    ball: BallInfo
    depth_assisted: DepthAssisted
    ball_flight: Optional[BallFlight] = None
