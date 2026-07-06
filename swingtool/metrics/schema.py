"""Output contract for metrics.json.

Mirrors the Phase-1 style: a schema_version, a reference to the source, and
per-metric values that each carry units and a quality/confidence flag. No
faked 3D - anything view-dependent or projected is labelled as such.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

METRICS_SCHEMA_VERSION = "1.0"

# reliable        - survives 2D projection (timing, image-plane displacement)
# view_dependent  - a real angle, but its magnitude depends on camera view
# approximate_2d  - projected only; cannot recover true 3D (rotation groundwork)
# low_confidence  - underlying keypoints too weak to trust this value
Quality = Literal["reliable", "view_dependent", "approximate_2d", "low_confidence"]


class MetricValue(BaseModel):
    value: Optional[float]
    unit: str
    quality: Quality
    confidence: float = Field(ge=0.0, le=1.0)
    notes: str = ""


class SwingEvent(BaseModel):
    frame_index: int
    timestamp_s: float
    confidence: float = Field(ge=0.0, le=1.0)
    interpolated: bool
    notes: str = ""


class SwingEvents(BaseModel):
    address: Optional[SwingEvent]
    top: Optional[SwingEvent]
    impact: Optional[SwingEvent]


class TempoMetrics(BaseModel):
    backswing_duration: MetricValue
    downswing_duration: MetricValue
    swing_duration: MetricValue
    tempo_ratio: MetricValue


class HeadStabilityMetrics(BaseModel):
    lateral_drift_px: MetricValue
    vertical_drift_px: MetricValue
    lateral_drift_frac: MetricValue
    vertical_drift_frac: MetricValue


class KneeFlexMetrics(BaseModel):
    lead_address: MetricValue
    trail_address: MetricValue
    lead_impact: MetricValue
    trail_impact: MetricValue


class SpineMetrics(BaseModel):
    tilt_from_vertical_address: MetricValue


class Rotation2DMetrics(BaseModel):
    shoulder_angle_address: MetricValue
    shoulder_angle_top: MetricValue
    shoulder_angle_impact: MetricValue
    hip_angle_address: MetricValue
    hip_angle_top: MetricValue
    hip_angle_impact: MetricValue


class Metrics(BaseModel):
    tempo: TempoMetrics
    head_stability: HeadStabilityMetrics
    knee_flex: KneeFlexMetrics
    spine: SpineMetrics
    rotation_2d: Rotation2DMetrics


class MetricsSource(BaseModel):
    keypoints_path: str
    source_video: str
    fps: float
    width: int
    height: int
    handed: str
    frames_analyzed: int


class MetricsResult(BaseModel):
    schema_version: str = METRICS_SCHEMA_VERSION
    source: MetricsSource
    events: SwingEvents
    metrics: Metrics
