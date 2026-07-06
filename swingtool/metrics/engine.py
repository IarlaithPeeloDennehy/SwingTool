"""Orchestration: keypoints.json -> events -> metrics -> metrics.json.

Pure geometry. No model/video/IO logic beyond reading the pose contract and
writing the metrics contract.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from swingtool.metrics.events import detect_events
from swingtool.metrics.geometry import (
    angle_at_vertex,
    line_angle_from_horizontal,
    midpoint,
    tilt_from_vertical,
)
from swingtool.metrics.schema import (
    METRICS_SCHEMA_VERSION,
    HeadStabilityMetrics,
    KneeFlexMetrics,
    MetricsResult,
    MetricsSource,
    MetricValue,
    Metrics,
    Rotation2DMetrics,
    SpineMetrics,
    SwingEvent,
    SwingEvents,
    TempoMetrics,
)
from swingtool.metrics.signals import (
    CONF_THRESHOLD,
    body_scale,
    confident_hands,
    frame_kp_dicts,
    interp_low_conf,
    series,
    smooth,
    timestamps,
)
from swingtool.schema import AnalysisResult

_LOW = "low_confidence"


def _none_metric(unit: str, note: str) -> MetricValue:
    return MetricValue(value=None, unit=unit, quality=_LOW, confidence=0.0, notes=note)


# --- individual metric helpers -------------------------------------------------

def _knee_metric(kd: list[dict], idx: int, side: str) -> MetricValue:
    d = kd[idx]
    hip, knee, ankle = f"{side}_hip", f"{side}_knee", f"{side}_ankle"
    if not all(k in d for k in (hip, knee, ankle)):
        return _none_metric("deg", "missing leg keypoints")
    angle = angle_at_vertex(d[hip], d[knee], d[ankle])
    conf = min(d[hip][2], d[knee][2], d[ankle][2])
    if angle is None:
        return _none_metric("deg", "degenerate leg geometry")
    quality = "view_dependent" if conf >= CONF_THRESHOLD else _LOW
    return MetricValue(value=round(angle, 1), unit="deg", quality=quality,
                       confidence=round(conf, 3),
                       notes="2D hip-knee-ankle angle; magnitude depends on camera view")


def _spine_metric(kd: list[dict], idx: int) -> MetricValue:
    d = kd[idx]
    needed = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    if not all(k in d for k in needed):
        return _none_metric("deg", "missing torso keypoints")
    sm = midpoint(d["left_shoulder"], d["right_shoulder"])
    hm = midpoint(d["left_hip"], d["right_hip"])
    tilt = tilt_from_vertical(hm, sm)
    conf = min(d[k][2] for k in needed)
    if tilt is None:
        return _none_metric("deg", "degenerate torso geometry")
    quality = "view_dependent" if conf >= CONF_THRESHOLD else _LOW
    return MetricValue(value=round(tilt, 1), unit="deg", quality=quality,
                       confidence=round(conf, 3),
                       notes="torso lean vs screen-vertical; down-the-line view, still 2D")


def _line_metric(kd: list[dict], idx: int, left: str, right: str, what: str) -> MetricValue:
    d = kd[idx]
    if left not in d or right not in d:
        return _none_metric("deg", f"missing {what} keypoints")
    angle = line_angle_from_horizontal(d[left], d[right])
    conf = min(d[left][2], d[right][2])
    if angle is None:
        return _none_metric("deg", f"degenerate {what} geometry")
    return MetricValue(value=round(angle, 1), unit="deg", quality="approximate_2d",
                       confidence=round(conf, 3),
                       notes=f"projected {what} angle; cannot recover 3D rotation (Phase 3)")


def _head_point(d: dict, thresh: float) -> Optional[tuple[float, float, float]]:
    """Best available head location: nose, else eye midpoint, else ear midpoint."""
    if "nose" in d and d["nose"][2] >= thresh:
        return d["nose"][0], d["nose"][1], d["nose"][2]
    if all(k in d for k in ("left_eye", "right_eye")) and min(d["left_eye"][2], d["right_eye"][2]) >= thresh:
        c = min(d["left_eye"][2], d["right_eye"][2])
        return (d["left_eye"][0] + d["right_eye"][0]) / 2, (d["left_eye"][1] + d["right_eye"][1]) / 2, c
    if all(k in d for k in ("left_ear", "right_ear")) and min(d["left_ear"][2], d["right_ear"][2]) >= thresh:
        c = min(d["left_ear"][2], d["right_ear"][2])
        return (d["left_ear"][0] + d["right_ear"][0]) / 2, (d["left_ear"][1] + d["right_ear"][1]) / 2, c
    return None


def _head_drift(kd: list[dict], t: np.ndarray, a: int, b: int, scale: float
                ) -> tuple[MetricValue, MetricValue, MetricValue, MetricValue]:
    lo, hi = min(a, b), max(a, b)
    xs, ys, cs = [], [], []
    for d in kd[lo : hi + 1]:
        hp = _head_point(d, CONF_THRESHOLD)
        if hp is None:
            xs.append(np.nan); ys.append(np.nan); cs.append(0.0)
        else:
            xs.append(hp[0]); ys.append(hp[1]); cs.append(hp[2])
    xs, ys, cs = np.array(xs), np.array(ys), np.array(cs)
    tw = t[lo : hi + 1]
    xs = interp_low_conf(xs, cs, tw)
    ys = interp_low_conf(ys, cs, tw)
    if not (np.isfinite(xs).any() and np.isfinite(ys).any()):
        m = _none_metric("px", "head not tracked across the swing window")
        return m, m, m, m
    lat = float(np.nanmax(xs) - np.nanmin(xs))
    vert = float(np.nanmax(ys) - np.nanmin(ys))
    conf = round(float(np.mean(cs[cs > 0])) if (cs > 0).any() else 0.0, 3)
    px_note = "peak-to-peak head travel, address->impact"
    lat_px = MetricValue(value=round(lat, 1), unit="px", quality="reliable", confidence=conf, notes=px_note)
    vert_px = MetricValue(value=round(vert, 1), unit="px", quality="reliable", confidence=conf, notes=px_note)
    if np.isfinite(scale) and scale > 0:
        frac_note = "as fraction of torso length (shoulder-mid to hip-mid)"
        lat_frac = MetricValue(value=round(lat / scale, 3), unit="torso_frac", quality="reliable", confidence=conf, notes=frac_note)
        vert_frac = MetricValue(value=round(vert / scale, 3), unit="torso_frac", quality="reliable", confidence=conf, notes=frac_note)
    else:
        lat_frac = _none_metric("torso_frac", "body scale unavailable")
        vert_frac = _none_metric("torso_frac", "body scale unavailable")
    return lat_px, vert_px, lat_frac, vert_frac


# --- top-level orchestration ---------------------------------------------------

def compute_metrics(result: AnalysisResult, keypoints_path: str, handed: str) -> MetricsResult:
    frames = result.frames
    kd = frame_kp_dicts(frames)
    t = timestamps(frames)
    scale = body_scale(kd)

    hx, hy, hconf = confident_hands(kd, t)
    hx_s = smooth(interp_low_conf(hx, hconf, t))
    hy_s = smooth(interp_low_conf(hy, hconf, t))
    ev = detect_events(t, hx_s, hy_s, hconf, scale)

    lead = "left" if handed == "right" else "right"
    trail = "right" if handed == "right" else "left"

    if ev is None:
        events = SwingEvents(address=None, top=None, impact=None)
        na_deg = _none_metric("deg", "no swing detected")
        na_s = _none_metric("s", "no swing detected")
        metrics = Metrics(
            tempo=TempoMetrics(backswing_duration=na_s, downswing_duration=na_s,
                               swing_duration=na_s, tempo_ratio=_none_metric("ratio", "no swing detected")),
            head_stability=HeadStabilityMetrics(
                lateral_drift_px=_none_metric("px", "no swing detected"),
                vertical_drift_px=_none_metric("px", "no swing detected"),
                lateral_drift_frac=_none_metric("torso_frac", "no swing detected"),
                vertical_drift_frac=_none_metric("torso_frac", "no swing detected")),
            knee_flex=KneeFlexMetrics(lead_address=na_deg, trail_address=na_deg,
                                      lead_impact=na_deg, trail_impact=na_deg),
            spine=SpineMetrics(tilt_from_vertical_address=na_deg),
            rotation_2d=Rotation2DMetrics(
                shoulder_angle_address=na_deg, shoulder_angle_top=na_deg, shoulder_angle_impact=na_deg,
                hip_angle_address=na_deg, hip_angle_top=na_deg, hip_angle_impact=na_deg),
        )
        return _assemble(result, keypoints_path, handed, events, metrics)

    ai, ti, ii = ev["address"]["index"], ev["top"]["index"], ev["impact"]["index"]

    def mk_event(e: dict, note: str) -> SwingEvent:
        idx = e["index"]
        return SwingEvent(frame_index=frames[idx].frame_index, timestamp_s=round(float(t[idx]), 3),
                          confidence=round(e["confidence"], 3), interpolated=e["interpolated"], notes=note)

    events = SwingEvents(
        address=mk_event(ev["address"], "last quiet frame before motion"),
        top=mk_event(ev["top"], "hands highest in backswing"),
        impact=mk_event(ev["impact"],
                        "hands lowest (ball); wrist confidence drops here from motion blur"),
    )

    # Tempo
    backswing = float(t[ti] - t[ai])
    downswing = float(t[ii] - t[ti])
    swing = float(t[ii] - t[ai])
    ac, tc, ic = ev["address"]["confidence"], ev["top"]["confidence"], ev["impact"]["confidence"]
    tempo = TempoMetrics(
        backswing_duration=MetricValue(value=round(backswing, 3), unit="s", quality="reliable",
                                       confidence=round(min(ac, tc), 3), notes="address to top"),
        downswing_duration=MetricValue(value=round(downswing, 3), unit="s", quality="reliable",
                                       confidence=round(min(tc, ic), 3), notes="top to impact"),
        swing_duration=MetricValue(value=round(swing, 3), unit="s", quality="reliable",
                                   confidence=round(min(ac, tc, ic), 3), notes="address to impact"),
        tempo_ratio=MetricValue(value=(round(backswing / downswing, 2) if downswing > 0 else None),
                                unit="ratio", quality="reliable", confidence=round(min(ac, tc, ic), 3),
                                notes="backswing:downswing; benchmark ~3:1"),
    )

    lat_px, vert_px, lat_frac, vert_frac = _head_drift(kd, t, ai, ii, scale)
    head = HeadStabilityMetrics(lateral_drift_px=lat_px, vertical_drift_px=vert_px,
                                lateral_drift_frac=lat_frac, vertical_drift_frac=vert_frac)

    knees = KneeFlexMetrics(
        lead_address=_knee_metric(kd, ai, lead), trail_address=_knee_metric(kd, ai, trail),
        lead_impact=_knee_metric(kd, ii, lead), trail_impact=_knee_metric(kd, ii, trail),
    )
    spine = SpineMetrics(tilt_from_vertical_address=_spine_metric(kd, ai))
    rotation = Rotation2DMetrics(
        shoulder_angle_address=_line_metric(kd, ai, "left_shoulder", "right_shoulder", "shoulder-line"),
        shoulder_angle_top=_line_metric(kd, ti, "left_shoulder", "right_shoulder", "shoulder-line"),
        shoulder_angle_impact=_line_metric(kd, ii, "left_shoulder", "right_shoulder", "shoulder-line"),
        hip_angle_address=_line_metric(kd, ai, "left_hip", "right_hip", "hip-line"),
        hip_angle_top=_line_metric(kd, ti, "left_hip", "right_hip", "hip-line"),
        hip_angle_impact=_line_metric(kd, ii, "left_hip", "right_hip", "hip-line"),
    )

    metrics = Metrics(tempo=tempo, head_stability=head, knee_flex=knees, spine=spine, rotation_2d=rotation)
    return _assemble(result, keypoints_path, handed, events, metrics)


def _assemble(result: AnalysisResult, keypoints_path: str, handed: str,
              events: SwingEvents, metrics: Metrics) -> MetricsResult:
    return MetricsResult(
        schema_version=METRICS_SCHEMA_VERSION,
        source=MetricsSource(
            keypoints_path=keypoints_path,
            source_video=result.video.source_path,
            fps=result.video.fps,
            width=result.video.width,
            height=result.video.height,
            handed=handed,
            frames_analyzed=len(result.frames),
        ),
        events=events,
        metrics=metrics,
    )


def run_metrics_stage(keypoints_path: Path, output_dir: Path, handed: str = "right") -> MetricsResult:
    keypoints_path = Path(keypoints_path)
    if not keypoints_path.exists():
        raise FileNotFoundError(f"Keypoints file not found: {keypoints_path}")
    data = json.loads(keypoints_path.read_text(encoding="utf-8"))
    result = AnalysisResult.model_validate(data)
    if not result.frames:
        raise ValueError(f"Keypoints file has no frames: {keypoints_path}")

    metrics = compute_metrics(result, str(keypoints_path), handed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "metrics.json"
    out_path.write_text(json.dumps(metrics.model_dump(), indent=2), encoding="utf-8")
    return metrics
