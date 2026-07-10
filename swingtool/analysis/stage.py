"""Derive stage: detections.json + keypoints.json -> analysis.json (pure geometry)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from swingtool.analysis.clubpath import (
    fill_short_gaps,
    relative_speed,
    select_address_ball,
    select_club_path,
    smooth_club_path,
)
from swingtool.analysis.schema import (
    AnalysisSource,
    BallAtAddress,
    BallInfo,
    ClubPathPoint,
    MetricValue,
    RelativeClubSpeed,
    SwingAnalysis,
)
from swingtool.detect.schema import DetectionsResult
from swingtool.metrics.events import detect_events
from swingtool.metrics.signals import (
    body_scale,
    confident_hands,
    frame_kp_dicts,
    interp_low_conf,
    smooth,
    timestamps,
)
from swingtool.schema import AnalysisResult


def _hands_by_frame(result: AnalysisResult) -> dict[int, Optional[tuple[float, float]]]:
    frames = result.frames
    kd = frame_kp_dicts(frames)
    t = timestamps(frames)
    hx, hy, hconf = confident_hands(kd, t)
    out: dict[int, Optional[tuple[float, float]]] = {}
    for i, f in enumerate(frames):
        if np.isfinite(hx[i]) and np.isfinite(hy[i]) and hconf[i] > 0:
            out[f.frame_index] = (float(hx[i]), float(hy[i]))
        else:
            out[f.frame_index] = None
    return out


def _events_window(result: AnalysisResult) -> Optional[tuple[int, int]]:
    frames = result.frames
    kd = frame_kp_dicts(frames)
    t = timestamps(frames)
    hx, hy, hconf = confident_hands(kd, t)
    ev = detect_events(t, smooth(interp_low_conf(hx, hconf, t)),
                       smooth(interp_low_conf(hy, hconf, t)), hconf, body_scale(kd))
    if ev is None:
        return None
    return frames[ev["top"]["index"]].frame_index, frames[ev["impact"]["index"]].frame_index


def _detections_to_plain(det: DetectionsResult) -> list[dict]:
    return [{"frame_index": f.frame_index, "timestamp_s": f.timestamp_s,
             "club": [(b.x1, b.y1, b.x2, b.y2, b.score) for b in f.club],
             "ball": [(b.x1, b.y1, b.x2, b.y2, b.score) for b in f.ball]}
            for f in det.frames]


def _launch_direction(frames_plain: list[dict], impact_fi: int,
                      address_ball: Optional[dict], body_scale_px: float) -> MetricValue:
    """Attempt launch direction (2D). The ball leaves the frame in ~1-2 blurred
    frames, so this usually returns not_detected - which is the honest answer,
    not a fabricated angle."""
    if address_ball is None or not np.isfinite(body_scale_px) or body_scale_px <= 0:
        return MetricValue(value=None, unit="deg", quality="not_detected",
                           confidence=0.0, notes="no address ball to reference")
    ab = (address_ball["x"], address_ball["y"])
    best = None
    best_d = 0.0
    for fr in frames_plain:
        if not (impact_fi < fr["frame_index"] <= impact_fi + 6):
            continue
        for b in fr["ball"]:
            c = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
            d = float(np.hypot(c[0] - ab[0], c[1] - ab[1]))
            if d > body_scale_px and d > best_d:      # clearly displaced from the tee ball
                best_d, best = d, c
    if best is None:
        return MetricValue(value=None, unit="deg", quality="not_detected", confidence=0.0,
                           notes="ball not tracked after impact (leaves frame / motion blur)")
    angle = float(np.degrees(np.arctan2(-(best[1] - ab[1]), best[0] - ab[0])))
    return MetricValue(value=round(angle, 1), unit="deg", quality="low_confidence",
                       confidence=0.3,
                       notes="2D launch vs image-horizontal; single displaced detection, treat as rough")


def run_derive_stage(keypoints_path: Path, detections_path: Path, output_dir: Path,
                     handed: str = "right") -> SwingAnalysis:
    keypoints_path = Path(keypoints_path)
    detections_path = Path(detections_path)
    for p in (keypoints_path, detections_path):
        if not p.exists():
            raise FileNotFoundError(f"Required input not found: {p}")

    kp = AnalysisResult.model_validate(json.loads(keypoints_path.read_text(encoding="utf-8")))
    det = DetectionsResult.model_validate(json.loads(detections_path.read_text(encoding="utf-8")))

    hands = _hands_by_frame(kp)
    scale = body_scale(frame_kp_dicts(kp.frames))
    frames_plain = _detections_to_plain(det)

    raw = select_club_path(frames_plain, hands, scale)
    club = smooth_club_path(fill_short_gaps(raw, max_gap=3))

    window = _events_window(kp) or (det.source.window_start, det.source.window_end)
    steps, peak = relative_speed(club, window, scale, kp.video.fps)
    impact_fi = window[1]

    ball_cluster = select_address_ball(frames_plain, impact_fi=impact_fi)
    if ball_cluster is not None:
        struck = ball_cluster.get("struck", False)
        if struck:
            note = (f"struck ball: present before impact, gone after (seen in "
                    f"{ball_cluster['count']} frames)")
        else:
            note = (f"most persistent ball, but could not confirm it as the struck ball "
                    f"(multiple balls in frame); seen in {ball_cluster['count']} frames")
        address_ball = BallAtAddress(
            frame_index=det.source.window_start,
            x=round(ball_cluster["x"], 1), y=round(ball_cluster["y"], 1),
            confidence=round(min(ball_cluster["mean_score"], 1.0), 3),
            quality="reliable_2d" if struck else "low_confidence",
            notes=note)
    else:
        address_ball = BallAtAddress(frame_index=None, x=None, y=None, confidence=0.0,
                                     quality="not_detected", notes="no ball candidates")

    launch = _launch_direction(frames_plain, impact_fi,
                               ball_cluster if ball_cluster else None, scale)

    speed_quality = "coarse" if peak is not None else "not_detected"
    speed_note = ("relative units (body-lengths/s); COARSE - downswing is ~7 frames "
                  "at 30fps, badly undersampled; NOT a physical speed")
    analysis = SwingAnalysis(
        source=AnalysisSource(
            video=det.source.video, keypoints_path=str(keypoints_path),
            detections_path=str(detections_path), detector_model=det.source.detector_model,
            handed=handed, body_scale_px=(round(scale, 1) if np.isfinite(scale) else None)),
        club_path=[ClubPathPoint(**p) for p in club],
        relative_club_speed=RelativeClubSpeed(
            window_frames=window,
            peak=MetricValue(value=peak, unit="body_len/s",
                             quality=speed_quality if peak is not None else "not_detected",
                             confidence=round(float(np.mean([s["confidence"] for s in steps if s["value"] is not None]) or 0.0), 3)
                             if any(s["value"] is not None for s in steps) else 0.0,
                             notes=speed_note),
            profile=[MetricValue(value=s["value"], unit="body_len/s",
                                 quality="coarse" if s["value"] is not None else "not_detected",
                                 confidence=s["confidence"], notes="") for s in steps]),
        ball=BallInfo(address=address_ball, launch_direction=launch),
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis.json").write_text(
        json.dumps(analysis.model_dump(), indent=2), encoding="utf-8")
    return analysis
