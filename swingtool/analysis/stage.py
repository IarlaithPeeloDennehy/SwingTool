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
from swingtool.analysis.ballflight import (
    classify_shape,
    estimate_path_direction,
    estimate_start_direction,
    predict_trajectory,
)
from swingtool.analysis.schema import (
    AnalysisSource,
    BallAtAddress,
    BallFlight,
    BallInfo,
    ClubPathPoint,
    DepthAssisted,
    MetricValue,
    ObservedBallPoint,
    PredictedPoint,
    RelativeClubSpeed,
    ShotShape,
    SwingAnalysis,
)
from swingtool.analysis.spatial import fit_plane_tilt, rotation_angle_top_down, xfactor
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


def _nd(unit: str, note: str) -> MetricValue:
    return MetricValue(value=None, unit=unit, quality="not_detected", confidence=0.0, notes=note)


def _post_impact_balls(frames_plain: list[dict], impact_fi: int,
                       address_ball: Optional[dict], body_scale_px: float,
                       max_after: int = 10) -> list[dict]:
    """Real ball detections after impact that are the struck ball IN FLIGHT.

    Down-the-line footage often has extra stationary balls (the tee ball, range
    balls on the grass). A post-impact detection only counts as flight if it is
    (a) clearly displaced from the address ball AND (b) not sitting on any
    stationary cluster that already existed before impact - otherwise a static
    range ball would be mislabelled as the struck ball's flight. Usually returns
    0-3 points because the ball leaves frame fast; that emptiness is honest."""
    from swingtool.analysis.clubpath import cluster_balls

    if address_ball is None:
        return []
    ax, ay = address_ball["x"], address_ball["y"]
    scale = body_scale_px if np.isfinite(body_scale_px) and body_scale_px > 0 else 40.0
    min_disp = 0.5 * scale
    near = max(30.0, 0.35 * scale)

    # Centres of balls that sit still through the pre-impact frames (tee/range).
    stationary = [(c["x"], c["y"]) for c in cluster_balls(frames_plain)
                  if any(f <= impact_fi for f in c["frames"])]

    out: list[dict] = []
    for fr in frames_plain:
        fi = fr["frame_index"]
        if not (impact_fi < fi <= impact_fi + max_after):
            continue
        best = None
        best_d = min_disp
        for b in fr["ball"]:
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            if any(np.hypot(cx - sx, cy - sy) <= near for sx, sy in stationary):
                continue                              # a static ball, not flight
            d = float(np.hypot(cx - ax, cy - ay))
            if d > best_d:
                best_d, best = d, (cx, cy, b[4])
        if best is not None:
            out.append({"frame_index": fi, "x": best[0], "y": best[1], "confidence": best[2]})
    return out


def _ball_flight(frames_plain: list[dict], club: list[dict], impact_fi: int,
                 address_ball: Optional[dict], body_scale_px: float,
                 width: int, height: int, handed: str) -> BallFlight:
    """Assemble the honest ball-flight block: real post-impact detections, a
    MODELLED predicted arc, and a low-confidence shot-shape estimate."""
    scale = body_scale_px if np.isfinite(body_scale_px) and body_scale_px > 0 else float("nan")
    observed = _post_impact_balls(frames_plain, impact_fi, address_ball, scale)

    addr_pt = (address_ball["x"], address_ball["y"]) if address_ball else None
    start_deg, start_conf, start_note = estimate_start_direction(addr_pt, observed)
    path_deg, path_conf, path_note = estimate_path_direction(club, impact_fi, scale)
    shape, f2p, shape_conf, shape_note = classify_shape(start_deg, path_deg, handed)

    shot = ShotShape(
        shape=shape, handed=handed,
        start_direction=MetricValue(
            value=(round(start_deg, 1) if start_deg is not None else None), unit="deg",
            quality="model_estimate" if start_deg is not None else "not_detected",
            confidence=round(start_conf, 3), notes=start_note),
        club_path_direction=MetricValue(
            value=(round(path_deg, 1) if path_deg is not None else None), unit="deg",
            quality="model_estimate" if path_deg is not None else "not_detected",
            confidence=round(path_conf, 3), notes=path_note),
        face_to_path=MetricValue(
            value=f2p, unit="deg",
            quality="model_estimate" if shape is not None else "not_detected",
            confidence=round(shape_conf, 3), notes=shape_note),
        confidence=round(shape_conf, 3),
        quality="model_estimate" if shape is not None else "not_detected",
        notes="shot shape is a MODEL ESTIMATE from start direction + club path; "
              "face angle and spin are not measured - confirm with a launch monitor")

    # Predict from the last observed ball if we have one, else from the tee ball.
    launch_pt = (observed[-1]["x"], observed[-1]["y"]) if observed else addr_pt
    predicted: list[PredictedPoint] = []
    if launch_pt is not None and shape is not None:
        arc = predict_trajectory(launch_pt, start_deg, f2p, width, height, scale)
        predicted = [PredictedPoint(x=round(x, 1), y=round(y, 1)) for x, y in arc]

    if not observed and not predicted:
        note = "ball never tracked in flight and no basis to predict"
    elif not observed:
        note = "flight not observed; arc is a MODEL prediction from the swing"
    else:
        note = f"{len(observed)} real post-impact detection(s) + predicted continuation"

    return BallFlight(impact_frame=impact_fi, observed=[ObservedBallPoint(**o) for o in observed],
                      predicted=predicted, shot_shape=shot, notes=note)


def _z_to_px(z: float, fd, scale: float) -> float:
    """Normalise a relative-depth sample within its frame and rescale to pixel-
    comparable units so it can be mixed with x/y coordinates."""
    body = scale if np.isfinite(scale) and scale > 0 else 1.0
    return (z - fd.frame_median) / (fd.frame_scale or 1.0) * body


def _depth_assisted(kp: AnalysisResult, depth: DepthResult, club: list[dict],
                    window: tuple[int, int], scale: float) -> DepthAssisted:
    dframes = {fd.frame_index: fd for fd in depth.frames}
    club_xy = {p["frame_index"]: (p["x"], p["y"]) for p in club if p["x"] is not None}

    # Swing plane: lift the club path with relative depth, fit a plane.
    pts3d = []
    for fi, fd in dframes.items():
        if fd.club_z is None or fi not in club_xy:
            continue
        x, y = club_xy[fi]
        pts3d.append((x, y, _z_to_px(fd.club_z, fd, scale)))
    tilt, planarity = fit_plane_tilt(pts3d)
    if tilt is None:
        plane = _nd("deg", "too few depth-lifted club points to fit a plane")
    else:
        plane = MetricValue(
            value=round(tilt, 1), unit="deg", quality="depth_assisted_approximate",
            confidence=round(min(max(planarity, 0.0), 1.0), 3),
            notes="angle the swing plane comes out of the 2D image, from RELATIVE "
                  "depth; approximate, not true 3D")

    # X-factor at the top of the backswing (window[0]).
    top_fi = window[0]
    fd = dframes.get(top_fi) or (min(dframes.values(), key=lambda d: abs(d.frame_index - top_fi))
                                 if dframes else None)
    xf_metric = _nd("deg", "no depth at the top-of-backswing frame")
    pose_by_frame = {f.frame_index: f for f in kp.frames}
    if fd is not None and fd.frame_index in pose_by_frame:
        zmap = {k.name: k.z for k in fd.keypoints}
        kx = {k.name: (k.x, k.y) for k in pose_by_frame[fd.frame_index].keypoints}
        need = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
        if all(n in zmap and n in kx and np.isfinite(zmap[n]) for n in need):
            sh = rotation_angle_top_down((kx["left_shoulder"][0], _z_to_px(zmap["left_shoulder"], fd, scale)),
                                         (kx["right_shoulder"][0], _z_to_px(zmap["right_shoulder"], fd, scale)))
            hp = rotation_angle_top_down((kx["left_hip"][0], _z_to_px(zmap["left_hip"], fd, scale)),
                                         (kx["right_hip"][0], _z_to_px(zmap["right_hip"], fd, scale)))
            xf = xfactor(sh, hp)
            if xf is not None:
                xf_metric = MetricValue(
                    value=round(xf, 1), unit="deg", quality="depth_assisted_approximate",
                    confidence=0.3,
                    notes=f"hip/shoulder separation at top (frame {fd.frame_index}) from "
                          "relative depth; approximate upgrade of the Phase-2 approximate_2d "
                          "rotation; not true 3D")
    return DepthAssisted(swing_plane_tilt=plane, xfactor=xf_metric)


def run_derive_stage(keypoints_path: Path, detections_path: Path, output_dir: Path,
                     handed: str = "right", depth_path: Optional[Path] = None) -> SwingAnalysis:
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

    addr_dict = ({"x": ball_cluster["x"], "y": ball_cluster["y"]} if ball_cluster else None)
    ball_flight = _ball_flight(frames_plain, club, impact_fi, addr_dict, scale,
                               kp.video.width, kp.video.height, handed)

    depth_model = None
    if depth_path is not None and Path(depth_path).exists():
        from swingtool.depth.schema import DepthResult

        depth = DepthResult.model_validate(json.loads(Path(depth_path).read_text(encoding="utf-8")))
        depth_model = depth.source.depth_model
        depth_assisted = _depth_assisted(kp, depth, club, window, scale)
    else:
        depth_assisted = DepthAssisted(
            swing_plane_tilt=_nd("deg", "no depth stage run (see `depth` subcommand)"),
            xfactor=_nd("deg", "no depth stage run (see `depth` subcommand)"))

    speed_quality = "coarse" if peak is not None else "not_detected"
    speed_note = ("relative units (body-lengths/s); COARSE - downswing is ~7 frames "
                  "at 30fps, badly undersampled; NOT a physical speed")
    analysis = SwingAnalysis(
        source=AnalysisSource(
            video=det.source.video, keypoints_path=str(keypoints_path),
            detections_path=str(detections_path), detector_model=det.source.detector_model,
            depth_model=depth_model, handed=handed,
            body_scale_px=(round(scale, 1) if np.isfinite(scale) else None)),
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
        depth_assisted=depth_assisted,
        ball_flight=ball_flight,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis.json").write_text(
        json.dumps(analysis.model_dump(), indent=2), encoding="utf-8")
    return analysis
