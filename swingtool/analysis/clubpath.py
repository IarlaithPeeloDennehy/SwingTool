"""Pure geometry for club-path selection, gap handling, relative speed, and
ball anchoring. No ML, no pydantic - operates on plain tuples/dicts so the
tests can feed synthetic detection sequences with known gaps.

Honesty rules enforced here:
  * a frame with no plausible club candidate becomes a GAP (point = None),
    never a fabricated position;
  * only SHORT gaps (<= max_gap) are interpolated, and those points are
    flagged interpolated=True with reduced confidence;
  * speed is RELATIVE only (normalised by body scale), never physical.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

Box = tuple[float, float, float, float, float]   # x1, y1, x2, y2, score
Point = tuple[float, float]


def box_center(b: Box) -> Point:
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def _dist(a: Point, b: Point) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def club_head_point(b: Box, hand_pt: Optional[Point]) -> Point:
    """Estimate the club-head location from a club box: the grip end is near
    the hands, so the head is the box's far extent in the direction pointing
    AWAY from the hands. We project from the box centre along that direction by
    half the box diagonal (smooth, unlike snapping to a jittery corner). With
    no hand reference we fall back to the box centre."""
    cx, cy = box_center(b)
    if hand_pt is None:
        return cx, cy
    dx, dy = cx - hand_pt[0], cy - hand_pt[1]
    n = float(np.hypot(dx, dy))
    if n == 0:
        return cx, cy
    half_diag = 0.5 * float(np.hypot(b[2] - b[0], b[3] - b[1]))
    return cx + dx / n * half_diag, cy + dy / n * half_diag


def select_club_path(
    frames: list[dict],
    hands: dict[int, Optional[Point]],
    body_scale: float,
    max_hand_dist: float = 1.8,
    continuity_weight: float = 1.0,
) -> list[dict]:
    """Pick one club point per frame (box centroid of the chosen candidate).

    frames: [{"frame_index", "timestamp_s", "club": [Box, ...]}, ...]
    hands:  frame_index -> wrist-midpoint (or None if hands not tracked)
    Selection = detection score, penalised by distance to the previous pick
    (temporal continuity) and gated by plausibility near the hands.

    Returns per-frame dicts with x/y = None on gaps.
    """
    scale = body_scale if (np.isfinite(body_scale) and body_scale > 0) else 1.0
    prev: Optional[Point] = None
    out: list[dict] = []

    for fr in frames:
        fi = fr["frame_index"]
        hand_pt = hands.get(fi)
        best: Optional[Point] = None
        best_conf = 0.0
        best_eff = -1e9

        for b in fr.get("club", []):
            c = box_center(b)
            if hand_pt is not None and _dist(c, hand_pt) > max_hand_dist * scale:
                continue                      # grip end implausibly far from hands -> reject
            head = club_head_point(b, hand_pt)   # the point we trace (far end)
            eff = b[4]
            if prev is not None:
                eff -= continuity_weight * (_dist(head, prev) / scale)
            if eff > best_eff:
                best_eff, best, best_conf = eff, head, b[4]

        if best is None:
            out.append({"frame_index": fi, "timestamp_s": fr["timestamp_s"],
                        "x": None, "y": None, "confidence": 0.0,
                        "detected": False, "interpolated": False})
        else:
            out.append({"frame_index": fi, "timestamp_s": fr["timestamp_s"],
                        "x": best[0], "y": best[1], "confidence": best_conf,
                        "detected": True, "interpolated": False})
            prev = best
    return out


def fill_short_gaps(points: list[dict], max_gap: int = 3) -> list[dict]:
    """Linearly interpolate runs of <= max_gap consecutive gaps between two
    detected points. Longer gaps are left as holes (never fabricated)."""
    pts = [dict(p) for p in points]
    n = len(pts)
    i = 0
    while i < n:
        if pts[i]["detected"]:
            i += 1
            continue
        j = i
        while j < n and not pts[j]["detected"]:
            j += 1
        left = i - 1
        right = j
        gap_len = j - i
        if left >= 0 and right < n and gap_len <= max_gap:
            lx, ly = pts[left]["x"], pts[left]["y"]
            rx, ry = pts[right]["x"], pts[right]["y"]
            lc, rc = pts[left]["confidence"], pts[right]["confidence"]
            for k in range(i, j):
                a = (k - left) / (right - left)
                pts[k].update(x=lx + a * (rx - lx), y=ly + a * (ry - ly),
                              confidence=round(min(lc, rc) * 0.5, 3),
                              detected=False, interpolated=True)
        i = j
    return pts


def smooth_club_path(points: list[dict], window: int = 5) -> list[dict]:
    """Temporally smooth the club-head trace to damp the jitter that comes from
    extrapolating past a noisy detection box. Smoothing is applied only within
    contiguous runs of present points, so genuine gaps stay gaps (unsmoothed,
    unbridged)."""
    from swingtool.metrics.signals import smooth

    pts = [dict(p) for p in points]

    def flush(run: list[int]) -> None:
        if len(run) < 3:
            return
        xs = smooth(np.array([pts[k]["x"] for k in run]), window)
        ys = smooth(np.array([pts[k]["y"] for k in run]), window)
        for idx, k in enumerate(run):
            pts[k]["x"] = float(xs[idx])
            pts[k]["y"] = float(ys[idx])

    run: list[int] = []
    for k, p in enumerate(pts):
        if p["x"] is not None:
            run.append(k)
        else:
            flush(run)
            run = []
    flush(run)
    return pts


def fit_clean_path(points: list[dict], window: tuple[int, int],
                   bandwidth: float = 3.0) -> list[dict]:
    """Confidence-weighted kernel smoother for a clean, continuous club-head
    curve (the money-shot line). Each output frame is a Gaussian-in-time
    weighted average of the detected points, weighted ALSO by detection
    confidence - so the low-confidence jitter through the top/impact is damped
    while the high-confidence anchors (address, takeaway, downswing, finish)
    dominate. Produces a value at every frame in the window -> one unbroken line.

    This is a visualisation of the honest per-frame detections (which keep their
    gaps/flags in analysis.json); it does not change the recorded data.
    """
    obs = [(p["frame_index"], p["x"], p["y"], max(p["confidence"], 0.05))
           for p in points if p["x"] is not None]
    if len(obs) < 2:
        return []
    fs = np.array([o[0] for o in obs], dtype=float)
    xs = np.array([o[1] for o in obs], dtype=float)
    ys = np.array([o[2] for o in obs], dtype=float)
    ws = np.array([o[3] for o in obs], dtype=float)

    out: list[dict] = []
    for t in range(int(window[0]), int(window[1]) + 1):
        wt = ws * np.exp(-0.5 * ((fs - t) / bandwidth) ** 2)
        sw = float(wt.sum())
        if sw <= 1e-9:
            continue
        out.append({"frame_index": t, "x": float((wt * xs).sum() / sw),
                    "y": float((wt * ys).sum() / sw)})
    return out


def relative_speed(points: list[dict], window: tuple[int, int], body_scale: float,
                   fps: float) -> tuple[list[dict], Optional[float]]:
    """Relative club speed across [start, end] frame window, in body-lengths
    per second. Steps spanning a gap are omitted (value None), never guessed.
    Returns (per-step list, peak) - RELATIVE units only, coarse."""
    scale = body_scale if (np.isfinite(body_scale) and body_scale > 0) else 1.0
    lo, hi = window
    seq = [p for p in points if lo <= p["frame_index"] <= hi]
    steps: list[dict] = []
    peak: Optional[float] = None
    for a, b in zip(seq[:-1], seq[1:]):
        usable = (a["x"] is not None and b["x"] is not None)
        dt = (b["frame_index"] - a["frame_index"]) / fps if fps else 0.0
        if not usable or dt <= 0:
            steps.append({"frame_index": b["frame_index"], "value": None,
                          "confidence": 0.0})
            continue
        v = _dist((a["x"], a["y"]), (b["x"], b["y"])) / scale / dt
        conf = round(min(a["confidence"], b["confidence"]), 3)
        steps.append({"frame_index": b["frame_index"], "value": round(v, 3),
                      "confidence": conf})
        peak = v if peak is None else max(peak, v)
    return steps, (round(peak, 3) if peak is not None else None)


def cluster_balls(frames: list[dict], eps: float = 25.0) -> list[dict]:
    """Group ball detections across frames by spatial proximity. Returns
    clusters sorted by persistence (frame count), each with a median position,
    mean score, and frame count - used to find the stationary struck ball."""
    obs = []
    for fr in frames:
        for b in fr.get("ball", []):
            obs.append((box_center(b), b[4], fr["frame_index"]))
    clusters: list[dict] = []
    for center, score, fi in obs:
        placed = False
        for cl in clusters:
            if _dist(center, cl["_seed"]) <= eps:
                cl["pts"].append(center); cl["scores"].append(score); cl["frames"].add(fi)
                placed = True
                break
        if not placed:
            clusters.append({"_seed": center, "pts": [center], "scores": [score], "frames": {fi}})
    result = []
    for cl in clusters:
        xs = [p[0] for p in cl["pts"]]; ys = [p[1] for p in cl["pts"]]
        result.append({"x": float(np.median(xs)), "y": float(np.median(ys)),
                       "mean_score": float(np.mean(cl["scores"])),
                       "count": len(cl["frames"]),
                       "frames": sorted(cl["frames"])})
    result.sort(key=lambda c: c["count"], reverse=True)
    return result


def select_address_ball(frames: list[dict], impact_fi: Optional[int] = None,
                        eps: float = 25.0) -> Optional[dict]:
    """Identify the struck ball among candidate clusters.

    Down-the-line footage often shows several real golf balls (the teed ball
    plus range balls on the grass), so persistence/confidence alone can't tell
    them apart. The honest discriminator is behavioural: the STRUCK ball is
    present before impact and DISAPPEARS after it, whereas a stationary range
    ball persists throughout. When that signal is available we use it and mark
    the pick reliable; otherwise we fall back to the most persistent cluster
    and flag it ambiguous (via 'struck': False).
    """
    clusters = cluster_balls(frames, eps=eps)
    if not clusters:
        return None

    frame_indices = [fr["frame_index"] for fr in frames]
    have_post = impact_fi is not None and any(fi > impact_fi for fi in frame_indices)
    if have_post:
        struck = []
        for c in clusters:
            pre = [f for f in c["frames"] if f <= impact_fi]
            post = [f for f in c["frames"] if f > impact_fi]
            if len(pre) >= 3 and len(post) <= max(1, 0.25 * len(pre)):
                struck.append(c)
        if struck:
            chosen = max(struck, key=lambda c: len([f for f in c["frames"] if f <= impact_fi])
                         * c["mean_score"])
            chosen = dict(chosen); chosen["struck"] = True
            return chosen

    chosen = dict(max(clusters, key=lambda c: c["count"] * c["mean_score"]))
    chosen["struck"] = False
    return chosen
