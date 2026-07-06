"""Swing-event detection from the conditioned wrist signal.

Sequential, not global-extremum: the follow-through finish puts the hands
HIGHER than the top of the backswing in a 2D down-the-line view, so a naive
"highest hands = top" is wrong. We instead find, in order:

  address - last quiet frame before sustained motion,
  top     - first prominent local minimum of hand height (hands highest),
  impact  - first local maximum of hand height after top (hands lowest).

Confidence-aware: raw wrist scores collapse through impact (motion blur), so
detection runs on the smoothed/interpolated signal and every event is tagged
with the raw confidence at its frame plus an `interpolated` flag.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# Raw confidence below this marks an event as interpolated / to be flagged.
EVENT_CONF_FLOOR = 0.5


def _local_extrema(y: np.ndarray) -> tuple[list[int], list[int]]:
    mins, maxs = [], []
    for i in range(1, len(y) - 1):
        if y[i] < y[i - 1] and y[i] <= y[i + 1]:
            mins.append(i)
        if y[i] > y[i - 1] and y[i] >= y[i + 1]:
            maxs.append(i)
    return mins, maxs


def _event(index: int, conf: np.ndarray) -> dict:
    c = float(conf[index]) if np.isfinite(conf[index]) else 0.0
    return {"index": int(index), "confidence": c, "interpolated": bool(c < EVENT_CONF_FLOOR)}


def detect_events(
    t: np.ndarray,
    hand_x: np.ndarray,
    hand_y: np.ndarray,
    hand_conf: np.ndarray,
    body_scale: float,
) -> Optional[dict]:
    """Detect address/top/impact from smoothed hand signals.

    hand_x/hand_y should already be interpolated + smoothed. Returns a dict
    with 'address', 'top', 'impact' (each {index, confidence, interpolated})
    or None if the sequence is too short / shows no motion.
    """
    n = len(t)
    if n < 5:
        return None

    scale = body_scale if (np.isfinite(body_scale) and body_scale > 0) else (float(np.nanstd(hand_y)) or 1.0)
    # No meaningful vertical hand travel -> there is no swing to segment.
    if (np.nanmax(hand_y) - np.nanmin(hand_y)) < 0.1 * scale:
        return None

    vx = np.gradient(hand_x, t)
    vy = np.gradient(hand_y, t)
    speed = np.hypot(vx, vy)
    peak_speed = float(np.nanmax(speed))
    if not np.isfinite(peak_speed) or peak_speed <= 0:
        return None

    motion_thr = 0.15 * peak_speed
    prominence = 0.3 * scale
    peak = int(np.nanargmax(speed))

    # Address height = median hand height over the quiet frames before the
    # swing's fastest moment (robust to a stray frame at the top, where the
    # hands momentarily stop and speed also dips).
    quiet = [i for i in range(peak) if speed[i] < motion_thr]
    baseline = float(np.median(hand_y[quiet])) if quiet else float(np.nanmedian(hand_y[: max(1, peak)]))

    mins, maxs = _local_extrema(hand_y)

    # Top: first local minimum that rose clearly above address height. The
    # pre-swing plateau has no local minimum, so this can't fire early; and
    # taking the FIRST keeps us out of the (higher) follow-through.
    top = next((i for i in mins if (baseline - hand_y[i]) >= prominence), None)
    if top is None:
        top = int(np.argmin(hand_y[: peak + 1])) if peak > 0 else int(np.argmin(hand_y))

    # Impact: first local maximum after top (hands returned to their lowest),
    # which precedes the finish, so the follow-through is never mistaken for it.
    impact = next((i for i in maxs if i > top), None)
    if impact is None:
        tail = hand_y[top + 1 :]
        impact = top + 1 + int(np.argmax(tail)) if len(tail) else min(top + 1, n - 1)

    # Address: walk back from the top past the backswing to the last quiet
    # frame - the settled posture just before takeaway.
    address = 0
    for i in range(top - 1, -1, -1):
        if speed[i] < motion_thr:
            address = i
            break

    return {
        "address": _event(address, hand_conf),
        "top": _event(top, hand_conf),
        "impact": _event(impact, hand_conf),
        "baseline_y": baseline,
    }
