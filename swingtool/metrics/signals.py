"""Time-series extraction and confidence-aware conditioning.

Turns per-frame keypoints into aligned numpy arrays, interpolates over
low-confidence samples, and smooths - so downstream event detection never
trusts a raw noisy frame (the wrists blur badly through impact).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from swingtool.schema import FramePose

# Keypoints below this score are treated as gaps and interpolated over.
CONF_THRESHOLD = 0.3


def frame_kp_dicts(frames: Sequence[FramePose]) -> list[dict[str, tuple[float, float, float]]]:
    """One dict per frame mapping keypoint name -> (x, y, score)."""
    return [{k.name: (k.x, k.y, k.score) for k in f.keypoints} for f in frames]


def timestamps(frames: Sequence[FramePose]) -> np.ndarray:
    return np.array([f.timestamp_s for f in frames], dtype=float)


def series(kd: list[dict], name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (x, y, score) arrays for one keypoint; NaN where absent."""
    x = np.array([d[name][0] if name in d else np.nan for d in kd])
    y = np.array([d[name][1] if name in d else np.nan for d in kd])
    s = np.array([d[name][2] if name in d else 0.0 for d in kd])
    return x, y, s


def interp_low_conf(values: np.ndarray, conf: np.ndarray, t: np.ndarray,
                    thresh: float = CONF_THRESHOLD) -> np.ndarray:
    """Linearly re-interpolate samples whose confidence is below thresh,
    anchored on the confident samples and indexed by timestamp (so uneven
    frame spacing is handled correctly)."""
    values = np.asarray(values, dtype=float).copy()
    good = np.isfinite(values) & (conf >= thresh)
    n_good = int(good.sum())
    if n_good >= 2:
        return np.interp(t, t[good], values[good])
    if n_good == 1:
        values[:] = values[good][0]
    return values


def smooth(y: np.ndarray, window: int = 9) -> np.ndarray:
    """Savitzky-Golay smoothing when scipy is available, else a moving
    average. Window is shrunk to fit short signals and forced odd."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 3:
        return y.copy()
    w = min(window, n if n % 2 == 1 else n - 1)
    if w % 2 == 0:
        w -= 1
    if w < 3:
        return y.copy()
    try:
        from scipy.signal import savgol_filter

        return savgol_filter(y, w, min(3, w - 1))
    except Exception:
        kernel = np.ones(w) / w
        padded = np.pad(y, w // 2, mode="edge")
        return np.convolve(padded, kernel, mode="valid")


def confident_hands(kd: list[dict], t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Confidence-weighted mean of both wrists -> (x, y, conf) per frame.
    conf is the better of the two wrist scores at that frame."""
    lwx, lwy, lws = series(kd, "left_wrist")
    rwx, rwy, rws = series(kd, "right_wrist")
    wsum = lws + rws
    safe = np.where(wsum > 0, wsum, 1.0)
    hx = np.where(wsum > 0, (lwx * lws + rwx * rws) / safe, np.nan)
    hy = np.where(wsum > 0, (lwy * lws + rwy * rws) / safe, np.nan)
    hconf = np.maximum(lws, rws)
    return hx, hy, hconf


def body_scale(kd: list[dict], min_score: float = 0.4) -> float:
    """Median torso length (shoulder-midpoint to hip-midpoint) in pixels,
    used to normalise drift and to scale detection thresholds. Robust to a
    few bad frames; falls back to NaN if the torso is never clearly seen."""
    lengths = []
    for d in kd:
        needed = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
        if not all(k in d for k in needed):
            continue
        ls, rs, lh, rh = (d[k] for k in needed)
        if min(ls[2], rs[2], lh[2], rh[2]) < min_score:
            continue
        sm = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
        hm = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
        lengths.append(float(np.hypot(sm[0] - hm[0], sm[1] - hm[1])))
    return float(np.median(lengths)) if lengths else float("nan")
