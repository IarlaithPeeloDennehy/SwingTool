"""Depth-assisted 3D-ish geometry: swing-plane fit and hip/shoulder separation.

Everything here is APPROXIMATE, not true 3D. Depth Anything gives relative,
per-frame-arbitrary depth, so these quantities describe orientation qualitatively
(labelled depth_assisted_approximate downstream). No metric scale is implied.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

Point3 = Sequence[float]


def fit_plane_tilt(points: Sequence[Point3]) -> tuple[Optional[float], float]:
    """Fit a plane to 3D-ish club points (x, y, depth-as-pixels) via PCA.

    Returns (tilt_deg, planarity):
      * tilt_deg   - angle between the fitted plane and the frontal image plane,
        i.e. how far the swing plane comes OUT of the 2D image using depth.
        0 = club moved purely in-image; grows as the arc spans depth.
      * planarity  - 0..1, how well the points lie on a single plane (a
        confidence proxy). Low planarity => the tilt is not trustworthy.
    """
    pts = np.asarray(points, dtype=float)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 3:
        return None, 0.0
    centered = pts - pts.mean(axis=0)
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1]
    planarity = float(1.0 - s[-1] / (s.sum() + 1e-9))
    cos = abs(float(np.dot(normal, [0.0, 0.0, 1.0]))) / (float(np.linalg.norm(normal)) + 1e-9)
    tilt = float(np.degrees(np.arccos(np.clip(cos, 0.0, 1.0))))
    return tilt, planarity


def rotation_angle_top_down(left_xz: Point3, right_xz: Point3) -> Optional[float]:
    """Angle (deg) of the left->right segment viewed from above, in the
    horizontal (x) vs depth (z) plane. A proxy for how much a body line
    (shoulders or hips) has rotated toward/away from the camera."""
    lx, lz = left_xz[0], left_xz[1]
    rx, rz = right_xz[0], right_xz[1]
    dx, dz = rx - lx, rz - lz
    if dx == 0 and dz == 0:
        return None
    return float(np.degrees(np.arctan2(dz, dx)))


def xfactor(shoulder_angle: Optional[float], hip_angle: Optional[float]) -> Optional[float]:
    """Hip/shoulder separation = shoulder rotation minus hip rotation (deg).
    Approximate: derived from relative depth, not true 3D."""
    if shoulder_angle is None or hip_angle is None:
        return None
    return float(shoulder_angle - hip_angle)
