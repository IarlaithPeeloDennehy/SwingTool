"""2D geometry helpers. Image coordinates: x right, y DOWN.

Every function returns a plain float in degrees, or None when the input is
degenerate (zero-length vector). No 3D is inferred anywhere.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

Point = Sequence[float]

_VERTICAL_UP = np.array([0.0, -1.0])  # screen-up, since y grows downward


def _xy(p: Point) -> np.ndarray:
    return np.asarray(p[:2], dtype=float)


def angle_at_vertex(a: Point, b: Point, c: Point) -> Optional[float]:
    """Interior angle at vertex b for the path a-b-c, in degrees.
    180 = straight (e.g. fully extended leg), smaller = more flexed."""
    v1 = _xy(a) - _xy(b)
    v2 = _xy(c) - _xy(b)
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 == 0.0 or n2 == 0.0:
        return None
    cos = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def tilt_from_vertical(bottom: Point, top: Point) -> Optional[float]:
    """Angle between the vector bottom->top and screen-vertical, in degrees.
    0 = perfectly upright; grows as the segment leans."""
    v = _xy(top) - _xy(bottom)
    n = float(np.linalg.norm(v))
    if n == 0.0:
        return None
    cos = np.clip(np.dot(v, _VERTICAL_UP) / n, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def line_angle_from_horizontal(p1: Point, p2: Point) -> Optional[float]:
    """Signed angle of the vector p1->p2 from the horizontal axis, in degrees,
    in (-180, 180]. Positive means p2 is higher on screen than p1."""
    a = _xy(p1)
    b = _xy(p2)
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    if dx == 0.0 and dy == 0.0:
        return None
    return float(np.degrees(np.arctan2(-dy, dx)))


def midpoint(p1: Point, p2: Point) -> np.ndarray:
    return (_xy(p1) + _xy(p2)) / 2.0
