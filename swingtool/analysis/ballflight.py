"""Ball-flight prediction and shot-shape classification (pure geometry).

WHAT THIS IS, HONESTLY
----------------------
Down-the-line phone footage shows the struck ball for only ~1-2 blurred frames
before it leaves frame, and this pipeline never measures club *face* angle or
spin. Shot shape (slice/hook/fade/draw) is governed by face-to-path angle and
spin axis, which a launch monitor measures directly and which we cannot observe
here. So everything in this module is a MODEL ESTIMATE from weak proxies, not a
measurement:

  * start direction  ~ face angle   (ball starts roughly where the face points)
  * club-path lean    ~ swing path   (crude 2D horizontal lean of the club-head
                                       trace through impact)
  * shape ~ sign/size of (start - path)   (standard ball-flight relationship)

Confidence is always low and every value is flagged `model_estimate`. The
predicted trajectory is a plausible image-space arc for visualisation, NOT a
tracked flight. No physical scale, distance, or spin number is ever produced.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

Point = tuple[float, float]

# Classification tolerances (OURS, not from a cited source) in degrees of
# estimated curvature (face-to-path). Deliberately coarse given the weak inputs.
CURVE_STRAIGHT_DEG = 3.0    # |face-to-path| below this reads as straight
CURVE_STRONG_DEG = 9.0      # above this reads as slice/hook rather than fade/draw

Shape = str  # "draw" | "fade" | "hook" | "slice" | "straight" | None


def _angle_from_vertical(dx: float, dy: float) -> float:
    """Angle (deg) of a vector versus the target line, which runs UP the image
    from the ball (behind-the-golfer view). Positive = leaning to the RIGHT of
    the image (right of the target line)."""
    return float(math.degrees(math.atan2(dx, -dy)))


def estimate_start_direction(
    address_ball: Optional[Point],
    post_impact_balls: list[dict],
) -> tuple[Optional[float], float, str]:
    """Start direction (~face) from the first clearly displaced post-impact ball
    detection relative to the address ball. Positive = started right of target.

    Returns (deg | None, confidence, note). None when the ball is never seen in
    flight (the common case) - that absence is honest, not fabricated.
    """
    if address_ball is None or not post_impact_balls:
        return None, 0.0, "ball not tracked after impact (leaves frame / blur)"
    first = min(post_impact_balls, key=lambda b: b["frame_index"])
    dx = first["x"] - address_ball[0]
    dy = first["y"] - address_ball[1]
    if math.hypot(dx, dy) < 1e-6:
        return None, 0.0, "post-impact ball coincident with address ball"
    deg = _angle_from_vertical(dx, dy)
    # More displaced detections -> slightly firmer (still low).
    conf = min(0.4, 0.2 + 0.05 * len(post_impact_balls))
    return deg, conf, f"from {len(post_impact_balls)} post-impact ball detection(s)"


def estimate_path_direction(
    club_path: list[dict],
    impact_frame: Optional[int],
    body_scale: float,
    span: int = 6,
) -> tuple[Optional[float], float, str]:
    """Crude swing-path lean from the horizontal drift of the club-head trace
    through impact. Positive = club moving right through the ball (in-to-out for
    a right-hander). This is a 2D proxy: the real in/out component is largely in
    DEPTH, which we don't use here, so confidence stays low.
    """
    if impact_frame is None:
        return None, 0.0, "no impact frame to anchor the path window"
    pts = [p for p in club_path
           if p.get("x") is not None
           and impact_frame - span <= p["frame_index"] <= impact_frame + 2]
    pts.sort(key=lambda p: p["frame_index"])
    if len(pts) < 2:
        return None, 0.0, "too few club-head points around impact"
    dx = pts[-1]["x"] - pts[0]["x"]
    dy = pts[-1]["y"] - pts[0]["y"]
    scale = body_scale if (np.isfinite(body_scale) and body_scale > 0) else 1.0
    # Lean of the club's horizontal drift versus its total travel.
    deg = float(math.degrees(math.atan2(dx, abs(dy) + 1e-6)))
    travel = math.hypot(dx, dy) / scale
    conf = min(0.3, 0.1 + 0.1 * travel)   # more travel -> a bit more trustworthy
    return deg, round(conf, 3), f"2D horizontal lean of club trace over {len(pts)} frames"


def classify_shape(
    start_deg: Optional[float],
    path_deg: Optional[float],
    handed: str,
) -> tuple[Optional[Shape], Optional[float], float, str]:
    """Map (start - path) to a shot shape using the standard ball-flight
    relationship: the ball curves away from the path toward the face.

    Right-handed: face right of path -> curves right -> fade/slice; face left of
    path -> curves left -> draw/hook. Left-handed mirrors the words.

    Returns (shape | None, face_to_path_deg | None, confidence, note).
    """
    if start_deg is None and path_deg is None:
        return None, None, 0.0, "no start or path signal available"

    # Face proxy = start direction. With no observed launch we fall back to a
    # square-face prior (0 deg) so a shape can still be predicted from path
    # alone - flagged, and with reduced confidence.
    if start_deg is None:
        face = 0.0
        base_conf = 0.12
        basis = "square-face prior (flight not seen); shape from club path only"
    else:
        face = start_deg
        base_conf = 0.3
        basis = "start direction as face proxy"

    path = path_deg if path_deg is not None else 0.0
    face_to_path = face - path   # >0: face open to path -> right curve (RH)

    right_curve = face_to_path > 0
    mag = abs(face_to_path)
    strong = mag >= CURVE_STRONG_DEG

    if mag < CURVE_STRAIGHT_DEG:
        shape: Shape = "straight"
    elif handed == "right":
        shape = ("slice" if strong else "fade") if right_curve else ("hook" if strong else "draw")
    else:
        shape = ("hook" if strong else "draw") if right_curve else ("slice" if strong else "fade")

    conf = base_conf
    if path_deg is None:
        conf *= 0.6
    conf = round(min(conf, 0.4), 3)
    note = f"{basis}; face-to-path {face_to_path:+.1f} deg (tolerance_ours)"
    return shape, round(face_to_path, 1), conf, note


def predict_trajectory(
    start_point: Point,
    start_deg: Optional[float],
    face_to_path_deg: Optional[float],
    width: int,
    height: int,
    body_scale: float,
    n: int = 48,
) -> list[Point]:
    """A plausible image-space flight arc for visualisation ONLY.

    Launches from `start_point` up the image, leaning by the start direction and
    bending by the estimated curvature. Height/scale are arbitrary (we have no
    metric scale), so the arc simply recedes toward the top of the frame with a
    little perspective foreshortening. This is a MODEL curve, drawn dashed and
    labelled as such - not a tracked ball.
    """
    sx, sy = start_point
    lean = math.tan(math.radians(start_deg)) if start_deg is not None else 0.0
    scale = body_scale if (np.isfinite(body_scale) and body_scale > 0) else float(min(width, height)) / 8.0

    # Rise to near the top of the frame; foreshorten so points bunch as they recede.
    rise = max(sy - 0.08 * height, 0.2 * height)
    curve_px = 0.0
    if face_to_path_deg is not None:
        # Sign: positive face-to-path -> ball bends right (+x) in the image.
        curve_px = math.copysign(min(abs(face_to_path_deg) / 12.0, 1.5) * scale, face_to_path_deg)

    pts: list[Point] = []
    for i in range(n + 1):
        t = i / n
        ease = t ** 0.7                      # perspective: fast near ball, slow far away
        y = sy - rise * ease
        x = sx + lean * (sy - y) + curve_px * (t ** 2)
        x = float(min(max(x, -0.5 * width), 1.5 * width))
        pts.append((x, float(y)))
    return pts
