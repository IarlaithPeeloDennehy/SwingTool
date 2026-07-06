import numpy as np

from swingtool.metrics.events import detect_events
from swingtool.metrics.signals import interp_low_conf, smooth


def synthetic_swing(top_frame=34, impact_frame=44, fps=30.0):
    """Build a wrist-height trajectory with a KNOWN top and impact.

    Address baseline, backswing rise to a minimum (top), downswing back to a
    local maximum (impact = hands lowest), then a follow-through that rises
    even HIGHER than the top - so a naive global-min detector would be wrong.
    """
    hy = []
    for i in range(60):
        if i < 20:
            y = 1000.0                          # address, hands low
        elif i <= top_frame:
            y = 1000.0 - (i - 20) * (300 / (top_frame - 20))   # backswing up
        elif i <= impact_frame:
            y = 700.0 + (i - top_frame) * (300 / (impact_frame - top_frame))  # down
        else:
            y = 1000.0 - (i - impact_frame) * (350 / (60 - impact_frame))     # finish higher
        hy.append(y)
    hy = np.array(hy)
    hx = np.full(60, 500.0)
    conf = np.full(60, 0.9)
    t = np.arange(60) / fps
    return t, hx, hy, conf


def test_detects_top_and_impact():
    t, hx, hy, conf = synthetic_swing(top_frame=34, impact_frame=44)
    ev = detect_events(t, smooth(hx), smooth(hy), conf, body_scale=100.0)
    assert ev is not None
    assert abs(ev["top"]["index"] - 34) <= 2
    assert abs(ev["impact"]["index"] - 44) <= 2
    assert ev["address"]["index"] < ev["top"]["index"] < ev["impact"]["index"]


def test_follow_through_not_mistaken_for_top():
    # global minimum of hand_y is the finish (frame 59), NOT the top.
    t, hx, hy, conf = synthetic_swing()
    assert int(np.argmin(hy)) >= 58          # finish is the global highest
    ev = detect_events(t, smooth(hx), smooth(hy), conf, body_scale=100.0)
    assert ev["top"]["index"] < 40           # top is the backswing peak, not the finish


def test_low_confidence_gap_at_impact_is_flagged():
    t, hx, hy, conf = synthetic_swing(top_frame=34, impact_frame=44)
    # Simulate motion blur: wrist confidence collapses through impact.
    conf[42:47] = 0.2
    # Position signal must be interpolated over the gap before detection.
    hy_i = smooth(interp_low_conf(hy, conf, t))
    hx_i = smooth(interp_low_conf(hx, conf, t))
    ev = detect_events(t, hx_i, hy_i, conf, body_scale=100.0)
    assert abs(ev["impact"]["index"] - 44) <= 2      # still found despite the gap
    assert ev["impact"]["interpolated"] is True      # and honestly flagged
    assert ev["impact"]["confidence"] < 0.5


def test_low_confidence_gap_at_top_still_detected():
    t, hx, hy, conf = synthetic_swing(top_frame=34, impact_frame=44)
    conf[32:37] = 0.15                                # wrists drop out at the top
    hy_i = smooth(interp_low_conf(hy, conf, t))
    ev = detect_events(t, smooth(hx), hy_i, conf, body_scale=100.0)
    assert abs(ev["top"]["index"] - 34) <= 2
    assert ev["top"]["interpolated"] is True


def test_returns_none_on_no_motion():
    t = np.arange(30) / 30.0
    flat = np.full(30, 1000.0)
    conf = np.full(30, 0.9)
    assert detect_events(t, flat, flat, conf, body_scale=100.0) is None
