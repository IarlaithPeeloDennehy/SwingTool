"""Ball-flight prediction + shot-shape classification (pure geometry)."""

from swingtool.analysis.ballflight import (
    classify_shape,
    estimate_path_direction,
    estimate_start_direction,
    predict_trajectory,
)


def _club(points):
    return [{"frame_index": fi, "x": x, "y": y} for fi, x, y in points]


def test_start_direction_right_and_left():
    addr = (500.0, 1000.0)
    right = estimate_start_direction(addr, [{"frame_index": 51, "x": 560.0, "y": 900.0}])
    left = estimate_start_direction(addr, [{"frame_index": 51, "x": 440.0, "y": 900.0}])
    assert right[0] is not None and right[0] > 0     # started right of target
    assert left[0] is not None and left[0] < 0       # started left
    assert 0 < right[1] <= 0.4                        # confidence stays low


def test_start_direction_absent_when_no_flight():
    deg, conf, _ = estimate_start_direction((500.0, 1000.0), [])
    assert deg is None and conf == 0.0               # honest absence, not fabricated


def test_path_direction_sign_tracks_horizontal_drift():
    out = _club([(45, 480, 1100), (48, 495, 1050), (50, 510, 1010)])
    deg, conf, _ = estimate_path_direction(out, impact_frame=50, body_scale=200.0)
    assert deg is not None and deg > 0               # club drifting right -> in-to-out (RH)
    assert conf <= 0.3


def test_classify_right_handed_four_shapes():
    # face (start) relative to path drives the curve; RH words.
    assert classify_shape(12.0, 0.0, "right")[0] == "slice"    # big right curve
    assert classify_shape(5.0, 0.0, "right")[0] == "fade"      # small right curve
    assert classify_shape(-12.0, 0.0, "right")[0] == "hook"    # big left curve
    assert classify_shape(-5.0, 0.0, "right")[0] == "draw"     # small left curve
    assert classify_shape(1.0, 0.0, "right")[0] == "straight"  # inside deadband


def test_classify_left_handed_mirrors():
    # Same geometry, left-hander: a right-bending ball is a hook, not a slice.
    assert classify_shape(12.0, 0.0, "left")[0] == "hook"
    assert classify_shape(-12.0, 0.0, "left")[0] == "slice"


def test_classify_confidence_is_low_and_flagged():
    shape, f2p, conf, note = classify_shape(12.0, 0.0, "right")
    assert conf <= 0.4                               # never presented as certain
    assert f2p == 12.0
    assert "tolerance_ours" in note


def test_classify_no_signal_returns_none():
    shape, f2p, conf, _ = classify_shape(None, None, "right")
    assert shape is None and f2p is None and conf == 0.0


def test_classify_square_face_prior_when_only_path():
    # No observed launch: falls back to square-face prior, lower confidence.
    with_start = classify_shape(0.0, 8.0, "right")
    prior = classify_shape(None, 8.0, "right")
    assert prior[0] is not None
    assert prior[2] < with_start[2]                  # prior is less confident


def test_predict_trajectory_shape_and_launch():
    arc = predict_trajectory((500.0, 1000.0), 10.0, 8.0, 1080, 1920, 200.0, n=20)
    assert len(arc) == 21
    assert arc[0] == (500.0, 1000.0)                 # launches from the ball
    assert arc[-1][1] < arc[0][1]                    # flies UP the image (y shrinks)
