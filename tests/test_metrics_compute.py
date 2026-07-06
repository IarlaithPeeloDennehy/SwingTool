import json

from swingtool.metrics import run_metrics_stage
from swingtool.metrics.engine import compute_metrics
from swingtool.metrics.schema import MetricsResult
from swingtool.schema import AnalysisResult, FramePose, Keypoint, VideoMeta


def _wrist_y(i: int) -> float:
    if i < 20:
        return 1000.0
    if i <= 34:
        return 1000.0 - (i - 20) * (300 / 14)      # backswing up, top at 34
    if i <= 44:
        return 700.0 + (i - 34) * (300 / 10)        # downswing, impact (lowest) at 44
    return 1000.0 - (i - 44) * (350 / 16)           # finish, even higher


def _frame(i: int, fps: float = 30.0) -> FramePose:
    wy = _wrist_y(i)
    # Static legs with a KNOWN 90-degree knee angle; static vertical torso.
    pts = {
        "nose": (500, 850), "left_eye": (490, 845), "right_eye": (510, 845),
        "left_ear": (480, 850), "right_ear": (520, 850),
        "left_shoulder": (460, 900), "right_shoulder": (540, 900),
        "left_elbow": (470, 960), "right_elbow": (530, 960),
        "left_wrist": (500, wy), "right_wrist": (500, wy),
        "left_hip": (470, 1000), "right_hip": (530, 1000),
        "left_knee": (470, 1100), "right_knee": (530, 1100),
        "left_ankle": (570, 1100), "right_ankle": (630, 1100),
    }
    kps = [Keypoint(name=n, x=float(x), y=float(y), score=0.9) for n, (x, y) in pts.items()]
    return FramePose(frame_index=i, timestamp_s=i / fps, box=(400.0, 800.0, 700.0, 1150.0),
                     box_score=0.95, keypoints=kps)


def _make_result() -> AnalysisResult:
    return AnalysisResult(
        video=VideoMeta(source_path="synthetic.mov", fps=30.0, width=1080, height=1920,
                        total_frames=60, frame_stride=1, max_dim=None,
                        pose_model="test", detector_model="test", device="cpu"),
        frames=[_frame(i) for i in range(60)],
    )


def test_known_knee_angle():
    r = compute_metrics(_make_result(), "synthetic.json", handed="right")
    # left_knee at (470,1100): knee->hip = (0,-100), knee->ankle = (100,0) -> 90 deg
    assert r.metrics.knee_flex.lead_address.value == 90.0
    assert r.metrics.knee_flex.lead_address.quality == "view_dependent"


def test_known_spine_tilt_is_vertical():
    r = compute_metrics(_make_result(), "synthetic.json", handed="right")
    # shoulder-mid (500,900) directly above hip-mid (500,1000) -> 0 deg tilt
    assert r.metrics.spine.tilt_from_vertical_address.value == 0.0


def test_events_have_expected_order_and_tempo():
    r = compute_metrics(_make_result(), "synthetic.json", handed="right")
    assert r.events.address.frame_index < r.events.top.frame_index < r.events.impact.frame_index
    assert abs(r.events.top.frame_index - 34) <= 2
    # backswing (~0.5s) longer than downswing (~0.33s) -> ratio > 1
    assert r.metrics.tempo.tempo_ratio.value > 1.0


def test_rotation_flagged_approximate():
    r = compute_metrics(_make_result(), "synthetic.json", handed="right")
    assert r.metrics.rotation_2d.shoulder_angle_address.quality == "approximate_2d"


def test_run_metrics_stage_writes_valid_json(tmp_path):
    kp_path = tmp_path / "keypoints.json"
    kp_path.write_text(json.dumps(_make_result().model_dump()), encoding="utf-8")
    out_dir = tmp_path / "out"

    result = run_metrics_stage(kp_path, out_dir, handed="right")

    written = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    restored = MetricsResult.model_validate(written)   # round-trips against the schema
    assert restored == result
    assert restored.schema_version == "1.0"


def test_missing_file_raises(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        run_metrics_stage(tmp_path / "nope.json", tmp_path, handed="right")
