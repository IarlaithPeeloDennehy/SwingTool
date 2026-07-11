import json

from swingtool.analysis import run_derive_stage
from swingtool.analysis.schema import SwingAnalysis
from swingtool.depth.schema import DepthResult, DepthSource, FrameDepth, KeypointDepth
from swingtool.detect.schema import (
    BoxDet,
    DetectionSource,
    DetectionsResult,
    FrameDetections,
)
from swingtool.schema import AnalysisResult, FramePose, Keypoint, VideoMeta

FORBIDDEN_UNITS = ("mph", "km/h", "kmh", "m/s", "meter", "metre", "mile")


def _wrist_y(i):
    if i < 20:
        return 1000.0
    if i <= 34:
        return 1000.0 - (i - 20) * (300 / 14)
    if i <= 44:
        return 700.0 + (i - 34) * (300 / 10)
    return 1000.0 - (i - 44) * (350 / 16)


def _kp_frame(i):
    wy = _wrist_y(i)
    pts = {
        "left_shoulder": (460, 900), "right_shoulder": (540, 900),
        "left_wrist": (500, wy), "right_wrist": (500, wy),
        "left_hip": (470, 1000), "right_hip": (530, 1000),
    }
    kps = [Keypoint(name=n, x=float(x), y=float(y), score=0.9) for n, (x, y) in pts.items()]
    return FramePose(frame_index=i, timestamp_s=i / 30.0, box=(400.0, 800.0, 700.0, 1150.0),
                     box_score=0.95, keypoints=kps)


def _make_keypoints():
    return AnalysisResult(
        video=VideoMeta(source_path="synthetic.mov", fps=30.0, width=1080, height=1920,
                        total_frames=60, frame_stride=1, max_dim=None,
                        pose_model="t", detector_model="t", device="cpu"),
        frames=[_kp_frame(i) for i in range(60)])


def _club_box(cx, cy, score):
    return BoxDet(x1=cx - 10, y1=cy - 10, x2=cx + 10, y2=cy + 10, score=score)


def _make_detections():
    """Club near the hands each frame, but a 4-frame gap through 'impact'
    (frames 41-44) to mimic downswing motion blur. Persistent ball off to
    the side."""
    frames = []
    for i in range(20, 51):
        wy = _wrist_y(i)
        club = [] if 41 <= i <= 44 else [_club_box(500, wy + 60, 0.5)]
        ball = [_club_box(760, 1330, 0.45), _club_box(500, 1060, 0.4)]
        frames.append(FrameDetections(frame_index=i, timestamp_s=i / 30.0, club=club, ball=ball))
    return DetectionsResult(
        source=DetectionSource(video="synthetic.mov", keypoints_path="k.json",
                               detector_model="t", device="cpu", prompt="p",
                               box_threshold=0.25, text_threshold=0.2,
                               window_start=20, window_end=50, frame_stride=1),
        frames=frames)


def _make_depth():
    """Synthetic depth: right shoulder/hip closer than left (a rotation signal),
    with more separation at the shoulders than hips (nonzero X-factor), and a
    club depth that grows so the swing plane tilts out of the image."""
    frames = []
    for i in range(20, 51):
        kps = [
            KeypointDepth(name="left_shoulder", z=1.00), KeypointDepth(name="right_shoulder", z=1.40),
            KeypointDepth(name="left_hip", z=1.00), KeypointDepth(name="right_hip", z=1.10),
            KeypointDepth(name="left_wrist", z=1.2), KeypointDepth(name="right_wrist", z=1.2),
        ]
        frames.append(FrameDepth(frame_index=i, timestamp_s=i / 30.0, frame_median=1.0,
                                 frame_scale=0.2, keypoints=kps, club_z=1.0 + 0.01 * i))
    return DepthResult(
        source=DepthSource(video="synthetic.mov", keypoints_path="k.json", detections_path="d.json",
                           depth_model="depth-test", device="cpu", window_start=20, window_end=50,
                           patch_radius=4),
        frames=frames)


def _run(tmp_path, with_depth=False):
    kp = tmp_path / "keypoints.json"
    det = tmp_path / "detections.json"
    kp.write_text(json.dumps(_make_keypoints().model_dump()), encoding="utf-8")
    det.write_text(json.dumps(_make_detections().model_dump()), encoding="utf-8")
    depth_path = None
    if with_depth:
        depth_path = tmp_path / "depth_samples.json"
        depth_path.write_text(json.dumps(_make_depth().model_dump()), encoding="utf-8")
    return run_derive_stage(kp, det, tmp_path / "out", handed="right", depth_path=depth_path)


def test_output_validates_and_is_written(tmp_path):
    analysis = _run(tmp_path)
    written = json.loads((tmp_path / "out" / "analysis.json").read_text(encoding="utf-8"))
    assert SwingAnalysis.model_validate(written) == analysis


def test_relative_only_labeling_present(tmp_path):
    analysis = _run(tmp_path)
    speed = analysis.relative_club_speed.peak
    assert speed.unit == "body_len/s"                 # normalised, not physical
    assert speed.quality in ("coarse", "not_detected")
    assert "NOT a physical speed" in speed.notes or speed.value is None


def test_no_physical_scale_units_anywhere(tmp_path):
    analysis = _run(tmp_path)
    blob = json.dumps(analysis.model_dump()).lower()
    for bad in FORBIDDEN_UNITS:
        assert bad not in blob, f"physical-scale unit leaked: {bad}"


def test_long_gap_preserved_as_holes(tmp_path):
    analysis = _run(tmp_path)
    gap_frames = [p for p in analysis.club_path if 41 <= p.frame_index <= 44]
    # 4-frame gap > max_gap(3): must remain unfilled, not fabricated
    assert all(p.x is None for p in gap_frames)
    assert all(not p.interpolated for p in gap_frames)


def test_ball_detected_at_address(tmp_path):
    analysis = _run(tmp_path)
    assert analysis.ball.address.x is not None        # persistent ball cluster found


def test_without_depth_metrics_are_not_detected(tmp_path):
    analysis = _run(tmp_path, with_depth=False)
    assert analysis.depth_assisted.swing_plane_tilt.quality == "not_detected"
    assert analysis.depth_assisted.xfactor.quality == "not_detected"
    assert analysis.source.depth_model is None


def test_with_depth_produces_approximate_metrics(tmp_path):
    analysis = _run(tmp_path, with_depth=True)
    sp = analysis.depth_assisted.swing_plane_tilt
    xf = analysis.depth_assisted.xfactor
    assert sp.value is not None and sp.quality == "depth_assisted_approximate"
    assert xf.value is not None and xf.quality == "depth_assisted_approximate"
    # shoulders rotated more than hips -> nonzero separation
    assert abs(xf.value) > 0.0
    assert analysis.source.depth_model == "depth-test"


def test_depth_metrics_claim_no_physical_scale(tmp_path):
    analysis = _run(tmp_path, with_depth=True)
    blob = json.dumps(analysis.model_dump()).lower()
    for bad in FORBIDDEN_UNITS:
        assert bad not in blob
    # depth-assisted values must be explicitly labelled approximate/not-3D
    assert "not true 3d" in analysis.depth_assisted.swing_plane_tilt.notes.lower()
