import json

from swingtool.schema import (
    COCO_KEYPOINTS,
    COCO_SKELETON,
    SCHEMA_VERSION,
    AnalysisResult,
    FramePose,
    Keypoint,
    VideoMeta,
)


def make_pose(frame_index: int = 0) -> FramePose:
    return FramePose(
        frame_index=frame_index,
        timestamp_s=frame_index / 30.0,
        box=(10.0, 20.0, 200.0, 400.0),
        box_score=0.95,
        keypoints=[
            Keypoint(name=name, x=float(i), y=float(i * 2), score=0.5)
            for i, name in enumerate(COCO_KEYPOINTS)
        ],
    )


def make_result() -> AnalysisResult:
    return AnalysisResult(
        video=VideoMeta(
            source_path="samples/swing.mov",
            fps=30.0,
            width=1080,
            height=1920,  # portrait
            total_frames=300,
            frame_stride=2,
            max_dim=1280,
            pose_model="usyd-community/vitpose-base-simple",
            detector_model="PekingU/rtdetr_r50vd_coco_o365",
            device="cuda",
        ),
        frames=[make_pose(0), make_pose(2)],
    )


def test_keypoint_count():
    assert len(COCO_KEYPOINTS) == 17


def test_skeleton_indices_valid():
    for a, b in COCO_SKELETON:
        assert 0 <= a < 17 and 0 <= b < 17


def test_json_round_trip():
    result = make_result()
    payload = json.loads(json.dumps(result.model_dump()))
    restored = AnalysisResult.model_validate(payload)
    assert restored == result
    assert restored.schema_version == SCHEMA_VERSION


def test_portrait_dimensions_preserved():
    result = make_result()
    assert result.video.height > result.video.width
