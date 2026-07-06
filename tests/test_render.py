import numpy as np

from swingtool.render import OverlayWriter, draw_skeleton
from swingtool.schema import COCO_KEYPOINTS, FramePose, Keypoint


def make_pose(score: float) -> FramePose:
    return FramePose(
        frame_index=0,
        timestamp_s=0.0,
        box=(10.0, 10.0, 90.0, 180.0),
        box_score=0.9,
        keypoints=[
            Keypoint(name=name, x=20.0 + i * 3, y=30.0 + i * 8, score=score)
            for i, name in enumerate(COCO_KEYPOINTS)
        ],
    )


def test_draw_skeleton_modifies_image():
    image = np.zeros((200, 100, 3), dtype=np.uint8)
    out = draw_skeleton(image, make_pose(score=0.9), min_score=0.3)
    assert out.any()


def test_low_score_keypoints_hidden():
    image = np.zeros((200, 100, 3), dtype=np.uint8)
    draw_skeleton(image, make_pose(score=0.1), min_score=0.3)
    # Only the detection box should be drawn; blank out its border and
    # nothing else should remain.
    image[8:13, :] = 0
    image[178:183, :] = 0
    image[:, 8:13] = 0
    image[:, 88:93] = 0
    assert not image.any()


def test_overlay_writer(tmp_path):
    path = tmp_path / "out" / "overlay.mp4"
    with OverlayWriter(path, fps=10.0, width=100, height=200) as writer:
        for _ in range(3):
            writer.write(np.zeros((200, 100, 3), dtype=np.uint8))
    assert path.exists() and path.stat().st_size > 0
