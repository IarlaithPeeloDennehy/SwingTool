"""The pipeline's data contract.

Every downstream stage (metrics, depth, rendering) consumes AnalysisResult.
All coordinates are pixels in the ORIGINAL video resolution, regardless of
any internal downscaling done for inference. Frame indices and timestamps
refer to the original video timeline, so --frame-stride never shifts time.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0"

# Canonical COCO-17 keypoint order. Left/right are ANATOMICAL: in footage
# filmed from behind the golfer, "left_shoulder" appears on the image's left.
COCO_KEYPOINTS: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

# Skeleton edges as index pairs into COCO_KEYPOINTS.
COCO_SKELETON: tuple[tuple[int, int], ...] = (
    (5, 7), (7, 9),          # left arm
    (6, 8), (8, 10),         # right arm
    (5, 6),                  # shoulders
    (5, 11), (6, 12),        # torso
    (11, 12),                # hips
    (11, 13), (13, 15),      # left leg
    (12, 14), (14, 16),      # right leg
    (0, 1), (0, 2), (1, 3), (2, 4),  # face
)


class Keypoint(BaseModel):
    name: str
    x: float
    y: float
    score: float = Field(ge=0.0, le=1.0)


class FramePose(BaseModel):
    frame_index: int
    timestamp_s: float
    box: tuple[float, float, float, float]  # person bbox (x1, y1, x2, y2)
    box_score: float
    keypoints: list[Keypoint]  # always all 17; filter by score downstream


class VideoMeta(BaseModel):
    source_path: str
    fps: float
    width: int   # after rotation metadata is applied (portrait clips are tall)
    height: int
    total_frames: int
    frame_stride: int
    max_dim: int | None
    pose_model: str
    detector_model: str
    device: str


class AnalysisResult(BaseModel):
    schema_version: str = SCHEMA_VERSION
    video: VideoMeta
    frames: list[FramePose]
