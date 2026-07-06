"""Pose stage: stream frames -> detect person -> estimate keypoints -> stream
overlay to disk. Models are loaded at stage start and VRAM is freed at stage
end, so future stages (depth, club detection) can follow in the same process
without exceeding 6GB.
"""

from __future__ import annotations

import json

import cv2
from tqdm import tqdm

from swingtool.config import AnalyzeConfig, resolve_device
from swingtool.ingest import VideoReader, compute_scale, downscale
from swingtool.render import OverlayWriter, draw_skeleton
from swingtool.schema import (
    COCO_KEYPOINTS,
    AnalysisResult,
    FramePose,
    Keypoint,
    VideoMeta,
)


def run_pose_stage(config: AnalyzeConfig) -> AnalysisResult:
    device = resolve_device(config.device)

    from swingtool.pose.detector import PersonDetector
    from swingtool.pose.estimator import PoseEstimator

    with VideoReader(config.video_path, config.frame_stride) as reader:
        info = reader.info
        scale = compute_scale(info.width, info.height, config.max_dim)

        detector = PersonDetector(config.detector_model, device)
        estimator = PoseEstimator(config.pose_model, device)

        frames: list[FramePose] = []
        effective_fps = info.fps / config.frame_stride
        expected = max(1, -(-info.total_frames // config.frame_stride))  # ceil

        config.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            with OverlayWriter(config.overlay_path, effective_fps,
                               info.width, info.height) as overlay:
                progress = tqdm(reader.frames(), total=expected, unit="frame",
                                desc="Analyzing")
                for frame in progress:
                    small = downscale(frame.image, scale)
                    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

                    detection = detector.detect_best_person(rgb, config.min_box_score)
                    if detection is not None:
                        box_small, box_score = detection
                        kps_small, kp_scores = estimator.estimate(rgb, box_small)

                        # Contract: everything in ORIGINAL-resolution pixels.
                        box = box_small / scale
                        kps = kps_small / scale
                        pose = FramePose(
                            frame_index=frame.index,
                            timestamp_s=frame.timestamp_s,
                            box=tuple(float(v) for v in box),
                            box_score=box_score,
                            keypoints=[
                                Keypoint(name=COCO_KEYPOINTS[i],
                                         x=float(kps[i, 0]), y=float(kps[i, 1]),
                                         score=float(kp_scores[i]))
                                for i in range(len(COCO_KEYPOINTS))
                            ],
                        )
                        frames.append(pose)
                        draw_skeleton(frame.image, pose, config.overlay_min_score)
                    overlay.write(frame.image)
        finally:
            detector.unload()
            estimator.unload()

        result = AnalysisResult(
            video=VideoMeta(
                source_path=str(config.video_path),
                fps=info.fps,
                width=info.width,
                height=info.height,
                total_frames=info.total_frames,
                frame_stride=config.frame_stride,
                max_dim=config.max_dim,
                pose_model=config.pose_model,
                detector_model=config.detector_model,
                device=device,
            ),
            frames=frames,
        )

    config.keypoints_path.write_text(
        json.dumps(result.model_dump(), indent=2), encoding="utf-8"
    )
    return result
