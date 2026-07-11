"""Depth stage: video + keypoints + detections -> depth_samples.json.

Samples relative depth at the pose keypoints and the club-head point per frame,
over the swing window. Runs on GPU (fp16); load -> run -> free.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from swingtool.analysis.clubpath import fill_short_gaps, select_club_path, smooth_club_path
from swingtool.analysis.stage import _detections_to_plain, _hands_by_frame
from swingtool.config import resolve_device
from swingtool.depth.depth_model import DepthEstimator
from swingtool.depth.schema import DepthResult, DepthSource, FrameDepth, KeypointDepth
from swingtool.detect.schema import DetectionsResult
from swingtool.ingest import VideoReader
from swingtool.metrics.signals import body_scale, frame_kp_dicts
from swingtool.schema import AnalysisResult

DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


def _sample_patch(depth: np.ndarray, x: float, y: float, r: int) -> float:
    h, w = depth.shape
    xi, yi = int(round(x)), int(round(y))
    if not (0 <= xi < w and 0 <= yi < h):
        return float("nan")
    patch = depth[max(0, yi - r):yi + r + 1, max(0, xi - r):xi + r + 1]
    return float(np.median(patch)) if patch.size else float("nan")


def _frame_norm(depth: np.ndarray, box) -> tuple[float, float]:
    """Median and robust scale (MAD) of depth inside the person box, used to
    normalise relative depth per frame for cross-frame comparison."""
    h, w = depth.shape
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    region = depth[y1:y2:4, x1:x2:4]
    if region.size == 0:
        return float(np.median(depth)), float(np.std(depth) or 1.0)
    med = float(np.median(region))
    mad = float(np.median(np.abs(region - med)))
    return med, (mad * 1.4826 if mad > 0 else float(np.std(region) or 1.0))


def run_depth_stage(video_path: Path, keypoints_path: Path, detections_path: Path,
                    output_dir: Path, device: str = "cuda", patch_radius: int = 4) -> DepthResult:
    keypoints_path, detections_path = Path(keypoints_path), Path(detections_path)
    for p in (keypoints_path, detections_path):
        if not p.exists():
            raise FileNotFoundError(f"Required input not found: {p}")

    result = AnalysisResult.model_validate(json.loads(keypoints_path.read_text(encoding="utf-8")))
    det = DetectionsResult.model_validate(json.loads(detections_path.read_text(encoding="utf-8")))
    device = resolve_device(device)

    # Reuse the club-head path so club depth aligns with the club_path in analysis.
    hands = _hands_by_frame(result)
    scale = body_scale(frame_kp_dicts(result.frames))
    club = smooth_club_path(fill_short_gaps(select_club_path(_detections_to_plain(det), hands, scale)))
    club_xy = {p["frame_index"]: (p["x"], p["y"]) for p in club if p["x"] is not None}

    win_start, win_end = det.source.window_start, det.source.window_end
    pose_by_frame = {f.frame_index: f for f in result.frames}

    estimator = DepthEstimator(DEPTH_MODEL_ID, device)
    frames_out: list[FrameDepth] = []
    try:
        with VideoReader(video_path, frame_stride=1) as reader:
            for frame in tqdm(reader.frames(), total=reader.info.total_frames,
                              unit="frame", desc="Depth"):
                if frame.index < win_start or frame.index > win_end:
                    continue
                pose = pose_by_frame.get(frame.index)
                if pose is None:
                    continue
                rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
                depth = estimator.depth_map(rgb)
                med, sc = _frame_norm(depth, pose.box)
                kps = [KeypointDepth(name=k.name, z=_sample_patch(depth, k.x, k.y, patch_radius))
                       for k in pose.keypoints]
                cxy = club_xy.get(frame.index)
                club_z = _sample_patch(depth, cxy[0], cxy[1], patch_radius) if cxy else None
                if club_z is not None and np.isnan(club_z):
                    club_z = None
                frames_out.append(FrameDepth(
                    frame_index=frame.index, timestamp_s=frame.timestamp_s,
                    frame_median=med, frame_scale=sc, keypoints=kps, club_z=club_z))
                if frame.index >= win_end:
                    break
    finally:
        estimator.unload()

    depth_result = DepthResult(
        source=DepthSource(
            video=str(video_path), keypoints_path=str(keypoints_path),
            detections_path=str(detections_path), depth_model=DEPTH_MODEL_ID,
            device=device, window_start=win_start, window_end=win_end,
            patch_radius=patch_radius),
        frames=frames_out,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "depth_samples.json").write_text(
        json.dumps(depth_result.model_dump(), indent=2), encoding="utf-8")
    return depth_result
