"""Phase 3B: monocular relative depth (Depth Anything V2 Small, Apache-2.0).

Depth Anything outputs RELATIVE (scale-free) inverse depth, never metric
distance - so nothing derived from it claims a physical scale. Depth runs on
GPU (fp16); unlike Grounding DINO it has no CUDA issue on this build. The stage
samples depth at the pose keypoints and the club-head point per frame, with
per-frame normalisation so cross-frame comparison is at least consistent.
"""

from swingtool.depth.stage import run_depth_stage

__all__ = ["run_depth_stage"]
