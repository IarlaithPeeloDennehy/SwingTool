"""Phase 3A: zero-shot club/ball detection (Grounding DINO).

Runs on CPU by default: Grounding DINO's multi-scale deformable attention
triggers an illegal-memory-access in grid_sample on this torch/CUDA build
(the Phase-1 models are unaffected - they don't use deformable attention).
CPU is correct and deterministic; --device cuda stays available for machines
where the GPU path works. To stay tractable at ~15s/frame, detection is
scoped to the swing window via the Phase-2 event detector.
"""

from swingtool.detect.stage import run_detect_stage, swing_window

__all__ = ["run_detect_stage", "swing_window"]
