"""Run configuration shared across pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

POSE_MODEL_ID = "usyd-community/vitpose-base-simple"
DETECTOR_MODEL_ID = "PekingU/rtdetr_r50vd_coco_o365"


@dataclass
class AnalyzeConfig:
    video_path: Path
    output_dir: Path = Path("output")
    device: str = "cuda"          # "cuda" or "cpu"; cuda missing is an error, not a fallback
    frame_stride: int = 1         # process every Nth frame
    max_dim: int | None = None    # downscale so max(w, h) <= this before inference
    min_box_score: float = 0.3    # person detection threshold
    overlay_min_score: float = 0.3  # hide keypoints below this in the overlay
    pose_model: str = POSE_MODEL_ID
    detector_model: str = DETECTOR_MODEL_ID

    @property
    def keypoints_path(self) -> Path:
        return self.output_dir / "keypoints.json"

    @property
    def overlay_path(self) -> Path:
        return self.output_dir / "overlay.mp4"


def resolve_device(requested: str) -> str:
    """Validate the requested device. A missing GPU is an explicit error:
    silently falling back to CPU would hide a broken CUDA install."""
    import torch

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but torch.cuda.is_available() is False. "
                "Likely a CPU-only torch build or driver problem - reinstall with "
                "the cu124 wheel (see README). Pass --device cpu to run on CPU "
                "deliberately."
            )
        return "cuda"
    if requested == "cpu":
        return "cpu"
    raise ValueError(f"Unknown device {requested!r}; expected 'cuda' or 'cpu'.")
