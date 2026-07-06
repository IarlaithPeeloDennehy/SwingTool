"""Keypoint estimation (ViTPose, Apache-2.0). Top-down: takes the frame plus
a person box, returns 17 COCO keypoints with confidences."""

from __future__ import annotations

import numpy as np
import torch


class PoseEstimator:
    def __init__(self, model_id: str, device: str):
        from transformers import AutoProcessor, VitPoseForPoseEstimation

        self.device = device
        self.dtype = torch.float16 if device == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = VitPoseForPoseEstimation.from_pretrained(
            model_id, torch_dtype=self.dtype
        ).to(device)
        self.model.eval()

    @torch.no_grad()
    def estimate(
        self, image_rgb: np.ndarray, box_xyxy: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (keypoints[17, 2], scores[17]) in the coordinates of image_rgb.

        ViTPose's processor expects COCO-format boxes (x, y, w, h).
        """
        x1, y1, x2, y2 = box_xyxy
        box_xywh = np.array([[x1, y1, x2 - x1, y2 - y1]], dtype=np.float32)

        inputs = self.processor(image_rgb, boxes=[box_xywh], return_tensors="pt").to(
            self.device, dtype=self.dtype
        )
        outputs = self.model(**inputs)
        # Heatmap decoding runs on CPU via scipy, which rejects float16.
        outputs.heatmaps = outputs.heatmaps.float()
        result = self.processor.post_process_pose_estimation(
            outputs, boxes=[box_xywh]
        )[0][0]  # first image, first (only) person

        keypoints = result["keypoints"].float().cpu().numpy()
        scores = result["scores"].float().cpu().numpy()
        return keypoints, np.clip(scores, 0.0, 1.0)

    def unload(self) -> None:
        del self.model
        if self.device == "cuda":
            torch.cuda.empty_cache()
