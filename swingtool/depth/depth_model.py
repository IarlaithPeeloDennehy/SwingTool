"""Depth Anything V2 Small wrapper. Outputs a per-frame RELATIVE depth map
(higher = closer), resized to the original frame. fp16 on GPU."""

from __future__ import annotations

import cv2
import numpy as np
import torch


class DepthEstimator:
    def __init__(self, model_id: str, device: str):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.device = device
        self.dtype = torch.float16 if device == "cuda" else torch.float32
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(
            model_id, torch_dtype=self.dtype
        ).to(device).eval()

    @torch.no_grad()
    def depth_map(self, image_rgb: np.ndarray) -> np.ndarray:
        """Relative inverse-depth map at the frame's resolution (float32).
        Values are scale-free - relative only, never metric distance.

        The model's low-res prediction is moved to CPU and upscaled there with
        cv2: a large GPU bicubic interpolate to 1080x1920 faults on this build,
        and the upscale is cheap on CPU anyway."""
        h, w = image_rgb.shape[:2]
        inputs = self.processor(images=image_rgb, return_tensors="pt").to(self.device, self.dtype)
        pred = self.model(**inputs).predicted_depth  # (1, Hs, Ws), small
        small = pred[0].float().cpu().numpy()
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)

    def unload(self) -> None:
        del self.model
        if self.device == "cuda":
            torch.cuda.empty_cache()
