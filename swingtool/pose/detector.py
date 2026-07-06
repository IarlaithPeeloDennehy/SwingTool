"""Person detection (RT-DETR, Apache-2.0) - supplies the bounding box that
top-down ViTPose requires."""

from __future__ import annotations

import numpy as np
import torch

_PERSON_LABEL = 0  # COCO class id for "person"


class PersonDetector:
    def __init__(self, model_id: str, device: str):
        from transformers import AutoProcessor, RTDetrForObjectDetection

        self.device = device
        self.dtype = torch.float16 if device == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = RTDetrForObjectDetection.from_pretrained(
            model_id, torch_dtype=self.dtype
        ).to(device)
        self.model.eval()

    @torch.no_grad()
    def detect_best_person(
        self, image_rgb: np.ndarray, threshold: float
    ) -> tuple[np.ndarray, float] | None:
        """Return (box_xyxy, score) for the highest-confidence person, or None."""
        h, w = image_rgb.shape[:2]
        inputs = self.processor(images=image_rgb, return_tensors="pt").to(
            self.device, dtype=self.dtype
        )
        outputs = self.model(**inputs)
        result = self.processor.post_process_object_detection(
            outputs, target_sizes=torch.tensor([(h, w)]), threshold=threshold
        )[0]

        mask = result["labels"] == _PERSON_LABEL
        if not bool(mask.any()):
            return None
        scores = result["scores"][mask].float().cpu()
        boxes = result["boxes"][mask].float().cpu()
        best = int(scores.argmax())
        return boxes[best].numpy(), float(scores[best])

    def unload(self) -> None:
        del self.model
        if self.device == "cuda":
            torch.cuda.empty_cache()
