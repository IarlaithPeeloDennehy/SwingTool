"""Grounding DINO zero-shot detector, wrapped to return club/ball candidates.

Prompt uses "a golf club. a golf ball." only. The tempting "club head" prompt
is dropped: it is a false friend that locks onto the golfer's (human) head.
"""

from __future__ import annotations

import numpy as np
import torch

from swingtool.detect.schema import BoxDet

PROMPT = "a golf club. a golf ball."


class ClubBallDetector:
    def __init__(self, model_id: str, device: str = "cpu"):
        from transformers import AutoProcessor, GroundingDinoForObjectDetection

        self.device = device
        # fp32 always: Grounding DINO's grid_sample rejects fp16 (Half vs Float).
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = GroundingDinoForObjectDetection.from_pretrained(
            model_id, torch_dtype=torch.float32
        ).to(device).eval()

    @torch.no_grad()
    def detect(self, image_rgb: np.ndarray, box_threshold: float, text_threshold: float,
               top_k: int = 5) -> tuple[list[BoxDet], list[BoxDet]]:
        h, w = image_rgb.shape[:2]
        inputs = self.processor(images=image_rgb, text=PROMPT, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            outputs, inputs["input_ids"], threshold=box_threshold,
            text_threshold=text_threshold, target_sizes=[(h, w)],
        )[0]

        club: list[BoxDet] = []
        ball: list[BoxDet] = []
        # Prefer string text_labels; "labels" is slated to become integer ids.
        labels = result.get("text_labels", result["labels"])
        for box, score, label in zip(result["boxes"], result["scores"], labels):
            b = [float(v) for v in box.tolist()]
            det = BoxDet(x1=b[0], y1=b[1], x2=b[2], y2=b[3], score=float(score))
            text = str(label).lower()
            if "ball" in text:
                ball.append(det)
            elif "club" in text:            # "club head" false friend already excluded from prompt
                club.append(det)
        club.sort(key=lambda d: d.score, reverse=True)
        ball.sort(key=lambda d: d.score, reverse=True)
        return club[:top_k], ball[:top_k]

    def unload(self) -> None:
        del self.model
        if self.device == "cuda":
            torch.cuda.empty_cache()
