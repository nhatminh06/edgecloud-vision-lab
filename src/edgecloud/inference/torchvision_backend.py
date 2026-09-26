from __future__ import annotations

import numpy as np
import torch
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large,
)
from torchvision.transforms.functional import to_tensor

from edgecloud.inference.engine import Image
from edgecloud.inference.models import BoundingBox, Detection


class TorchvisionSSDLiteBackend:
    """CPU-compatible pretrained SSDLite object detector."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)
        weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        self.categories = tuple(weights.meta["categories"])
        self.model = ssdlite320_mobilenet_v3_large(weights=weights).to(self.device).eval()

    def preprocess(self, image: Image) -> torch.Tensor:
        # OpenCV provides BGR while torchvision detectors are trained on RGB.
        rgb = np.ascontiguousarray(image[:, :, ::-1])
        return to_tensor(rgb).to(self.device)

    def infer(self, model_input: torch.Tensor) -> dict[str, torch.Tensor]:
        with torch.inference_mode():
            return self.model([model_input])[0]

    def postprocess(
        self, raw_output: dict[str, torch.Tensor], image_shape: tuple[int, ...], confidence: float
    ) -> tuple[Detection, ...]:
        del image_shape
        boxes = raw_output["boxes"].detach().cpu().tolist()
        labels = raw_output["labels"].detach().cpu().tolist()
        scores = raw_output["scores"].detach().cpu().tolist()
        detections: list[Detection] = []
        for box, class_id, score in zip(boxes, labels, scores, strict=True):
            if score < confidence:
                continue
            label = self.categories[class_id] if class_id < len(self.categories) else str(class_id)
            detections.append(
                Detection(BoundingBox(*map(float, box)), label, float(score), int(class_id))
            )
        return tuple(detections)
