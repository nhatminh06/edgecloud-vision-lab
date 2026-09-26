from __future__ import annotations

import cv2

from edgecloud.inference.engine import Image
from edgecloud.inference.models import Detection


def render_detections(image: Image, detections: tuple[Detection, ...]) -> Image:
    rendered = image.copy()
    for detection in detections:
        box = detection.box
        start = (round(box.x1), round(box.y1))
        end = (round(box.x2), round(box.y2))
        cv2.rectangle(rendered, start, end, (0, 200, 0), 2)
        text = f"{detection.label} {detection.score:.2f}"
        cv2.putText(
            rendered,
            text,
            (start[0], max(15, start[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 200, 0),
            1,
            cv2.LINE_AA,
        )
    return rendered
