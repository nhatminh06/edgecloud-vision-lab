from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2

from edgecloud.inference.engine import Image


@dataclass(frozen=True, slots=True)
class Frame:
    image: Image
    index: int
    source: str


def image_frames(path: Path) -> Iterator[Frame]:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"could not read image: {path}")
    yield Frame(image=image, index=0, source=str(path))


def video_frames(path: Path) -> Iterator[Frame]:
    yield from _capture_frames(cv2.VideoCapture(str(path)), str(path))


def webcam_frames(device: int = 0) -> Iterator[Frame]:
    yield from _capture_frames(cv2.VideoCapture(device), f"webcam:{device}")


def _capture_frames(capture: cv2.VideoCapture, source: str) -> Iterator[Frame]:
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"could not open video source: {source}")
    try:
        index = 0
        while True:
            available, image = capture.read()
            if not available:
                break
            yield Frame(image=image, index=index, source=source)
            index += 1
    finally:
        capture.release()
