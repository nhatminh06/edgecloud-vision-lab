from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError("bounding box maximums must not be less than minimums")


@dataclass(frozen=True, slots=True)
class Detection:
    box: BoundingBox
    label: str
    score: float
    class_id: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("detection score must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class TimingMetrics:
    preprocessing_ms: float
    inference_ms: float
    postprocessing_ms: float
    total_ms: float

    def __post_init__(self) -> None:
        values = (
            self.preprocessing_ms,
            self.inference_ms,
            self.postprocessing_ms,
            self.total_ms,
        )
        if any(value < 0 for value in values):
            raise ValueError("timing values cannot be negative")


@dataclass(frozen=True, slots=True)
class InferenceResult:
    detections: tuple[Detection, ...]
    timing: TimingMetrics
    source: str | None = None
    frame_index: int | None = None

    @property
    def detection_count(self) -> int:
        return len(self.detections)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["detection_count"] = self.detection_count
        return value
