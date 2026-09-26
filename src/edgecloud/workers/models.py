from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Protocol

from edgecloud.inference.engine import Image
from edgecloud.inference.models import BoundingBox, Detection, TimingMetrics


class InferenceWorker(Protocol):
    def infer(self, image: Image) -> WorkerResult: ...


@dataclass(frozen=True, slots=True)
class WorkerResult:
    detections: tuple[Detection, ...]
    timing: TimingMetrics
    worker_id: str
    worker_type: str
    backend: str
    round_trip_ms: float | None = None

    def __post_init__(self) -> None:
        if not self.worker_id:
            raise ValueError("worker_id cannot be empty")
        if not self.worker_type:
            raise ValueError("worker_type cannot be empty")
        if not self.backend:
            raise ValueError("backend cannot be empty")
        if self.round_trip_ms is not None and self.round_trip_ms < 0:
            raise ValueError("round_trip_ms cannot be negative")

    @property
    def detection_count(self) -> int:
        return len(self.detections)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["detection_count"] = self.detection_count
        return value

    @classmethod
    def from_dict(cls, value: Any, *, round_trip_ms: float | None = None) -> WorkerResult:
        if not isinstance(value, dict):
            raise ValueError("worker result must be an object")
        try:
            raw_detections = value["detections"]
            raw_timing = value["timing"]
            if not isinstance(raw_detections, list) or not isinstance(raw_timing, dict):
                raise ValueError("invalid detections or timing")
            detections = tuple(_parse_detection(item) for item in raw_detections)
            timing = TimingMetrics(
                preprocessing_ms=_number(raw_timing["preprocessing_ms"]),
                inference_ms=_number(raw_timing["inference_ms"]),
                postprocessing_ms=_number(raw_timing["postprocessing_ms"]),
                total_ms=_number(raw_timing["total_ms"]),
            )
            return cls(
                detections=detections,
                timing=timing,
                worker_id=_text(value["worker_id"]),
                worker_type=_text(value["worker_type"]),
                backend=_text(value["backend"]),
                round_trip_ms=round_trip_ms,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid worker result structure") from exc


def _parse_detection(value: Any) -> Detection:
    if not isinstance(value, dict) or not isinstance(value.get("box"), dict):
        raise ValueError("invalid detection structure")
    box = value["box"]
    return Detection(
        box=BoundingBox(
            x1=_number(box["x1"]),
            y1=_number(box["y1"]),
            x2=_number(box["x2"]),
            y2=_number(box["y2"]),
        ),
        label=_text(value["label"]),
        score=_number(value["score"]),
        class_id=_integer(value["class_id"]),
    )


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("expected a number")
    return float(value)


def _integer(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("expected an integer")
    return value


def _text(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("expected non-empty text")
    return value


class HealthStatus(StrEnum):
    READY = "ready"
    NOT_READY = "not_ready"
    UNREACHABLE = "unreachable"


@dataclass(frozen=True, slots=True)
class HealthResult:
    status: HealthStatus
    worker_id: str | None = None
    backend: str | None = None
    round_trip_ms: float | None = None

    @property
    def ready(self) -> bool:
        return self.status is HealthStatus.READY
