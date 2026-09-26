from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Protocol

from edgecloud.inference.engine import Image
from edgecloud.workers.models import WorkerResult


class SchedulerStrategy(StrEnum):
    EDGE_ONLY = "edge_only"
    REMOTE_ONLY = "remote_only"
    ROUND_ROBIN = "round_robin"


@dataclass(frozen=True, slots=True)
class ScheduledResult:
    strategy: SchedulerStrategy
    selected_worker_id: str
    selected_worker_type: str
    scheduler_latency_ms: float
    worker_result: WorkerResult

    def __post_init__(self) -> None:
        if self.scheduler_latency_ms < 0:
            raise ValueError("scheduler_latency_ms cannot be negative")
        if self.selected_worker_id != self.worker_result.worker_id:
            raise ValueError("selected worker ID must match the worker result")
        if self.selected_worker_type != self.worker_result.worker_type:
            raise ValueError("selected worker type must match the worker result")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SchedulerMetrics:
    total_requests: int = 0
    edge_selections: int = 0
    remote_selections: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: float = 0.0

    def record(self, worker_type: str, latency_ms: float, *, succeeded: bool) -> None:
        self.total_requests += 1
        if worker_type == "edge":
            self.edge_selections += 1
        elif worker_type == "remote":
            self.remote_selections += 1
        if succeeded:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        self.total_latency_ms += latency_ms

    @property
    def mean_latency_ms(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_latency_ms / self.total_requests

    def selection_percentage(self, worker_type: str) -> float:
        if self.total_requests == 0:
            return 0.0
        selections = self.edge_selections if worker_type == "edge" else self.remote_selections
        return selections / self.total_requests * 100

    def to_dict(self) -> dict[str, int | float]:
        value = asdict(self)
        value["mean_latency_ms"] = self.mean_latency_ms
        value["edge_selection_percentage"] = self.selection_percentage("edge")
        value["remote_selection_percentage"] = self.selection_percentage("remote")
        return value


class Scheduler(Protocol):
    strategy: SchedulerStrategy
    metrics: SchedulerMetrics

    def infer(self, image: Image) -> ScheduledResult: ...
