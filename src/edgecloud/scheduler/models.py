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
    LATENCY_AWARE = "latency_aware"
    RESOURCE_AWARE = "resource_aware"
    RESILIENT = "resilient"


@dataclass(frozen=True, slots=True)
class LatencyDecision:
    edge_latency_estimate_ms: float | None
    remote_latency_estimate_ms: float | None
    cold_start: bool
    reason: str


@dataclass(frozen=True, slots=True)
class ResourceDecision:
    edge_latency_estimate_ms: float | None
    remote_latency_estimate_ms: float | None
    edge_resource_pressure: float | None
    remote_resource_pressure: float | None
    edge_telemetry_status: str
    remote_telemetry_status: str
    edge_telemetry_error: str | None
    remote_telemetry_error: str | None
    edge_execution_resource: str
    remote_execution_resource: str
    cold_start: bool
    reason: str


@dataclass(frozen=True, slots=True)
class ResilientDecision:
    initial_worker_id: str
    initial_worker_type: str
    final_worker_id: str
    final_worker_type: str
    fallback_occurred: bool
    primary_failure_type: str | None
    primary_failure_code: str | None
    primary_attempt_ms: float
    fallback_attempt_ms: float | None
    total_logical_latency_ms: float
    edge_health: str
    remote_health: str
    edge_health_fresh: bool
    remote_health_fresh: bool
    reason: str


@dataclass(frozen=True, slots=True)
class ScheduledResult:
    strategy: SchedulerStrategy
    selected_worker_id: str
    selected_worker_type: str
    scheduler_latency_ms: float
    worker_result: WorkerResult
    executed_worker_id: str | None = None
    executed_worker_type: str | None = None
    fallback: bool = False
    fallback_reason: str | None = None
    edge_injected_delay_ms: float = 0.0
    remote_injected_delay_ms: float = 0.0
    latency_decision: LatencyDecision | None = None
    resource_decision: ResourceDecision | None = None
    resilient_decision: ResilientDecision | None = None

    def __post_init__(self) -> None:
        if self.scheduler_latency_ms < 0:
            raise ValueError("scheduler_latency_ms cannot be negative")
        if self.executed_worker_id is None:
            object.__setattr__(self, "executed_worker_id", self.worker_result.worker_id)
        if self.executed_worker_type is None:
            object.__setattr__(self, "executed_worker_type", self.worker_result.worker_type)
        if self.executed_worker_id != self.worker_result.worker_id:
            raise ValueError("executed worker ID must match the worker result")
        if self.executed_worker_type != self.worker_result.worker_type:
            raise ValueError("executed worker type must match the worker result")
        if self.fallback != (self.selected_worker_type != self.executed_worker_type):
            raise ValueError("fallback must reflect selected and executed worker types")
        if self.fallback and self.fallback_reason is None:
            raise ValueError("fallback_reason is required when fallback occurs")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["observed_latency_ms"] = self.scheduler_latency_ms
        value["detections"] = self.worker_result.detection_count
        if self.latency_decision is None:
            value.pop("latency_decision")
        if self.resource_decision is None:
            value.pop("resource_decision")
        if self.resilient_decision is None:
            value.pop("resilient_decision")
        return value


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


@dataclass(slots=True)
class ResilientMetrics:
    logical_requests: int = 0
    successful_logical_requests: int = 0
    failed_logical_requests: int = 0
    worker_attempts: int = 0
    primary_edge_selections: int = 0
    primary_remote_selections: int = 0
    primary_worker_failures: int = 0
    fallback_attempts: int = 0
    successful_fallbacks: int = 0
    failed_fallbacks: int = 0
    edge_to_remote_fallbacks: int = 0
    remote_to_edge_fallbacks: int = 0
    final_edge_results: int = 0
    final_remote_results: int = 0
    total_logical_latency_ms: float = 0.0

    @property
    def mean_logical_latency_ms(self) -> float:
        if self.logical_requests == 0:
            return 0.0
        return self.total_logical_latency_ms / self.logical_requests

    def to_dict(self) -> dict[str, int | float]:
        value = asdict(self)
        value["mean_logical_latency_ms"] = self.mean_logical_latency_ms
        return value


class Scheduler(Protocol):
    strategy: SchedulerStrategy
    metrics: SchedulerMetrics | ResilientMetrics

    def infer(self, image: Image) -> ScheduledResult: ...

    def summary(self) -> dict[str, int | float | None]: ...
