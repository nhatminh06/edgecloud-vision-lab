from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Lock
from time import perf_counter
from typing import Protocol

from edgecloud.inference.engine import Image
from edgecloud.scheduler.errors import ResilientRequestError, SchedulerUnavailableError
from edgecloud.scheduler.estimator import EwmaEstimator
from edgecloud.scheduler.health import CachedHealth
from edgecloud.scheduler.models import (
    ResilientDecision,
    ResilientMetrics,
    ResourceDecision,
    ScheduledResult,
    SchedulerStrategy,
)
from edgecloud.scheduler.resource import (
    ExecutionResource,
    WorkerResource,
    choose_resource_preference,
    resource_pressure,
)
from edgecloud.telemetry.cache import CachedTelemetry
from edgecloud.workers.errors import (
    WorkerConnectionError,
    WorkerExecutionError,
    WorkerHTTPError,
    WorkerTimeoutError,
    WorkerUnavailableError,
)
from edgecloud.workers.models import InferenceWorker, WorkerResult


class TelemetryStateSource(Protocol):
    def edge(self) -> CachedTelemetry: ...

    def remote(self) -> CachedTelemetry: ...


class HealthStateSource(Protocol):
    def edge(self) -> CachedHealth: ...

    def remote(self) -> CachedHealth: ...

    def mark_healthy(self, worker_type: str, detail: str | None = None) -> None: ...

    def mark_unhealthy(self, worker_type: str, detail: str | None = None) -> None: ...


@dataclass(slots=True)
class ResilientScheduler:
    edge_worker: InferenceWorker
    remote_worker: InferenceWorker
    telemetry: TelemetryStateSource
    health: HealthStateSource
    alpha: float = 0.3
    pressure_threshold: float = 0.85
    edge_resource: WorkerResource = field(
        default_factory=lambda: WorkerResource(ExecutionResource.CPU)
    )
    remote_resource: WorkerResource = field(
        default_factory=lambda: WorkerResource(ExecutionResource.CPU)
    )
    clock: Callable[[], float] = perf_counter
    strategy: SchedulerStrategy = field(init=False, default=SchedulerStrategy.RESILIENT)
    metrics: ResilientMetrics = field(default_factory=ResilientMetrics)
    _state_lock: Lock = field(init=False, default_factory=Lock, repr=False)
    _edge_estimator: EwmaEstimator = field(init=False, repr=False)
    _remote_estimator: EwmaEstimator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.pressure_threshold <= 1.0:
            raise ValueError("resource pressure threshold must be between 0 and 1")
        self._edge_estimator = EwmaEstimator(self.alpha)
        self._remote_estimator = EwmaEstimator(self.alpha)

    @property
    def edge_latency_estimate_ms(self) -> float | None:
        return self._edge_estimator.estimate

    @property
    def remote_latency_estimate_ms(self) -> float | None:
        return self._remote_estimator.estimate

    def infer(self, image: Image) -> ScheduledResult:
        edge_telemetry = self.telemetry.edge()
        remote_telemetry = self.telemetry.remote()
        edge_health = self.health.edge()
        remote_health = self.health.remote()
        edge_pressure = resource_pressure(edge_telemetry, self.edge_resource)
        remote_pressure = resource_pressure(remote_telemetry, self.remote_resource)

        with self._state_lock:
            self.metrics.logical_requests += 1
            if edge_health.blocks_routing and remote_health.blocks_routing:
                self.metrics.failed_logical_requests += 1
                raise SchedulerUnavailableError("edge and remote workers are unhealthy")

            preference = choose_resource_preference(
                edge_sample_count=self._edge_estimator.sample_count,
                remote_sample_count=self._remote_estimator.sample_count,
                edge_latency_ms=self.edge_latency_estimate_ms,
                remote_latency_ms=self.remote_latency_estimate_ms,
                edge_telemetry_status=edge_telemetry.status,
                remote_telemetry_status=remote_telemetry.status,
                edge_pressure=edge_pressure,
                remote_pressure=remote_pressure,
                pressure_threshold=self.pressure_threshold,
            )
            primary_type = preference.worker_type
            reason = preference.reason
            if primary_type == "edge" and edge_health.blocks_routing:
                primary_type = "remote"
                reason = "edge_unhealthy_remote_selected"
            elif primary_type == "remote" and remote_health.blocks_routing:
                primary_type = "edge"
                reason = "remote_unhealthy_edge_selected"

            primary = self._worker(primary_type)
            fallback_type = "remote" if primary_type == "edge" else "edge"
            fallback = self._worker(fallback_type)
            fallback_health = remote_health if fallback_type == "remote" else edge_health
            self._record_primary_selection(primary_type)

            try:
                worker_result, primary_ms = self._attempt(primary, primary_type, image)
            except _AttemptFailure as failure:
                self.metrics.primary_worker_failures += 1
                if not failure.fallback_eligible:
                    self._record_logical_failure(failure.elapsed_ms)
                    raise failure.error from failure
                self.health.mark_unhealthy(
                    primary_type, f"{type(failure.error).__name__}: {failure.error}"
                )
                if fallback_health.blocks_routing:
                    self._record_logical_failure(failure.elapsed_ms)
                    raise ResilientRequestError(
                        f"{primary_type} failed and {fallback_type} is unhealthy"
                    ) from failure.error
                self._record_fallback(primary_type)
                try:
                    worker_result, fallback_ms = self._attempt(fallback, fallback_type, image)
                except _AttemptFailure as fallback_failure:
                    if fallback_failure.fallback_eligible:
                        self.health.mark_unhealthy(
                            fallback_type,
                            f"{type(fallback_failure.error).__name__}: {fallback_failure.error}",
                        )
                    total_ms = failure.elapsed_ms + fallback_failure.elapsed_ms
                    self.metrics.failed_fallbacks += 1
                    self._record_logical_failure(total_ms)
                    raise ResilientRequestError(
                        f"{primary_type} primary and {fallback_type} fallback failed"
                    ) from fallback_failure.error
                self.health.mark_healthy(fallback_type, "inference succeeded")
                self._estimator(fallback_type).observe(fallback_ms)
                total_ms = failure.elapsed_ms + fallback_ms
                self.metrics.successful_fallbacks += 1
                self._record_logical_success(fallback_type, total_ms)
                return self._result(
                    worker_result,
                    primary,
                    primary_type,
                    failure.elapsed_ms,
                    fallback_ms,
                    total_ms,
                    edge_health,
                    remote_health,
                    edge_telemetry,
                    remote_telemetry,
                    edge_pressure,
                    remote_pressure,
                    preference.cold_start,
                    f"{reason};{primary_type}_failed_fallback_{fallback_type}",
                    failure.error,
                )

            self.health.mark_healthy(primary_type, "inference succeeded")
            self._estimator(primary_type).observe(primary_ms)
            self._record_logical_success(primary_type, primary_ms)
            return self._result(
                worker_result,
                primary,
                primary_type,
                primary_ms,
                None,
                primary_ms,
                edge_health,
                remote_health,
                edge_telemetry,
                remote_telemetry,
                edge_pressure,
                remote_pressure,
                preference.cold_start,
                reason,
                None,
            )

    def _attempt(
        self, worker: InferenceWorker, worker_type: str, image: Image
    ) -> tuple[WorkerResult, float]:
        started = self.clock()
        self.metrics.worker_attempts += 1
        try:
            result = worker.infer(image)
        except (
            WorkerConnectionError,
            WorkerExecutionError,
            WorkerHTTPError,
            WorkerTimeoutError,
            WorkerUnavailableError,
        ) as exc:
            elapsed_ms = (self.clock() - started) * 1000
            raise _AttemptFailure(exc, elapsed_ms, _can_fallback(exc, worker_type)) from exc
        except (OSError, RuntimeError, ValueError) as exc:
            elapsed_ms = (self.clock() - started) * 1000
            raise _AttemptFailure(exc, elapsed_ms, False) from exc
        return result, (self.clock() - started) * 1000

    def _result(
        self,
        worker_result: WorkerResult,
        primary: InferenceWorker,
        primary_type: str,
        primary_ms: float,
        fallback_ms: float | None,
        total_ms: float,
        edge_health: CachedHealth,
        remote_health: CachedHealth,
        edge_telemetry: CachedTelemetry,
        remote_telemetry: CachedTelemetry,
        edge_pressure: float | None,
        remote_pressure: float | None,
        cold_start: bool,
        reason: str,
        primary_error: Exception | None,
    ) -> ScheduledResult:
        return ScheduledResult(
            strategy=self.strategy,
            selected_worker_id=worker_result.worker_id,
            selected_worker_type=worker_result.worker_type,
            scheduler_latency_ms=total_ms,
            worker_result=worker_result,
            resource_decision=ResourceDecision(
                self.edge_latency_estimate_ms,
                self.remote_latency_estimate_ms,
                edge_pressure,
                remote_pressure,
                edge_telemetry.status.value,
                remote_telemetry.status.value,
                edge_telemetry.last_error,
                remote_telemetry.last_error,
                self.edge_resource.resource_type.value,
                self.remote_resource.resource_type.value,
                cold_start,
                reason,
            ),
            resilient_decision=ResilientDecision(
                initial_worker_id=getattr(primary, "worker_id", primary_type),
                initial_worker_type=primary_type,
                final_worker_id=worker_result.worker_id,
                final_worker_type=worker_result.worker_type,
                fallback_occurred=fallback_ms is not None,
                primary_failure_type=(
                    type(primary_error).__name__ if primary_error is not None else None
                ),
                primary_failure_code=_failure_code(primary_error),
                primary_attempt_ms=primary_ms,
                fallback_attempt_ms=fallback_ms,
                total_logical_latency_ms=total_ms,
                edge_health=edge_health.state.value,
                remote_health=remote_health.state.value,
                edge_health_fresh=edge_health.fresh,
                remote_health_fresh=remote_health.fresh,
                reason=reason,
            ),
        )

    def _worker(self, worker_type: str) -> InferenceWorker:
        return self.edge_worker if worker_type == "edge" else self.remote_worker

    def _estimator(self, worker_type: str) -> EwmaEstimator:
        return self._edge_estimator if worker_type == "edge" else self._remote_estimator

    def _record_primary_selection(self, worker_type: str) -> None:
        if worker_type == "edge":
            self.metrics.primary_edge_selections += 1
        else:
            self.metrics.primary_remote_selections += 1

    def _record_fallback(self, primary_type: str) -> None:
        self.metrics.fallback_attempts += 1
        if primary_type == "edge":
            self.metrics.edge_to_remote_fallbacks += 1
        else:
            self.metrics.remote_to_edge_fallbacks += 1

    def _record_logical_success(self, final_type: str, total_ms: float) -> None:
        self.metrics.successful_logical_requests += 1
        self.metrics.total_logical_latency_ms += total_ms
        if final_type == "edge":
            self.metrics.final_edge_results += 1
        else:
            self.metrics.final_remote_results += 1

    def _record_logical_failure(self, total_ms: float) -> None:
        self.metrics.failed_logical_requests += 1
        self.metrics.total_logical_latency_ms += total_ms

    def summary(self) -> dict[str, int | float | str | bool | None]:
        edge_health = self.health.edge()
        remote_health = self.health.remote()
        value: dict[str, int | float | str | bool | None] = self.metrics.to_dict()
        value.update(
            {
                "edge_latency_estimate_ms": self.edge_latency_estimate_ms,
                "remote_latency_estimate_ms": self.remote_latency_estimate_ms,
                "edge_sample_count": self._edge_estimator.sample_count,
                "remote_sample_count": self._remote_estimator.sample_count,
                "edge_health": edge_health.state.value,
                "remote_health": remote_health.state.value,
                "edge_health_fresh": edge_health.fresh,
                "remote_health_fresh": remote_health.fresh,
            }
        )
        return value


@dataclass(frozen=True, slots=True)
class _AttemptFailure(Exception):
    error: Exception
    elapsed_ms: float
    fallback_eligible: bool


def _can_fallback(error: Exception, worker_type: str) -> bool:
    if isinstance(error, WorkerExecutionError):
        return worker_type == "edge"
    if worker_type != "remote":
        return False
    if isinstance(error, WorkerHTTPError):
        return error.status_code == 429 or error.status_code >= 500
    return isinstance(
        error,
        (WorkerConnectionError, WorkerTimeoutError, WorkerUnavailableError),
    )


def _failure_code(error: Exception | None) -> str | None:
    if error is None:
        return None
    if isinstance(error, WorkerHTTPError):
        return f"http_{error.status_code}"
    codes = {
        WorkerConnectionError: "connection_error",
        WorkerExecutionError: "execution_error",
        WorkerTimeoutError: "timeout",
        WorkerUnavailableError: "unavailable",
    }
    for error_type, code in codes.items():
        if isinstance(error, error_type):
            return code
    return "non_recoverable"
