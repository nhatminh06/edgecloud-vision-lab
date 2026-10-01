from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Lock
from time import perf_counter
from typing import Protocol

from edgecloud.inference.engine import Image
from edgecloud.scheduler.estimator import EwmaEstimator
from edgecloud.scheduler.models import (
    LatencyDecision,
    ResourceDecision,
    ScheduledResult,
    Scheduler,
    SchedulerMetrics,
    SchedulerStrategy,
)
from edgecloud.scheduler.resilient import HealthStateSource, ResilientScheduler
from edgecloud.scheduler.resource import (
    ExecutionResource,
    WorkerResource,
    choose_resource_preference,
    resource_pressure,
)
from edgecloud.telemetry.cache import CachedTelemetry
from edgecloud.workers.errors import (
    WorkerConnectionError,
    WorkerError,
    WorkerExecutionError,
    WorkerHTTPError,
    WorkerTimeoutError,
    WorkerUnavailableError,
)
from edgecloud.workers.models import InferenceWorker, WorkerResult


def _infer_with(
    worker: InferenceWorker,
    strategy: SchedulerStrategy,
    metrics: SchedulerMetrics,
    image: Image,
    selected_worker_type: str,
) -> ScheduledResult:
    started = perf_counter()
    try:
        worker_result = worker.infer(image)
    except (OSError, RuntimeError, ValueError):
        latency_ms = (perf_counter() - started) * 1000
        metrics.record(selected_worker_type, latency_ms, succeeded=False)
        raise
    latency_ms = (perf_counter() - started) * 1000
    metrics.record(worker_result.worker_type, latency_ms, succeeded=True)
    return ScheduledResult(
        strategy=strategy,
        selected_worker_id=worker_result.worker_id,
        selected_worker_type=worker_result.worker_type,
        scheduler_latency_ms=latency_ms,
        worker_result=worker_result,
    )


@dataclass(slots=True)
class EdgeOnlyScheduler:
    edge_worker: InferenceWorker
    strategy: SchedulerStrategy = field(init=False, default=SchedulerStrategy.EDGE_ONLY)
    metrics: SchedulerMetrics = field(default_factory=SchedulerMetrics)

    def infer(self, image: Image) -> ScheduledResult:
        return _infer_with(self.edge_worker, self.strategy, self.metrics, image, "edge")

    def summary(self) -> dict[str, int | float | None]:
        return self.metrics.to_dict()


@dataclass(slots=True)
class RemoteOnlyScheduler:
    remote_worker: InferenceWorker
    strategy: SchedulerStrategy = field(init=False, default=SchedulerStrategy.REMOTE_ONLY)
    metrics: SchedulerMetrics = field(default_factory=SchedulerMetrics)

    def infer(self, image: Image) -> ScheduledResult:
        return _infer_with(self.remote_worker, self.strategy, self.metrics, image, "remote")

    def summary(self) -> dict[str, int | float | None]:
        return self.metrics.to_dict()


@dataclass(slots=True)
class RoundRobinScheduler:
    edge_worker: InferenceWorker
    remote_worker: InferenceWorker
    strategy: SchedulerStrategy = field(init=False, default=SchedulerStrategy.ROUND_ROBIN)
    metrics: SchedulerMetrics = field(default_factory=SchedulerMetrics)
    _next_index: int = field(init=False, default=0)
    _selection_lock: Lock = field(init=False, default_factory=Lock, repr=False)

    def infer(self, image: Image) -> ScheduledResult:
        with self._selection_lock:
            worker = (self.edge_worker, self.remote_worker)[self._next_index]
            worker_type = ("edge", "remote")[self._next_index]
            self._next_index = (self._next_index + 1) % 2
        return _infer_with(worker, self.strategy, self.metrics, image, worker_type)

    def summary(self) -> dict[str, int | float | None]:
        return self.metrics.to_dict()


@dataclass(slots=True)
class LatencyAwareScheduler:
    edge_worker: InferenceWorker
    remote_worker: InferenceWorker
    alpha: float = 0.3
    clock: Callable[[], float] = perf_counter
    strategy: SchedulerStrategy = field(init=False, default=SchedulerStrategy.LATENCY_AWARE)
    metrics: SchedulerMetrics = field(default_factory=SchedulerMetrics)
    _state_lock: Lock = field(init=False, default_factory=Lock, repr=False)
    _edge_estimator: EwmaEstimator = field(init=False, repr=False)
    _remote_estimator: EwmaEstimator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._edge_estimator = EwmaEstimator(self.alpha)
        self._remote_estimator = EwmaEstimator(self.alpha)

    @property
    def edge_latency_estimate_ms(self) -> float | None:
        return self._edge_estimator.estimate

    @property
    def remote_latency_estimate_ms(self) -> float | None:
        return self._remote_estimator.estimate

    @property
    def edge_sample_count(self) -> int:
        return self._edge_estimator.sample_count

    @property
    def remote_sample_count(self) -> int:
        return self._remote_estimator.sample_count

    def infer(self, image: Image) -> ScheduledResult:
        # Serializing selection and update prevents concurrent requests from making decisions
        # against the same stale warm-up state. Phase 4 intentionally dispatches synchronously.
        with self._state_lock:
            worker, worker_type, cold_start, reason = self._select_worker()
            fallback_worker = self.remote_worker if worker_type == "edge" else self.edge_worker
            fallback_type = "remote" if worker_type == "edge" else "edge"
            request_started = self.clock()
            attempt_started = request_started
            try:
                worker_result = worker.infer(image)
            except WorkerError as primary_error:
                primary_elapsed_ms = (self.clock() - attempt_started) * 1000
                if not _can_latency_fallback(primary_error, worker_type):
                    self.metrics.record(worker_type, primary_elapsed_ms, succeeded=False)
                    raise
                fallback_started = self.clock()
                try:
                    worker_result = fallback_worker.infer(image)
                except WorkerError as fallback_error:
                    latency_ms = (self.clock() - request_started) * 1000
                    self.metrics.record(worker_type, latency_ms, succeeded=False)
                    raise WorkerUnavailableError(
                        f"{worker_type} primary failed ({primary_error}); "
                        f"{fallback_type} fallback failed ({fallback_error})"
                    ) from fallback_error
                fallback_finished = self.clock()
                fallback_elapsed_ms = (fallback_finished - fallback_started) * 1000
                self._estimator(fallback_type).observe(fallback_elapsed_ms)
                latency_ms = (fallback_finished - request_started) * 1000
                self.metrics.record(worker_type, latency_ms, succeeded=True)
                return self._result(
                    worker_result=worker_result,
                    selected_worker_type=worker_type,
                    latency_ms=latency_ms,
                    cold_start=cold_start,
                    reason=reason,
                    fallback_reason=f"{type(primary_error).__name__}: {primary_error}",
                )
            except (OSError, RuntimeError, ValueError):
                latency_ms = (self.clock() - request_started) * 1000
                self.metrics.record(worker_type, latency_ms, succeeded=False)
                raise
            latency_ms = (self.clock() - attempt_started) * 1000
            self.metrics.record(worker_type, latency_ms, succeeded=True)
            self._estimator(worker_type).observe(latency_ms)
            return self._result(
                worker_result=worker_result,
                selected_worker_type=worker_type,
                latency_ms=latency_ms,
                cold_start=cold_start,
                reason=reason,
            )

    def _estimator(self, worker_type: str) -> EwmaEstimator:
        return self._edge_estimator if worker_type == "edge" else self._remote_estimator

    def _result(
        self,
        *,
        worker_result: WorkerResult,
        selected_worker_type: str,
        latency_ms: float,
        cold_start: bool,
        reason: str,
        fallback_reason: str | None = None,
    ) -> ScheduledResult:
        # Kept as a separate constructor so successful primary and fallback paths serialize
        # exactly the same decision state.
        selected_worker = self.edge_worker if selected_worker_type == "edge" else self.remote_worker
        fallback = selected_worker_type != worker_result.worker_type
        return ScheduledResult(
            strategy=self.strategy,
            selected_worker_id=selected_worker.worker_id,
            selected_worker_type=selected_worker_type,
            executed_worker_id=worker_result.worker_id,
            executed_worker_type=worker_result.worker_type,
            fallback=fallback,
            fallback_reason=fallback_reason,
            edge_injected_delay_ms=getattr(self.edge_worker, "delay_ms", 0.0),
            remote_injected_delay_ms=getattr(self.remote_worker, "delay_ms", 0.0),
            scheduler_latency_ms=latency_ms,
            worker_result=worker_result,
            latency_decision=LatencyDecision(
                edge_latency_estimate_ms=self.edge_latency_estimate_ms,
                remote_latency_estimate_ms=self.remote_latency_estimate_ms,
                cold_start=cold_start,
                reason=reason,
            ),
        )

    def _select_worker(self) -> tuple[InferenceWorker, str, bool, str]:
        if self.edge_sample_count == 0:
            return self.edge_worker, "edge", True, "cold_start_edge"
        if self.remote_sample_count == 0:
            return self.remote_worker, "remote", True, "cold_start_remote"
        edge_estimate = self.edge_latency_estimate_ms
        remote_estimate = self.remote_latency_estimate_ms
        if edge_estimate is None or remote_estimate is None:
            raise RuntimeError("latency estimates missing after warm-up")
        if edge_estimate < remote_estimate:
            return self.edge_worker, "edge", False, "edge_lower_estimated_latency"
        if remote_estimate < edge_estimate:
            return self.remote_worker, "remote", False, "remote_lower_estimated_latency"
        return self.edge_worker, "edge", False, "latency_tie_edge"

    def summary(self) -> dict[str, int | float | None]:
        value: dict[str, int | float | None] = self.metrics.to_dict()
        value.update(
            {
                "edge_latency_estimate_ms": self.edge_latency_estimate_ms,
                "remote_latency_estimate_ms": self.remote_latency_estimate_ms,
                "edge_sample_count": self.edge_sample_count,
                "remote_sample_count": self.remote_sample_count,
            }
        )
        return value


def _can_latency_fallback(error: WorkerError, worker_type: str) -> bool:
    if isinstance(error, WorkerExecutionError):
        return worker_type == "edge"
    if worker_type != "remote":
        return False
    if isinstance(error, WorkerHTTPError):
        return error.status_code == 429 or error.status_code >= 500
    return isinstance(error, (WorkerConnectionError, WorkerTimeoutError, WorkerUnavailableError))


class TelemetryStateSource(Protocol):
    def edge(self) -> CachedTelemetry: ...

    def remote(self) -> CachedTelemetry: ...


@dataclass(slots=True)
class ResourceAwareScheduler:
    edge_worker: InferenceWorker
    remote_worker: InferenceWorker
    telemetry: TelemetryStateSource
    alpha: float = 0.3
    pressure_threshold: float = 0.85
    edge_resource: WorkerResource = field(
        default_factory=lambda: WorkerResource(ExecutionResource.CPU)
    )
    remote_resource: WorkerResource = field(
        default_factory=lambda: WorkerResource(ExecutionResource.CPU)
    )
    clock: Callable[[], float] = perf_counter
    strategy: SchedulerStrategy = field(init=False, default=SchedulerStrategy.RESOURCE_AWARE)
    metrics: SchedulerMetrics = field(default_factory=SchedulerMetrics)
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

    @property
    def edge_sample_count(self) -> int:
        return self._edge_estimator.sample_count

    @property
    def remote_sample_count(self) -> int:
        return self._remote_estimator.sample_count

    def infer(self, image: Image) -> ScheduledResult:
        # Cache reads happen before taking the scheduler lock. Background HTTP sampling can
        # therefore never block latency state or inference selection.
        edge_telemetry = self.telemetry.edge()
        remote_telemetry = self.telemetry.remote()
        edge_pressure = resource_pressure(edge_telemetry, self.edge_resource)
        remote_pressure = resource_pressure(remote_telemetry, self.remote_resource)
        with self._state_lock:
            worker, worker_type, cold_start, reason = self._select_worker(
                edge_telemetry, remote_telemetry, edge_pressure, remote_pressure
            )
            started = self.clock()
            try:
                worker_result = worker.infer(image)
            except (OSError, RuntimeError, ValueError):
                latency_ms = (self.clock() - started) * 1000
                self.metrics.record(worker_type, latency_ms, succeeded=False)
                raise
            latency_ms = (self.clock() - started) * 1000
            self.metrics.record(worker_type, latency_ms, succeeded=True)
            estimator = self._edge_estimator if worker_type == "edge" else self._remote_estimator
            estimator.observe(latency_ms)
            return ScheduledResult(
                strategy=self.strategy,
                selected_worker_id=worker_result.worker_id,
                selected_worker_type=worker_result.worker_type,
                scheduler_latency_ms=latency_ms,
                worker_result=worker_result,
                resource_decision=ResourceDecision(
                    edge_latency_estimate_ms=self.edge_latency_estimate_ms,
                    remote_latency_estimate_ms=self.remote_latency_estimate_ms,
                    edge_resource_pressure=edge_pressure,
                    remote_resource_pressure=remote_pressure,
                    edge_telemetry_status=edge_telemetry.status.value,
                    remote_telemetry_status=remote_telemetry.status.value,
                    edge_telemetry_error=edge_telemetry.last_error,
                    remote_telemetry_error=remote_telemetry.last_error,
                    edge_execution_resource=self.edge_resource.resource_type.value,
                    remote_execution_resource=self.remote_resource.resource_type.value,
                    cold_start=cold_start,
                    reason=reason,
                ),
            )

    def _select_worker(
        self,
        edge_telemetry: CachedTelemetry,
        remote_telemetry: CachedTelemetry,
        edge_pressure: float | None,
        remote_pressure: float | None,
    ) -> tuple[InferenceWorker, str, bool, str]:
        preference = choose_resource_preference(
            edge_sample_count=self.edge_sample_count,
            remote_sample_count=self.remote_sample_count,
            edge_latency_ms=self.edge_latency_estimate_ms,
            remote_latency_ms=self.remote_latency_estimate_ms,
            edge_telemetry_status=edge_telemetry.status,
            remote_telemetry_status=remote_telemetry.status,
            edge_pressure=edge_pressure,
            remote_pressure=remote_pressure,
            pressure_threshold=self.pressure_threshold,
        )
        worker = self.edge_worker if preference.worker_type == "edge" else self.remote_worker
        return worker, preference.worker_type, preference.cold_start, preference.reason

    def summary(self) -> dict[str, int | float | str | None]:
        edge_telemetry = self.telemetry.edge()
        remote_telemetry = self.telemetry.remote()
        value: dict[str, int | float | str | None] = self.metrics.to_dict()
        value.update(
            {
                "edge_latency_estimate_ms": self.edge_latency_estimate_ms,
                "remote_latency_estimate_ms": self.remote_latency_estimate_ms,
                "edge_sample_count": self.edge_sample_count,
                "remote_sample_count": self.remote_sample_count,
                "edge_resource_pressure": resource_pressure(edge_telemetry, self.edge_resource),
                "remote_resource_pressure": resource_pressure(
                    remote_telemetry, self.remote_resource
                ),
                "edge_telemetry_status": edge_telemetry.status.value,
                "remote_telemetry_status": remote_telemetry.status.value,
                "edge_telemetry_error": edge_telemetry.last_error,
                "remote_telemetry_error": remote_telemetry.last_error,
            }
        )
        return value


def create_scheduler(
    strategy: str | SchedulerStrategy,
    *,
    edge_worker: InferenceWorker | None = None,
    remote_worker: InferenceWorker | None = None,
    latency_alpha: float = 0.3,
    telemetry_state: TelemetryStateSource | None = None,
    resource_pressure_threshold: float = 0.85,
    edge_resource: WorkerResource | None = None,
    remote_resource: WorkerResource | None = None,
    health_state: HealthStateSource | None = None,
) -> Scheduler:
    try:
        selected = SchedulerStrategy(strategy)
    except ValueError as exc:
        choices = ", ".join(item.value for item in SchedulerStrategy)
        raise ValueError(
            f"unknown scheduler strategy {strategy!r}; choose from: {choices}"
        ) from exc

    if selected is SchedulerStrategy.EDGE_ONLY:
        if edge_worker is None:
            raise ValueError("edge_only requires an edge worker")
        return EdgeOnlyScheduler(edge_worker)
    if selected is SchedulerStrategy.REMOTE_ONLY:
        if remote_worker is None:
            raise ValueError("remote_only requires a remote worker")
        return RemoteOnlyScheduler(remote_worker)
    if edge_worker is None or remote_worker is None:
        raise ValueError(f"{selected.value} requires edge and remote workers")
    if selected is SchedulerStrategy.ROUND_ROBIN:
        return RoundRobinScheduler(edge_worker, remote_worker)
    if selected is SchedulerStrategy.LATENCY_AWARE:
        return LatencyAwareScheduler(edge_worker, remote_worker, alpha=latency_alpha)
    if telemetry_state is None:
        raise ValueError(f"{selected.value} requires telemetry state")
    if selected is SchedulerStrategy.RESILIENT:
        if health_state is None:
            raise ValueError("resilient requires health state")
        return ResilientScheduler(
            edge_worker,
            remote_worker,
            telemetry_state,
            health_state,
            alpha=latency_alpha,
            pressure_threshold=resource_pressure_threshold,
            edge_resource=edge_resource or WorkerResource(ExecutionResource.CPU),
            remote_resource=remote_resource or WorkerResource(ExecutionResource.CPU),
        )
    return ResourceAwareScheduler(
        edge_worker,
        remote_worker,
        telemetry_state,
        alpha=latency_alpha,
        pressure_threshold=resource_pressure_threshold,
        edge_resource=edge_resource or WorkerResource(ExecutionResource.CPU),
        remote_resource=remote_resource or WorkerResource(ExecutionResource.CPU),
    )
