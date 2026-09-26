from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from time import perf_counter

from edgecloud.inference.engine import Image
from edgecloud.scheduler.models import (
    ScheduledResult,
    Scheduler,
    SchedulerMetrics,
    SchedulerStrategy,
)
from edgecloud.workers.models import InferenceWorker


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


@dataclass(slots=True)
class RemoteOnlyScheduler:
    remote_worker: InferenceWorker
    strategy: SchedulerStrategy = field(init=False, default=SchedulerStrategy.REMOTE_ONLY)
    metrics: SchedulerMetrics = field(default_factory=SchedulerMetrics)

    def infer(self, image: Image) -> ScheduledResult:
        return _infer_with(self.remote_worker, self.strategy, self.metrics, image, "remote")


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


def create_scheduler(
    strategy: str | SchedulerStrategy,
    *,
    edge_worker: InferenceWorker | None = None,
    remote_worker: InferenceWorker | None = None,
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
        raise ValueError("round_robin requires edge and remote workers")
    return RoundRobinScheduler(edge_worker, remote_worker)
