from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pytest

from edgecloud.inference.models import TimingMetrics
from edgecloud.scheduler import ResilientScheduler, WorkerHealth, create_scheduler
from edgecloud.scheduler.errors import ResilientRequestError, SchedulerUnavailableError
from edgecloud.scheduler.health import CachedHealth
from edgecloud.telemetry.cache import CachedTelemetry, TelemetryStatus
from edgecloud.telemetry.models import SystemMetrics, TelemetrySnapshot
from edgecloud.workers.errors import (
    WorkerExecutionError,
    WorkerHTTPError,
    WorkerResponseError,
    WorkerTimeoutError,
)
from edgecloud.workers.models import WorkerResult


@dataclass
class FakeWorker:
    worker_id: str
    worker_type: str
    failure: Exception | None = None
    calls: int = 0

    def infer(self, image: np.ndarray) -> WorkerResult:
        self.calls += 1
        if self.failure is not None:
            raise self.failure
        return WorkerResult((), TimingMetrics(1, 2, 3, 6), self.worker_id, self.worker_type, "fake")


class SequenceClock:
    def __init__(self, *values: float) -> None:
        self.values: Iterator[float] = iter(values)

    def __call__(self) -> float:
        return next(self.values)


class FakeTelemetry:
    def __init__(self, edge_cpu: float = 20, remote_cpu: float = 20) -> None:
        self.edge_value = telemetry(edge_cpu)
        self.remote_value = telemetry(remote_cpu)

    def edge(self) -> CachedTelemetry:
        return self.edge_value

    def remote(self) -> CachedTelemetry:
        return self.remote_value


class FakeHealth:
    def __init__(
        self,
        edge: WorkerHealth = WorkerHealth.UNKNOWN,
        remote: WorkerHealth = WorkerHealth.UNKNOWN,
        *,
        fresh: bool = True,
    ) -> None:
        self.values = {
            "edge": CachedHealth(edge, fresh, 0),
            "remote": CachedHealth(remote, fresh, 0),
        }

    def edge(self) -> CachedHealth:
        return self.values["edge"]

    def remote(self) -> CachedHealth:
        return self.values["remote"]

    def mark_healthy(self, worker_type: str, detail: str | None = None) -> None:
        self.values[worker_type] = CachedHealth(WorkerHealth.HEALTHY, True, 0, detail)

    def mark_unhealthy(self, worker_type: str, detail: str | None = None) -> None:
        self.values[worker_type] = CachedHealth(WorkerHealth.UNHEALTHY, True, 0, detail)


def telemetry(cpu: float) -> CachedTelemetry:
    snapshot = TelemetrySnapshot(
        datetime(2026, 1, 1, tzinfo=UTC),
        SystemMetrics(cpu, 1, 2, 20),
        (),
        False,
    )
    return CachedTelemetry(snapshot, TelemetryStatus.AVAILABLE, 0, None)


def frame() -> np.ndarray:
    return np.zeros((8, 8, 3), dtype=np.uint8)


def test_unknown_health_allows_cold_start_primary() -> None:
    scheduler = ResilientScheduler(
        FakeWorker("edge-1", "edge"),
        FakeWorker("remote-1", "remote"),
        FakeTelemetry(),
        FakeHealth(fresh=False),
        clock=SequenceClock(0, 0.01),
    )

    result = scheduler.infer(frame())

    assert result.selected_worker_type == "edge"
    assert result.resilient_decision is not None
    assert result.resilient_decision.edge_health == "unknown"
    assert result.resilient_decision.reason == "cold_start_edge"


def test_fresh_unhealthy_primary_is_not_dispatched() -> None:
    edge = FakeWorker("edge-1", "edge")
    remote = FakeWorker("remote-1", "remote")
    scheduler = ResilientScheduler(
        edge,
        remote,
        FakeTelemetry(),
        FakeHealth(WorkerHealth.UNHEALTHY, WorkerHealth.UNKNOWN),
        clock=SequenceClock(0, 0.02),
    )

    result = scheduler.infer(frame())

    assert edge.calls == 0
    assert remote.calls == 1
    assert result.resilient_decision is not None
    assert result.resilient_decision.reason == "edge_unhealthy_remote_selected"


def test_both_fresh_unhealthy_fails_without_attempt() -> None:
    edge = FakeWorker("edge-1", "edge")
    remote = FakeWorker("remote-1", "remote")
    scheduler = ResilientScheduler(
        edge,
        remote,
        FakeTelemetry(),
        FakeHealth(WorkerHealth.UNHEALTHY, WorkerHealth.UNHEALTHY),
    )

    with pytest.raises(SchedulerUnavailableError, match="edge and remote"):
        scheduler.infer(frame())

    assert edge.calls == remote.calls == 0
    assert scheduler.metrics.logical_requests == 1
    assert scheduler.metrics.worker_attempts == 0


def test_local_execution_failure_falls_back_once_and_updates_only_remote_ewma() -> None:
    edge = FakeWorker("edge-1", "edge", WorkerExecutionError("backend failed"))
    remote = FakeWorker("remote-1", "remote")
    health = FakeHealth()
    scheduler = ResilientScheduler(
        edge,
        remote,
        FakeTelemetry(),
        health,
        clock=SequenceClock(0, 0.01, 1, 1.02),
    )

    result = scheduler.infer(frame())

    assert result.selected_worker_type == "remote"
    assert result.scheduler_latency_ms == pytest.approx(30)
    assert scheduler.edge_latency_estimate_ms is None
    assert scheduler.remote_latency_estimate_ms == pytest.approx(20)
    assert health.edge().state is WorkerHealth.UNHEALTHY
    assert result.resilient_decision is not None
    assert result.resilient_decision.primary_failure_code == "execution_error"
    assert result.resilient_decision.fallback_occurred is True
    assert scheduler.metrics.worker_attempts == 2
    assert scheduler.metrics.successful_fallbacks == 1


def test_remote_timeout_falls_back_to_edge() -> None:
    edge = FakeWorker("edge-1", "edge")
    remote = FakeWorker("remote-1", "remote")
    scheduler = ResilientScheduler(
        edge,
        remote,
        FakeTelemetry(),
        FakeHealth(),
        clock=SequenceClock(0, 0.01, 1, 1.02, 2, 2.015),
    )
    scheduler.infer(frame())
    remote.failure = WorkerTimeoutError("timeout")

    result = scheduler.infer(frame())

    assert result.selected_worker_type == "edge"
    assert result.resilient_decision is not None
    assert result.resilient_decision.initial_worker_type == "remote"
    assert result.resilient_decision.primary_failure_code == "timeout"
    assert scheduler.metrics.remote_to_edge_fallbacks == 1
    assert scheduler.metrics.logical_requests == 2
    assert scheduler.metrics.worker_attempts == 3


@pytest.mark.parametrize(
    "failure",
    [WorkerResponseError("malformed"), WorkerHTTPError(400), ValueError("bad input")],
)
def test_non_recoverable_primary_failure_does_not_fallback(failure: Exception) -> None:
    edge = FakeWorker("edge-1", "edge", failure)
    remote = FakeWorker("remote-1", "remote")
    scheduler = ResilientScheduler(
        edge,
        remote,
        FakeTelemetry(),
        FakeHealth(),
        clock=SequenceClock(0, 0.01),
    )

    with pytest.raises(type(failure)):
        scheduler.infer(frame())

    assert remote.calls == 0
    assert scheduler.metrics.fallback_attempts == 0


def test_failed_fallback_stops_after_two_attempts() -> None:
    edge = FakeWorker("edge-1", "edge", WorkerExecutionError("edge failed"))
    remote = FakeWorker("remote-1", "remote", WorkerTimeoutError("remote failed"))
    scheduler = ResilientScheduler(
        edge,
        remote,
        FakeTelemetry(),
        FakeHealth(),
        clock=SequenceClock(0, 0.01, 1, 1.02),
    )

    with pytest.raises(ResilientRequestError):
        scheduler.infer(frame())

    assert edge.calls == remote.calls == 1
    assert scheduler.metrics.worker_attempts == 2
    assert scheduler.metrics.failed_fallbacks == 1
    assert scheduler.metrics.failed_logical_requests == 1


def test_create_resilient_scheduler_requires_health_and_telemetry() -> None:
    edge = FakeWorker("edge-1", "edge")
    remote = FakeWorker("remote-1", "remote")
    with pytest.raises(ValueError, match="telemetry state"):
        create_scheduler("resilient", edge_worker=edge, remote_worker=remote)
    with pytest.raises(ValueError, match="health state"):
        create_scheduler(
            "resilient",
            edge_worker=edge,
            remote_worker=remote,
            telemetry_state=FakeTelemetry(),
        )

    scheduler = create_scheduler(
        "resilient",
        edge_worker=edge,
        remote_worker=remote,
        telemetry_state=FakeTelemetry(),
        health_state=FakeHealth(),
    )

    assert isinstance(scheduler, ResilientScheduler)
