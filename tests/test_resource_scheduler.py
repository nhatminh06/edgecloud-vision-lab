from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pytest

from edgecloud.inference.models import TimingMetrics
from edgecloud.scheduler import (
    ExecutionResource,
    ResourceAwareScheduler,
    WorkerResource,
    create_scheduler,
    resource_pressure,
)
from edgecloud.telemetry.cache import CachedTelemetry, TelemetryStatus
from edgecloud.telemetry.models import GpuMetrics, SystemMetrics, TelemetrySnapshot
from edgecloud.workers.errors import WorkerTimeoutError
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
        return WorkerResult(
            detections=(),
            timing=TimingMetrics(1, 2, 3, 6),
            worker_id=self.worker_id,
            worker_type=self.worker_type,
            backend="fake",
        )


class SequenceClock:
    def __init__(self, *values: float) -> None:
        self.values: Iterator[float] = iter(values)

    def __call__(self) -> float:
        return next(self.values)


class FakeTelemetryState:
    def __init__(self, edge: CachedTelemetry, remote: CachedTelemetry) -> None:
        self.edge_value = edge
        self.remote_value = remote

    def edge(self) -> CachedTelemetry:
        return self.edge_value

    def remote(self) -> CachedTelemetry:
        return self.remote_value


def snapshot(
    cpu: float,
    memory: float,
    *,
    gpu: float | None = None,
    gpu_memory: float | None = None,
) -> TelemetrySnapshot:
    gpus = ()
    if gpu is not None or gpu_memory is not None:
        gpus = (GpuMetrics(0, "GPU", gpu, 1, 2, gpu_memory, None, None),)
    return TelemetrySnapshot(
        datetime(2026, 1, 1, tzinfo=UTC),
        SystemMetrics(cpu, 4_000, 10_000, memory),
        gpus,
        bool(gpus),
    )


def cached(
    cpu: float,
    memory: float,
    *,
    status: TelemetryStatus = TelemetryStatus.AVAILABLE,
    gpu: float | None = None,
    gpu_memory: float | None = None,
) -> CachedTelemetry:
    return CachedTelemetry(snapshot(cpu, memory, gpu=gpu, gpu_memory=gpu_memory), status, 0, None)


def unavailable() -> CachedTelemetry:
    return CachedTelemetry(None, TelemetryStatus.UNAVAILABLE, None, "sample failed")


def frame() -> np.ndarray:
    return np.zeros((8, 8, 3), dtype=np.uint8)


@pytest.mark.parametrize(
    ("cpu", "memory", "expected"), [(20, 30, 0.3), (95, 30, 0.95), (20, 90, 0.9)]
)
def test_cpu_resource_pressure_uses_larger_fraction(
    cpu: float, memory: float, expected: float
) -> None:
    assert resource_pressure(cached(cpu, memory), WorkerResource(ExecutionResource.CPU)) == expected


@pytest.mark.parametrize(
    ("gpu", "gpu_memory", "expected"), [(90, 20, 0.9), (20, 95, 0.95), (None, None, None)]
)
def test_gpu_resource_pressure_uses_available_gpu_metrics(
    gpu: float | None, gpu_memory: float | None, expected: float | None
) -> None:
    telemetry = cached(10, 10, gpu=gpu, gpu_memory=gpu_memory)
    assert (
        resource_pressure(telemetry, WorkerResource(ExecutionResource.GPU, gpu_index=0)) == expected
    )


def test_resource_pressure_is_missing_for_stale_or_absent_gpu() -> None:
    assert resource_pressure(unavailable(), WorkerResource(ExecutionResource.CPU)) is None
    assert (
        resource_pressure(
            cached(10, 10, status=TelemetryStatus.STALE),
            WorkerResource(ExecutionResource.CPU),
        )
        is None
    )
    assert (
        resource_pressure(cached(10, 10), WorkerResource(ExecutionResource.GPU, gpu_index=0))
        is None
    )


def scheduler_for(
    telemetry: FakeTelemetryState,
    *clock_values: float,
    edge_resource: WorkerResource | None = None,
    remote_resource: WorkerResource | None = None,
) -> ResourceAwareScheduler:
    return ResourceAwareScheduler(
        FakeWorker("edge-test", "edge"),
        FakeWorker("remote-test", "remote"),
        telemetry,
        edge_resource=edge_resource or WorkerResource(ExecutionResource.CPU),
        remote_resource=remote_resource or WorkerResource(ExecutionResource.CPU),
        clock=SequenceClock(*clock_values),
    )


def warm_up(scheduler: ResourceAwareScheduler) -> None:
    assert scheduler.infer(frame()).selected_worker_type == "edge"
    assert scheduler.infer(frame()).selected_worker_type == "remote"


def test_cold_start_samples_edge_then_remote() -> None:
    telemetry = FakeTelemetryState(cached(10, 10), cached(10, 10))
    scheduler = scheduler_for(telemetry, 0, 0.010, 1, 1.020)

    first = scheduler.infer(frame())
    second = scheduler.infer(frame())

    assert first.resource_decision is not None
    assert first.resource_decision.reason == "cold_start_edge"
    assert second.resource_decision is not None
    assert second.resource_decision.reason == "cold_start_remote"


@pytest.mark.parametrize(
    ("edge_ms", "remote_ms", "expected", "reason"),
    [(10, 20, "edge", "edge_lower_latency"), (20, 10, "remote", "remote_lower_latency")],
)
def test_lower_latency_wins_under_normal_pressure(
    edge_ms: float, remote_ms: float, expected: str, reason: str
) -> None:
    telemetry = FakeTelemetryState(cached(20, 20), cached(30, 30))
    scheduler = scheduler_for(telemetry, 0, edge_ms / 1000, 1, 1 + remote_ms / 1000, 2, 2.015)
    warm_up(scheduler)

    result = scheduler.infer(frame())

    assert result.selected_worker_type == expected
    assert result.resource_decision is not None
    assert result.resource_decision.reason == reason


def test_high_edge_pressure_prefers_remote_despite_lower_edge_latency() -> None:
    telemetry = FakeTelemetryState(cached(95, 20), cached(20, 20))
    scheduler = scheduler_for(telemetry, 0, 0.010, 1, 1.020, 2, 2.015)
    warm_up(scheduler)

    result = scheduler.infer(frame())

    assert result.selected_worker_type == "remote"
    assert result.resource_decision is not None
    assert result.resource_decision.reason == "edge_high_resource_pressure"


def test_high_remote_pressure_prefers_edge_despite_lower_remote_latency() -> None:
    telemetry = FakeTelemetryState(cached(20, 20), cached(95, 20))
    scheduler = scheduler_for(telemetry, 0, 0.020, 1, 1.010, 2, 2.015)
    warm_up(scheduler)

    result = scheduler.infer(frame())

    assert result.selected_worker_type == "edge"
    assert result.resource_decision is not None
    assert result.resource_decision.reason == "remote_high_resource_pressure"


def test_equal_high_pressure_uses_deterministic_latency_choice() -> None:
    telemetry = FakeTelemetryState(cached(90, 20), cached(90, 20))
    scheduler = scheduler_for(telemetry, 0, 0.010, 1, 1.020, 2, 2.015)
    warm_up(scheduler)

    assert scheduler.infer(frame()).selected_worker_type == "edge"


def test_latency_tie_selects_edge() -> None:
    telemetry = FakeTelemetryState(cached(20, 20), cached(20, 20))
    scheduler = scheduler_for(telemetry, 0, 1, 10, 11, 20, 21)
    warm_up(scheduler)

    result = scheduler.infer(frame())

    assert result.selected_worker_type == "edge"
    assert result.resource_decision is not None
    assert result.resource_decision.reason == "latency_tie_edge"


@pytest.mark.parametrize(
    ("edge_value", "remote_value", "reason"),
    [
        (unavailable(), cached(20, 20), "telemetry_unavailable_latency_fallback"),
        (
            cached(20, 20, status=TelemetryStatus.STALE),
            cached(20, 20),
            "telemetry_stale_latency_fallback",
        ),
    ],
)
def test_missing_or_stale_telemetry_falls_back_to_latency(
    edge_value: CachedTelemetry, remote_value: CachedTelemetry, reason: str
) -> None:
    scheduler = scheduler_for(
        FakeTelemetryState(edge_value, remote_value), 0, 0.010, 1, 1.020, 2, 2.015
    )
    warm_up(scheduler)

    result = scheduler.infer(frame())

    assert result.selected_worker_type == "edge"
    assert result.resource_decision is not None
    assert result.resource_decision.reason == reason


def test_cpu_worker_ignores_unrelated_gpu_pressure() -> None:
    telemetry = FakeTelemetryState(
        cached(20, 20, gpu=100, gpu_memory=100), cached(30, 30, gpu=0, gpu_memory=0)
    )
    scheduler = scheduler_for(telemetry, 0, 0.010, 1, 1.020, 2, 2.015)
    warm_up(scheduler)

    result = scheduler.infer(frame())

    assert result.selected_worker_type == "edge"
    assert result.resource_decision is not None
    assert result.resource_decision.edge_resource_pressure == 0.2


def test_gpu_worker_uses_gpu_pressure() -> None:
    telemetry = FakeTelemetryState(
        cached(10, 10, gpu=95, gpu_memory=20), cached(20, 20, gpu=10, gpu_memory=10)
    )
    gpu_resource = WorkerResource(ExecutionResource.GPU, gpu_index=0)
    scheduler = scheduler_for(
        telemetry,
        0,
        0.010,
        1,
        1.020,
        2,
        2.015,
        edge_resource=gpu_resource,
        remote_resource=gpu_resource,
    )
    warm_up(scheduler)

    result = scheduler.infer(frame())

    assert result.selected_worker_type == "remote"
    assert result.resource_decision is not None
    assert result.resource_decision.reason == "edge_high_resource_pressure"


def test_inference_failure_propagates_without_fallback() -> None:
    edge = FakeWorker("edge-test", "edge", WorkerTimeoutError("failed"))
    remote = FakeWorker("remote-test", "remote")
    scheduler = ResourceAwareScheduler(
        edge,
        remote,
        FakeTelemetryState(cached(10, 10), cached(10, 10)),
        clock=SequenceClock(0, 0.01),
    )

    with pytest.raises(WorkerTimeoutError):
        scheduler.infer(frame())

    assert edge.calls == 1
    assert remote.calls == 0
    assert scheduler.edge_sample_count == 0


def test_sampling_error_on_valid_cache_does_not_fail_inference() -> None:
    valid_with_error = CachedTelemetry(
        snapshot(20, 20), TelemetryStatus.AVAILABLE, 1, "remote telemetry timeout"
    )
    telemetry = FakeTelemetryState(valid_with_error, cached(20, 20))
    scheduler = scheduler_for(telemetry, 0, 0.010, 1, 1.020, 2, 2.015)
    warm_up(scheduler)

    result = scheduler.infer(frame())

    assert result.selected_worker_type == "edge"
    assert scheduler.metrics.failed_requests == 0


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_resource_pressure_threshold_validation(threshold: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        ResourceAwareScheduler(
            FakeWorker("edge", "edge"),
            FakeWorker("remote", "remote"),
            FakeTelemetryState(cached(20, 20), cached(20, 20)),
            pressure_threshold=threshold,
        )


def test_create_resource_aware_scheduler_requires_and_accepts_telemetry_state() -> None:
    edge = FakeWorker("edge", "edge")
    remote = FakeWorker("remote", "remote")
    with pytest.raises(ValueError, match="telemetry state"):
        create_scheduler("resource_aware", edge_worker=edge, remote_worker=remote)

    scheduler = create_scheduler(
        "resource_aware",
        edge_worker=edge,
        remote_worker=remote,
        telemetry_state=FakeTelemetryState(cached(20, 20), cached(20, 20)),
    )

    assert isinstance(scheduler, ResourceAwareScheduler)


def test_pressure_changes_switch_and_restore_latency_routing() -> None:
    telemetry = FakeTelemetryState(cached(20, 20), cached(30, 30))
    scheduler = scheduler_for(telemetry, 0, 0.010, 1, 1.020, 2, 2.012, 3, 3.018, 4, 4.011)
    warm_up(scheduler)
    normal = scheduler.infer(frame())
    telemetry.edge_value = cached(95, 20)
    pressured = scheduler.infer(frame())
    telemetry.edge_value = cached(20, 20)
    restored = scheduler.infer(frame())

    assert [
        normal.selected_worker_type,
        pressured.selected_worker_type,
        restored.selected_worker_type,
    ] == ["edge", "remote", "edge"]
