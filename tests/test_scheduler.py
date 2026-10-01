from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from edgecloud.inference.models import BoundingBox, Detection, TimingMetrics
from edgecloud.scheduler import (
    EdgeOnlyScheduler,
    LatencyAwareScheduler,
    RemoteOnlyScheduler,
    RoundRobinScheduler,
    SchedulerStrategy,
    create_scheduler,
)
from edgecloud.workers.errors import (
    WorkerConnectionError,
    WorkerTimeoutError,
    WorkerUnavailableError,
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
        return WorkerResult(
            detections=(Detection(BoundingBox(1, 2, 3, 4), "object", 0.9, 1),),
            timing=TimingMetrics(1, 2, 3, 6),
            worker_id=self.worker_id,
            worker_type=self.worker_type,
            backend="fake",
        )


def frame() -> np.ndarray:
    return np.zeros((8, 8, 3), dtype=np.uint8)


def test_edge_only_always_selects_edge() -> None:
    edge = FakeWorker("edge-test", "edge")
    scheduler = EdgeOnlyScheduler(edge)

    results = [scheduler.infer(frame()) for _ in range(3)]

    assert [result.selected_worker_type for result in results] == ["edge"] * 3
    assert edge.calls == 3
    assert scheduler.metrics.edge_selections == 3


def test_edge_only_propagates_failure_without_remote_attempt() -> None:
    failure = RuntimeError("edge failed")
    edge = FakeWorker("edge-test", "edge", failure)
    scheduler = EdgeOnlyScheduler(edge)

    with pytest.raises(RuntimeError, match="edge failed"):
        scheduler.infer(frame())

    assert scheduler.metrics.failed_requests == 1
    assert scheduler.metrics.edge_selections == 1


@pytest.mark.parametrize(
    "failure",
    [
        WorkerTimeoutError("timeout"),
        WorkerUnavailableError("unavailable"),
        WorkerConnectionError("connection failed"),
    ],
)
def test_remote_only_propagates_worker_failure(failure: Exception) -> None:
    remote = FakeWorker("remote-test", "remote", failure)
    scheduler = RemoteOnlyScheduler(remote)

    with pytest.raises(type(failure)):
        scheduler.infer(frame())

    assert remote.calls == 1
    assert scheduler.metrics.failed_requests == 1
    assert scheduler.metrics.remote_selections == 1


def test_remote_only_always_selects_remote() -> None:
    remote = FakeWorker("remote-test", "remote")
    scheduler = RemoteOnlyScheduler(remote)

    results = [scheduler.infer(frame()) for _ in range(2)]

    assert [result.selected_worker_type for result in results] == ["remote", "remote"]


def test_round_robin_starts_with_edge_and_alternates() -> None:
    scheduler = RoundRobinScheduler(
        FakeWorker("edge-test", "edge"), FakeWorker("remote-test", "remote")
    )

    results = [scheduler.infer(frame()) for _ in range(6)]

    assert [result.selected_worker_type for result in results] == [
        "edge",
        "remote",
        "edge",
        "remote",
        "edge",
        "remote",
    ]


def test_round_robin_failure_advances_sequence_without_fallback() -> None:
    edge = FakeWorker("edge-test", "edge")
    remote = FakeWorker("remote-test", "remote", WorkerTimeoutError("timeout"))
    scheduler = RoundRobinScheduler(edge, remote)

    first = scheduler.infer(frame())
    with pytest.raises(WorkerTimeoutError):
        scheduler.infer(frame())
    third = scheduler.infer(frame())

    assert first.selected_worker_type == "edge"
    assert third.selected_worker_type == "edge"
    assert edge.calls == 2
    assert remote.calls == 1
    assert scheduler.metrics.total_requests == 3
    assert scheduler.metrics.failed_requests == 1


def test_scheduled_result_composes_worker_result_and_metadata() -> None:
    result = EdgeOnlyScheduler(FakeWorker("edge-test", "edge")).infer(frame())

    assert result.strategy is SchedulerStrategy.EDGE_ONLY
    assert result.selected_worker_id == "edge-test"
    assert result.selected_worker_type == "edge"
    assert result.scheduler_latency_ms >= 0
    assert result.worker_result.detection_count == 1
    assert "latency_decision" not in result.to_dict()


def test_scheduler_metrics_provide_totals_mean_and_percentages() -> None:
    scheduler = RoundRobinScheduler(
        FakeWorker("edge-test", "edge"), FakeWorker("remote-test", "remote")
    )

    for _ in range(4):
        scheduler.infer(frame())

    metrics = scheduler.metrics.to_dict()
    assert metrics["total_requests"] == 4
    assert metrics["successful_requests"] == 4
    assert metrics["failed_requests"] == 0
    assert metrics["edge_selection_percentage"] == 50
    assert metrics["remote_selection_percentage"] == 50
    assert metrics["mean_latency_ms"] >= 0


@pytest.mark.parametrize(
    ("name", "expected_type"),
    [
        ("edge_only", EdgeOnlyScheduler),
        ("remote_only", RemoteOnlyScheduler),
        ("round_robin", RoundRobinScheduler),
        ("latency_aware", LatencyAwareScheduler),
    ],
)
def test_create_scheduler_accepts_valid_names(name: str, expected_type: type) -> None:
    scheduler = create_scheduler(
        name,
        edge_worker=FakeWorker("edge-test", "edge"),
        remote_worker=FakeWorker("remote-test", "remote"),
    )

    assert isinstance(scheduler, expected_type)


def test_create_scheduler_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown scheduler strategy"):
        create_scheduler("adaptive")
