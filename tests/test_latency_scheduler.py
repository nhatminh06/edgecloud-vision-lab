from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pytest

from edgecloud.inference.engine import InferenceEngine
from edgecloud.inference.models import TimingMetrics
from edgecloud.scheduler import EwmaEstimator, LatencyAwareScheduler
from edgecloud.workers.edge import EdgeWorker
from edgecloud.workers.errors import (
    WorkerExecutionError,
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
            detections=(),
            timing=TimingMetrics(1, 2, 3, 6),
            worker_id=self.worker_id,
            worker_type=self.worker_type,
            backend="fake",
        )


@dataclass
class IdentityFromResultWorker:
    worker_type: str

    def infer(self, image: np.ndarray) -> WorkerResult:
        return WorkerResult(
            detections=(),
            timing=TimingMetrics(1, 2, 3, 6),
            worker_id=f"{self.worker_type}-reported",
            worker_type=self.worker_type,
            backend="fake",
        )


class SequenceClock:
    def __init__(self, *values: float) -> None:
        self._values: Iterator[float] = iter(values)

    def __call__(self) -> float:
        return next(self._values)


def frame() -> np.ndarray:
    return np.zeros((8, 8, 3), dtype=np.uint8)


def test_ewma_first_and_repeated_observations() -> None:
    estimator = EwmaEstimator(alpha=0.25)

    assert estimator.observe(10) == 10
    assert estimator.observe(30) == 15
    assert estimator.observe(15) == 15
    assert estimator.sample_count == 3


@pytest.mark.parametrize("alpha", [-0.1, 0.0, 1.1])
def test_ewma_rejects_invalid_alpha(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        EwmaEstimator(alpha)


def test_cold_start_samples_edge_then_remote_from_observations() -> None:
    scheduler = LatencyAwareScheduler(
        FakeWorker("edge-test", "edge"),
        FakeWorker("remote-test", "remote"),
        clock=SequenceClock(0, 0.010, 1, 1.020),
    )

    first = scheduler.infer(frame())
    second = scheduler.infer(frame())

    assert first.selected_worker_type == "edge"
    assert first.latency_decision is not None
    assert first.latency_decision.reason == "cold_start_edge"
    assert first.latency_decision.cold_start is True
    assert first.latency_decision.edge_latency_estimate_ms == pytest.approx(10)
    assert first.latency_decision.remote_latency_estimate_ms is None
    assert second.selected_worker_type == "remote"
    assert second.latency_decision is not None
    assert second.latency_decision.reason == "cold_start_remote"
    assert scheduler.edge_sample_count == 1
    assert scheduler.remote_sample_count == 1


def test_worker_identity_may_be_reported_only_in_result() -> None:
    scheduler = LatencyAwareScheduler(
        IdentityFromResultWorker("edge"),
        IdentityFromResultWorker("remote"),
        clock=SequenceClock(0, 0.010, 1, 1.020),
    )

    edge_result = scheduler.infer(frame())
    remote_result = scheduler.infer(frame())

    assert edge_result.selected_worker_id == "edge-reported"
    assert remote_result.selected_worker_id == "remote-reported"


@pytest.mark.parametrize(
    ("edge_ms", "remote_ms", "expected_worker", "expected_reason"),
    [
        (10, 20, "edge", "edge_lower_estimated_latency"),
        (20, 10, "remote", "remote_lower_estimated_latency"),
    ],
)
def test_selection_compares_estimates(
    edge_ms: float, remote_ms: float, expected_worker: str, expected_reason: str
) -> None:
    scheduler = LatencyAwareScheduler(
        FakeWorker("edge-test", "edge"),
        FakeWorker("remote-test", "remote"),
        clock=SequenceClock(
            0,
            edge_ms / 1000,
            1,
            1 + remote_ms / 1000,
            2,
            2.015,
        ),
    )
    scheduler.infer(frame())
    scheduler.infer(frame())

    result = scheduler.infer(frame())

    assert result.selected_worker_type == expected_worker
    assert result.latency_decision is not None
    assert result.latency_decision.reason == expected_reason
    assert result.latency_decision.cold_start is False


def test_equal_estimates_select_edge() -> None:
    scheduler = LatencyAwareScheduler(
        FakeWorker("edge-test", "edge"),
        FakeWorker("remote-test", "remote"),
        clock=SequenceClock(0, 1, 10, 11, 20, 21),
    )
    scheduler.infer(frame())
    scheduler.infer(frame())

    result = scheduler.infer(frame())

    assert result.selected_worker_type == "edge"
    assert result.latency_decision is not None
    assert result.latency_decision.reason == "latency_tie_edge"


def test_new_observation_can_change_next_selection() -> None:
    scheduler = LatencyAwareScheduler(
        FakeWorker("edge-test", "edge"),
        FakeWorker("remote-test", "remote"),
        alpha=1.0,
        clock=SequenceClock(0, 0.010, 1, 1.020, 2, 2.030, 3, 3.015),
    )

    selected = [scheduler.infer(frame()).selected_worker_type for _ in range(4)]

    assert selected == ["edge", "remote", "edge", "remote"]
    assert scheduler.edge_latency_estimate_ms == pytest.approx(30)
    assert scheduler.remote_latency_estimate_ms == pytest.approx(15)


def test_edge_failure_falls_back_to_remote_without_failed_observation() -> None:
    edge = FakeWorker("edge-test", "edge", WorkerExecutionError("failed"))
    remote = FakeWorker("remote-test", "remote")
    scheduler = LatencyAwareScheduler(edge, remote, clock=SequenceClock(0, 0.012, 1, 1.020))

    result = scheduler.infer(frame())

    assert edge.calls == 1
    assert remote.calls == 1
    assert scheduler.edge_sample_count == 0
    assert scheduler.edge_latency_estimate_ms is None
    assert scheduler.remote_latency_estimate_ms == pytest.approx(20)
    assert result.selected_worker_type == "edge"
    assert result.executed_worker_type == "remote"
    assert result.fallback is True
    assert result.fallback_reason == "WorkerExecutionError: failed"


def test_remote_failure_falls_back_to_edge() -> None:
    edge = FakeWorker("edge-test", "edge")
    remote = FakeWorker("remote-test", "remote")
    scheduler = LatencyAwareScheduler(
        edge,
        remote,
        clock=SequenceClock(0, 0.010, 1, 1.020, 2, 2.025),
    )
    scheduler.infer(frame())
    remote.failure = WorkerTimeoutError("remote unavailable")

    result = scheduler.infer(frame())

    assert result.selected_worker_type == "remote"
    assert result.executed_worker_type == "edge"
    assert result.fallback is True
    assert scheduler.edge_latency_estimate_ms == pytest.approx(14.5)
    assert scheduler.remote_latency_estimate_ms is None


def test_both_worker_failures_raise_clear_worker_error() -> None:
    scheduler = LatencyAwareScheduler(
        FakeWorker("edge-test", "edge", WorkerExecutionError("edge failed")),
        FakeWorker("remote-test", "remote", WorkerTimeoutError("remote failed")),
        clock=SequenceClock(0, 0.010, 1, 1.020),
    )

    with pytest.raises(WorkerUnavailableError, match="edge primary.*remote fallback"):
        scheduler.infer(frame())

    assert scheduler.metrics.failed_requests == 1


def test_repeated_slow_edge_observations_route_to_remote() -> None:
    scheduler = LatencyAwareScheduler(
        FakeWorker("edge-test", "edge"),
        FakeWorker("remote-test", "remote"),
        clock=SequenceClock(0, 0.010, 1, 1.100, 2, 2.200, 3, 3.200, 4, 4.100),
    )

    selected = [scheduler.infer(frame()).selected_worker_type for _ in range(5)]

    assert selected == ["edge", "remote", "edge", "edge", "remote"]


def test_repeated_slow_remote_observations_route_to_edge() -> None:
    scheduler = LatencyAwareScheduler(
        FakeWorker("edge-test", "edge"),
        FakeWorker("remote-test", "remote"),
        clock=SequenceClock(0, 0.100, 1, 1.010, 2, 2.200, 3, 3.200, 4, 4.100),
    )

    selected = [scheduler.infer(frame()).selected_worker_type for _ in range(5)]

    assert selected == ["edge", "remote", "remote", "remote", "edge"]


class FakeBackend:
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        return image

    def infer(self, model_input: np.ndarray) -> np.ndarray:
        return model_input

    def postprocess(
        self, raw_output: np.ndarray, image_shape: tuple[int, ...], confidence: float
    ) -> tuple[()]:
        return ()


def test_latency_aware_integrates_with_edge_worker_adapters() -> None:
    edge = EdgeWorker(InferenceEngine(FakeBackend()), worker_id="edge-test")
    remote = EdgeWorker(
        InferenceEngine(FakeBackend()), worker_id="remote-test", worker_type="remote"
    )
    scheduler = LatencyAwareScheduler(
        edge,
        remote,
        clock=SequenceClock(0, 0.008, 1, 1.018, 2, 2.009),
    )

    results = [scheduler.infer(frame()) for _ in range(3)]

    assert [result.selected_worker_type for result in results] == ["edge", "remote", "edge"]
    assert scheduler.edge_sample_count == 2
    assert scheduler.remote_sample_count == 1
    assert scheduler.summary()["edge_latency_estimate_ms"] == pytest.approx(8.3)
