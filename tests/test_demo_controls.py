from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from edgecloud.demo import DEMO_PROFILES, ControlledWorker
from edgecloud.inference.models import TimingMetrics
from edgecloud.scheduler import LatencyAwareScheduler
from edgecloud.workers.errors import WorkerUnavailableError
from edgecloud.workers.models import WorkerResult


@dataclass
class FakeWorker:
    worker_id: str
    worker_type: str

    def infer(self, image: np.ndarray) -> WorkerResult:
        return WorkerResult((), TimingMetrics(1, 2, 3, 6), self.worker_id, self.worker_type, "fake")


def frame() -> np.ndarray:
    return np.zeros((4, 4, 3), dtype=np.uint8)


def test_controlled_worker_exposes_injected_delay_separately() -> None:
    sleeps: list[float] = []
    worker = ControlledWorker(
        FakeWorker("edge-1", "edge"), "edge", delay_ms=150, sleeper=sleeps.append
    )

    result = worker.infer(frame())

    assert sleeps == [0.15]
    assert result.timing.total_ms == 6
    assert result.injected_delay_ms == 150


def test_controlled_worker_failure_and_reset_are_deterministic() -> None:
    worker = ControlledWorker(FakeWorker("remote-1", "remote"), "remote", available=False)

    with pytest.raises(WorkerUnavailableError, match="demo profile"):
        worker.infer(frame())
    worker.reset()

    assert worker.infer(frame()).worker_type == "remote"


def test_profiles_have_documented_controls() -> None:
    assert DEMO_PROFILES["normal"].edge_delay_ms == 0
    assert DEMO_PROFILES["edge_hot"].edge_delay_ms == 150
    assert DEMO_PROFILES["cloud_slow"].remote_delay_ms == 200
    assert DEMO_PROFILES["cloud_down"].remote_available is False
    assert DEMO_PROFILES["recovered"] == DEMO_PROFILES["normal"]


class SequenceClock:
    def __init__(self, *values: float) -> None:
        self.values = iter(values)

    def __call__(self) -> float:
        return next(self.values)


def test_serialized_adaptive_result_contains_replay_fields() -> None:
    edge = ControlledWorker(
        FakeWorker("edge-1", "edge"), "edge", delay_ms=150, sleeper=lambda _: None
    )
    remote = ControlledWorker(FakeWorker("remote-1", "remote"), "remote", sleeper=lambda _: None)
    scheduler = LatencyAwareScheduler(edge, remote, clock=SequenceClock(0, 0.160))

    payload = scheduler.infer(frame()).to_dict()

    assert payload["selected_worker_type"] == "edge"
    assert payload["executed_worker_type"] == "edge"
    assert payload["observed_latency_ms"] == pytest.approx(160)
    assert payload["fallback"] is False
    assert payload["detections"] == 0
    assert payload["edge_injected_delay_ms"] == 150
    assert payload["remote_injected_delay_ms"] == 0
    assert payload["latency_decision"]["edge_latency_estimate_ms"] == pytest.approx(160)
