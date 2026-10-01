from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from edgecloud.experiments.scenarios import CpuPressure
from edgecloud.experiments.workers import DelayWorker, FailureWindowWorker, RequestTrace
from edgecloud.inference.models import TimingMetrics
from edgecloud.workers.errors import WorkerConnectionError
from edgecloud.workers.models import WorkerResult


@dataclass
class FakeWorker:
    calls: int = 0

    def infer(self, image: np.ndarray) -> WorkerResult:
        self.calls += 1
        return WorkerResult((), TimingMetrics(1, 2, 3, 6), "remote-1", "remote", "fake")


def frame() -> np.ndarray:
    return np.zeros((2, 2, 3), dtype=np.uint8)


def test_delay_worker_requests_total_delay_without_waiting_and_preserves_result() -> None:
    delays: list[float] = []
    worker = FakeWorker()

    result = DelayWorker(worker, 50, sleeper=delays.append).infer(frame())

    assert delays == [0.05]
    assert result.worker_id == "remote-1"
    assert worker.calls == 1


def test_failure_window_transitions_available_unavailable_recovered() -> None:
    trace = RequestTrace()
    worker = FakeWorker()
    scenario = FailureWindowWorker(worker, trace, 1, 1)

    trace.begin(0)
    assert scenario.infer(frame()).worker_type == "remote"
    trace.begin(1)
    with pytest.raises(WorkerConnectionError, match="configured remote outage"):
        scenario.infer(frame())
    trace.begin(2)
    assert scenario.infer(frame()).worker_type == "remote"

    assert worker.calls == 2


def test_cpu_pressure_starts_and_stops_bounded_threads() -> None:
    pressure = CpuPressure(1).start()
    assert len(pressure._threads) == 1
    pressure.close()
    assert pressure._threads == []
