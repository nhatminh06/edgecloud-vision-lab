from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from time import perf_counter, sleep

from edgecloud.inference.engine import Image
from edgecloud.workers.errors import WorkerConnectionError
from edgecloud.workers.models import InferenceWorker, WorkerResult


@dataclass(frozen=True, slots=True)
class AttemptRecord:
    worker_id: str
    worker_type: str
    elapsed_ms: float
    success: bool
    error_type: str | None = None


@dataclass(slots=True)
class RequestTrace:
    request_index: int = 0
    attempts: list[AttemptRecord] = field(default_factory=list)

    def begin(self, request_index: int) -> None:
        self.request_index = request_index
        self.attempts.clear()


@dataclass(slots=True)
class DelayWorker:
    worker: InferenceWorker
    delay_ms: float
    sleeper: Callable[[float], None] = sleep

    def __post_init__(self) -> None:
        if self.delay_ms < 0:
            raise ValueError("delay cannot be negative")

    def infer(self, image: Image) -> WorkerResult:
        self.sleeper(self.delay_ms / 1000)
        return self.worker.infer(image)


@dataclass(slots=True)
class FailureWindowWorker:
    worker: InferenceWorker
    trace: RequestTrace
    start_request: int
    end_request: int

    def infer(self, image: Image) -> WorkerResult:
        if self.start_request <= self.trace.request_index <= self.end_request:
            raise WorkerConnectionError("experiment-configured remote outage")
        return self.worker.infer(image)


@dataclass(slots=True)
class RecordingWorker:
    worker: InferenceWorker
    worker_id: str
    worker_type: str
    trace: RequestTrace
    clock: Callable[[], float] = perf_counter

    def infer(self, image: Image) -> WorkerResult:
        started = self.clock()
        try:
            result = self.worker.infer(image)
        except (OSError, RuntimeError, ValueError) as exc:
            self.trace.attempts.append(
                AttemptRecord(
                    self.worker_id,
                    self.worker_type,
                    (self.clock() - started) * 1000,
                    False,
                    type(exc).__name__,
                )
            )
            raise
        self.trace.attempts.append(
            AttemptRecord(
                self.worker_id,
                self.worker_type,
                (self.clock() - started) * 1000,
                True,
            )
        )
        return result
