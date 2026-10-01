from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from time import sleep
from typing import Any

from edgecloud.demo.profiles import DemoProfile
from edgecloud.inference.engine import Image
from edgecloud.workers.errors import WorkerUnavailableError
from edgecloud.workers.models import InferenceWorker, WorkerResult


@dataclass(slots=True)
class ControlledWorker:
    """Wrap a real worker with explicit, deterministic demonstration controls."""

    worker: InferenceWorker
    worker_type: str
    delay_ms: float = 0.0
    available: bool = True
    sleeper: Callable[[float], None] = sleep

    @property
    def worker_id(self) -> str:
        return self.worker.worker_id

    def __getattr__(self, name: str) -> Any:
        # Preserve optional worker capabilities such as remote health, telemetry, and close.
        return getattr(self.worker, name)

    def __post_init__(self) -> None:
        if self.delay_ms < 0:
            raise ValueError("injected delay cannot be negative")

    def infer(self, image: Image) -> WorkerResult:
        if not self.available:
            raise WorkerUnavailableError(f"{self.worker_type} disabled by demo profile")
        if self.delay_ms:
            self.sleeper(self.delay_ms / 1000)
        result = self.worker.infer(image)
        return replace(result, injected_delay_ms=result.injected_delay_ms + self.delay_ms)

    def reset(self) -> None:
        self.delay_ms = 0.0
        self.available = True

    def apply(self, profile: DemoProfile) -> None:
        if self.worker_type == "edge":
            self.delay_ms = profile.edge_delay_ms
            self.available = profile.edge_available
        else:
            self.delay_ms = profile.remote_delay_ms
            self.available = profile.remote_available
