from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter


@dataclass(slots=True)
class StreamMetrics:
    started_at: float
    frame_count: int = 0

    @classmethod
    def start(cls) -> StreamMetrics:
        return cls(started_at=perf_counter())

    def record_frame(self) -> None:
        self.frame_count += 1

    def fps(self, now: float | None = None) -> float:
        elapsed = (perf_counter() if now is None else now) - self.started_at
        return self.frame_count / elapsed if elapsed > 0 else 0.0
