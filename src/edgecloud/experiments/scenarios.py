from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from threading import Event, Thread


@dataclass(slots=True)
class CpuPressure(AbstractContextManager["CpuPressure"]):
    """Bounded experiment-only CPU pressure stopped through a shared event."""

    worker_count: int = 1
    _stop: Event = field(init=False, default_factory=Event, repr=False)
    _threads: list[Thread] = field(init=False, default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if self.worker_count <= 0:
            raise ValueError("CPU pressure worker count must be positive")

    def start(self) -> CpuPressure:
        if self._threads:
            return self
        self._stop.clear()
        self._threads = [
            Thread(target=self._load, name=f"experiment-cpu-pressure-{index}", daemon=True)
            for index in range(self.worker_count)
        ]
        for thread in self._threads:
            thread.start()
        return self

    def close(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join()
        self._threads.clear()

    def __enter__(self) -> CpuPressure:
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.close()

    def _load(self) -> None:
        value = 1
        while not self._stop.is_set():
            value = (value * 1_664_525 + 1_013_904_223) & 0xFFFFFFFF


class NoPressure(AbstractContextManager["NoPressure"]):
    def __enter__(self) -> NoPressure:
        return self

    def __exit__(self, *_: object) -> None:
        return None
