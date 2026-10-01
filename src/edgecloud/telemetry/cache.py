from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from threading import Event, Lock, Thread
from time import monotonic

from edgecloud.telemetry.models import TelemetrySnapshot


class TelemetryStatus(StrEnum):
    AVAILABLE = "available"
    STALE = "stale"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class CachedTelemetry:
    snapshot: TelemetrySnapshot | None
    status: TelemetryStatus
    age_seconds: float | None
    last_error: str | None


class SnapshotCache:
    def __init__(
        self,
        source: Callable[[], TelemetrySnapshot],
        *,
        max_age_seconds: float,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if max_age_seconds <= 0:
            raise ValueError("telemetry max age must be positive")
        self._source = source
        self._max_age_seconds = max_age_seconds
        self._clock = clock
        self._lock = Lock()
        self._snapshot: TelemetrySnapshot | None = None
        self._received_at: float | None = None
        self._last_error: str | None = None

    def refresh(self) -> None:
        try:
            snapshot = self._source()
        except (OSError, RuntimeError, ValueError) as exc:
            with self._lock:
                self._last_error = str(exc)
            return
        received_at = self._clock()
        with self._lock:
            self._snapshot = snapshot
            self._received_at = received_at
            self._last_error = None

    def get(self) -> CachedTelemetry:
        now = self._clock()
        with self._lock:
            snapshot = self._snapshot
            received_at = self._received_at
            error = self._last_error
        if snapshot is None or received_at is None:
            return CachedTelemetry(None, TelemetryStatus.UNAVAILABLE, None, error)
        age = max(0.0, now - received_at)
        status = (
            TelemetryStatus.AVAILABLE if age <= self._max_age_seconds else TelemetryStatus.STALE
        )
        return CachedTelemetry(snapshot, status, age, error)


class TelemetryState:
    def __init__(
        self,
        edge_source: Callable[[], TelemetrySnapshot],
        remote_source: Callable[[], TelemetrySnapshot],
        *,
        interval_seconds: float = 1.0,
        max_age_seconds: float = 3.0,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("telemetry interval must be positive")
        self.interval_seconds = interval_seconds
        self._edge = SnapshotCache(edge_source, max_age_seconds=max_age_seconds, clock=clock)
        self._remote = SnapshotCache(remote_source, max_age_seconds=max_age_seconds, clock=clock)
        self._stop = Event()
        self._threads: list[Thread] = []

    def edge(self) -> CachedTelemetry:
        return self._edge.get()

    def remote(self) -> CachedTelemetry:
        return self._remote.get()

    def refresh_edge(self) -> None:
        self._edge.refresh()

    def refresh_remote(self) -> None:
        self._remote.refresh()

    def start(self) -> TelemetryState:
        if self._threads:
            return self
        self._stop.clear()
        for name, refresh in (
            ("edge", self.refresh_edge),
            ("remote", self.refresh_remote),
        ):
            thread = Thread(
                target=self._sample_loop,
                args=(refresh,),
                name=f"telemetry-{name}",
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)
        return self

    def _sample_loop(self, refresh: Callable[[], None]) -> None:
        while not self._stop.is_set():
            refresh()
            if self._stop.wait(self.interval_seconds):
                break

    def close(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join()
        self._threads.clear()

    @property
    def running(self) -> bool:
        return any(thread.is_alive() for thread in self._threads)

    def __enter__(self) -> TelemetryState:
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.close()
