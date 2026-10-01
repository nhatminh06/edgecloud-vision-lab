from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from threading import Event, Lock, Thread
from time import monotonic


class WorkerHealth(StrEnum):
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"


@dataclass(frozen=True, slots=True)
class HealthSample:
    state: WorkerHealth
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class CachedHealth:
    state: WorkerHealth
    fresh: bool
    age_seconds: float | None
    detail: str | None = None

    @property
    def blocks_routing(self) -> bool:
        return self.fresh and self.state is WorkerHealth.UNHEALTHY


@dataclass(slots=True)
class HealthCache:
    max_age_seconds: float
    clock: Callable[[], float] = monotonic
    _sample: HealthSample | None = field(init=False, default=None, repr=False)
    _received_at: float | None = field(init=False, default=None, repr=False)
    _lock: Lock = field(init=False, default_factory=Lock, repr=False)

    def __post_init__(self) -> None:
        if self.max_age_seconds <= 0:
            raise ValueError("health maximum age must be positive")

    def update(self, state: WorkerHealth, detail: str | None = None) -> None:
        received_at = self.clock()
        with self._lock:
            self._sample = HealthSample(state, detail)
            self._received_at = received_at

    def mark_healthy(self, detail: str | None = None) -> None:
        self.update(WorkerHealth.HEALTHY, detail)

    def mark_unhealthy(self, detail: str | None = None) -> None:
        self.update(WorkerHealth.UNHEALTHY, detail)

    def get(self) -> CachedHealth:
        with self._lock:
            sample = self._sample
            received_at = self._received_at
        if sample is None or received_at is None:
            return CachedHealth(WorkerHealth.UNKNOWN, False, None)
        age = max(0.0, self.clock() - received_at)
        return CachedHealth(sample.state, age <= self.max_age_seconds, age, sample.detail)


class HealthState:
    def __init__(
        self,
        edge_source: Callable[[], HealthSample],
        remote_source: Callable[[], HealthSample],
        *,
        interval_seconds: float = 1.0,
        max_age_seconds: float = 3.0,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("health interval must be positive")
        self._edge_source = edge_source
        self._remote_source = remote_source
        self._interval_seconds = interval_seconds
        self._edge_cache = HealthCache(max_age_seconds, clock)
        self._remote_cache = HealthCache(max_age_seconds, clock)
        self._stop = Event()
        self._threads: list[Thread] = []
        self._lifecycle_lock = Lock()

    def edge(self) -> CachedHealth:
        return self._edge_cache.get()

    def remote(self) -> CachedHealth:
        return self._remote_cache.get()

    def mark_healthy(self, worker_type: str, detail: str | None = None) -> None:
        self._cache(worker_type).mark_healthy(detail)

    def mark_unhealthy(self, worker_type: str, detail: str | None = None) -> None:
        self._cache(worker_type).mark_unhealthy(detail)

    def refresh_edge(self) -> None:
        self._refresh(self._edge_source, self._edge_cache)

    def refresh_remote(self) -> None:
        self._refresh(self._remote_source, self._remote_cache)

    def start(self) -> HealthState:
        with self._lifecycle_lock:
            if self._threads:
                return self
            self._stop.clear()
            self._threads = [
                Thread(
                    target=self._sample_loop,
                    args=(self._edge_source, self._edge_cache),
                    name="edge-health-sampler",
                    daemon=True,
                ),
                Thread(
                    target=self._sample_loop,
                    args=(self._remote_source, self._remote_cache),
                    name="remote-health-sampler",
                    daemon=True,
                ),
            ]
            for thread in self._threads:
                thread.start()
        return self

    def close(self) -> None:
        with self._lifecycle_lock:
            threads = self._threads
            self._threads = []
            self._stop.set()
        for thread in threads:
            thread.join()

    @property
    def running(self) -> bool:
        with self._lifecycle_lock:
            return bool(self._threads) and all(thread.is_alive() for thread in self._threads)

    def _cache(self, worker_type: str) -> HealthCache:
        if worker_type == "edge":
            return self._edge_cache
        if worker_type == "remote":
            return self._remote_cache
        raise ValueError(f"unknown worker type: {worker_type}")

    @staticmethod
    def _refresh(source: Callable[[], HealthSample], cache: HealthCache) -> None:
        try:
            sample = source()
        except (OSError, RuntimeError, ValueError) as exc:
            cache.mark_unhealthy(f"{type(exc).__name__}: {exc}")
            return
        cache.update(sample.state, sample.detail)

    def _sample_loop(self, source: Callable[[], HealthSample], cache: HealthCache) -> None:
        while not self._stop.is_set():
            self._refresh(source, cache)
            self._stop.wait(self._interval_seconds)
