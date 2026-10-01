from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol

from edgecloud.telemetry.models import GpuMetrics, SystemMetrics, TelemetrySnapshot
from edgecloud.telemetry.nvidia import NvidiaCollector
from edgecloud.telemetry.system import SystemCollector


class SystemMetricsSource(Protocol):
    def collect(self) -> SystemMetrics: ...


class GpuMetricsSource(Protocol):
    available: bool
    error: str | None

    def collect(self) -> tuple[GpuMetrics, ...]: ...

    def close(self) -> None: ...


class TelemetryCollector:
    def __init__(
        self,
        system: SystemMetricsSource | None = None,
        gpu: GpuMetricsSource | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._system = system or SystemCollector()
        self._gpu = gpu or NvidiaCollector()
        self._clock = clock or (lambda: datetime.now(UTC))

    def collect(self) -> TelemetrySnapshot:
        system = self._system.collect()
        gpus = self._gpu.collect()
        return TelemetrySnapshot(
            timestamp=self._clock(),
            system=system,
            gpus=gpus,
            gpu_telemetry_available=self._gpu.available,
            gpu_telemetry_error=self._gpu.error,
        )

    def close(self) -> None:
        self._gpu.close()

    def __enter__(self) -> TelemetryCollector:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
