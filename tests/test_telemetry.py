from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from edgecloud.telemetry import (
    GpuMetrics,
    NvidiaCollector,
    SystemCollector,
    SystemMetrics,
    TelemetryCollector,
    TelemetrySnapshot,
)


class FakePsutil:
    def __init__(self, cpu_percent: float = 25.5) -> None:
        self.cpu_value = cpu_percent
        self.interval = "unset"

    def cpu_percent(self, interval=None):
        self.interval = interval
        return self.cpu_value

    def virtual_memory(self):
        return SimpleNamespace(used=4_000, total=10_000, percent=40.0)


class FakeNvmlError(Exception):
    pass


class FakeNvml:
    NVMLError = FakeNvmlError
    NVML_TEMPERATURE_GPU = 0

    def __init__(self, count: int = 1, *, init_failure: bool = False) -> None:
        self.count = count
        self.init_failure = init_failure
        self.fail_metrics: set[str] = set()
        self.initialized = 0
        self.shutdown = 0

    def nvmlInit(self) -> None:
        if self.init_failure:
            raise FakeNvmlError("driver unavailable")
        self.initialized += 1

    def nvmlShutdown(self) -> None:
        self.shutdown += 1

    def nvmlDeviceGetCount(self) -> int:
        return self.count

    def nvmlDeviceGetHandleByIndex(self, index: int) -> int:
        return index

    def _value(self, metric: str, value):
        if metric in self.fail_metrics:
            raise FakeNvmlError("not supported")
        return value

    def nvmlDeviceGetName(self, handle: int) -> bytes:
        return self._value("name", f"GPU {handle}".encode())

    def nvmlDeviceGetUtilizationRates(self, handle: int):
        return self._value("utilization", SimpleNamespace(gpu=50 + handle, memory=10))

    def nvmlDeviceGetMemoryInfo(self, handle: int):
        return self._value("memory", SimpleNamespace(used=2_000, total=8_000))

    def nvmlDeviceGetTemperature(self, handle: int, sensor: int) -> int:
        return self._value("temperature", 60 + handle)

    def nvmlDeviceGetPowerUsage(self, handle: int) -> int:
        return self._value("power", 42_500)


def system_metrics() -> SystemMetrics:
    return SystemMetrics(25.5, 4_000, 10_000, 40.0)


def gpu_metrics() -> GpuMetrics:
    return GpuMetrics(0, "GPU 0", 50, 2_000, 8_000, 25, 60, 42.5)


def test_system_collector_uses_nonblocking_cpu_and_memory_bytes() -> None:
    fake = FakePsutil()

    result = SystemCollector(fake).collect()

    assert fake.interval is None
    assert result.cpu_utilization_percent == 25.5
    assert result.memory_used_bytes == 4_000
    assert result.memory_total_bytes == 10_000
    assert result.memory_utilization_percent == 40.0


@pytest.mark.parametrize("value", [-1, 101])
def test_system_metrics_reject_invalid_cpu_percentage(value: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 100"):
        SystemMetrics(value, 1, 2, 50)


def test_snapshot_serialization_round_trip() -> None:
    snapshot = TelemetrySnapshot(
        datetime(2026, 1, 2, 3, 4, tzinfo=UTC), system_metrics(), (gpu_metrics(),), True
    )

    serialized = snapshot.to_dict()
    restored = TelemetrySnapshot.from_dict(serialized)

    assert serialized["timestamp"] == "2026-01-02T03:04:00+00:00"
    assert restored == snapshot


def test_nvidia_collector_supports_no_and_multiple_gpus() -> None:
    no_gpus = NvidiaCollector(FakeNvml(count=0))
    multiple = NvidiaCollector(FakeNvml(count=2))

    assert no_gpus.available is True
    assert no_gpus.collect() == ()
    results = multiple.collect()
    assert [gpu.name for gpu in results] == ["GPU 0", "GPU 1"]
    assert results[0].memory_utilization_percent == 25


@pytest.mark.parametrize(
    ("metric", "field"),
    [("power", "power_draw_watts"), ("temperature", "temperature_celsius")],
)
def test_nvidia_collector_marks_unsupported_optional_metric_unavailable(
    metric: str, field: str
) -> None:
    nvml = FakeNvml()
    nvml.fail_metrics.add(metric)

    result = NvidiaCollector(nvml).collect()[0]

    assert getattr(result, field) is None
    assert result.utilization_percent == 50


def test_nvidia_initialization_failure_degrades_gracefully() -> None:
    collector = NvidiaCollector(FakeNvml(init_failure=True))

    assert collector.available is False
    assert collector.collect() == ()
    assert collector.error is not None


def test_nvidia_individual_metric_failure_does_not_drop_device() -> None:
    nvml = FakeNvml()
    nvml.fail_metrics.add("utilization")

    result = NvidiaCollector(nvml).collect()[0]

    assert result.name == "GPU 0"
    assert result.utilization_percent is None
    assert result.memory_used_bytes == 2_000


class FakeSystemSource:
    def collect(self) -> SystemMetrics:
        return system_metrics()


class FakeGpuSource:
    def __init__(self, available: bool, gpus: tuple[GpuMetrics, ...] = ()) -> None:
        self.available = available
        self.error = None if available else "unavailable"
        self.gpus = gpus
        self.closed = False

    def collect(self) -> tuple[GpuMetrics, ...]:
        return self.gpus

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize(
    ("gpu_source", "expected_count", "available"),
    [(FakeGpuSource(False), 0, False), (FakeGpuSource(True, (gpu_metrics(),)), 1, True)],
)
def test_telemetry_collector_combines_system_and_optional_gpu(
    gpu_source: FakeGpuSource, expected_count: int, available: bool
) -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    collector = TelemetryCollector(FakeSystemSource(), gpu_source, clock=lambda: now)

    snapshot = collector.collect()

    assert snapshot.system == system_metrics()
    assert len(snapshot.gpus) == expected_count
    assert snapshot.gpu_telemetry_available is available
    assert snapshot.timestamp == now
