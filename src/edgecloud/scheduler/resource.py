from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from edgecloud.telemetry.cache import CachedTelemetry, TelemetryStatus


class ExecutionResource(StrEnum):
    CPU = "cpu"
    GPU = "gpu"


@dataclass(frozen=True, slots=True)
class WorkerResource:
    resource_type: ExecutionResource
    gpu_index: int | None = None

    def __post_init__(self) -> None:
        if self.resource_type is ExecutionResource.GPU and self.gpu_index is None:
            raise ValueError("GPU execution requires a GPU index")
        if self.gpu_index is not None and self.gpu_index < 0:
            raise ValueError("GPU index cannot be negative")


@dataclass(frozen=True, slots=True)
class ResourcePreference:
    worker_type: str
    cold_start: bool
    reason: str


def choose_resource_preference(
    *,
    edge_sample_count: int,
    remote_sample_count: int,
    edge_latency_ms: float | None,
    remote_latency_ms: float | None,
    edge_telemetry_status: TelemetryStatus,
    remote_telemetry_status: TelemetryStatus,
    edge_pressure: float | None,
    remote_pressure: float | None,
    pressure_threshold: float,
) -> ResourcePreference:
    """Apply the deterministic resource-aware preference rules without dispatching."""
    if edge_sample_count == 0:
        return ResourcePreference("edge", True, "cold_start_edge")
    if remote_sample_count == 0:
        return ResourcePreference("remote", True, "cold_start_remote")
    if edge_latency_ms is None or remote_latency_ms is None:
        raise RuntimeError("latency estimates missing after warm-up")
    if (
        edge_telemetry_status is TelemetryStatus.STALE
        or remote_telemetry_status is TelemetryStatus.STALE
    ):
        return ResourcePreference(
            _lower_latency(edge_latency_ms, remote_latency_ms)[0],
            False,
            "telemetry_stale_latency_fallback",
        )
    if edge_pressure is None or remote_pressure is None:
        return ResourcePreference(
            _lower_latency(edge_latency_ms, remote_latency_ms)[0],
            False,
            "telemetry_unavailable_latency_fallback",
        )
    if edge_pressure >= pressure_threshold and remote_pressure < edge_pressure:
        return ResourcePreference("remote", False, "edge_high_resource_pressure")
    if remote_pressure >= pressure_threshold and edge_pressure < remote_pressure:
        return ResourcePreference("edge", False, "remote_high_resource_pressure")
    worker_type, reason = _lower_latency(edge_latency_ms, remote_latency_ms)
    return ResourcePreference(worker_type, False, reason)


def _lower_latency(edge_latency_ms: float, remote_latency_ms: float) -> tuple[str, str]:
    if edge_latency_ms < remote_latency_ms:
        return "edge", "edge_lower_latency"
    if remote_latency_ms < edge_latency_ms:
        return "remote", "remote_lower_latency"
    return "edge", "latency_tie_edge"


def resource_pressure(telemetry: CachedTelemetry, resource: WorkerResource) -> float | None:
    if telemetry.status is not TelemetryStatus.AVAILABLE or telemetry.snapshot is None:
        return None
    snapshot = telemetry.snapshot
    if resource.resource_type is ExecutionResource.CPU:
        return (
            max(
                snapshot.system.cpu_utilization_percent,
                snapshot.system.memory_utilization_percent,
            )
            / 100
        )
    gpu = next((item for item in snapshot.gpus if item.index == resource.gpu_index), None)
    if gpu is None or not snapshot.gpu_telemetry_available:
        return None
    values = [
        value
        for value in (gpu.utilization_percent, gpu.memory_utilization_percent)
        if value is not None
    ]
    return max(values) / 100 if values else None
