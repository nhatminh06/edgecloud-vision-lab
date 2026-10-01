from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from edgecloud.scheduler import SchedulerStrategy


@dataclass(frozen=True, slots=True)
class FailureWindow:
    start_request: int
    end_request: int

    def __post_init__(self) -> None:
        if self.start_request < 0 or self.end_request < self.start_request:
            raise ValueError("failure window must be a non-negative inclusive range")

    def contains(self, request_index: int) -> bool:
        return self.start_request <= request_index <= self.end_request


@dataclass(frozen=True, slots=True)
class CloudEnvironment:
    provider: str
    region: str
    instance_type: str
    operating_system: str
    architecture: str
    cpu_model: str
    vcpu_count: int
    total_memory_bytes: int

    def __post_init__(self) -> None:
        text_values = (
            self.provider,
            self.region,
            self.instance_type,
            self.operating_system,
            self.architecture,
            self.cpu_model,
        )
        if any(not value.strip() for value in text_values):
            raise ValueError("cloud environment text fields cannot be empty")
        if self.vcpu_count <= 0 or self.total_memory_bytes <= 0:
            raise ValueError("cloud vCPU count and memory must be positive")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    name: str
    scheduler: SchedulerStrategy
    input_path: Path
    measured_requests: int
    warmup_requests: int = 0
    repetitions: int = 1
    remote_url: str = "http://127.0.0.1:8000"
    latency_alpha: float = 0.3
    resource_threshold: float = 0.85
    telemetry_interval: float = 1.0
    health_interval: float = 1.0
    delay_ms: float = 0.0
    resource_scenario: str = "normal"
    cpu_pressure_workers: int = 1
    failure_window: FailureWindow | None = None
    output_dir: Path = Path("results")
    label: str = "development validation"
    cloud_environment: CloudEnvironment | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("experiment name cannot be empty")
        if self.measured_requests <= 0:
            raise ValueError("measured requests must be positive")
        if self.warmup_requests < 0 or self.repetitions <= 0:
            raise ValueError("warm-up must be non-negative and repetitions must be positive")
        if self.delay_ms < 0:
            raise ValueError("delay cannot be negative")
        if self.resource_scenario not in {"normal", "edge_cpu_pressure"}:
            raise ValueError("resource scenario must be normal or edge_cpu_pressure")
        if self.cpu_pressure_workers <= 0:
            raise ValueError("CPU pressure worker count must be positive")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["scheduler"] = self.scheduler.value
        value["input_path"] = str(self.input_path)
        value["output_dir"] = str(self.output_dir)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ExperimentConfig:
        failure = value.get("failure_window")
        cloud = value.get("cloud_environment")
        return cls(
            name=str(value["name"]),
            scheduler=SchedulerStrategy(value["scheduler"]),
            input_path=Path(value["input_path"]),
            measured_requests=int(value["measured_requests"]),
            warmup_requests=int(value.get("warmup_requests", 0)),
            repetitions=int(value.get("repetitions", 1)),
            remote_url=str(value.get("remote_url", "http://127.0.0.1:8000")),
            latency_alpha=float(value.get("latency_alpha", 0.3)),
            resource_threshold=float(value.get("resource_threshold", 0.85)),
            telemetry_interval=float(value.get("telemetry_interval", 1.0)),
            health_interval=float(value.get("health_interval", 1.0)),
            delay_ms=float(value.get("delay_ms", 0)),
            resource_scenario=str(value.get("resource_scenario", "normal")),
            cpu_pressure_workers=int(value.get("cpu_pressure_workers", 1)),
            failure_window=FailureWindow(**failure) if failure else None,
            output_dir=Path(value.get("output_dir", "results")),
            label=str(value.get("label", "development validation")),
            cloud_environment=CloudEnvironment(**cloud) if cloud else None,
        )


@dataclass(frozen=True, slots=True)
class RawObservation:
    experiment_id: str
    run_id: int
    scenario: str
    scheduler: str
    request_index: int
    warmup: bool
    relative_time_seconds: float
    initial_worker: str | None
    final_worker: str | None
    decision_reason: str | None
    success: bool
    error_type: str | None
    fallback_occurred: bool
    primary_failure_type: str | None
    worker_attempts: int
    total_latency_ms: float
    primary_attempt_ms: float | None
    fallback_attempt_ms: float | None
    edge_latency_estimate_ms: float | None
    remote_latency_estimate_ms: float | None
    edge_resource_pressure: float | None
    remote_resource_pressure: float | None
    edge_health: str | None
    remote_health: str | None
    worker_inference_ms: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
