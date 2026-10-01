from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any


def _validate_percent(value: float | None, field_name: str) -> None:
    if value is not None and not 0.0 <= value <= 100.0:
        raise ValueError(f"{field_name} must be between 0 and 100")


@dataclass(frozen=True, slots=True)
class SystemMetrics:
    cpu_utilization_percent: float
    memory_used_bytes: int
    memory_total_bytes: int
    memory_utilization_percent: float

    def __post_init__(self) -> None:
        _validate_percent(self.cpu_utilization_percent, "cpu_utilization_percent")
        _validate_percent(self.memory_utilization_percent, "memory_utilization_percent")
        if self.memory_used_bytes < 0 or self.memory_total_bytes <= 0:
            raise ValueError("system memory values must be positive")
        if self.memory_used_bytes > self.memory_total_bytes:
            raise ValueError("used system memory cannot exceed total memory")


@dataclass(frozen=True, slots=True)
class GpuMetrics:
    index: int
    name: str
    utilization_percent: float | None
    memory_used_bytes: int | None
    memory_total_bytes: int | None
    memory_utilization_percent: float | None
    temperature_celsius: float | None
    power_draw_watts: float | None

    def __post_init__(self) -> None:
        if self.index < 0 or not self.name:
            raise ValueError("GPU index and name must be valid")
        _validate_percent(self.utilization_percent, "utilization_percent")
        _validate_percent(self.memory_utilization_percent, "memory_utilization_percent")
        if self.memory_used_bytes is not None and self.memory_used_bytes < 0:
            raise ValueError("GPU memory used cannot be negative")
        if self.memory_total_bytes is not None and self.memory_total_bytes <= 0:
            raise ValueError("GPU memory total must be positive")
        if (
            self.memory_used_bytes is not None
            and self.memory_total_bytes is not None
            and self.memory_used_bytes > self.memory_total_bytes
        ):
            raise ValueError("used GPU memory cannot exceed total memory")


@dataclass(frozen=True, slots=True)
class TelemetrySnapshot:
    timestamp: datetime
    system: SystemMetrics
    gpus: tuple[GpuMetrics, ...]
    gpu_telemetry_available: bool
    gpu_telemetry_error: str | None = None

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError("telemetry timestamp must include a timezone")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["timestamp"] = self.timestamp.isoformat()
        value["gpus"] = list(value["gpus"])
        return value

    @classmethod
    def from_dict(cls, value: Any) -> TelemetrySnapshot:
        if not isinstance(value, dict):
            raise ValueError("telemetry snapshot must be an object")
        try:
            raw_system = value["system"]
            raw_gpus = value["gpus"]
            if not isinstance(raw_system, dict) or not isinstance(raw_gpus, list):
                raise ValueError("invalid telemetry structure")
            available = value["gpu_telemetry_available"]
            error = value.get("gpu_telemetry_error")
            if not isinstance(available, bool) or error is not None and not isinstance(error, str):
                raise ValueError("invalid GPU telemetry status")
            return cls(
                timestamp=datetime.fromisoformat(_text(value["timestamp"])),
                system=SystemMetrics(
                    cpu_utilization_percent=_number(raw_system["cpu_utilization_percent"]),
                    memory_used_bytes=_integer(raw_system["memory_used_bytes"]),
                    memory_total_bytes=_integer(raw_system["memory_total_bytes"]),
                    memory_utilization_percent=_number(raw_system["memory_utilization_percent"]),
                ),
                gpus=tuple(_parse_gpu(item) for item in raw_gpus),
                gpu_telemetry_available=available,
                gpu_telemetry_error=error,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid telemetry snapshot structure") from exc


def _parse_gpu(value: Any) -> GpuMetrics:
    if not isinstance(value, dict):
        raise ValueError("GPU telemetry must be an object")
    return GpuMetrics(
        index=_integer(value["index"]),
        name=_text(value["name"]),
        utilization_percent=_optional_number(value["utilization_percent"]),
        memory_used_bytes=_optional_integer(value["memory_used_bytes"]),
        memory_total_bytes=_optional_integer(value["memory_total_bytes"]),
        memory_utilization_percent=_optional_number(value["memory_utilization_percent"]),
        temperature_celsius=_optional_number(value["temperature_celsius"]),
        power_draw_watts=_optional_number(value["power_draw_watts"]),
    )


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("expected a number")
    return float(value)


def _integer(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("expected an integer")
    return value


def _optional_number(value: Any) -> float | None:
    return None if value is None else _number(value)


def _optional_integer(value: Any) -> int | None:
    return None if value is None else _integer(value)


def _text(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("expected non-empty text")
    return value
