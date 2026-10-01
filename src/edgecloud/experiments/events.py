from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from edgecloud.demo.profiles import DemoProfile
from edgecloud.scheduler.models import ScheduledResult

SCHEMA_VERSION = 1
EVENT_TYPES = frozenset({"run_start", "inference", "profile_change", "run_end"})
WORKER_TYPES = frozenset({"edge", "remote"})


@dataclass(frozen=True, slots=True)
class ExperimentEvent:
    event_type: str
    run_id: str
    sequence: int
    elapsed_ms: float
    fields: dict[str, Any]
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.event_type not in EVENT_TYPES:
            raise ValueError(f"unsupported event type: {self.event_type}")
        if not self.run_id:
            raise ValueError("run ID cannot be empty")
        if self.sequence < 0 or self.elapsed_ms < 0:
            raise ValueError("event sequence and elapsed time cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        value = {
            "schema_version": self.schema_version,
            "event_type": self.event_type,
            "run_id": self.run_id,
            "sequence": self.sequence,
            "elapsed_ms": self.elapsed_ms,
            **self.fields,
        }
        _validate_json_value(value)
        return value

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"), allow_nan=False)


def run_start_event(
    *,
    run_id: str,
    scheduler: str,
    latency_alpha: float,
    source_type: str,
    source_name: str,
    scenario: str | None,
    initial_profile: str,
) -> ExperimentEvent:
    return ExperimentEvent(
        "run_start",
        run_id,
        0,
        0.0,
        {
            "scheduler": scheduler,
            "latency_alpha": latency_alpha,
            "source": {"type": source_type, "name": source_name},
            "scenario": scenario,
            "initial_profile": initial_profile,
            "model": "ssdlite320_mobilenet_v3_large",
            "notes": {"observed_timing_may_contain_injected_delay": True},
        },
    )


def inference_event(
    *,
    run_id: str,
    sequence: int,
    elapsed_ms: float,
    frame_index: int,
    profile_name: str,
    profile: DemoProfile,
    scheduler: str,
    result: ScheduledResult | None,
    error: Exception | None = None,
) -> ExperimentEvent:
    latency = result.latency_decision if result is not None else None
    resource = result.resource_decision if result is not None else None
    resilient = result.resilient_decision if result is not None else None
    reason = (
        latency.reason
        if latency
        else resource.reason
        if resource
        else resilient.reason
        if resilient
        else None
    )
    if reason == "latency_tie_edge":
        reason = "tie_prefer_edge"
    if result is not None and result.fallback:
        reason = f"fallback_to_{result.executed_worker_type}"
    edge_estimate = (
        latency.edge_latency_estimate_ms
        if latency
        else resource.edge_latency_estimate_ms
        if resource
        else None
    )
    remote_estimate = (
        latency.remote_latency_estimate_ms
        if latency
        else resource.remote_latency_estimate_ms
        if resource
        else None
    )
    selected = result.selected_worker_type if result is not None else None
    executed = result.executed_worker_type if result is not None else None
    worker_result = result.worker_result if result is not None else None
    return ExperimentEvent(
        "inference",
        run_id,
        sequence,
        elapsed_ms,
        {
            "frame_index": frame_index,
            "profile": profile_name,
            "scheduler": scheduler,
            "decision": {
                "selected_worker": selected,
                "executed_worker": executed,
                "reason": reason,
                "edge_estimate_ms": edge_estimate,
                "remote_estimate_ms": remote_estimate,
            },
            "timing": {
                "observed_request_ms": result.scheduler_latency_ms if result else None,
                "model_total_ms": worker_result.timing.total_ms if worker_result else None,
                "model_inference_ms": worker_result.timing.inference_ms if worker_result else None,
                "round_trip_ms": worker_result.round_trip_ms if worker_result else None,
            },
            "fallback": {
                "used": result.fallback if result else False,
                "reason": result.fallback_reason if result else None,
            },
            "workers": {
                "edge": {
                    "available": profile.edge_available,
                    "injected_delay_ms": profile.edge_delay_ms,
                },
                "remote": {
                    "available": profile.remote_available,
                    "injected_delay_ms": profile.remote_delay_ms,
                },
            },
            "detections": {"count": worker_result.detection_count if worker_result else 0},
            "error": ({"type": type(error).__name__, "message": str(error)} if error else None),
        },
    )


def profile_change_event(
    *,
    run_id: str,
    sequence: int,
    elapsed_ms: float,
    frame_index: int,
    from_profile: str,
    to_profile: str,
) -> ExperimentEvent:
    return ExperimentEvent(
        "profile_change",
        run_id,
        sequence,
        elapsed_ms,
        {
            "frame_index": frame_index,
            "from_profile": from_profile,
            "to_profile": to_profile,
        },
    )


def run_end_event(
    *,
    run_id: str,
    sequence: int,
    elapsed_ms: float,
    total_frames: int,
    successful_frames: int,
    failed_frames: int,
    edge_executions: int,
    remote_executions: int,
    fallback_count: int,
    worker_switches: int,
    mean_observed_latency_ms: float | None,
) -> ExperimentEvent:
    return ExperimentEvent(
        "run_end",
        run_id,
        sequence,
        elapsed_ms,
        {
            "summary": {
                "total_frames": total_frames,
                "successful_frames": successful_frames,
                "failed_frames": failed_frames,
                "edge_executions": edge_executions,
                "remote_executions": remote_executions,
                "fallback_count": fallback_count,
                "worker_switches": worker_switches,
                "mean_observed_latency_ms": mean_observed_latency_ms,
                "scenario_duration_ms": elapsed_ms,
            }
        },
    )


def _validate_json_value(value: Any) -> None:
    if value is None or isinstance(value, str | bool | int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("event contains a non-finite number")
        return
    if isinstance(value, list | tuple):
        for item in value:
            _validate_json_value(item)
        return
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("event object keys must be strings")
        for item in value.values():
            _validate_json_value(item)
        return
    raise ValueError(f"event contains a non-JSON value: {type(value).__name__}")
