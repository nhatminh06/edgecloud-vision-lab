from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from edgecloud.demo.profiles import DEMO_PROFILES
from edgecloud.experiments.events import EVENT_TYPES, SCHEMA_VERSION, WORKER_TYPES


def load_and_validate_jsonl(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"event on line {line_number} must be an object")
            events.append(value)
    validate_events(events)
    return events


def validate_events(events: list[dict[str, Any]]) -> None:
    if not events:
        raise ValueError("recording is empty")
    starts = [event for event in events if event.get("event_type") == "run_start"]
    ends = [event for event in events if event.get("event_type") == "run_end"]
    if len(starts) != 1 or len(ends) != 1:
        raise ValueError("recording must contain exactly one run_start and one run_end")
    if events[0].get("event_type") != "run_start" or events[-1].get("event_type") != "run_end":
        raise ValueError("run_start must be first and run_end must be last")

    run_id = starts[0].get("run_id")
    previous_sequence = -1
    previous_frame = -1
    for event in events:
        if event.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema version: {event.get('schema_version')}")
        if event.get("event_type") not in EVENT_TYPES:
            raise ValueError(f"unsupported event type: {event.get('event_type')}")
        if event.get("run_id") != run_id or not isinstance(run_id, str) or not run_id:
            raise ValueError("event run IDs must match")
        sequence = _non_negative_number(event, "sequence", integer=True)
        if sequence <= previous_sequence:
            raise ValueError("event sequences must be strictly increasing")
        previous_sequence = sequence
        _non_negative_number(event, "elapsed_ms")

        event_type = event["event_type"]
        if event_type == "run_start":
            if sequence != 0:
                raise ValueError("run_start sequence must be zero")
            _profile(event, "initial_profile")
            if not isinstance(event.get("scheduler"), str) or not event["scheduler"]:
                raise ValueError("run_start requires a scheduler")
            _non_negative_number(event, "latency_alpha")
            source = _object(event, "source")
            if any(
                not isinstance(source.get(key), str) or not source[key] for key in ("type", "name")
            ):
                raise ValueError("run_start source requires type and name")
        elif event_type == "inference":
            frame = _non_negative_number(event, "frame_index", integer=True)
            if frame < previous_frame:
                raise ValueError("inference frame indices cannot regress")
            previous_frame = frame
            _validate_inference(event)
        elif event_type == "profile_change":
            _non_negative_number(event, "frame_index", integer=True)
            _profile(event, "from_profile")
            _profile(event, "to_profile")
        elif event_type == "run_end":
            summary = event.get("summary")
            if not isinstance(summary, dict):
                raise ValueError("run_end requires a summary object")
            for key in (
                "total_frames",
                "successful_frames",
                "failed_frames",
                "edge_executions",
                "remote_executions",
                "fallback_count",
                "worker_switches",
            ):
                _non_negative_number(summary, key, integer=True)
            _optional_non_negative(summary, "mean_observed_latency_ms")
            _non_negative_number(summary, "scenario_duration_ms")


def export_replay(source: Path, destination: Path) -> dict[str, Any]:
    events = load_and_validate_jsonl(source)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run": events[0],
        "events": events[1:],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def _validate_inference(event: dict[str, Any]) -> None:
    _profile(event, "profile")
    decision = _object(event, "decision")
    for key in ("selected_worker", "executed_worker"):
        worker = decision.get(key)
        if worker is not None and worker not in WORKER_TYPES:
            raise ValueError(f"invalid worker name: {worker}")
    if not isinstance(decision.get("reason"), str | type(None)):
        raise ValueError("decision reason must be a string or null")
    _optional_non_negative(decision, "edge_estimate_ms")
    _optional_non_negative(decision, "remote_estimate_ms")
    timing = _object(event, "timing")
    for key in ("observed_request_ms", "model_total_ms", "model_inference_ms", "round_trip_ms"):
        _optional_non_negative(timing, key)
    fallback = _object(event, "fallback")
    if not isinstance(fallback.get("used"), bool):
        raise ValueError("fallback used must be boolean")
    workers = _object(event, "workers")
    for worker_name in WORKER_TYPES:
        worker = _object(workers, worker_name)
        if not isinstance(worker.get("available"), bool):
            raise ValueError("worker availability must be boolean")
        _non_negative_number(worker, "injected_delay_ms")
    detections = _object(event, "detections")
    _non_negative_number(detections, "count", integer=True)


def _object(value: dict[str, Any], key: str) -> dict[str, Any]:
    result = value.get(key)
    if not isinstance(result, dict):
        raise ValueError(f"{key} must be an object")
    return result


def _profile(value: dict[str, Any], key: str) -> str:
    profile = value.get(key)
    if profile not in DEMO_PROFILES:
        raise ValueError(f"invalid profile: {profile}")
    return profile


def _optional_non_negative(value: dict[str, Any], key: str) -> float | None:
    if value.get(key) is None:
        return None
    return float(_non_negative_number(value, key))


def _non_negative_number(value: dict[str, Any], key: str, *, integer: bool = False) -> int | float:
    number = value.get(key)
    valid_type = isinstance(number, int) if integer else isinstance(number, int | float)
    if isinstance(number, bool) or not valid_type or number < 0 or not math.isfinite(number):
        raise ValueError(f"{key} must be a non-negative {'integer' if integer else 'number'}")
    return number
