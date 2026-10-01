from __future__ import annotations

import json
from pathlib import Path

import pytest

from edgecloud.demo import SCENARIOS, Scenario, ScenarioPhase, get_demo_profile
from edgecloud.experiments.events import (
    ExperimentEvent,
    inference_event,
    profile_change_event,
    run_end_event,
    run_start_event,
)
from edgecloud.experiments.recording import JsonlRecorder
from edgecloud.experiments.replay import export_replay, load_and_validate_jsonl, validate_events
from edgecloud.inference.models import TimingMetrics
from edgecloud.scheduler.models import LatencyDecision, ScheduledResult, SchedulerStrategy
from edgecloud.scheduler_cli import _validate_control_options, build_parser
from edgecloud.workers.models import WorkerResult

FIXTURE = Path(__file__).parent / "fixtures/replay/sample-synthetic.jsonl"


def result(*, fallback: bool = False) -> ScheduledResult:
    worker_type = "remote" if fallback else "edge"
    return ScheduledResult(
        strategy=SchedulerStrategy.LATENCY_AWARE,
        selected_worker_id="edge-1",
        selected_worker_type="edge",
        executed_worker_id=f"{worker_type}-1",
        executed_worker_type=worker_type,
        fallback=fallback,
        fallback_reason="WorkerExecutionError: failed" if fallback else None,
        scheduler_latency_ms=12,
        worker_result=WorkerResult(
            (), TimingMetrics(1, 2, 3, 6), f"{worker_type}-1", worker_type, "fake"
        ),
        latency_decision=LatencyDecision(12, 20, False, "edge_lower_estimated_latency"),
    )


def complete_events() -> list[dict[str, object]]:
    profile = get_demo_profile("normal")
    return [
        run_start_event(
            run_id="run-1",
            scheduler="latency_aware",
            latency_alpha=0.3,
            source_type="video",
            source_name="input.mp4",
            scenario="adaptive-failover",
            initial_profile="normal",
        ).to_dict(),
        inference_event(
            run_id="run-1",
            sequence=1,
            elapsed_ms=12,
            frame_index=0,
            profile_name="normal",
            profile=profile,
            scheduler="latency_aware",
            result=result(),
        ).to_dict(),
        run_end_event(
            run_id="run-1",
            sequence=2,
            elapsed_ms=12,
            total_frames=1,
            successful_frames=1,
            failed_frames=0,
            edge_executions=1,
            remote_executions=0,
            fallback_count=0,
            worker_switches=0,
            mean_observed_latency_ms=12,
        ).to_dict(),
    ]


def test_event_serializes_to_compact_valid_json() -> None:
    event = ExperimentEvent(
        "profile_change",
        "run-1",
        1,
        2.5,
        {"frame_index": 2, "from_profile": "normal", "to_profile": "edge_hot"},
    )
    assert json.loads(event.to_json()) == event.to_dict()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_event_rejects_non_finite_numbers(value: float) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        ExperimentEvent("profile_change", "run-1", 1, 0, {"value": value}).to_json()


def test_scenario_selection_and_exact_transitions() -> None:
    scenario = Scenario("test", (ScenarioPhase(0, "normal"), ScenarioPhase(2, "cloud_down")))
    assert [scenario.profile_at(index) for index in range(4)] == [
        "normal",
        "normal",
        "cloud_down",
        "cloud_down",
    ]


def test_canonical_scenario_file_matches_runtime_definition() -> None:
    path = Path(__file__).parent.parent / "demo-scenarios/adaptive-failover.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    scenario = SCENARIOS["adaptive-failover"]
    assert value["name"] == scenario.name
    assert value["phases"] == [
        {"start_frame": phase.start_frame, "profile": phase.profile} for phase in scenario.phases
    ]


def test_scenario_requires_frame_zero() -> None:
    with pytest.raises(ValueError, match="frame 0"):
        Scenario("bad", (ScenarioPhase(1, "normal"),))


def test_scenario_rejects_duplicate_or_decreasing_starts() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        Scenario("bad", (ScenarioPhase(0, "normal"), ScenarioPhase(0, "recovered")))


def test_scenario_rejects_unknown_profile() -> None:
    with pytest.raises(ValueError, match="unknown demo profile"):
        ScenarioPhase(0, "missing")


def test_static_profile_and_scenario_conflict() -> None:
    with pytest.raises(ValueError, match="cannot be used together"):
        _validate_control_options("normal", "adaptive-failover")


def test_recording_is_optional_in_cli() -> None:
    args = build_parser().parse_args(["--image", "frame.jpg", "--scheduler", "edge_only"])
    assert args.record is None
    assert args.scenario is None


def test_jsonl_recorder_emits_start_inference_and_end(tmp_path: Path) -> None:
    path = tmp_path / "nested/run.jsonl"
    events = complete_events()
    with JsonlRecorder(path) as recorder:
        for event in events:
            recorder.record(
                ExperimentEvent(
                    event["event_type"],
                    event["run_id"],
                    event["sequence"],
                    event["elapsed_ms"],
                    {
                        key: value
                        for key, value in event.items()
                        if key
                        not in {"schema_version", "event_type", "run_id", "sequence", "elapsed_ms"}
                    },
                )
            )
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert [item["event_type"] for item in lines] == ["run_start", "inference", "run_end"]
    assert [item["sequence"] for item in lines] == [0, 1, 2]


def test_replay_export_preserves_order(tmp_path: Path) -> None:
    output = tmp_path / "replay.json"
    payload = export_replay(FIXTURE, output)
    assert payload["run"]["event_type"] == "run_start"
    assert [event["sequence"] for event in payload["events"]] == list(range(1, 13))
    assert json.loads(output.read_text()) == payload


def test_validator_rejects_malformed_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("{bad}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        load_and_validate_jsonl(path)


def test_validator_rejects_mismatched_run_ids() -> None:
    events = complete_events()
    events[1]["run_id"] = "other"
    with pytest.raises(ValueError, match="run IDs"):
        validate_events(events)


def test_validator_rejects_unsupported_schema() -> None:
    events = complete_events()
    events[1]["schema_version"] = 2
    with pytest.raises(ValueError, match="unsupported schema"):
        validate_events(events)


def test_inference_keeps_injected_and_model_timing_separate() -> None:
    event = inference_event(
        run_id="run-1",
        sequence=1,
        elapsed_ms=162,
        frame_index=0,
        profile_name="edge_hot",
        profile=get_demo_profile("edge_hot"),
        scheduler="latency_aware",
        result=result(),
    ).to_dict()
    assert event["timing"]["model_total_ms"] == 6
    assert event["workers"]["edge"]["injected_delay_ms"] == 150


def test_fallback_preserves_selected_and_executed_workers() -> None:
    event = inference_event(
        run_id="run-1",
        sequence=1,
        elapsed_ms=12,
        frame_index=0,
        profile_name="normal",
        profile=get_demo_profile("normal"),
        scheduler="latency_aware",
        result=result(fallback=True),
    ).to_dict()
    assert event["decision"]["selected_worker"] == "edge"
    assert event["decision"]["executed_worker"] == "remote"
    assert event["decision"]["reason"] == "fallback_to_remote"
    assert event["fallback"]["used"] is True


def test_profile_change_has_explicit_profiles() -> None:
    event = profile_change_event(
        run_id="run-1",
        sequence=2,
        elapsed_ms=20,
        frame_index=20,
        from_profile="normal",
        to_profile="edge_hot",
    ).to_dict()
    assert event["from_profile"] == "normal"
    assert event["to_profile"] == "edge_hot"


def test_synthetic_fixture_validates() -> None:
    assert len(load_and_validate_jsonl(FIXTURE)) == 13
