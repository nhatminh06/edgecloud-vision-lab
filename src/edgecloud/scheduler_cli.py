from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from time import perf_counter, time_ns

from edgecloud.capture import Frame, image_frames, video_frames
from edgecloud.demo import (
    DEMO_PROFILES,
    SCENARIOS,
    ControlledWorker,
    get_demo_profile,
    get_scenario,
)
from edgecloud.experiments.events import (
    inference_event,
    profile_change_event,
    run_end_event,
    run_start_event,
)
from edgecloud.experiments.recording import JsonlRecorder
from edgecloud.inference.engine import InferenceEngine
from edgecloud.inference.torchvision_backend import TorchvisionSSDLiteBackend
from edgecloud.scheduler import (
    HealthSample,
    HealthState,
    SchedulerStrategy,
    WorkerHealth,
    create_scheduler,
)
from edgecloud.telemetry import TelemetryCollector, TelemetryState
from edgecloud.workers.edge import EdgeWorker
from edgecloud.workers.errors import WorkerError
from edgecloud.workers.models import HealthStatus
from edgecloud.workers.remote import RemoteWorker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run inference through a scheduler")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path)
    source.add_argument("--video", type=Path)
    parser.add_argument(
        "--scheduler",
        choices=[strategy.value for strategy in SchedulerStrategy],
        required=True,
    )
    parser.add_argument("--remote-url", default="http://127.0.0.1:8000")
    parser.add_argument("--remote-timeout", type=float, default=10.0)
    parser.add_argument("--latency-alpha", type=float, default=0.3)
    parser.add_argument("--telemetry-interval", type=float, default=1.0)
    parser.add_argument("--telemetry-max-age", type=float, default=3.0)
    parser.add_argument("--health-interval", type=float, default=1.0)
    parser.add_argument("--health-max-age", type=float, default=3.0)
    parser.add_argument("--resource-pressure-threshold", type=float, default=0.85)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--demo-profile", choices=DEMO_PROFILES)
    parser.add_argument("--scenario", choices=SCENARIOS)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--run-id")
    return parser


def _frames(args: argparse.Namespace) -> Iterable[Frame]:
    if args.image is not None:
        return image_frames(args.image)
    return video_frames(args.video)


def run(args: argparse.Namespace) -> int:
    _validate_control_options(args.demo_profile, args.scenario)
    strategy = SchedulerStrategy(args.scheduler)
    edge_worker = None
    remote_worker = None
    if strategy in (
        SchedulerStrategy.EDGE_ONLY,
        SchedulerStrategy.ROUND_ROBIN,
        SchedulerStrategy.LATENCY_AWARE,
        SchedulerStrategy.RESOURCE_AWARE,
        SchedulerStrategy.RESILIENT,
    ):
        engine = InferenceEngine(
            TorchvisionSSDLiteBackend(device=args.device), confidence=args.confidence
        )
        edge_worker = EdgeWorker(engine)
    if strategy in (
        SchedulerStrategy.REMOTE_ONLY,
        SchedulerStrategy.ROUND_ROBIN,
        SchedulerStrategy.LATENCY_AWARE,
        SchedulerStrategy.RESOURCE_AWARE,
        SchedulerStrategy.RESILIENT,
    ):
        remote_worker = RemoteWorker(args.remote_url, timeout_seconds=args.remote_timeout)

    scenario = get_scenario(args.scenario) if args.scenario is not None else None
    active_profile_name = scenario.profile_at(0) if scenario else args.demo_profile or "normal"
    active_profile = get_demo_profile(active_profile_name)
    edge_control = None
    remote_control = None
    if args.demo_profile is not None or scenario is not None:
        if edge_worker is not None:
            edge_control = ControlledWorker(
                edge_worker,
                "edge",
                delay_ms=active_profile.edge_delay_ms,
                available=active_profile.edge_available,
            )
            edge_worker = edge_control
        if remote_worker is not None:
            remote_control = ControlledWorker(
                remote_worker,
                "remote",
                delay_ms=active_profile.remote_delay_ms,
                available=active_profile.remote_available,
            )
            remote_worker = remote_control

    local_telemetry = None
    telemetry_state = None
    if strategy in (SchedulerStrategy.RESOURCE_AWARE, SchedulerStrategy.RESILIENT):
        if remote_worker is None:
            raise RuntimeError("resource-aware scheduler requires a remote worker")
        local_telemetry = TelemetryCollector()
        telemetry_state = TelemetryState(
            local_telemetry.collect,
            remote_worker.telemetry,
            interval_seconds=args.telemetry_interval,
            max_age_seconds=args.telemetry_max_age,
        ).start()

    health_state = None
    if strategy is SchedulerStrategy.RESILIENT:
        if remote_worker is None:
            raise RuntimeError("resilient scheduler requires a remote worker")

        def remote_health() -> HealthSample:
            result = remote_worker.health()
            state = (
                WorkerHealth.HEALTHY
                if result.status is HealthStatus.READY
                else WorkerHealth.UNHEALTHY
            )
            return HealthSample(state, result.status.value)

        health_state = HealthState(
            lambda: HealthSample(WorkerHealth.HEALTHY, "local worker initialized"),
            remote_health,
            interval_seconds=args.health_interval,
            max_age_seconds=args.health_max_age,
        ).start()

    recorder = None
    try:
        scheduler = create_scheduler(
            strategy,
            edge_worker=edge_worker,
            remote_worker=remote_worker,
            latency_alpha=args.latency_alpha,
            telemetry_state=telemetry_state,
            resource_pressure_threshold=args.resource_pressure_threshold,
            health_state=health_state,
        )
        recorder = JsonlRecorder(args.record) if args.record is not None else None
        run_id = args.run_id or f"run-{time_ns()}"
        run_started = perf_counter()
        sequence = 0
        total_frames = successful_frames = failed_frames = 0
        edge_executions = remote_executions = fallback_count = worker_switches = 0
        observed_latencies: list[float] = []
        previous_executed: str | None = None
        source_type = "image" if args.image is not None else "video"
        source_path = args.image if args.image is not None else args.video
        if recorder is not None:
            recorder.record(
                run_start_event(
                    run_id=run_id,
                    scheduler=strategy.value,
                    latency_alpha=args.latency_alpha,
                    source_type=source_type,
                    source_name=source_path.name,
                    scenario=args.scenario,
                    initial_profile=active_profile_name,
                )
            )
        exit_code = 0
        for frame in _frames(args):
            desired_profile_name = (
                scenario.profile_at(frame.index) if scenario else active_profile_name
            )
            if desired_profile_name != active_profile_name:
                previous_profile = active_profile_name
                active_profile_name = desired_profile_name
                active_profile = get_demo_profile(active_profile_name)
                if edge_control is not None:
                    edge_control.apply(active_profile)
                if remote_control is not None:
                    remote_control.apply(active_profile)
                sequence += 1
                if recorder is not None:
                    recorder.record(
                        profile_change_event(
                            run_id=run_id,
                            sequence=sequence,
                            elapsed_ms=(perf_counter() - run_started) * 1000,
                            frame_index=frame.index,
                            from_profile=previous_profile,
                            to_profile=active_profile_name,
                        )
                    )
            total_frames += 1
            result = None
            error = None
            try:
                result = scheduler.infer(frame.image)
            except (OSError, RuntimeError, ValueError, WorkerError) as exc:
                error = exc
                failed_frames += 1
                print(
                    json.dumps(
                        {
                            "frame_index": frame.index,
                            "strategy": strategy.value,
                            "error": str(exc),
                        }
                    )
                )
                exit_code = 2
            else:
                successful_frames += 1
                executed = result.executed_worker_type
                edge_executions += executed == "edge"
                remote_executions += executed == "remote"
                fallback_count += result.fallback
                worker_switches += previous_executed is not None and previous_executed != executed
                previous_executed = executed
                observed_latencies.append(result.scheduler_latency_ms)
                payload = result.to_dict()
                payload["frame_index"] = frame.index
                print(json.dumps(payload), flush=True)
            sequence += 1
            if recorder is not None:
                recorder.record(
                    inference_event(
                        run_id=run_id,
                        sequence=sequence,
                        elapsed_ms=(perf_counter() - run_started) * 1000,
                        frame_index=frame.index,
                        profile_name=active_profile_name,
                        profile=active_profile,
                        scheduler=strategy.value,
                        result=result,
                        error=error,
                    )
                )
            if error is not None:
                break
        sequence += 1
        if recorder is not None:
            recorder.record(
                run_end_event(
                    run_id=run_id,
                    sequence=sequence,
                    elapsed_ms=(perf_counter() - run_started) * 1000,
                    total_frames=total_frames,
                    successful_frames=successful_frames,
                    failed_frames=failed_frames,
                    edge_executions=edge_executions,
                    remote_executions=remote_executions,
                    fallback_count=fallback_count,
                    worker_switches=worker_switches,
                    mean_observed_latency_ms=(
                        sum(observed_latencies) / len(observed_latencies)
                        if observed_latencies
                        else None
                    ),
                )
            )
            recorder.close()
        print(json.dumps({"summary": scheduler.summary()}), flush=True)
        return exit_code
    finally:
        if telemetry_state is not None:
            telemetry_state.close()
        if health_state is not None:
            health_state.close()
        if local_telemetry is not None:
            local_telemetry.close()
        if remote_worker is not None:
            remote_worker.close()
        if recorder is not None:
            recorder.close()


def _validate_control_options(profile: str | None, scenario: str | None) -> None:
    if profile is not None and scenario is not None:
        raise ValueError("--demo-profile and --scenario cannot be used together")


def main() -> None:
    try:
        raise SystemExit(run(build_parser().parse_args()))
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
