from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path

from edgecloud.capture import Frame, image_frames, video_frames
from edgecloud.demo import DEMO_PROFILES, ControlledWorker, get_demo_profile
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
    return parser


def _frames(args: argparse.Namespace) -> Iterable[Frame]:
    if args.image is not None:
        return image_frames(args.image)
    return video_frames(args.video)


def run(args: argparse.Namespace) -> int:
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

    if args.demo_profile is not None:
        profile = get_demo_profile(args.demo_profile)
        if edge_worker is not None:
            edge_worker = ControlledWorker(
                edge_worker,
                "edge",
                delay_ms=profile.edge_delay_ms,
                available=profile.edge_available,
            )
        if remote_worker is not None:
            remote_worker = ControlledWorker(
                remote_worker,
                "remote",
                delay_ms=profile.remote_delay_ms,
                available=profile.remote_available,
            )

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
        for frame in _frames(args):
            try:
                result = scheduler.infer(frame.image)
            except (OSError, RuntimeError, ValueError, WorkerError) as exc:
                print(
                    json.dumps(
                        {
                            "frame_index": frame.index,
                            "strategy": strategy.value,
                            "error": str(exc),
                        }
                    )
                )
                print(json.dumps({"summary": scheduler.summary()}))
                return 2
            payload = result.to_dict()
            payload["frame_index"] = frame.index
            print(json.dumps(payload), flush=True)
        print(json.dumps({"summary": scheduler.summary()}), flush=True)
        return 0
    finally:
        if telemetry_state is not None:
            telemetry_state.close()
        if health_state is not None:
            health_state.close()
        if local_telemetry is not None:
            local_telemetry.close()
        if remote_worker is not None:
            remote_worker.close()


def main() -> None:
    try:
        raise SystemExit(run(build_parser().parse_args()))
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
