from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path

from edgecloud.capture import image_frames, video_frames
from edgecloud.experiments.environment import collect_environment
from edgecloud.experiments.models import ExperimentConfig, FailureWindow
from edgecloud.experiments.output import write_results
from edgecloud.experiments.plots import generate_plots
from edgecloud.experiments.runner import ExperimentRunner
from edgecloud.experiments.scenarios import CpuPressure, NoPressure
from edgecloud.experiments.workers import (
    DelayWorker,
    FailureWindowWorker,
    RecordingWorker,
    RequestTrace,
)
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
from edgecloud.workers.models import HealthStatus
from edgecloud.workers.remote import RemoteWorker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run reproducible scheduler experiments")
    subcommands = parser.add_subparsers(dest="command", required=True)
    run_parser = subcommands.add_parser("run", help="run one experiment configuration")
    run_parser.add_argument("--name", default="development-validation")
    run_parser.add_argument(
        "--scheduler", choices=[item.value for item in SchedulerStrategy], required=True
    )
    run_parser.add_argument("--image", type=Path, required=True)
    run_parser.add_argument("--requests", type=int, required=True)
    run_parser.add_argument("--warmup", type=int, default=0)
    run_parser.add_argument("--repetitions", type=int, default=1)
    run_parser.add_argument(
        "--remote-url", default=os.environ.get("EDGECLOUD_REMOTE_URL", "http://127.0.0.1:8000")
    )
    run_parser.add_argument("--delay-ms", type=float, default=0)
    run_parser.add_argument("--latency-alpha", type=float, default=0.3)
    run_parser.add_argument("--resource-threshold", type=float, default=0.85)
    run_parser.add_argument("--telemetry-interval", type=float, default=1.0)
    run_parser.add_argument("--health-interval", type=float, default=1.0)
    run_parser.add_argument(
        "--resource-scenario", choices=("normal", "edge_cpu_pressure"), default="normal"
    )
    run_parser.add_argument("--cpu-pressure-workers", type=int, default=1)
    run_parser.add_argument("--failure-start", type=int)
    run_parser.add_argument("--failure-end", type=int)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--device", default="cpu")
    run_parser.add_argument("--confidence", type=float, default=0.5)

    matrix_parser = subcommands.add_parser("matrix", help="run configurations from JSON")
    matrix_parser.add_argument("--config", type=Path, required=True)
    matrix_parser.add_argument("--remote-url", default=os.environ.get("EDGECLOUD_REMOTE_URL"))
    matrix_parser.add_argument("--device", default="cpu")
    matrix_parser.add_argument("--confidence", type=float, default=0.5)

    plot_parser = subcommands.add_parser("plot", help="generate plots from result directories")
    plot_parser.add_argument("results", nargs="+", type=Path)
    plot_parser.add_argument("--output", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> int:
    if args.command == "plot":
        for path in generate_plots(args.results, args.output):
            print(path)
        return 0
    if args.command == "matrix":
        payload = json.loads(args.config.read_text(encoding="utf-8"))
        configs = _matrix_configs(payload, remote_url_override=args.remote_url)
        for config in configs:
            _run_config(config, device=args.device, confidence=args.confidence)
        return 0
    failure_window = _failure_window(args.failure_start, args.failure_end)
    config = ExperimentConfig(
        name=args.name,
        scheduler=SchedulerStrategy(args.scheduler),
        input_path=args.image,
        measured_requests=args.requests,
        warmup_requests=args.warmup,
        repetitions=args.repetitions,
        remote_url=args.remote_url,
        latency_alpha=args.latency_alpha,
        resource_threshold=args.resource_threshold,
        telemetry_interval=args.telemetry_interval,
        health_interval=args.health_interval,
        delay_ms=args.delay_ms,
        resource_scenario=args.resource_scenario,
        cpu_pressure_workers=args.cpu_pressure_workers,
        failure_window=failure_window,
        output_dir=args.output,
    )
    _run_config(config, device=args.device, confidence=args.confidence)
    return 0


def _run_config(config: ExperimentConfig, *, device: str, confidence: float) -> None:
    frames = _load_frames(config.input_path)
    trace = RequestTrace()
    needs_edge = config.scheduler is not SchedulerStrategy.REMOTE_ONLY
    needs_remote = config.scheduler is not SchedulerStrategy.EDGE_ONLY
    with ExitStack() as stack:
        edge_worker = None
        if needs_edge:
            engine = InferenceEngine(
                TorchvisionSSDLiteBackend(device=device), confidence=confidence
            )
            edge_worker = RecordingWorker(EdgeWorker(engine), "edge-1", "edge", trace)

        remote_client = None
        remote_worker = None
        if needs_remote:
            remote_client = RemoteWorker(config.remote_url)
            stack.callback(remote_client.close)
            wrapped_remote = DelayWorker(remote_client, config.delay_ms)
            if config.failure_window is not None:
                wrapped_remote = FailureWindowWorker(
                    wrapped_remote,
                    trace,
                    config.failure_window.start_request,
                    config.failure_window.end_request,
                )
            remote_worker = RecordingWorker(wrapped_remote, "remote-1", "remote", trace)

        telemetry_state = None
        if config.scheduler in (SchedulerStrategy.RESOURCE_AWARE, SchedulerStrategy.RESILIENT):
            if remote_client is None:
                raise RuntimeError("adaptive experiment requires a remote worker")
            local_telemetry = TelemetryCollector()
            stack.callback(local_telemetry.close)
            telemetry_state = TelemetryState(
                local_telemetry.collect,
                remote_client.telemetry,
                interval_seconds=config.telemetry_interval,
                max_age_seconds=max(3.0, config.telemetry_interval * 3),
            ).start()
            stack.callback(telemetry_state.close)

        health_state = None
        if config.scheduler is SchedulerStrategy.RESILIENT:
            if remote_client is None:
                raise RuntimeError("resilient experiment requires a remote worker")

            def remote_health() -> HealthSample:
                result = remote_client.health()
                state = (
                    WorkerHealth.HEALTHY
                    if result.status is HealthStatus.READY
                    else WorkerHealth.UNHEALTHY
                )
                return HealthSample(state, result.status.value)

            health_state = HealthState(
                lambda: HealthSample(WorkerHealth.HEALTHY, "local worker initialized"),
                remote_health,
                interval_seconds=config.health_interval,
                max_age_seconds=max(3.0, config.health_interval * 3),
            ).start()
            stack.callback(health_state.close)

        def scheduler_factory():
            return create_scheduler(
                config.scheduler,
                edge_worker=edge_worker,
                remote_worker=remote_worker,
                latency_alpha=config.latency_alpha,
                telemetry_state=telemetry_state,
                resource_pressure_threshold=config.resource_threshold,
                health_state=health_state,
            )

        pressure = (
            CpuPressure(config.cpu_pressure_workers)
            if config.resource_scenario == "edge_cpu_pressure"
            else NoPressure()
        )
        with pressure:
            observations, summary = ExperimentRunner(
                config,
                scheduler_factory,
                lambda index: frames[index % len(frames)],
                trace,
            ).run()
        with TelemetryCollector() as environment_telemetry:
            environment_snapshot = environment_telemetry.collect()
        gpu_name = environment_snapshot.gpus[0].name if environment_snapshot.gpus else None
        summary["environment"] = {
            "local": collect_environment(
                backend="torchvision-ssdlite320-mobilenet-v3",
                execution_resource=device,
                gpu_name=gpu_name,
            ),
            "cloud": (
                asdict(config.cloud_environment) if config.cloud_environment is not None else None
            ),
        }
        summary["limitations"] = [
            "edge and remote may share physical hardware in local emulation",
            "artificial delay is total added request-path delay before remote inference",
        ]
        write_results(config.output_dir, config, observations, summary)
        print(config.output_dir)


def _load_frames(path: Path):
    image_suffixes = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
    source = image_frames(path) if path.suffix.lower() in image_suffixes else video_frames(path)
    frames = [frame.image for frame in source]
    if not frames:
        raise ValueError(f"input contains no frames: {path}")
    return frames


def _failure_window(start: int | None, end: int | None) -> FailureWindow | None:
    if start is None and end is None:
        return None
    if start is None or end is None:
        raise ValueError("failure start and end must be provided together")
    return FailureWindow(start, end)


def _matrix_configs(
    payload: dict[str, object], remote_url_override: str | None = None
) -> list[ExperimentConfig]:
    require_remote_url = payload.get("require_remote_url") is True
    configured_remote_url = payload.get("remote_url")
    remote_url = remote_url_override or (
        str(configured_remote_url) if configured_remote_url is not None else None
    )
    if require_remote_url and not remote_url:
        raise ValueError("matrix requires a remote URL via --remote-url or EDGECLOUD_REMOTE_URL")
    experiments = payload.get("experiments")
    if isinstance(experiments, list):
        items = [dict(item) for item in experiments]
        if remote_url is not None:
            for item in items:
                item["remote_url"] = remote_url
        return [ExperimentConfig.from_dict(item) for item in items]
    schedulers = payload.get("schedulers")
    delays = payload.get("delays_ms")
    if not isinstance(schedulers, list) or not isinstance(delays, list):
        raise ValueError("matrix config requires experiments or schedulers and delays_ms")
    base = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "schedulers",
            "delays_ms",
            "output_dir",
            "name",
            "require_remote_url",
        }
    }
    matrix_name = str(payload.get("name", "matrix"))
    output_root = Path(str(payload.get("output_dir", "results")))
    configs = []
    for scheduler in schedulers:
        for delay in delays:
            item = dict(base)
            item.update(
                {
                    "name": f"{matrix_name}-{scheduler}-delay-{delay}",
                    "scheduler": scheduler,
                    "delay_ms": delay,
                    "output_dir": str(output_root / f"{scheduler}-delay-{delay}"),
                }
            )
            if remote_url is not None:
                item["remote_url"] = remote_url
            configs.append(ExperimentConfig.from_dict(item))
    return configs


def main() -> None:
    try:
        raise SystemExit(run(build_parser().parse_args()))
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
