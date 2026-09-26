from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path

from edgecloud.capture import Frame, image_frames, video_frames
from edgecloud.inference.engine import InferenceEngine
from edgecloud.inference.torchvision_backend import TorchvisionSSDLiteBackend
from edgecloud.scheduler import SchedulerStrategy, create_scheduler
from edgecloud.workers.edge import EdgeWorker
from edgecloud.workers.errors import WorkerError
from edgecloud.workers.remote import RemoteWorker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run inference through a baseline scheduler")
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
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    return parser


def _frames(args: argparse.Namespace) -> Iterable[Frame]:
    if args.image is not None:
        return image_frames(args.image)
    return video_frames(args.video)


def run(args: argparse.Namespace) -> int:
    strategy = SchedulerStrategy(args.scheduler)
    edge_worker = None
    remote_worker = None
    if strategy in (SchedulerStrategy.EDGE_ONLY, SchedulerStrategy.ROUND_ROBIN):
        engine = InferenceEngine(
            TorchvisionSSDLiteBackend(device=args.device), confidence=args.confidence
        )
        edge_worker = EdgeWorker(engine)
    if strategy in (SchedulerStrategy.REMOTE_ONLY, SchedulerStrategy.ROUND_ROBIN):
        remote_worker = RemoteWorker(args.remote_url, timeout_seconds=args.remote_timeout)

    scheduler = create_scheduler(strategy, edge_worker=edge_worker, remote_worker=remote_worker)
    try:
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
                print(json.dumps({"summary": scheduler.metrics.to_dict()}))
                return 2
            payload = result.to_dict()
            payload["frame_index"] = frame.index
            print(json.dumps(payload), flush=True)
        print(json.dumps({"summary": scheduler.metrics.to_dict()}), flush=True)
        return 0
    finally:
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
