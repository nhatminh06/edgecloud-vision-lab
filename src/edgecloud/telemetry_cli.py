from __future__ import annotations

import argparse
import json
import time

from edgecloud.telemetry import TelemetryCollector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect machine telemetry as JSON Lines")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--count", type=int, default=1)
    return parser


def run(args: argparse.Namespace) -> int:
    if args.count <= 0:
        raise ValueError("count must be positive")
    if args.interval < 0:
        raise ValueError("interval cannot be negative")
    with TelemetryCollector() as collector:
        for index in range(args.count):
            print(json.dumps(collector.collect().to_dict()), flush=True)
            if index + 1 < args.count:
                time.sleep(args.interval)
    return 0


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
