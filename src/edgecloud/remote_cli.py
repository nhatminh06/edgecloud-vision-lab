from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from edgecloud.capture import image_frames
from edgecloud.workers.errors import WorkerError
from edgecloud.workers.remote import RemoteWorker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Send an image to an HTTP inference worker")
    parser.add_argument("image", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=10.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        frame = next(iter(image_frames(args.image)))
        with RemoteWorker(args.url, timeout_seconds=args.timeout) as worker:
            result = worker.infer(frame.image)
        print(json.dumps(result.to_dict()))
    except (OSError, ValueError, WorkerError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
