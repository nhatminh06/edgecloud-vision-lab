from __future__ import annotations

import argparse

import uvicorn

from edgecloud.api import ServiceSettings, create_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the HTTP inference worker")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--worker-id", default="remote-1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--confidence", type=float, default=0.5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = ServiceSettings(
        worker_id=args.worker_id, device=args.device, confidence=args.confidence
    )
    uvicorn.run(create_app(settings=settings), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
