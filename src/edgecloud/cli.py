from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path

import cv2

from edgecloud.capture import Frame, image_frames, video_frames, webcam_frames
from edgecloud.inference.engine import InferenceEngine
from edgecloud.inference.torchvision_backend import TorchvisionSSDLiteBackend
from edgecloud.metrics import StreamMetrics
from edgecloud.render import render_detections


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local object detection")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path)
    source.add_argument("--video", type=Path)
    source.add_argument("--webcam", type=int, nargs="?", const=0)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--device", default="cpu", help="torch device, for example cpu or cuda")
    parser.add_argument("--output", type=Path, help="rendered image or video path")
    return parser


def _frames(args: argparse.Namespace) -> Iterable[Frame]:
    if args.image is not None:
        return image_frames(args.image)
    if args.video is not None:
        return video_frames(args.video)
    return webcam_frames(args.webcam)


def run(args: argparse.Namespace) -> int:
    engine = InferenceEngine(
        TorchvisionSSDLiteBackend(device=args.device), confidence=args.confidence
    )
    frames = _frames(args)
    metrics = StreamMetrics.start()
    writer: cv2.VideoWriter | None = None
    try:
        for frame in frames:
            result = engine.run(frame.image, source=frame.source, frame_index=frame.index)
            metrics.record_frame()
            payload = result.to_dict()
            payload["stream_fps"] = metrics.fps()
            print(json.dumps(payload), flush=True)
            if args.output is not None:
                rendered = render_detections(frame.image, result.detections)
                if args.image is not None:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(args.output), rendered):
                        raise OSError(f"could not write image: {args.output}")
                else:
                    if writer is None:
                        args.output.parent.mkdir(parents=True, exist_ok=True)
                        height, width = rendered.shape[:2]
                        writer = cv2.VideoWriter(
                            str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height)
                        )
                        if not writer.isOpened():
                            raise OSError(f"could not open video output: {args.output}")
                    writer.write(rendered)
    finally:
        if writer is not None:
            writer.release()
    return 0


def main() -> None:
    parser = build_parser()
    try:
        raise SystemExit(run(parser.parse_args()))
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
