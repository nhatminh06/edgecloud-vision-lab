from __future__ import annotations

import argparse
import sys
from pathlib import Path

from edgecloud.experiments.replay import export_replay


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate JSONL and export static replay JSON")
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        export_replay(args.source, args.destination)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
