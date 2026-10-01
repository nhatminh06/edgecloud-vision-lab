from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from edgecloud.experiments.models import ExperimentConfig, RawObservation

CSV_FIELDS = (
    "scenario",
    "scheduler",
    "delay_ms",
    "run_id",
    "measured_requests",
    "success_rate",
    "mean_latency_ms",
    "p50_latency_ms",
    "p95_latency_ms",
    "p99_latency_ms",
    "throughput_rps",
    "edge_selection_percent",
    "remote_selection_percent",
    "fallback_count",
)


def write_results(
    output_dir: Path,
    config: ExperimentConfig,
    observations: list[RawObservation],
    summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "config.json", config.to_dict())
    with (output_dir / "raw.jsonl").open("w", encoding="utf-8") as handle:
        for observation in observations:
            handle.write(json.dumps(observation.to_dict(), sort_keys=True) + "\n")
    _write_json(output_dir / "summary.json", summary)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for run in summary["runs"]:
            metrics = run["metrics"]
            latency = metrics["latency"]
            writer.writerow(
                {
                    "scenario": config.name,
                    "scheduler": config.scheduler.value,
                    "delay_ms": config.delay_ms,
                    "run_id": run["run_id"],
                    "measured_requests": metrics["total_logical_requests"],
                    "success_rate": metrics["success_rate"],
                    "mean_latency_ms": latency["mean_ms"],
                    "p50_latency_ms": latency["p50_ms"],
                    "p95_latency_ms": latency["p95_ms"],
                    "p99_latency_ms": latency["p99_ms"],
                    "throughput_rps": metrics["throughput_rps"],
                    "edge_selection_percent": metrics["edge_selection_percentage"],
                    "remote_selection_percent": metrics["remote_selection_percentage"],
                    "fallback_count": metrics["fallback_attempts"],
                }
            )


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
