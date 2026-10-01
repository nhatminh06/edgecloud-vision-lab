from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def generate_plots(result_dirs: list[Path], output_dir: Path) -> list[Path]:
    matplotlib_config = Path(tempfile.gettempdir()) / "edgecloud-matplotlib"
    matplotlib_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config))
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("plot generation requires the 'experiment' extra") from exc

    records = [_load_summary(path) for path in result_dirs]
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = [
        _line_plot(
            plt,
            records,
            output_dir / "latency-vs-delay.png",
            "Artificial request-path delay (ms)",
            "Latency (ms)",
            (("p50", "p50_ms"), ("p95", "p95_ms")),
        ),
        _line_plot(
            plt,
            records,
            output_dir / "throughput-vs-delay.png",
            "Artificial request-path delay (ms)",
            "Successful requests/second",
            (("throughput", "throughput_rps"),),
        ),
        _routing_plot(plt, records, output_dir / "routing-vs-delay.png"),
    ]
    failure_path = _failure_plot(plt, result_dirs, output_dir / "failure-timeline.png")
    if failure_path is not None:
        generated.append(failure_path)
    return generated


def _load_summary(path: Path) -> dict[str, Any]:
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    metrics = summary["cross_run"]
    return {
        "path": path,
        "scheduler": summary["configuration"]["scheduler"],
        "delay_ms": summary["configuration"]["delay_ms"],
        "latency": metrics["latency"],
        "throughput_rps": metrics["throughput_rps"],
        "edge": metrics["edge_selection_percentage"],
        "remote": metrics["remote_selection_percentage"],
    }


def _line_plot(
    plt: Any,
    records: list[dict[str, Any]],
    path: Path,
    xlabel: str,
    ylabel: str,
    series: tuple[tuple[str, str], ...],
) -> Path:
    figure, axis = plt.subplots()
    schedulers = sorted({record["scheduler"] for record in records})
    for scheduler in schedulers:
        selected = sorted(
            (record for record in records if record["scheduler"] == scheduler),
            key=lambda item: item["delay_ms"],
        )
        for label, key in series:
            values = [
                record["latency"].get(key) if key.endswith("_ms") else record[key]
                for record in selected
            ]
            axis.plot(
                [record["delay_ms"] for record in selected],
                values,
                marker="o",
                label=f"{scheduler} {label}",
            )
    axis.set(xlabel=xlabel, ylabel=ylabel)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path


def _routing_plot(plt: Any, records: list[dict[str, Any]], path: Path) -> Path:
    figure, axis = plt.subplots()
    for scheduler in sorted({record["scheduler"] for record in records}):
        selected = sorted(
            (record for record in records if record["scheduler"] == scheduler),
            key=lambda item: item["delay_ms"],
        )
        axis.plot(
            [record["delay_ms"] for record in selected],
            [record["remote"] for record in selected],
            marker="o",
            label=f"{scheduler} remote",
        )
    axis.set(
        xlabel="Artificial request-path delay (ms)",
        ylabel="Initial remote selection (%)",
        ylim=(0, 100),
    )
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path


def _failure_plot(plt: Any, result_dirs: list[Path], path: Path) -> Path | None:
    points: list[dict[str, Any]] = []
    for result_dir in result_dirs:
        raw_path = result_dir / "raw.jsonl"
        if not raw_path.exists():
            continue
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            if not item["warmup"]:
                points.append(item)
    if not points:
        return None
    figure, axis = plt.subplots()
    worker_value = {"edge": 0, "remote": 1, None: -1}
    axis.scatter(
        [item["request_index"] for item in points],
        [worker_value.get(item["final_worker"], -1) for item in points],
        c=["tab:orange" if item["fallback_occurred"] else "tab:blue" for item in points],
    )
    axis.set(
        xlabel="Measured request index",
        ylabel="Final worker",
        yticks=[-1, 0, 1],
        yticklabels=["failed", "edge", "remote"],
    )
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path
