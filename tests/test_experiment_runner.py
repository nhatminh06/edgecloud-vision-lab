from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from edgecloud.experiment_cli import _matrix_configs
from edgecloud.experiments.models import CloudEnvironment, ExperimentConfig
from edgecloud.experiments.output import write_results
from edgecloud.experiments.runner import ExperimentRunner
from edgecloud.experiments.workers import RecordingWorker, RequestTrace
from edgecloud.inference.models import TimingMetrics
from edgecloud.scheduler import EdgeOnlyScheduler, RoundRobinScheduler, SchedulerStrategy
from edgecloud.workers.models import WorkerResult


@dataclass
class FakeWorker:
    worker_id: str
    worker_type: str

    def infer(self, image: np.ndarray) -> WorkerResult:
        return WorkerResult((), TimingMetrics(1, 2, 3, 6), self.worker_id, self.worker_type, "fake")


@pytest.mark.parametrize("strategy", [SchedulerStrategy.EDGE_ONLY, SchedulerStrategy.ROUND_ROBIN])
def test_runner_supports_policies_warmup_and_repeated_runs(
    strategy: SchedulerStrategy, tmp_path: Path
) -> None:
    trace = RequestTrace()
    edge = RecordingWorker(FakeWorker("edge-1", "edge"), "edge-1", "edge", trace)
    remote = RecordingWorker(FakeWorker("remote-1", "remote"), "remote-1", "remote", trace)
    config = ExperimentConfig(
        "runner-test",
        strategy,
        tmp_path / "input.jpg",
        measured_requests=2,
        warmup_requests=1,
        repetitions=2,
        output_dir=tmp_path / "results",
    )

    def scheduler_factory():
        if strategy is SchedulerStrategy.EDGE_ONLY:
            return EdgeOnlyScheduler(edge)
        return RoundRobinScheduler(edge, remote)

    observations, summary = ExperimentRunner(
        config,
        scheduler_factory,
        lambda _: np.zeros((2, 2, 3), dtype=np.uint8),
        trace,
    ).run()

    assert len(observations) == 6
    assert sum(not item.warmup for item in observations) == 4
    assert len(summary["runs"]) == 2
    assert all(run["metrics"]["total_logical_requests"] == 2 for run in summary["runs"])
    assert summary["cross_run"]["total_logical_requests"] == 4


def test_output_writes_config_raw_jsonl_summary_json_and_csv(tmp_path: Path) -> None:
    trace = RequestTrace()
    edge = RecordingWorker(FakeWorker("edge-1", "edge"), "edge-1", "edge", trace)
    config = ExperimentConfig(
        "output-test",
        SchedulerStrategy.EDGE_ONLY,
        tmp_path / "input.jpg",
        measured_requests=1,
        output_dir=tmp_path / "results",
    )
    observations, summary = ExperimentRunner(
        config,
        lambda: EdgeOnlyScheduler(edge),
        lambda _: np.zeros((2, 2, 3), dtype=np.uint8),
        trace,
    ).run()
    summary["environment"] = {"git_commit": None}

    write_results(config.output_dir, config, observations, summary)

    assert json.loads((config.output_dir / "config.json").read_text())["name"] == "output-test"
    raw = (config.output_dir / "raw.jsonl").read_text().splitlines()
    assert len(raw) == 1
    assert json.loads(raw[0])["success"] is True
    assert (
        json.loads((config.output_dir / "summary.json").read_text())["environment"]["git_commit"]
        is None
    )
    with (config.output_dir / "summary.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["scheduler"] == "edge_only"


def test_compact_matrix_expands_scheduler_delay_cross_product(tmp_path: Path) -> None:
    configs = _matrix_configs(
        {
            "name": "matrix-test",
            "input_path": str(tmp_path / "input.jpg"),
            "measured_requests": 2,
            "schedulers": ["edge_only", "remote_only"],
            "delays_ms": [0, 25],
            "output_dir": str(tmp_path / "results"),
        }
    )

    assert len(configs) == 4
    assert {(item.scheduler.value, item.delay_ms) for item in configs} == {
        ("edge_only", 0),
        ("edge_only", 25),
        ("remote_only", 0),
        ("remote_only", 25),
    }
    assert configs[0].output_dir.parent == tmp_path / "results"


def test_real_cloud_matrix_requires_explicit_remote_url(tmp_path: Path) -> None:
    payload = {
        "name": "cloud-test",
        "input_path": str(tmp_path / "input.jpg"),
        "measured_requests": 2,
        "schedulers": ["remote_only"],
        "delays_ms": [0],
        "require_remote_url": True,
    }

    with pytest.raises(ValueError, match="requires a remote URL"):
        _matrix_configs(payload)

    config = _matrix_configs(payload, remote_url_override="http://cloud.example:8000")[0]
    assert config.remote_url == "http://cloud.example:8000"


def test_cloud_environment_serialization_round_trip(tmp_path: Path) -> None:
    cloud = CloudEnvironment(
        provider="provider",
        region="region-1",
        instance_type="cpu-small",
        operating_system="Ubuntu 24.04",
        architecture="x86_64",
        cpu_model="Example CPU",
        vcpu_count=2,
        total_memory_bytes=4_294_967_296,
    )
    config = ExperimentConfig(
        "cloud-test",
        SchedulerStrategy.REMOTE_ONLY,
        tmp_path / "input.jpg",
        measured_requests=2,
        remote_url="http://cloud.example:8000",
        cloud_environment=cloud,
    )

    restored = ExperimentConfig.from_dict(config.to_dict())

    assert restored.cloud_environment == cloud
