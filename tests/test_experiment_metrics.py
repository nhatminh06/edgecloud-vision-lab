from __future__ import annotations

import pytest

from edgecloud.experiments.metrics import aggregate_run, latency_statistics, percentile
from edgecloud.experiments.models import RawObservation


def observation(
    index: int,
    latency: float,
    *,
    success: bool = True,
    warmup: bool = False,
    initial: str = "edge",
    final: str | None = "edge",
    fallback: bool = False,
    primary_failure: str | None = None,
    reason: str = "test_reason",
) -> RawObservation:
    return RawObservation(
        "test",
        1,
        "scenario",
        "resilient",
        index,
        warmup,
        float(index),
        initial,
        final,
        reason,
        success,
        None if success else "RuntimeError",
        fallback,
        primary_failure,
        2 if fallback else 1,
        latency,
        latency,
        None,
        10,
        20,
        0.2,
        0.3,
        "healthy",
        "healthy",
        5,
    )


def test_percentile_uses_linear_interpolation() -> None:
    values = [1, 2, 3, 4]
    assert percentile(values, 0.5) == 2.5
    assert percentile(values, 0.95) == pytest.approx(3.85)
    assert percentile(values, 0.99) == pytest.approx(3.97)
    assert percentile([], 0.5) is None


def test_latency_statistics_empty_and_single_sample() -> None:
    assert latency_statistics([])["mean_ms"] is None
    single = latency_statistics([7])
    assert single["mean_ms"] == 7
    assert single["standard_deviation_ms"] == 0
    assert single["p99_ms"] == 7


def test_aggregation_excludes_warmup_and_measures_throughput_and_reliability() -> None:
    observations = [
        observation(-1, 1000, warmup=True),
        observation(0, 10),
        observation(
            1,
            30,
            initial="remote",
            final="edge",
            fallback=True,
            primary_failure="WorkerTimeoutError",
            reason="remote_failed_fallback_edge",
        ),
        observation(2, 50, success=False, initial="remote", final=None),
    ]

    result = aggregate_run(observations, measured_duration_seconds=2)

    assert result["total_logical_requests"] == 3
    assert result["successful_logical_requests"] == 2
    assert result["failed_logical_requests"] == 1
    assert result["worker_attempts"] == 4
    assert result["success_rate"] == pytest.approx(2 / 3)
    assert result["throughput_rps"] == 1
    assert result["latency"]["mean_ms"] == 20
    assert result["fallback_attempts"] == 1
    assert result["successful_fallbacks"] == 1
    assert result["remote_to_edge_fallbacks"] == 1
    assert result["initial_edge_selections"] == 1
    assert result["initial_remote_selections"] == 2
    assert result["decision_reason_counts"]["test_reason"] == 2
