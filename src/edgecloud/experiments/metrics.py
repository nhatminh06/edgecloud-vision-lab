from __future__ import annotations

from collections import Counter
from statistics import fmean, pstdev
from typing import Any

from edgecloud.experiments.models import RawObservation


def percentile(values: list[float], quantile: float) -> float | None:
    """Return a linearly interpolated percentile using rank (n - 1) * quantile."""
    if not values:
        return None
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be between zero and one")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * quantile
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def aggregate_run(
    observations: list[RawObservation], measured_duration_seconds: float
) -> dict[str, Any]:
    measured = [item for item in observations if not item.warmup]
    successful = [item for item in measured if item.success]
    latencies = [item.total_latency_ms for item in successful]
    initial = Counter(item.initial_worker for item in measured)
    final = Counter(item.final_worker for item in successful)
    reasons = Counter(item.decision_reason for item in measured if item.decision_reason)
    primary_failures = sum(item.primary_failure_type is not None for item in measured)
    fallback_attempts = sum(item.fallback_occurred for item in measured)
    successful_fallbacks = sum(item.fallback_occurred and item.success for item in measured)
    failed_fallbacks = fallback_attempts - successful_fallbacks
    edge_to_remote = sum(
        item.fallback_occurred and item.initial_worker == "edge" for item in measured
    )
    remote_to_edge = sum(
        item.fallback_occurred and item.initial_worker == "remote" for item in measured
    )
    count = len(measured)
    success_count = len(successful)
    return {
        "total_logical_requests": count,
        "successful_logical_requests": success_count,
        "failed_logical_requests": count - success_count,
        "worker_attempts": sum(item.worker_attempts for item in measured),
        "success_rate": success_count / count if count else 0.0,
        "primary_failures": primary_failures,
        "fallback_attempts": fallback_attempts,
        "successful_fallbacks": successful_fallbacks,
        "failed_fallbacks": failed_fallbacks,
        "edge_to_remote_fallbacks": edge_to_remote,
        "remote_to_edge_fallbacks": remote_to_edge,
        "initial_edge_selections": initial["edge"],
        "initial_remote_selections": initial["remote"],
        "final_edge_executions": final["edge"],
        "final_remote_executions": final["remote"],
        "edge_selection_percentage": initial["edge"] / count * 100 if count else 0.0,
        "remote_selection_percentage": initial["remote"] / count * 100 if count else 0.0,
        "decision_reason_counts": dict(sorted(reasons.items())),
        "measured_duration_seconds": measured_duration_seconds,
        "throughput_rps": (
            success_count / measured_duration_seconds if measured_duration_seconds > 0 else 0.0
        ),
        "latency": latency_statistics(latencies),
    }


def latency_statistics(values: list[float]) -> dict[str, int | float | None]:
    return {
        "count": len(values),
        "mean_ms": fmean(values) if values else None,
        "min_ms": min(values) if values else None,
        "max_ms": max(values) if values else None,
        "standard_deviation_ms": pstdev(values) if values else None,
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
    }
