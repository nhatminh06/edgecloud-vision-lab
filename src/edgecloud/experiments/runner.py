from __future__ import annotations

from collections.abc import Callable
from time import perf_counter
from typing import Any

from edgecloud.experiments.metrics import aggregate_run
from edgecloud.experiments.models import ExperimentConfig, RawObservation
from edgecloud.experiments.workers import RequestTrace
from edgecloud.inference.engine import Image
from edgecloud.scheduler.models import ScheduledResult, Scheduler


class ExperimentRunner:
    def __init__(
        self,
        config: ExperimentConfig,
        scheduler_factory: Callable[[], Scheduler],
        workload: Callable[[int], Image],
        trace: RequestTrace,
        *,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        self.config = config
        self._scheduler_factory = scheduler_factory
        self._workload = workload
        self._trace = trace
        self._clock = clock

    def run(self) -> tuple[list[RawObservation], dict[str, Any]]:
        observations: list[RawObservation] = []
        runs: list[dict[str, Any]] = []
        durations: list[float] = []
        experiment_started = self._clock()
        for run_id in range(1, self.config.repetitions + 1):
            scheduler = self._scheduler_factory()
            run_observations: list[RawObservation] = []
            for warmup_index in range(self.config.warmup_requests):
                request_index = warmup_index - self.config.warmup_requests
                run_observations.append(
                    self._execute(
                        scheduler,
                        run_id,
                        request_index,
                        True,
                        experiment_started,
                    )
                )
            measured_started = self._clock()
            for request_index in range(self.config.measured_requests):
                run_observations.append(
                    self._execute(
                        scheduler,
                        run_id,
                        request_index,
                        False,
                        experiment_started,
                    )
                )
            measured_duration = max(0.0, self._clock() - measured_started)
            durations.append(measured_duration)
            observations.extend(run_observations)
            runs.append(
                {
                    "run_id": run_id,
                    "metrics": aggregate_run(run_observations, measured_duration),
                }
            )
        summary = {
            "configuration": self.config.to_dict(),
            "runs": runs,
            "cross_run": aggregate_run(observations, sum(durations)),
            "experiment_duration_seconds": max(0.0, self._clock() - experiment_started),
        }
        return observations, summary

    def _execute(
        self,
        scheduler: Scheduler,
        run_id: int,
        request_index: int,
        warmup: bool,
        experiment_started: float,
    ) -> RawObservation:
        self._trace.begin(request_index)
        relative_time = max(0.0, self._clock() - experiment_started)
        started = self._clock()
        result: ScheduledResult | None = None
        error: Exception | None = None
        try:
            result = scheduler.infer(self._workload(request_index))
        except (OSError, RuntimeError, ValueError) as exc:
            error = exc
        elapsed_ms = max(0.0, (self._clock() - started) * 1000)
        return self._observation(
            result,
            error,
            run_id,
            request_index,
            warmup,
            relative_time,
            elapsed_ms,
        )

    def _observation(
        self,
        result: ScheduledResult | None,
        error: Exception | None,
        run_id: int,
        request_index: int,
        warmup: bool,
        relative_time: float,
        elapsed_ms: float,
    ) -> RawObservation:
        attempts = self._trace.attempts
        resilient = result.resilient_decision if result is not None else None
        resource = result.resource_decision if result is not None else None
        latency = result.latency_decision if result is not None else None
        initial_worker = (
            resilient.initial_worker_type
            if resilient is not None
            else attempts[0].worker_type
            if attempts
            else None
        )
        final_worker = result.selected_worker_type if result is not None else None
        reason = (
            resilient.reason
            if resilient is not None
            else resource.reason
            if resource is not None
            else latency.reason
            if latency is not None
            else None
        )
        fallback = resilient.fallback_occurred if resilient is not None else len(attempts) > 1
        primary_failure = (
            resilient.primary_failure_type
            if resilient is not None
            else attempts[0].error_type
            if attempts and not attempts[0].success
            else None
        )
        total_latency = result.scheduler_latency_ms if result is not None else elapsed_ms
        primary_ms = (
            resilient.primary_attempt_ms if resilient is not None else _attempt_ms(attempts, 0)
        )
        fallback_ms = (
            resilient.fallback_attempt_ms if resilient is not None else _attempt_ms(attempts, 1)
        )
        return RawObservation(
            experiment_id=self.config.name,
            run_id=run_id,
            scenario=self.config.name,
            scheduler=self.config.scheduler.value,
            request_index=request_index,
            warmup=warmup,
            relative_time_seconds=relative_time,
            initial_worker=initial_worker,
            final_worker=final_worker,
            decision_reason=reason,
            success=result is not None,
            error_type=type(error).__name__ if error is not None else None,
            fallback_occurred=fallback,
            primary_failure_type=primary_failure,
            worker_attempts=len(attempts),
            total_latency_ms=total_latency,
            primary_attempt_ms=primary_ms,
            fallback_attempt_ms=fallback_ms,
            edge_latency_estimate_ms=(
                resource.edge_latency_estimate_ms
                if resource is not None
                else latency.edge_latency_estimate_ms
                if latency is not None
                else None
            ),
            remote_latency_estimate_ms=(
                resource.remote_latency_estimate_ms
                if resource is not None
                else latency.remote_latency_estimate_ms
                if latency is not None
                else None
            ),
            edge_resource_pressure=(
                resource.edge_resource_pressure if resource is not None else None
            ),
            remote_resource_pressure=(
                resource.remote_resource_pressure if resource is not None else None
            ),
            edge_health=resilient.edge_health if resilient is not None else None,
            remote_health=resilient.remote_health if resilient is not None else None,
            worker_inference_ms=(
                result.worker_result.timing.inference_ms if result is not None else None
            ),
        )


def _attempt_ms(attempts: list[Any], index: int) -> float | None:
    return attempts[index].elapsed_ms if len(attempts) > index else None
