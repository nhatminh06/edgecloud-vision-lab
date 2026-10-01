"""Deterministic inference schedulers."""

from edgecloud.scheduler.estimator import EwmaEstimator
from edgecloud.scheduler.health import (
    CachedHealth,
    HealthCache,
    HealthSample,
    HealthState,
    WorkerHealth,
)
from edgecloud.scheduler.models import (
    LatencyDecision,
    ResilientDecision,
    ResourceDecision,
    ScheduledResult,
    Scheduler,
    SchedulerStrategy,
)
from edgecloud.scheduler.resilient import ResilientScheduler
from edgecloud.scheduler.resource import ExecutionResource, WorkerResource, resource_pressure
from edgecloud.scheduler.strategies import (
    EdgeOnlyScheduler,
    LatencyAwareScheduler,
    RemoteOnlyScheduler,
    ResourceAwareScheduler,
    RoundRobinScheduler,
    create_scheduler,
)

__all__ = [
    "EdgeOnlyScheduler",
    "EwmaEstimator",
    "LatencyAwareScheduler",
    "LatencyDecision",
    "ExecutionResource",
    "RemoteOnlyScheduler",
    "ResourceAwareScheduler",
    "ResourceDecision",
    "ResilientDecision",
    "ResilientScheduler",
    "RoundRobinScheduler",
    "ScheduledResult",
    "Scheduler",
    "SchedulerStrategy",
    "WorkerResource",
    "CachedHealth",
    "HealthCache",
    "HealthSample",
    "HealthState",
    "WorkerHealth",
    "create_scheduler",
    "resource_pressure",
]
