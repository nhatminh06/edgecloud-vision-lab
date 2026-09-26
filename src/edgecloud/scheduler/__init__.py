"""Deterministic baseline inference schedulers."""

from edgecloud.scheduler.models import ScheduledResult, Scheduler, SchedulerStrategy
from edgecloud.scheduler.strategies import (
    EdgeOnlyScheduler,
    RemoteOnlyScheduler,
    RoundRobinScheduler,
    create_scheduler,
)

__all__ = [
    "EdgeOnlyScheduler",
    "RemoteOnlyScheduler",
    "RoundRobinScheduler",
    "ScheduledResult",
    "Scheduler",
    "SchedulerStrategy",
    "create_scheduler",
]
