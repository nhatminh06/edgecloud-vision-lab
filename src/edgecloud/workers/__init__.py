"""Local and remote inference workers."""

from edgecloud.workers.edge import EdgeWorker
from edgecloud.workers.models import HealthResult, HealthStatus, InferenceWorker, WorkerResult
from edgecloud.workers.remote import RemoteWorker

__all__ = [
    "EdgeWorker",
    "HealthResult",
    "HealthStatus",
    "InferenceWorker",
    "RemoteWorker",
    "WorkerResult",
]
