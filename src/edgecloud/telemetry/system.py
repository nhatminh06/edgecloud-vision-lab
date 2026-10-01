from __future__ import annotations

from typing import Any

import psutil

from edgecloud.telemetry.models import SystemMetrics


class SystemCollector:
    """Collect non-blocking CPU percentage and system virtual-memory metrics.

    psutil's interval=None reports CPU activity since its previous call. The first call reports
    activity since module import, avoiding a blocking sample interval on every snapshot.
    """

    def __init__(self, psutil_module: Any = psutil) -> None:
        self._psutil = psutil_module

    def collect(self) -> SystemMetrics:
        cpu_percent = float(self._psutil.cpu_percent(interval=None))
        memory = self._psutil.virtual_memory()
        return SystemMetrics(
            cpu_utilization_percent=cpu_percent,
            memory_used_bytes=int(memory.used),
            memory_total_bytes=int(memory.total),
            memory_utilization_percent=float(memory.percent),
        )
