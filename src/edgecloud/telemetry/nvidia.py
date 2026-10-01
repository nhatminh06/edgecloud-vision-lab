from __future__ import annotations

from typing import Any

from edgecloud.telemetry.models import GpuMetrics


class NvidiaCollector:
    def __init__(self, nvml_module: Any | None = None) -> None:
        if nvml_module is None:
            import pynvml

            nvml_module = pynvml
        self._nvml = nvml_module
        self.available = False
        self.error: str | None = None
        self._initialized = False
        try:
            self._nvml.nvmlInit()
            self._initialized = True
            self.available = True
        except self._nvml.NVMLError as exc:
            self.error = f"NVML unavailable: {exc}"

    def collect(self) -> tuple[GpuMetrics, ...]:
        if not self.available:
            return ()
        try:
            count = self._nvml.nvmlDeviceGetCount()
        except self._nvml.NVMLError as exc:
            self.available = False
            self.error = f"NVML device enumeration failed: {exc}"
            return ()
        devices: list[GpuMetrics] = []
        for index in range(count):
            try:
                devices.append(self._collect_device(index))
            except self._nvml.NVMLError:
                devices.append(
                    GpuMetrics(
                        index=index,
                        name=f"NVIDIA GPU {index}",
                        utilization_percent=None,
                        memory_used_bytes=None,
                        memory_total_bytes=None,
                        memory_utilization_percent=None,
                        temperature_celsius=None,
                        power_draw_watts=None,
                    )
                )
        return tuple(devices)

    def _collect_device(self, index: int) -> GpuMetrics:
        handle = self._nvml.nvmlDeviceGetHandleByIndex(index)
        name = self._optional_metric(self._nvml.nvmlDeviceGetName, handle)
        utilization = self._optional_metric(self._nvml.nvmlDeviceGetUtilizationRates, handle)
        memory = self._optional_metric(self._nvml.nvmlDeviceGetMemoryInfo, handle)
        temperature = self._optional_metric(
            self._nvml.nvmlDeviceGetTemperature,
            handle,
            self._nvml.NVML_TEMPERATURE_GPU,
        )
        power_milliwatts = self._optional_metric(self._nvml.nvmlDeviceGetPowerUsage, handle)
        decoded_name = name.decode() if isinstance(name, bytes) else name
        memory_used = int(memory.used) if memory is not None else None
        memory_total = int(memory.total) if memory is not None else None
        memory_percent = None
        if memory_used is not None and memory_total:
            memory_percent = memory_used / memory_total * 100
        return GpuMetrics(
            index=index,
            name=decoded_name or f"NVIDIA GPU {index}",
            utilization_percent=(float(utilization.gpu) if utilization is not None else None),
            memory_used_bytes=memory_used,
            memory_total_bytes=memory_total,
            memory_utilization_percent=memory_percent,
            temperature_celsius=float(temperature) if temperature is not None else None,
            power_draw_watts=(
                float(power_milliwatts) / 1000 if power_milliwatts is not None else None
            ),
        )

    def _optional_metric(self, function: Any, *args: Any) -> Any | None:
        try:
            return function(*args)
        except self._nvml.NVMLError:
            return None

    def close(self) -> None:
        if not self._initialized:
            return
        try:
            self._nvml.nvmlShutdown()
        except self._nvml.NVMLError as exc:
            self.error = f"NVML shutdown failed: {exc}"
        finally:
            self._initialized = False

    def __enter__(self) -> NvidiaCollector:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
