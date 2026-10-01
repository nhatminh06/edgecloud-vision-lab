from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import cv2
import httpx
import numpy as np
import pytest

from edgecloud.api import ServiceSettings, create_app
from edgecloud.inference.models import BoundingBox, Detection, TimingMetrics
from edgecloud.telemetry.models import SystemMetrics, TelemetrySnapshot
from edgecloud.workers.models import WorkerResult

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class FakeWorker:
    def infer(self, image: np.ndarray) -> WorkerResult:
        assert image.shape == (6, 8, 3)
        return sample_result()


class FailingWorker:
    def infer(self, image: np.ndarray) -> WorkerResult:
        raise RuntimeError("internal implementation details")


class FakeTelemetry:
    def __init__(self, *, failure: bool = False) -> None:
        self.failure = failure
        self.closed = False

    def collect(self) -> TelemetrySnapshot:
        if self.failure:
            raise RuntimeError("system metrics failed")
        return TelemetrySnapshot(
            timestamp=datetime(2026, 1, 1, tzinfo=UTC),
            system=SystemMetrics(20, 4_000, 10_000, 40),
            gpus=(),
            gpu_telemetry_available=False,
            gpu_telemetry_error="NVML unavailable",
        )

    def close(self) -> None:
        self.closed = True


def sample_result() -> WorkerResult:
    return WorkerResult(
        detections=(Detection(BoundingBox(1, 2, 3, 4), "object", 0.9, 1),),
        timing=TimingMetrics(1, 2, 3, 6),
        worker_id="remote-test",
        worker_type="remote",
        backend="fake",
    )


def encoded_image() -> bytes:
    success, encoded = cv2.imencode(".png", np.zeros((6, 8, 3), dtype=np.uint8))
    assert success
    return encoded.tobytes()


@asynccontextmanager
async def api_client(
    worker_factory, settings=None, telemetry_factory=lambda: FakeTelemetry()
) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(worker_factory, settings, telemetry_factory)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://worker.test") as client:
            yield client


async def test_health_reports_ready_after_initialization() -> None:
    settings = ServiceSettings(worker_id="remote-test", backend="fake")
    async with api_client(lambda: FakeWorker(), settings) as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "worker_id": "remote-test",
        "worker_type": "remote",
        "backend": "fake",
        "model_ready": True,
    }


async def test_health_reports_not_ready_when_initialization_fails() -> None:
    def fail() -> FakeWorker:
        raise RuntimeError("weights unavailable")

    async with api_client(fail) as client:
        response = await client.get("/health")
        inference = await client.post(
            "/infer", content=encoded_image(), headers={"content-type": "image/png"}
        )

    assert response.json()["status"] == "not_ready"
    assert response.json()["model_ready"] is False
    assert inference.status_code == 503


async def test_infer_returns_worker_result() -> None:
    async with api_client(lambda: FakeWorker()) as client:
        response = await client.post(
            "/infer", content=encoded_image(), headers={"content-type": "image/png"}
        )

    assert response.status_code == 200
    assert response.json()["worker_id"] == "remote-test"
    assert response.json()["detection_count"] == 1
    assert response.json()["timing"]["total_ms"] == 6


async def test_infer_rejects_missing_or_empty_input() -> None:
    async with api_client(lambda: FakeWorker()) as client:
        missing = await client.post("/infer")
        empty = await client.post("/infer", content=b"", headers={"content-type": "image/jpeg"})

    assert missing.status_code == 400
    assert empty.status_code == 400


async def test_infer_rejects_invalid_image() -> None:
    async with api_client(lambda: FakeWorker()) as client:
        response = await client.post(
            "/infer", content=b"not an image", headers={"content-type": "image/jpeg"}
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "invalid encoded image"


async def test_infer_does_not_expose_inference_failure_details() -> None:
    async with api_client(lambda: FailingWorker()) as client:
        response = await client.post(
            "/infer", content=encoded_image(), headers={"content-type": "image/png"}
        )

    assert response.status_code == 500
    assert response.json() == {"detail": "inference failed"}


async def test_telemetry_returns_system_only_snapshot() -> None:
    async with api_client(lambda: FakeWorker()) as client:
        response = await client.get("/telemetry")

    assert response.status_code == 200
    assert response.json()["system"]["cpu_utilization_percent"] == 20
    assert response.json()["gpus"] == []
    assert response.json()["gpu_telemetry_available"] is False


async def test_telemetry_collector_failure_returns_service_error() -> None:
    async with api_client(
        lambda: FakeWorker(), telemetry_factory=lambda: FakeTelemetry(failure=True)
    ) as client:
        response = await client.get("/telemetry")

    assert response.status_code == 500
    assert response.json() == {"detail": "telemetry collection failed"}
