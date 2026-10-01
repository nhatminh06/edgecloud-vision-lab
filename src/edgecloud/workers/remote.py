from __future__ import annotations

import json
from dataclasses import dataclass, field
from time import perf_counter

import cv2
import httpx

from edgecloud.inference.engine import Image
from edgecloud.telemetry.models import TelemetrySnapshot
from edgecloud.workers.errors import (
    WorkerConnectionError,
    WorkerHTTPError,
    WorkerResponseError,
    WorkerTimeoutError,
    WorkerUnavailableError,
)
from edgecloud.workers.models import HealthResult, HealthStatus, WorkerResult


@dataclass(slots=True)
class RemoteWorker:
    base_url: str
    timeout_seconds: float = 10.0
    client: httpx.Client | None = None
    _client: httpx.Client = field(init=False, repr=False)
    _owns_client: bool = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._owns_client = self.client is None
        self._client = self.client or httpx.Client(
            base_url=self.base_url.rstrip("/"), timeout=self.timeout_seconds
        )

    def infer(self, image: Image) -> WorkerResult:
        encoded, buffer = cv2.imencode(".jpg", image)
        if not encoded:
            raise ValueError("image could not be encoded")
        started = perf_counter()
        try:
            response = self._client.post(
                "/infer", content=buffer.tobytes(), headers={"content-type": "image/jpeg"}
            )
        except httpx.TimeoutException as exc:
            raise WorkerTimeoutError("remote inference timed out") from exc
        except httpx.RequestError as exc:
            raise WorkerConnectionError("remote worker connection failed") from exc
        round_trip_ms = (perf_counter() - started) * 1000
        if response.status_code == 503:
            raise WorkerUnavailableError("remote worker is not ready")
        if not response.is_success:
            raise WorkerHTTPError(response.status_code)
        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise WorkerResponseError("remote worker returned malformed JSON") from exc
        try:
            return WorkerResult.from_dict(payload, round_trip_ms=round_trip_ms)
        except ValueError as exc:
            raise WorkerResponseError("remote worker returned an invalid result") from exc

    def health(self) -> HealthResult:
        started = perf_counter()
        try:
            response = self._client.get("/health")
        except (httpx.TimeoutException, httpx.RequestError):
            return HealthResult(status=HealthStatus.UNREACHABLE)
        round_trip_ms = (perf_counter() - started) * 1000
        if not response.is_success:
            return HealthResult(status=HealthStatus.NOT_READY, round_trip_ms=round_trip_ms)
        try:
            payload = response.json()
            ready = payload["status"] == "ready" and payload["model_ready"] is True
            worker_id = _optional_text(payload.get("worker_id"))
            backend = _optional_text(payload.get("backend"))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return HealthResult(status=HealthStatus.NOT_READY, round_trip_ms=round_trip_ms)
        return HealthResult(
            status=HealthStatus.READY if ready else HealthStatus.NOT_READY,
            worker_id=worker_id,
            backend=backend,
            round_trip_ms=round_trip_ms,
        )

    def telemetry(self) -> TelemetrySnapshot:
        try:
            response = self._client.get("/telemetry")
        except httpx.TimeoutException as exc:
            raise WorkerTimeoutError("remote telemetry request timed out") from exc
        except httpx.RequestError as exc:
            raise WorkerConnectionError("remote telemetry connection failed") from exc
        if not response.is_success:
            raise WorkerHTTPError(response.status_code)
        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise WorkerResponseError("remote worker returned malformed telemetry JSON") from exc
        try:
            return TelemetrySnapshot.from_dict(payload)
        except ValueError as exc:
            raise WorkerResponseError("remote worker returned invalid telemetry") from exc

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> RemoteWorker:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("expected non-empty text")
    return value
