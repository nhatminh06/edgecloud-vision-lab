from __future__ import annotations

import httpx
import numpy as np
import pytest

from edgecloud.workers.errors import (
    WorkerConnectionError,
    WorkerHTTPError,
    WorkerResponseError,
    WorkerTimeoutError,
    WorkerUnavailableError,
)
from edgecloud.workers.models import HealthStatus
from edgecloud.workers.remote import RemoteWorker


def result_payload() -> dict[str, object]:
    return {
        "detections": [
            {
                "box": {"x1": 1, "y1": 2, "x2": 3, "y2": 4},
                "label": "object",
                "score": 0.9,
                "class_id": 1,
            }
        ],
        "timing": {
            "preprocessing_ms": 1,
            "inference_ms": 2,
            "postprocessing_ms": 3,
            "total_ms": 6,
        },
        "worker_id": "remote-test",
        "worker_type": "remote",
        "backend": "fake",
        "detection_count": 1,
    }


def worker_for(handler: httpx.MockTransport) -> RemoteWorker:
    client = httpx.Client(transport=handler, base_url="http://worker.test")
    return RemoteWorker("http://worker.test", client=client)


def test_remote_worker_parses_successful_result_and_records_round_trip() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=result_payload()))
    worker = worker_for(transport)

    result = worker.infer(np.zeros((8, 8, 3), dtype=np.uint8))

    assert result.worker_id == "remote-test"
    assert result.detection_count == 1
    assert result.timing.total_ms == 6
    assert result.round_trip_ms is not None
    assert result.round_trip_ms >= 0


def test_remote_worker_reports_timeout() -> None:
    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    with pytest.raises(WorkerTimeoutError):
        worker_for(httpx.MockTransport(timeout)).infer(np.zeros((8, 8, 3), dtype=np.uint8))


def test_remote_worker_reports_connection_failure() -> None:
    def connection_error(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("failed", request=request)

    with pytest.raises(WorkerConnectionError):
        worker_for(httpx.MockTransport(connection_error)).infer(np.zeros((8, 8, 3), dtype=np.uint8))


def test_remote_worker_reports_http_error() -> None:
    worker = worker_for(httpx.MockTransport(lambda request: httpx.Response(429)))

    with pytest.raises(WorkerHTTPError) as raised:
        worker.infer(np.zeros((8, 8, 3), dtype=np.uint8))

    assert raised.value.status_code == 429


def test_remote_worker_reports_unhealthy_service() -> None:
    worker = worker_for(httpx.MockTransport(lambda request: httpx.Response(503)))

    with pytest.raises(WorkerUnavailableError):
        worker.infer(np.zeros((8, 8, 3), dtype=np.uint8))


def test_remote_worker_reports_malformed_json() -> None:
    response = httpx.Response(200, content=b"not-json")
    worker = worker_for(httpx.MockTransport(lambda request: response))

    with pytest.raises(WorkerResponseError, match="malformed JSON"):
        worker.infer(np.zeros((8, 8, 3), dtype=np.uint8))


def test_remote_worker_reports_structurally_invalid_response() -> None:
    worker = worker_for(httpx.MockTransport(lambda request: httpx.Response(200, json={})))

    with pytest.raises(WorkerResponseError, match="invalid result"):
        worker.infer(np.zeros((8, 8, 3), dtype=np.uint8))


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (
            {
                "status": "ready",
                "model_ready": True,
                "worker_id": "remote-test",
                "backend": "fake",
            },
            HealthStatus.READY,
        ),
        (
            {"status": "not_ready", "model_ready": False, "worker_id": "remote-test"},
            HealthStatus.NOT_READY,
        ),
    ],
)
def test_remote_health_distinguishes_ready_state(
    payload: dict[str, object], expected: HealthStatus
) -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=payload))

    assert worker_for(transport).health().status is expected


def test_remote_health_reports_unreachable() -> None:
    def connection_error(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("failed", request=request)

    result = worker_for(httpx.MockTransport(connection_error)).health()

    assert result.status is HealthStatus.UNREACHABLE
    assert result.round_trip_ms is None
