from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass

import cv2
import numpy as np
from fastapi import Body, FastAPI, HTTPException, Request

from edgecloud.inference.engine import InferenceEngine
from edgecloud.inference.torchvision_backend import TorchvisionSSDLiteBackend
from edgecloud.workers.edge import EdgeWorker
from edgecloud.workers.models import InferenceWorker

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ServiceSettings:
    worker_id: str = "remote-1"
    worker_type: str = "remote"
    backend: str = "torchvision-ssdlite320-mobilenet-v3"
    device: str = "cpu"
    confidence: float = 0.5


WorkerFactory = Callable[[], InferenceWorker]


def default_worker_factory(settings: ServiceSettings) -> WorkerFactory:
    def create_worker() -> EdgeWorker:
        engine = InferenceEngine(
            TorchvisionSSDLiteBackend(device=settings.device), confidence=settings.confidence
        )
        return EdgeWorker(
            engine=engine,
            worker_id=settings.worker_id,
            worker_type=settings.worker_type,
            backend=settings.backend,
        )

    return create_worker


def create_app(
    worker_factory: WorkerFactory | None = None,
    settings: ServiceSettings | None = None,
) -> FastAPI:
    configured = settings or ServiceSettings()
    factory = worker_factory or default_worker_factory(configured)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            app.state.worker = factory()
            app.state.initialization_error = None
        except (OSError, RuntimeError, ValueError) as exc:
            app.state.worker = None
            app.state.initialization_error = str(exc)
            LOGGER.exception("inference worker initialization failed")
        yield

    app = FastAPI(title="EdgeCloud Vision Worker", lifespan=lifespan)
    app.state.worker = None
    app.state.initialization_error = None

    @app.get("/health")
    async def health() -> dict[str, object]:
        ready = app.state.worker is not None
        return {
            "status": "ready" if ready else "not_ready",
            "worker_id": configured.worker_id,
            "worker_type": configured.worker_type,
            "backend": configured.backend,
            "model_ready": ready,
        }

    @app.post("/infer")
    async def infer(request: Request, image_bytes: bytes = Body(default=b"")) -> dict[str, object]:
        worker = app.state.worker
        if worker is None:
            raise HTTPException(status_code=503, detail="inference worker is not ready")
        content_type = request.headers.get("content-type", "").split(";", maxsplit=1)[0]
        if (
            content_type
            and content_type != "application/octet-stream"
            and not content_type.startswith("image/")
        ):
            raise HTTPException(status_code=415, detail="unsupported media type")
        if not image_bytes:
            raise HTTPException(status_code=400, detail="image body is empty")
        encoded = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="invalid encoded image")
        try:
            return worker.infer(image).to_dict()
        except (OSError, RuntimeError, ValueError) as exc:
            LOGGER.exception("inference request failed")
            raise HTTPException(status_code=500, detail="inference failed") from exc

    return app


app = create_app()
