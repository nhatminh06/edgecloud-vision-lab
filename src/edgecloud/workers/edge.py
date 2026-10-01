from __future__ import annotations

from dataclasses import dataclass

from edgecloud.inference.engine import Image, InferenceEngine
from edgecloud.workers.errors import WorkerExecutionError
from edgecloud.workers.models import WorkerResult


@dataclass(slots=True)
class EdgeWorker:
    engine: InferenceEngine
    worker_id: str = "edge-1"
    backend: str = "torchvision-ssdlite320-mobilenet-v3"
    worker_type: str = "edge"

    def infer(self, image: Image) -> WorkerResult:
        try:
            result = self.engine.run(image)
        except (OSError, RuntimeError) as exc:
            raise WorkerExecutionError("edge worker inference execution failed") from exc
        return WorkerResult(
            detections=result.detections,
            timing=result.timing,
            worker_id=self.worker_id,
            worker_type=self.worker_type,
            backend=self.backend,
        )
