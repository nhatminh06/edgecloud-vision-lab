from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from edgecloud.inference.engine import InferenceEngine
from edgecloud.inference.models import BoundingBox, Detection
from edgecloud.workers.edge import EdgeWorker


class FakeBackend:
    def preprocess(self, image: np.ndarray) -> tuple[int, ...]:
        return image.shape

    def infer(self, model_input: tuple[int, ...]) -> tuple[int, ...]:
        return model_input

    def postprocess(
        self, raw_output: tuple[int, ...], image_shape: tuple[int, ...], confidence: float
    ) -> tuple[Detection, ...]:
        return (Detection(BoundingBox(1, 2, 3, 4), "object", 0.8, 1),)


def ticking_clock() -> Iterator[float]:
    yield from (1.0, 1.001, 1.003, 1.006)


def test_edge_worker_adapts_inference_result() -> None:
    ticks = ticking_clock()
    worker = EdgeWorker(
        InferenceEngine(FakeBackend(), clock=lambda: next(ticks)),
        worker_id="edge-test",
        backend="fake",
    )

    result = worker.infer(np.zeros((8, 8, 3), dtype=np.uint8))

    assert result.worker_id == "edge-test"
    assert result.worker_type == "edge"
    assert result.backend == "fake"
    assert result.detection_count == 1
    assert result.timing.total_ms == pytest.approx(6.0)
    assert result.round_trip_ms is None
