from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from edgecloud.inference.engine import InferenceEngine
from edgecloud.inference.models import BoundingBox, Detection


class FakeBackend:
    def preprocess(self, image: np.ndarray) -> tuple[int, ...]:
        return image.shape

    def infer(self, model_input: tuple[int, ...]) -> tuple[int, ...]:
        return model_input

    def postprocess(
        self, raw_output: tuple[int, ...], image_shape: tuple[int, ...], confidence: float
    ) -> tuple[Detection, ...]:
        assert raw_output == image_shape
        assert confidence == 0.5
        return (Detection(BoundingBox(1, 2, 3, 4), "object", 0.9, 1),)


def ticking_clock() -> Iterator[float]:
    yield from (10.0, 10.001, 10.004, 10.006)


def test_engine_returns_structured_result_and_phase_timings() -> None:
    ticks = ticking_clock()
    engine = InferenceEngine(FakeBackend(), clock=lambda: next(ticks))

    result = engine.run(np.zeros((8, 12, 3), dtype=np.uint8), source="sample", frame_index=7)

    assert result.detection_count == 1
    assert result.source == "sample"
    assert result.frame_index == 7
    assert result.timing.preprocessing_ms == pytest.approx(1.0)
    assert result.timing.inference_ms == pytest.approx(3.0)
    assert result.timing.postprocessing_ms == pytest.approx(2.0)
    assert result.timing.total_ms == pytest.approx(6.0)


def test_engine_rejects_non_color_input() -> None:
    engine = InferenceEngine(FakeBackend())

    with pytest.raises(ValueError, match="HxWx3"):
        engine.run(np.zeros((8, 12), dtype=np.uint8))
