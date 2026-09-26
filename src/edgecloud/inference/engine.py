from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from edgecloud.inference.models import Detection, InferenceResult, TimingMetrics

Image = NDArray[np.uint8]


class InferenceBackend(Protocol):
    """Model-specific operations used by the measured inference pipeline."""

    def preprocess(self, image: Image) -> Any: ...

    def infer(self, model_input: Any) -> Any: ...

    def postprocess(
        self, raw_output: Any, image_shape: tuple[int, ...], confidence: float
    ) -> tuple[Detection, ...]: ...


@dataclass(slots=True)
class InferenceEngine:
    backend: InferenceBackend
    confidence: float = 0.5
    clock: Callable[[], float] = perf_counter

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")

    def run(
        self, image: Image, *, source: str | None = None, frame_index: int | None = None
    ) -> InferenceResult:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("expected an HxWx3 BGR image")

        started = self.clock()
        model_input = self.backend.preprocess(image)
        preprocessed = self.clock()
        raw_output = self.backend.infer(model_input)
        inferred = self.clock()
        detections = self.backend.postprocess(raw_output, image.shape, self.confidence)
        completed = self.clock()

        timing = TimingMetrics(
            preprocessing_ms=(preprocessed - started) * 1000,
            inference_ms=(inferred - preprocessed) * 1000,
            postprocessing_ms=(completed - inferred) * 1000,
            total_ms=(completed - started) * 1000,
        )
        return InferenceResult(detections, timing, source, frame_index)
