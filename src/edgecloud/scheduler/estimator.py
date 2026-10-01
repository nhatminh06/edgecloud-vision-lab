from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class EwmaEstimator:
    alpha: float
    estimate: float | None = None
    sample_count: int = 0

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be greater than 0 and at most 1")

    def observe(self, value: float) -> float:
        if value < 0:
            raise ValueError("EWMA observations cannot be negative")
        if self.estimate is None:
            self.estimate = value
        else:
            self.estimate = self.alpha * value + (1 - self.alpha) * self.estimate
        self.sample_count += 1
        return self.estimate
