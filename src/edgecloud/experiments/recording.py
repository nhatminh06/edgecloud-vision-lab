from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import Protocol, TextIO

from edgecloud.experiments.events import ExperimentEvent


class ExperimentRecorder(Protocol):
    def record(self, event: ExperimentEvent) -> None: ...

    def close(self) -> None: ...


class JsonlRecorder:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._handle: TextIO = path.open("w", encoding="utf-8", newline="\n")
        self._closed = False

    def record(self, event: ExperimentEvent) -> None:
        if self._closed:
            raise ValueError("recorder is closed")
        self._handle.write(event.to_json() + "\n")
        self._handle.flush()

    def close(self) -> None:
        if not self._closed:
            self._handle.close()
            self._closed = True

    def __enter__(self) -> JsonlRecorder:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
