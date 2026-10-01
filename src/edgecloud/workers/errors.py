class WorkerError(RuntimeError):
    """Base class for worker failures."""


class WorkerExecutionError(WorkerError):
    """A worker accepted an inference request but could not execute it."""


class WorkerTimeoutError(WorkerError):
    pass


class WorkerConnectionError(WorkerError):
    pass


class WorkerHTTPError(WorkerError):
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(f"remote worker returned HTTP {status_code}")


class WorkerResponseError(WorkerError):
    pass


class WorkerUnavailableError(WorkerError):
    pass
