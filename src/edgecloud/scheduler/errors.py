class SchedulerUnavailableError(RuntimeError):
    """No worker is currently eligible to accept a request."""


class ResilientRequestError(RuntimeError):
    """A resilient request exhausted its permitted worker attempts."""
