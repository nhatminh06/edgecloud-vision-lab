from __future__ import annotations

from datetime import UTC, datetime
from threading import Event

from edgecloud.telemetry.cache import SnapshotCache, TelemetryState, TelemetryStatus
from edgecloud.telemetry.models import SystemMetrics, TelemetrySnapshot


class FakeClock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


def snapshot(cpu: float = 10) -> TelemetrySnapshot:
    return TelemetrySnapshot(
        datetime(2026, 1, 1, tzinfo=UTC),
        SystemMetrics(cpu, 1, 2, 50),
        (),
        False,
        "unavailable",
    )


def test_cache_stores_latest_snapshot_and_reports_age() -> None:
    clock = FakeClock(10)
    cache = SnapshotCache(lambda: snapshot(25), max_age_seconds=5, clock=clock)

    cache.refresh()
    clock.value = 12
    result = cache.get()

    assert result.snapshot == snapshot(25)
    assert result.status is TelemetryStatus.AVAILABLE
    assert result.age_seconds == 2


def test_cache_detects_stale_snapshot() -> None:
    clock = FakeClock(0)
    cache = SnapshotCache(snapshot, max_age_seconds=2, clock=clock)
    cache.refresh()
    clock.value = 3

    assert cache.get().status is TelemetryStatus.STALE


def test_failed_refresh_preserves_previous_snapshot_and_records_error() -> None:
    clock = FakeClock()
    failure = False

    def source() -> TelemetrySnapshot:
        if failure:
            raise RuntimeError("remote timeout")
        return snapshot()

    cache = SnapshotCache(source, max_age_seconds=2, clock=clock)
    cache.refresh()
    failure = True
    clock.value = 1
    cache.refresh()
    result = cache.get()

    assert result.snapshot == snapshot()
    assert result.status is TelemetryStatus.AVAILABLE
    assert result.last_error == "remote timeout"


def test_failed_initial_refresh_remains_unavailable() -> None:
    def fail() -> TelemetrySnapshot:
        raise RuntimeError("unavailable")

    cache = SnapshotCache(fail, max_age_seconds=2)
    cache.refresh()

    result = cache.get()
    assert result.snapshot is None
    assert result.status is TelemetryStatus.UNAVAILABLE
    assert result.last_error == "unavailable"


def test_telemetry_state_manual_refresh_and_clean_idempotent_shutdown() -> None:
    state = TelemetryState(snapshot, snapshot, interval_seconds=1, max_age_seconds=2)

    state.refresh_edge()
    state.refresh_remote()
    state.close()
    state.close()

    assert state.edge().status is TelemetryStatus.AVAILABLE
    assert state.remote().status is TelemetryStatus.AVAILABLE


def test_telemetry_state_background_sampler_starts_and_stops_cleanly() -> None:
    edge_sampled = Event()
    remote_sampled = Event()

    def edge_source() -> TelemetrySnapshot:
        edge_sampled.set()
        return snapshot()

    def remote_source() -> TelemetrySnapshot:
        remote_sampled.set()
        return snapshot()

    state = TelemetryState(
        edge_source, remote_source, interval_seconds=60, max_age_seconds=120
    ).start()
    assert edge_sampled.wait(1)
    assert remote_sampled.wait(1)
    assert state.running is True

    state.close()

    assert state.running is False
