from __future__ import annotations

from threading import Event

from edgecloud.scheduler import HealthCache, HealthSample, HealthState, WorkerHealth


class FakeClock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


def test_health_cache_starts_unknown_and_uses_local_receipt_freshness() -> None:
    clock = FakeClock(10)
    cache = HealthCache(2, clock)

    assert cache.get().state is WorkerHealth.UNKNOWN
    cache.mark_unhealthy("timeout")
    clock.value = 11
    fresh = cache.get()
    clock.value = 13
    stale = cache.get()

    assert fresh.blocks_routing is True
    assert fresh.age_seconds == 1
    assert stale.state is WorkerHealth.UNHEALTHY
    assert stale.fresh is False
    assert stale.blocks_routing is False


def test_manual_health_refresh_recovers_worker_without_sleeping() -> None:
    remote_state = WorkerHealth.UNHEALTHY

    def remote_source() -> HealthSample:
        return HealthSample(remote_state)

    state = HealthState(
        lambda: HealthSample(WorkerHealth.HEALTHY),
        remote_source,
        interval_seconds=1,
        max_age_seconds=2,
    )
    state.refresh_remote()
    assert state.remote().state is WorkerHealth.UNHEALTHY

    remote_state = WorkerHealth.HEALTHY
    state.refresh_remote()

    assert state.remote().state is WorkerHealth.HEALTHY
    assert state.remote().fresh is True


def test_health_sampler_starts_and_stops_cleanly() -> None:
    edge_sampled = Event()
    remote_sampled = Event()

    def sample(event: Event) -> HealthSample:
        event.set()
        return HealthSample(WorkerHealth.HEALTHY)

    state = HealthState(
        lambda: sample(edge_sampled),
        lambda: sample(remote_sampled),
        interval_seconds=60,
        max_age_seconds=120,
    ).start()
    assert edge_sampled.wait(1)
    assert remote_sampled.wait(1)
    assert state.running is True

    state.close()
    state.close()

    assert state.running is False
