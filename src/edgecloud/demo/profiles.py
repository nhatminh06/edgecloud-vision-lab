from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DemoProfile:
    edge_delay_ms: float = 0.0
    remote_delay_ms: float = 0.0
    edge_available: bool = True
    remote_available: bool = True


DEMO_PROFILES: dict[str, DemoProfile] = {
    "normal": DemoProfile(),
    "edge_hot": DemoProfile(edge_delay_ms=150.0),
    "cloud_slow": DemoProfile(remote_delay_ms=200.0),
    "cloud_down": DemoProfile(remote_available=False),
    "recovered": DemoProfile(),
}


def get_demo_profile(name: str) -> DemoProfile:
    try:
        return DEMO_PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(DEMO_PROFILES)
        raise ValueError(f"unknown demo profile {name!r}; choose from: {choices}") from exc
