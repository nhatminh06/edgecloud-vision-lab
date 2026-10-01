from __future__ import annotations

from dataclasses import dataclass

from edgecloud.demo.profiles import DEMO_PROFILES


@dataclass(frozen=True, slots=True)
class ScenarioPhase:
    start_frame: int
    profile: str

    def __post_init__(self) -> None:
        if self.start_frame < 0:
            raise ValueError("scenario phase frame cannot be negative")
        if self.profile not in DEMO_PROFILES:
            raise ValueError(f"unknown demo profile: {self.profile}")


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    phases: tuple[ScenarioPhase, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("scenario name cannot be empty")
        if not self.phases or self.phases[0].start_frame != 0:
            raise ValueError("scenario must begin at frame 0")
        starts = [phase.start_frame for phase in self.phases]
        if any(current <= previous for previous, current in zip(starts, starts[1:], strict=False)):
            raise ValueError("scenario phase frames must be strictly increasing")

    def profile_at(self, frame_index: int) -> str:
        if frame_index < 0:
            raise ValueError("frame index cannot be negative")
        active = self.phases[0].profile
        for phase in self.phases[1:]:
            if phase.start_frame > frame_index:
                break
            active = phase.profile
        return active


ADAPTIVE_FAILOVER = Scenario(
    name="adaptive-failover",
    phases=(
        ScenarioPhase(0, "normal"),
        ScenarioPhase(20, "edge_hot"),
        ScenarioPhase(50, "cloud_down"),
        ScenarioPhase(70, "recovered"),
    ),
)

SCENARIOS = {ADAPTIVE_FAILOVER.name: ADAPTIVE_FAILOVER}


def get_scenario(name: str) -> Scenario:
    try:
        return SCENARIOS[name]
    except KeyError as exc:
        choices = ", ".join(SCENARIOS)
        raise ValueError(f"unknown scenario {name!r}; choose from: {choices}") from exc
