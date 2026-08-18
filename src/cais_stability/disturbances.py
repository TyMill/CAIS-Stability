"""Runtime disturbance generation for reproducible CAIS-Stability experiments."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random


@dataclass(frozen=True, slots=True)
class DisturbanceConfig:
    intensity: float = 0.5
    onset: int = 10
    duration: int = 6
    observation_noise: float = 0.05
    monitoring_delay: int = 0
    risk_estimation_error: float = 0.0
    agent_degradation: float = 0.10

    def __post_init__(self) -> None:
        bounded_values = (
            ("intensity", self.intensity),
            ("observation_noise", self.observation_noise),
            ("agent_degradation", self.agent_degradation),
        )
        for name, value in bounded_values:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.onset < 0:
            raise ValueError("onset must be non-negative")
        if self.duration <= 0:
            raise ValueError("duration must be positive")
        if self.monitoring_delay < 0:
            raise ValueError("monitoring_delay must be non-negative")
        if not -1.0 <= self.risk_estimation_error <= 1.0:
            raise ValueError("risk_estimation_error must lie in [-1, 1]")

    @property
    def end(self) -> int:
        return self.onset + self.duration


@dataclass(frozen=True, slots=True)
class DisturbanceSample:
    active: bool
    shock: float
    observation_error: float


def sample_disturbance(
    step: int,
    config: DisturbanceConfig,
    rng: Random,
) -> DisturbanceSample:
    active = config.onset <= step < config.end
    if active:
        shock = min(1.0, config.intensity * (0.85 + 0.30 * rng.random()))
    else:
        shock = 0.0
    observation_error = config.observation_noise * rng.uniform(-1.0, 1.0)
    return DisturbanceSample(active, shock, observation_error)
