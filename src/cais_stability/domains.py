"""Deterministic domain adapters for the initial cross-domain benchmark scaffold."""

from __future__ import annotations

from dataclasses import dataclass

from .core import SystemState


@dataclass(frozen=True, slots=True)
class DomainScenario:
    name: str
    initial_risk: float
    initial_degradation: float
    uncertainty: float

    def state(self) -> SystemState:
        return SystemState(
            risk=self.initial_risk,
            degradation=self.initial_degradation,
            uncertainty=self.uncertainty,
        )


def default_scenarios() -> tuple[DomainScenario, ...]:
    """Return one normalized smoke scenario per target application domain."""
    return (
        DomainScenario("maritime", 0.42, 0.20, 0.10),
        DomainScenario("supply_chain", 0.50, 0.45, 0.15),
        DomainScenario("smart_grid", 0.68, 0.60, 0.20),
    )
