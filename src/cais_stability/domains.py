"""Normalized domain definitions for cross-domain CAIS-Stability benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .core import SystemState


class DomainName(str, Enum):
    """Application domains used in the article benchmark."""

    MARITIME = "maritime"
    SUPPLY_CHAIN = "supply_chain"
    SMART_GRID = "smart_grid"


@dataclass(frozen=True, slots=True)
class DomainParameters:
    """Normalized dynamics shared by the generic benchmark environment."""

    risk_persistence: float
    degradation_persistence: float
    disturbance_risk: float
    disturbance_degradation: float
    positive_action_risk: float
    risk_to_degradation: float
    recovery_risk: float
    recovery_degradation: float
    uncertainty_risk: float
    natural_recovery: float
    nominal_risk: float
    nominal_degradation: float
    unsafe_risk: float
    unsafe_degradation: float
    initial_state: SystemState


@dataclass(frozen=True, slots=True)
class DomainScenario:
    """Named domain scenario retained for smoke verification and public API use."""

    name: DomainName
    parameters: DomainParameters

    def state(self) -> SystemState:
        return self.parameters.initial_state


DOMAIN_PARAMETERS: dict[DomainName, DomainParameters] = {
    DomainName.MARITIME: DomainParameters(
        risk_persistence=0.68,
        degradation_persistence=0.72,
        disturbance_risk=0.34,
        disturbance_degradation=0.18,
        positive_action_risk=0.14,
        risk_to_degradation=0.14,
        recovery_risk=0.34,
        recovery_degradation=0.26,
        uncertainty_risk=0.10,
        natural_recovery=0.025,
        nominal_risk=0.30,
        nominal_degradation=0.25,
        unsafe_risk=0.92,
        unsafe_degradation=0.96,
        initial_state=SystemState(0.18, 0.10, 0.04),
    ),
    DomainName.SUPPLY_CHAIN: DomainParameters(
        risk_persistence=0.76,
        degradation_persistence=0.82,
        disturbance_risk=0.24,
        disturbance_degradation=0.34,
        positive_action_risk=0.09,
        risk_to_degradation=0.18,
        recovery_risk=0.22,
        recovery_degradation=0.39,
        uncertainty_risk=0.08,
        natural_recovery=0.020,
        nominal_risk=0.30,
        nominal_degradation=0.25,
        unsafe_risk=0.92,
        unsafe_degradation=0.96,
        initial_state=SystemState(0.16, 0.12, 0.05),
    ),
    DomainName.SMART_GRID: DomainParameters(
        risk_persistence=0.72,
        degradation_persistence=0.76,
        disturbance_risk=0.39,
        disturbance_degradation=0.23,
        positive_action_risk=0.17,
        risk_to_degradation=0.16,
        recovery_risk=0.41,
        recovery_degradation=0.27,
        uncertainty_risk=0.12,
        natural_recovery=0.025,
        nominal_risk=0.30,
        nominal_degradation=0.25,
        unsafe_risk=0.92,
        unsafe_degradation=0.96,
        initial_state=SystemState(0.17, 0.09, 0.05),
    ),
}


def default_scenarios() -> tuple[DomainScenario, ...]:
    """Return one normalized scenario for each article benchmark domain."""
    return tuple(DomainScenario(name, DOMAIN_PARAMETERS[name]) for name in DomainName)
