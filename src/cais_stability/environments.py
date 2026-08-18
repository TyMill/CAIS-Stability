"""Cross-domain normalized benchmark dynamics for CAIS-Stability."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

from .core import SystemState
from .domains import DOMAIN_PARAMETERS, DomainName, DomainParameters


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass(frozen=True, slots=True)
class Transition:
    state: SystemState
    task_utility: float
    hard_violation: bool


class NormalizedEnvironment:
    """Deterministic-under-seed normalized dynamics shared across all domains."""

    def __init__(self, domain: DomainName) -> None:
        self.domain = domain
        self.parameters = DOMAIN_PARAMETERS[domain]
        self.state = self.parameters.initial_state

    def reset(self) -> SystemState:
        self.state = self.parameters.initial_state
        return self.state

    def is_nominal(self, state: SystemState | None = None) -> bool:
        current = state or self.state
        p = self.parameters
        return current.risk <= p.nominal_risk and current.degradation <= p.nominal_degradation

    def is_unsafe(self, state: SystemState | None = None) -> bool:
        current = state or self.state
        p = self.parameters
        return current.risk >= p.unsafe_risk or current.degradation >= p.unsafe_degradation

    def observe(self, observation_error: float, risk_bias: float = 0.0) -> SystemState:
        risk = _clip01(self.state.risk + observation_error + risk_bias)
        degradation = _clip01(self.state.degradation + 0.5 * observation_error)
        uncertainty = _clip01(self.state.uncertainty + abs(observation_error))
        return SystemState(risk, degradation, uncertainty)

    def agent_proposal(self, rng: Random, agent_degradation: float = 0.0) -> float:
        jitter = rng.uniform(-0.08, 0.08) * (1.0 + agent_degradation)
        aggressiveness = 0.76 + 0.22 * agent_degradation
        risk_response = 0.20 * self.state.risk * (1.0 - agent_degradation)
        return max(-1.0, min(1.0, aggressiveness - risk_response + jitter))

    def step(self, action: float, shock: float, observation_noise: float) -> Transition:
        p = self.parameters
        positive = max(0.0, action)
        mitigation = max(0.0, -action)
        next_risk = _clip01(
            p.risk_persistence * self.state.risk
            + p.disturbance_risk * shock
            + p.positive_action_risk * positive
            + p.uncertainty_risk * self.state.uncertainty
            - p.recovery_risk * mitigation
            - p.natural_recovery
        )
        excess_risk = max(0.0, self.state.risk - p.nominal_risk)
        next_degradation = _clip01(
            p.degradation_persistence * self.state.degradation
            + p.disturbance_degradation * shock
            + p.risk_to_degradation * excess_risk
            - p.recovery_degradation * mitigation
            - p.natural_recovery
        )
        next_uncertainty = _clip01(0.65 * self.state.uncertainty + abs(observation_noise))
        next_state = SystemState(next_risk, next_degradation, next_uncertainty)
        desired_action = 0.75
        action_utility = max(0.0, 1.0 - abs(action - desired_action) / 1.75)
        state_utility = max(0.0, 1.0 - 0.45 * next_risk - 0.35 * next_degradation)
        utility = action_utility * state_utility
        self.state = next_state
        return Transition(next_state, utility, self.is_unsafe(next_state))

    @property
    def parameter_view(self) -> DomainParameters:
        return self.parameters
