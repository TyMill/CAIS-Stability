"""Reference governance baselines for the CAIS-Stability article benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .core import DynamicGovernance, GovernanceConfig, GovernanceDecision, GovernanceMode, SystemState


class GovernancePolicy(Protocol):
    name: str
    def reset(self) -> None: ...
    def decide(self, state: SystemState, proposed_action: float) -> GovernanceDecision: ...


def _clip_action(action: float, max_action: float) -> float:
    return max(-max_action, min(max_action, action))


class UngovernedPolicy:
    name = "ungoverned"
    def __init__(self, max_action: float = 1.0) -> None:
        self.max_action = max_action
    def reset(self) -> None:
        return None
    def decide(self, state: SystemState, proposed_action: float) -> GovernanceDecision:
        del state
        action = _clip_action(proposed_action, self.max_action)
        return GovernanceDecision(action, proposed_action, GovernanceMode.AUTO, abs(action-proposed_action)>1e-12)


class StaticCAISPolicy:
    name = "static_cais"
    def __init__(self, config: GovernanceConfig | None = None) -> None:
        self.config = config or GovernanceConfig()
    def reset(self) -> None:
        return None
    def decide(self, state: SystemState, proposed_action: float) -> GovernanceDecision:
        cfg = self.config
        proposal = _clip_action(proposed_action, cfg.max_action)
        risk = DynamicGovernance.effective_risk(state)
        if risk >= cfg.constrain_threshold:
            action, mode = proposal * max(0.0, 1.0-risk), GovernanceMode.CONSTRAIN
        else:
            action, mode = proposal, GovernanceMode.AUTO
        return GovernanceDecision(action, proposed_action, mode, abs(action-proposed_action)>1e-12)


class BinaryRTAPolicy:
    name = "binary_rta"
    def __init__(self, threshold: float = 0.65, fallback_action: float = -1.0, max_action: float = 1.0) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must lie in [0, 1]")
        self.threshold, self.fallback_action, self.max_action = threshold, fallback_action, max_action
    def reset(self) -> None:
        return None
    def decide(self, state: SystemState, proposed_action: float) -> GovernanceDecision:
        risk = DynamicGovernance.effective_risk(state)
        if max(risk, state.degradation) >= self.threshold:
            action, mode = self.fallback_action, GovernanceMode.FALLBACK
        else:
            action, mode = _clip_action(proposed_action, self.max_action), GovernanceMode.AUTO
        return GovernanceDecision(action, proposed_action, mode, abs(action-proposed_action)>1e-12)


@dataclass(frozen=True, slots=True)
class DynamicPolicyOptions:
    use_uncertainty: bool = True
    enable_recovery: bool = True
    enable_fallback: bool = True
    use_hysteresis: bool = True


class DynamicCAISPolicy:
    name = "dynamic_cais"
    def __init__(self, config: GovernanceConfig | None = None, *, options: DynamicPolicyOptions | None = None, name: str | None = None) -> None:
        self.governance = DynamicGovernance(config)
        self.options = options or DynamicPolicyOptions()
        self.name = name or type(self).name
        self._previous_mode: GovernanceMode | None = None
    def reset(self) -> None:
        self._previous_mode = None
    def decide(self, state: SystemState, proposed_action: float) -> GovernanceDecision:
        control_state = state if self.options.use_uncertainty else SystemState(state.risk, state.degradation, 0.0)
        previous_mode = self._previous_mode if self.options.use_hysteresis else None
        decision = self.governance.decide(control_state, proposed_action, previous_mode)
        if decision.mode is GovernanceMode.FALLBACK and not self.options.enable_fallback:
            if self.options.enable_recovery:
                decision = GovernanceDecision(self.governance.config.recovery_action, proposed_action, GovernanceMode.RECOVER, True)
            else:
                risk = self.governance.effective_risk(control_state)
                proposal = _clip_action(proposed_action, self.governance.config.max_action)
                action = proposal * max(0.0, 1.0-risk)
                decision = GovernanceDecision(action, proposed_action, GovernanceMode.CONSTRAIN, abs(action-proposed_action)>1e-12)
        elif decision.mode is GovernanceMode.RECOVER and not self.options.enable_recovery:
            risk = self.governance.effective_risk(control_state)
            proposal = _clip_action(proposed_action, self.governance.config.max_action)
            action = proposal * max(0.0, 1.0-risk)
            decision = GovernanceDecision(action, proposed_action, GovernanceMode.CONSTRAIN, abs(action-proposed_action)>1e-12)
        self._previous_mode = decision.mode
        return decision


def ablation_policies() -> tuple[GovernancePolicy, ...]:
    return (
        DynamicCAISPolicy(),
        DynamicCAISPolicy(options=DynamicPolicyOptions(enable_recovery=False), name="dynamic_no_recovery"),
        DynamicCAISPolicy(options=DynamicPolicyOptions(use_uncertainty=False), name="dynamic_no_uncertainty"),
        DynamicCAISPolicy(options=DynamicPolicyOptions(use_hysteresis=False), name="dynamic_no_hysteresis"),
        DynamicCAISPolicy(options=DynamicPolicyOptions(enable_fallback=False), name="dynamic_no_fallback"),
    )


def default_policies() -> tuple[GovernancePolicy, ...]:
    return (UngovernedPolicy(), StaticCAISPolicy(), BinaryRTAPolicy(), DynamicCAISPolicy())
