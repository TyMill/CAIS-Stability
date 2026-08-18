"""Core trajectory-level governance primitives for CAIS-Stability."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class GovernanceMode(str, Enum):
    AUTO = "auto"
    CONSTRAIN = "constrain"
    RECOVER = "recover"
    FALLBACK = "fallback"


@dataclass(frozen=True, slots=True)
class SystemState:
    risk: float
    degradation: float
    uncertainty: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (("risk", self.risk), ("degradation", self.degradation), ("uncertainty", self.uncertainty)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")


@dataclass(frozen=True, slots=True)
class GovernanceConfig:
    constrain_threshold: float = 0.35
    recover_threshold: float = 0.55
    fallback_threshold: float = 0.85
    max_action: float = 1.0
    recovery_action: float = -0.6
    fallback_action: float = -1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.constrain_threshold <= self.recover_threshold:
            raise ValueError("invalid constrain/recover thresholds")
        if not self.recover_threshold <= self.fallback_threshold <= 1.0:
            raise ValueError("invalid fallback threshold")
        if self.max_action <= 0.0:
            raise ValueError("max_action must be positive")


@dataclass(frozen=True, slots=True)
class GovernanceDecision:
    action: float
    proposed_action: float
    mode: GovernanceMode
    modified: bool


class DynamicGovernance:
    """Domain-independent deterministic reference governance operator."""

    def __init__(self, config: GovernanceConfig | None = None) -> None:
        self.config = config or GovernanceConfig()

    @staticmethod
    def effective_risk(state: SystemState) -> float:
        return min(1.0, state.risk + 0.25 * state.degradation + 0.25 * state.uncertainty)

    def decide(self, state: SystemState, proposed_action: float) -> GovernanceDecision:
        cfg = self.config
        risk = self.effective_risk(state)
        proposal = max(-cfg.max_action, min(cfg.max_action, proposed_action))

        if risk >= cfg.fallback_threshold:
            action, mode = cfg.fallback_action, GovernanceMode.FALLBACK
        elif risk >= cfg.recover_threshold or state.degradation >= cfg.recover_threshold:
            action, mode = cfg.recovery_action, GovernanceMode.RECOVER
        elif risk >= cfg.constrain_threshold:
            action, mode = proposal * max(0.0, 1.0 - risk), GovernanceMode.CONSTRAIN
        else:
            action, mode = proposal, GovernanceMode.AUTO

        return GovernanceDecision(
            action=action,
            proposed_action=proposed_action,
            mode=mode,
            modified=abs(action - proposed_action) > 1e-12,
        )
