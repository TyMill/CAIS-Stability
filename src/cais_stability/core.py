"""Core trajectory-level governance primitives for CAIS-Stability."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class GovernanceMode(StrEnum):
    """Execution modes used by the dynamic governance operator."""

    AUTO = "auto"
    CONSTRAIN = "constrain"
    RECOVER = "recover"
    FALLBACK = "fallback"


@dataclass(frozen=True, slots=True)
class SystemState:
    """Normalized state exposed to the domain-independent governance layer."""

    risk: float
    degradation: float
    uncertainty: float = 0.0

    def __post_init__(self) -> None:
        values = (
            ("risk", self.risk),
            ("degradation", self.degradation),
            ("uncertainty", self.uncertainty),
        )
        for name, value in values:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")


@dataclass(frozen=True, slots=True)
class GovernanceConfig:
    """Thresholds and actions for the reference dynamic CAIS operator."""

    constrain_threshold: float = 0.35
    recover_threshold: float = 0.55
    fallback_threshold: float = 0.85
    hysteresis: float = 0.05
    max_action: float = 1.0
    recovery_action: float = -0.6
    fallback_action: float = -1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.constrain_threshold <= self.recover_threshold:
            raise ValueError("invalid constrain/recover thresholds")
        if not self.recover_threshold <= self.fallback_threshold <= 1.0:
            raise ValueError("invalid fallback threshold")
        if not 0.0 <= self.hysteresis < self.constrain_threshold:
            raise ValueError(
                "hysteresis must be non-negative and smaller than constrain threshold"
            )
        if self.max_action <= 0.0:
            raise ValueError("max_action must be positive")
        if not -self.max_action <= self.recovery_action <= self.max_action:
            raise ValueError("recovery_action must lie within action bounds")
        if not -self.max_action <= self.fallback_action <= self.max_action:
            raise ValueError("fallback_action must lie within action bounds")


@dataclass(frozen=True, slots=True)
class GovernanceDecision:
    """One governance decision and its relationship to the agent proposal."""

    action: float
    proposed_action: float
    mode: GovernanceMode
    modified: bool


class DynamicGovernance:
    """Domain-independent deterministic trajectory-aware governance operator."""

    def __init__(self, config: GovernanceConfig | None = None) -> None:
        self.config = config or GovernanceConfig()

    @staticmethod
    def effective_risk(state: SystemState) -> float:
        """Fuse risk, degradation, and uncertainty into a normalized control signal."""
        return min(
            1.0,
            state.risk + 0.25 * state.degradation + 0.25 * state.uncertainty,
        )

    def decide(
        self,
        state: SystemState,
        proposed_action: float,
        previous_mode: GovernanceMode | None = None,
    ) -> GovernanceDecision:
        """Select an execution mode and governed action."""
        cfg = self.config
        risk = self.effective_risk(state)
        proposal = max(-cfg.max_action, min(cfg.max_action, proposed_action))

        hold_fallback = (
            previous_mode is GovernanceMode.FALLBACK
            and risk >= cfg.fallback_threshold - cfg.hysteresis
        )
        hold_recovery = (
            previous_mode in {GovernanceMode.RECOVER, GovernanceMode.FALLBACK}
            and max(risk, state.degradation) >= cfg.recover_threshold - cfg.hysteresis
        )
        hold_constraint = (
            previous_mode
            in {
                GovernanceMode.CONSTRAIN,
                GovernanceMode.RECOVER,
                GovernanceMode.FALLBACK,
            }
            and risk >= cfg.constrain_threshold - cfg.hysteresis
        )

        if risk >= cfg.fallback_threshold or hold_fallback:
            action, mode = cfg.fallback_action, GovernanceMode.FALLBACK
        elif (
            risk >= cfg.recover_threshold
            or state.degradation >= cfg.recover_threshold
            or hold_recovery
        ):
            action, mode = cfg.recovery_action, GovernanceMode.RECOVER
        elif risk >= cfg.constrain_threshold or hold_constraint:
            action, mode = proposal * max(0.0, 1.0 - risk), GovernanceMode.CONSTRAIN
        else:
            action, mode = proposal, GovernanceMode.AUTO

        return GovernanceDecision(
            action=action,
            proposed_action=proposed_action,
            mode=mode,
            modified=abs(action - proposed_action) > 1e-12,
        )
