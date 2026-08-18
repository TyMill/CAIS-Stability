"""Common trajectory-level metrics used across CAIS-Stability benchmarks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .core import GovernanceDecision, GovernanceMode, SystemState


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    hard_violation_rate: float
    recovery_success: bool
    recovery_time: int | None
    autonomy_retention: float
    intervention_rate: float
    fallback_rate: float
    mean_task_utility: float
    peak_risk: float
    peak_degradation: float


def summarize_episode(
    decisions: Sequence[GovernanceDecision],
    hard_violations: int,
    recovery_time: int | None,
    recovery_deadline: int,
    utilities: Sequence[float] | None = None,
    states: Sequence[SystemState] | None = None,
) -> EpisodeMetrics:
    if not decisions:
        raise ValueError("decisions must not be empty")
    if hard_violations < 0:
        raise ValueError("hard_violations must be non-negative")
    if recovery_deadline < 0:
        raise ValueError("recovery_deadline must be non-negative")
    if utilities is not None and len(utilities) != len(decisions):
        raise ValueError("utilities and decisions must have equal length")
    if states is not None and len(states) != len(decisions):
        raise ValueError("states and decisions must have equal length")
    n = len(decisions)
    autonomy = sum(d.mode is GovernanceMode.AUTO for d in decisions) / n
    interventions = sum(d.modified for d in decisions) / n
    fallbacks = sum(d.mode is GovernanceMode.FALLBACK for d in decisions) / n
    recovered = recovery_time is not None and recovery_time <= recovery_deadline
    mean_utility = sum(utilities) / n if utilities is not None else 0.0
    peak_risk = max((state.risk for state in states), default=0.0)
    peak_degradation = max((state.degradation for state in states), default=0.0)
    return EpisodeMetrics(
        hard_violation_rate=hard_violations / n,
        recovery_success=recovered,
        recovery_time=recovery_time,
        autonomy_retention=autonomy,
        intervention_rate=interventions,
        fallback_rate=fallbacks,
        mean_task_utility=mean_utility,
        peak_risk=peak_risk,
        peak_degradation=peak_degradation,
    )
