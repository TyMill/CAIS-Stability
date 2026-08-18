"""Common trajectory-level metrics used across CAIS-Stability benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

from .core import GovernanceDecision, GovernanceMode


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    hard_violation_rate: float
    recovery_success: bool
    recovery_time: int | None
    autonomy_retention: float
    intervention_rate: float
    fallback_rate: float


def summarize_episode(
    decisions: list[GovernanceDecision],
    hard_violations: int,
    recovery_time: int | None,
    recovery_deadline: int,
) -> EpisodeMetrics:
    """Summarize one benchmark episode with domain-independent metrics."""
    if not decisions:
        raise ValueError("decisions must not be empty")
    if hard_violations < 0:
        raise ValueError("hard_violations must be non-negative")
    if recovery_deadline < 0:
        raise ValueError("recovery_deadline must be non-negative")

    n = len(decisions)
    autonomy = sum(d.mode is GovernanceMode.AUTO for d in decisions) / n
    interventions = sum(d.modified for d in decisions) / n
    fallbacks = sum(d.mode is GovernanceMode.FALLBACK for d in decisions) / n
    recovered = recovery_time is not None and recovery_time <= recovery_deadline

    return EpisodeMetrics(
        hard_violation_rate=hard_violations / n,
        recovery_success=recovered,
        recovery_time=recovery_time,
        autonomy_retention=autonomy,
        intervention_rate=interventions,
        fallback_rate=fallbacks,
    )
