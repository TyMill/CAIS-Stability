from __future__ import annotations

import pytest

from cais_stability.core import DynamicGovernance, SystemState
from cais_stability.metrics import summarize_episode


def test_episode_metrics_are_bounded() -> None:
    gov = DynamicGovernance()
    decisions = [
        gov.decide(SystemState(0.1, 0.1), 0.5),
        gov.decide(SystemState(0.5, 0.2), 0.5),
        gov.decide(SystemState(0.8, 0.2, 0.2), 0.5),
    ]
    metrics = summarize_episode(decisions, hard_violations=0, recovery_time=2, recovery_deadline=3)
    assert metrics.recovery_success
    assert 0.0 <= metrics.autonomy_retention <= 1.0
    assert 0.0 <= metrics.intervention_rate <= 1.0
    assert 0.0 <= metrics.fallback_rate <= 1.0
    assert metrics.hard_violation_rate == pytest.approx(0.0)


def test_empty_episode_rejected() -> None:
    with pytest.raises(ValueError):
        summarize_episode([], hard_violations=0, recovery_time=None, recovery_deadline=3)
