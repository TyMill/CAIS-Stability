from __future__ import annotations

import pytest

from cais_stability.core import DynamicGovernance, GovernanceMode, SystemState


def test_auto_mode_preserves_safe_action() -> None:
    decision = DynamicGovernance().decide(SystemState(0.10, 0.10), 0.5)
    assert decision.mode is GovernanceMode.AUTO
    assert decision.action == pytest.approx(0.5)
    assert not decision.modified


def test_constrain_mode_attenuates_action() -> None:
    decision = DynamicGovernance().decide(SystemState(0.40, 0.10), 0.8)
    assert decision.mode is GovernanceMode.CONSTRAIN
    assert 0.0 < decision.action < 0.8
    assert decision.modified


def test_recover_mode_uses_recovery_action() -> None:
    decision = DynamicGovernance().decide(SystemState(0.50, 0.60), 0.8)
    assert decision.mode is GovernanceMode.RECOVER
    assert decision.action < 0.0


def test_fallback_mode_for_high_effective_risk() -> None:
    decision = DynamicGovernance().decide(SystemState(0.80, 0.20, 0.20), 0.8)
    assert decision.mode is GovernanceMode.FALLBACK
    assert decision.action == pytest.approx(-1.0)


def test_invalid_state_rejected() -> None:
    with pytest.raises(ValueError):
        SystemState(1.1, 0.0)
