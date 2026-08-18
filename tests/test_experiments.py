from __future__ import annotations

from cais_stability.baselines import DynamicCAISPolicy
from cais_stability.disturbances import DisturbanceConfig
from cais_stability.domains import DomainName
from cais_stability.experiments import (
    CampaignConfig,
    EpisodeConfig,
    aggregate_records,
    run_campaign,
    run_episode,
)


def test_episode_is_reproducible_under_seed() -> None:
    disturbance = DisturbanceConfig(intensity=0.75, onset=4, duration=3)
    config = EpisodeConfig(horizon=25, recovery_deadline=15, seed=4)
    first = run_episode(
        DomainName.MARITIME,
        DynamicCAISPolicy(),
        disturbance,
        config,
    )
    second = run_episode(
        DomainName.MARITIME,
        DynamicCAISPolicy(),
        disturbance,
        config,
    )
    assert first == second


def test_campaign_cardinality() -> None:
    config = CampaignConfig(
        domains=(DomainName.MARITIME,),
        intensities=(0.25, 0.75),
        seeds=(0, 1),
        horizon=20,
        disturbance_onset=4,
        disturbance_duration=3,
    )
    records = run_campaign(config)
    assert len(records) == 16


def test_campaign_metrics_are_bounded() -> None:
    config = CampaignConfig(
        intensities=(0.5,),
        seeds=(0,),
        horizon=25,
        disturbance_onset=4,
        disturbance_duration=3,
    )
    records = run_campaign(config)
    for record in records:
        assert 0.0 <= record.hard_violation_rate <= 1.0
        assert 0.0 <= record.autonomy_retention <= 1.0
        assert 0.0 <= record.intervention_rate <= 1.0
        assert 0.0 <= record.mean_task_utility <= 1.0


def test_aggregation_produces_one_row_per_method() -> None:
    config = CampaignConfig(
        domains=(DomainName.SMART_GRID,),
        intensities=(0.5,),
        seeds=(0, 1),
        horizon=20,
        disturbance_onset=4,
        disturbance_duration=3,
    )
    records = run_campaign(config)
    rows = aggregate_records(records)
    assert len(rows) == 4
    assert all(row["n"] == 2 for row in rows)
