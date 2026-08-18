from __future__ import annotations

from cais_stability.calibration import calibration_points, recommend_intensities
from cais_stability.domains import DomainName
from cais_stability.experiments import CampaignConfig, run_campaign


def _records():
    config = CampaignConfig(
        domains=(DomainName.MARITIME,),
        intensities=(0.0, 0.5, 1.0),
        seeds=(0, 1),
        horizon=25,
        disturbance_onset=4,
        disturbance_duration=3,
    )
    return run_campaign(config)


def test_calibration_points_cover_each_intensity() -> None:
    points = calibration_points(_records())
    assert {point.intensity for point in points} == {0.0, 0.5, 1.0}
    assert all(0.0 <= point.informative_score <= 1.0 for point in points)


def test_recommendation_keeps_boundary_intensities() -> None:
    recommendation = recommend_intensities(_records(), count=2)
    assert recommendation.intensities == (0.0, 1.0)


def test_recommendation_is_deterministic() -> None:
    records = _records()
    first = recommend_intensities(records, count=3)
    second = recommend_intensities(records, count=3)
    assert first == second
