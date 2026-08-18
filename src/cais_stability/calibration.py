"""Calibration utilities for selecting informative disturbance intensities."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean

from .experiments import EpisodeRecord


@dataclass(frozen=True, slots=True)
class CalibrationPoint:
    """Cross-method separation observed at one domain/intensity pair."""

    domain: str
    intensity: float
    hvr_spread: float
    recovery_spread: float
    autonomy_spread: float
    utility_spread: float
    informative_score: float


@dataclass(frozen=True, slots=True)
class CalibrationRecommendation:
    """Recommended intensities and the evidence used to select them."""

    intensities: tuple[float, ...]
    points: tuple[CalibrationPoint, ...]


def _spread(values: list[float]) -> float:
    return max(values) - min(values) if values else 0.0


def calibration_points(records: list[EpisodeRecord]) -> list[CalibrationPoint]:
    """Measure method separation for each domain and disturbance intensity."""
    if not records:
        raise ValueError("records must not be empty")

    groups: dict[tuple[str, float, str], list[EpisodeRecord]] = {}
    for record in records:
        key = (record.domain, record.intensity, record.method)
        groups.setdefault(key, []).append(record)

    domain_intensities = sorted({(row.domain, row.intensity) for row in records})
    points: list[CalibrationPoint] = []
    for domain, intensity in domain_intensities:
        methods = sorted(
            method
            for grouped_domain, grouped_intensity, method in groups
            if grouped_domain == domain and grouped_intensity == intensity
        )
        hvr = [
            fmean(row.hard_violation_rate for row in groups[(domain, intensity, method)])
            for method in methods
        ]
        recovery = [
            fmean(float(row.recovery_success) for row in groups[(domain, intensity, method)])
            for method in methods
        ]
        autonomy = [
            fmean(row.autonomy_retention for row in groups[(domain, intensity, method)])
            for method in methods
        ]
        utility = [
            fmean(row.mean_task_utility for row in groups[(domain, intensity, method)])
            for method in methods
        ]

        hvr_spread = _spread(hvr)
        recovery_spread = _spread(recovery)
        autonomy_spread = _spread(autonomy)
        utility_spread = _spread(utility)
        score = (
            0.35 * hvr_spread
            + 0.30 * recovery_spread
            + 0.25 * autonomy_spread
            + 0.10 * utility_spread
        )
        points.append(
            CalibrationPoint(
                domain=domain,
                intensity=intensity,
                hvr_spread=hvr_spread,
                recovery_spread=recovery_spread,
                autonomy_spread=autonomy_spread,
                utility_spread=utility_spread,
                informative_score=score,
            )
        )
    return points


def recommend_intensities(
    records: list[EpisodeRecord],
    count: int = 5,
) -> CalibrationRecommendation:
    """Select globally informative intensity levels while retaining severity coverage."""
    if count <= 0:
        raise ValueError("count must be positive")

    points = calibration_points(records)
    intensities = sorted({point.intensity for point in points})
    if count >= len(intensities):
        selected = tuple(intensities)
    else:
        scores: dict[float, float] = {}
        for intensity in intensities:
            matching = [
                point.informative_score
                for point in points
                if point.intensity == intensity
            ]
            scores[intensity] = fmean(matching)

        mandatory = {intensities[0], intensities[-1]}
        ranked = sorted(
            (intensity for intensity in intensities if intensity not in mandatory),
            key=lambda intensity: (-scores[intensity], intensity),
        )
        selected_set = mandatory | set(ranked[: max(0, count - len(mandatory))])
        selected = tuple(sorted(selected_set))

    return CalibrationRecommendation(
        intensities=selected,
        points=tuple(points),
    )
