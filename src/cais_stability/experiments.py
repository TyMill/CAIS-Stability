"""Reproducible episode and campaign runners for CAIS-Stability."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random
from statistics import fmean

from .baselines import GovernancePolicy, ablation_policies, default_policies
from .core import GovernanceDecision, SystemState
from .disturbances import DisturbanceConfig, sample_disturbance
from .domains import DomainName
from .environments import NormalizedEnvironment
from .metrics import EpisodeMetrics, summarize_episode


@dataclass(frozen=True, slots=True)
class EpisodeConfig:
    horizon: int = 60
    recovery_deadline: int = 25
    seed: int = 0

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.recovery_deadline < 0:
            raise ValueError("recovery_deadline must be non-negative")


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    domain: str
    method: str
    seed: int
    intensity: float
    hard_violation_rate: float
    recovery_success: bool
    recovery_time: int | None
    autonomy_retention: float
    intervention_rate: float
    fallback_rate: float
    mean_task_utility: float
    peak_risk: float
    peak_degradation: float


@dataclass(frozen=True, slots=True)
class CampaignConfig:
    domains: tuple[DomainName, ...] = tuple(DomainName)
    intensities: tuple[float, ...] = (0.0, 0.25, 0.50, 0.75, 1.0)
    seeds: tuple[int, ...] = tuple(range(10))
    horizon: int = 60
    recovery_deadline: int = 25
    disturbance_onset: int = 10
    disturbance_duration: int = 6
    observation_noise: float = 0.05
    monitoring_delay: int = 0
    risk_estimation_error: float = 0.0
    agent_degradation: float = 0.10


@dataclass(frozen=True, slots=True)
class RobustnessConfig:
    domains: tuple[DomainName, ...] = tuple(DomainName)
    seeds: tuple[int, ...] = tuple(range(10))
    intensity: float = 0.75
    observation_noise: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20)
    monitoring_delay: tuple[int, ...] = (0, 1, 3, 5)
    agent_degradation: tuple[float, ...] = (0.0, 0.25, 0.50)
    horizon: int = 60


def _domain_seed(domain: DomainName, seed: int) -> int:
    return {DomainName.MARITIME: 10_000, DomainName.SUPPLY_CHAIN: 20_000, DomainName.SMART_GRID: 30_000}[domain] + seed


def _delayed(history: list[SystemState], delay: int) -> SystemState:
    return history[max(0, len(history) - 1 - delay)]


def run_episode(domain: DomainName, policy: GovernancePolicy, disturbance: DisturbanceConfig, config: EpisodeConfig | None = None) -> EpisodeRecord:
    episode = config or EpisodeConfig()
    env = NormalizedEnvironment(domain)
    env.reset()
    policy.reset()
    rng = Random(_domain_seed(domain, episode.seed))
    decisions: list[GovernanceDecision] = []
    states: list[SystemState] = []
    utilities: list[float] = []
    observations: list[SystemState] = [env.state]
    hard_violations = 0
    degraded_start: int | None = None
    recovery_time: int | None = None

    for step in range(episode.horizon):
        sample = sample_disturbance(step, disturbance, rng)
        observed = env.observe(sample.observation_error, disturbance.risk_estimation_error)
        observations.append(observed)
        proposal = env.agent_proposal(rng, disturbance.agent_degradation)
        decision = policy.decide(_delayed(observations, disturbance.monitoring_delay), proposal)
        transition = env.step(decision.action, sample.shock, disturbance.observation_noise)
        decisions.append(decision)
        states.append(transition.state)
        utilities.append(transition.task_utility)
        hard_violations += int(transition.hard_violation)
        if step >= disturbance.onset and degraded_start is None and not env.is_nominal():
            degraded_start = step
        if degraded_start is not None and recovery_time is None and step >= disturbance.end and env.is_nominal():
            recovery_time = step - degraded_start

    if degraded_start is None:
        recovery_time = 0
    metrics: EpisodeMetrics = summarize_episode(decisions, hard_violations, recovery_time, episode.recovery_deadline, utilities, states)
    return EpisodeRecord(
        domain.value,
        policy.name,
        episode.seed,
        disturbance.intensity,
        metrics.hard_violation_rate,
        metrics.recovery_success,
        metrics.recovery_time,
        metrics.autonomy_retention,
        metrics.intervention_rate,
        metrics.fallback_rate,
        metrics.mean_task_utility,
        metrics.peak_risk,
        metrics.peak_degradation,
    )


def run_campaign(config: CampaignConfig | None = None) -> list[EpisodeRecord]:
    campaign = config or CampaignConfig()
    records: list[EpisodeRecord] = []
    for domain in campaign.domains:
        for intensity in campaign.intensities:
            disturbance = DisturbanceConfig(
                intensity=intensity,
                onset=campaign.disturbance_onset,
                duration=campaign.disturbance_duration,
                observation_noise=campaign.observation_noise,
                monitoring_delay=campaign.monitoring_delay,
                risk_estimation_error=campaign.risk_estimation_error,
                agent_degradation=campaign.agent_degradation,
            )
            for seed in campaign.seeds:
                for policy in default_policies():
                    records.append(run_episode(domain, policy, disturbance, EpisodeConfig(campaign.horizon, campaign.recovery_deadline, seed)))
    return records


def run_robustness_campaign(config: RobustnessConfig | None = None) -> list[EpisodeRecord]:
    campaign = config or RobustnessConfig()
    records: list[EpisodeRecord] = []
    for domain in campaign.domains:
        for noise in campaign.observation_noise:
            for delay in campaign.monitoring_delay:
                for degradation in campaign.agent_degradation:
                    disturbance = DisturbanceConfig(campaign.intensity, 10, 6, noise, delay, 0.0, degradation)
                    for seed in campaign.seeds:
                        for policy in default_policies():
                            records.append(run_episode(domain, policy, disturbance, EpisodeConfig(campaign.horizon, 25, seed)))
    return records


def run_ablation_campaign(config: CampaignConfig | None = None) -> list[EpisodeRecord]:
    campaign = config or CampaignConfig(intensities=(0.50, 0.75, 1.0), observation_noise=0.10, monitoring_delay=1, agent_degradation=0.25)
    records: list[EpisodeRecord] = []
    for domain in campaign.domains:
        for intensity in campaign.intensities:
            disturbance = DisturbanceConfig(intensity, campaign.disturbance_onset, campaign.disturbance_duration, campaign.observation_noise, campaign.monitoring_delay, campaign.risk_estimation_error, campaign.agent_degradation)
            for seed in campaign.seeds:
                for policy in ablation_policies():
                    records.append(run_episode(domain, policy, disturbance, EpisodeConfig(campaign.horizon, campaign.recovery_deadline, seed)))
    return records


def write_records_csv(records: list[EpisodeRecord], path: str | Path) -> Path:
    if not records:
        raise ValueError("records must not be empty")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))
    return output


def aggregate_records(records: list[EpisodeRecord]) -> list[dict[str, str | float | int]]:
    groups: dict[tuple[str, str, float], list[EpisodeRecord]] = {}
    for record in records:
        groups.setdefault((record.domain, record.method, record.intensity), []).append(record)
    rows: list[dict[str, str | float | int]] = []
    for (domain, method, intensity), group in sorted(groups.items()):
        recovery_times = [r.recovery_time for r in group if r.recovery_time is not None]
        rows.append({
            "domain": domain,
            "method": method,
            "intensity": intensity,
            "n": len(group),
            "hard_violation_rate_mean": fmean(r.hard_violation_rate for r in group),
            "recovery_success_rate": fmean(float(r.recovery_success) for r in group),
            "recovery_time_mean": fmean(recovery_times) if recovery_times else -1.0,
            "autonomy_retention_mean": fmean(r.autonomy_retention for r in group),
            "intervention_rate_mean": fmean(r.intervention_rate for r in group),
            "fallback_rate_mean": fmean(r.fallback_rate for r in group),
            "task_utility_mean": fmean(r.mean_task_utility for r in group),
            "peak_risk_mean": fmean(r.peak_risk for r in group),
            "peak_degradation_mean": fmean(r.peak_degradation for r in group),
        })
    return rows


def write_aggregate_csv(rows: list[dict[str, str | float | int]], path: str | Path) -> Path:
    if not rows:
        raise ValueError("rows must not be empty")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return output
