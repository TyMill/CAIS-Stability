"""Fast deterministic verification of the cross-domain experiment pipeline."""

from __future__ import annotations

from collections import Counter

from cais_stability.experiments import CampaignConfig, run_campaign


def main() -> None:
    records = run_campaign(CampaignConfig(intensities=(0.50,), seeds=(0,), horizon=30, recovery_deadline=20, disturbance_onset=6, disturbance_duration=4))
    if len(records) != 12:
        raise RuntimeError(f"expected 12 records, received {len(records)}")
    methods = Counter(record.method for record in records)
    if set(methods) != {"ungoverned", "static_cais", "binary_rta", "dynamic_cais"}:
        raise RuntimeError(f"unexpected method set: {sorted(methods)}")
    for record in records:
        bounded = (record.hard_violation_rate, record.autonomy_retention, record.intervention_rate, record.fallback_rate, record.mean_task_utility, record.peak_risk, record.peak_degradation)
        if any(not 0.0 <= value <= 1.0 for value in bounded):
            raise RuntimeError(f"unbounded metric in {record}")
    print("CAIS-Stability smoke verification")
    for record in records:
        print(f"{record.domain:12s} {record.method:12s} HVR={record.hard_violation_rate:.3f} RS={int(record.recovery_success)} AR={record.autonomy_retention:.3f} U={record.mean_task_utility:.3f}")
    print("CAIS-Stability smoke verification complete")


if __name__ == "__main__":
    main()
