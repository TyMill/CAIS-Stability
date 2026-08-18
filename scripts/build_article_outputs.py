"""Build deterministic aggregate tables from raw CAIS-Stability article records."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from cais_stability.experiments import EpisodeRecord, aggregate_records, write_aggregate_csv


def _optional_int(value: str) -> int | None:
    return None if value == "" else int(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/raw/article_results.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/aggregated/article_summary.csv"))
    args = parser.parse_args()
    records: list[EpisodeRecord] = []
    with args.input.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            records.append(EpisodeRecord(
                domain=row["domain"], method=row["method"], seed=int(row["seed"]), intensity=float(row["intensity"]),
                hard_violation_rate=float(row["hard_violation_rate"]), recovery_success=row["recovery_success"] == "True",
                recovery_time=_optional_int(row["recovery_time"]), autonomy_retention=float(row["autonomy_retention"]),
                intervention_rate=float(row["intervention_rate"]), fallback_rate=float(row["fallback_rate"]),
                mean_task_utility=float(row["mean_task_utility"]), peak_risk=float(row["peak_risk"]),
                peak_degradation=float(row["peak_degradation"]),
            ))
    rows = aggregate_records(records)
    output = write_aggregate_csv(rows, args.output)
    print(f"wrote {len(rows)} aggregate rows to {output}")


if __name__ == "__main__":
    main()
