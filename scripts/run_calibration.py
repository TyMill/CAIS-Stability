"""Run the CAIS-Stability calibration campaign and recommend article intensities."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

from cais_stability.calibration import recommend_intensities
from cais_stability.experiments import CampaignConfig, run_campaign, write_records_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument(
        "--intensities",
        type=float,
        nargs="+",
        default=[index / 20 for index in range(21)],
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=Path("results/calibration/calibration_raw.csv"),
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("results/calibration/calibration_report.csv"),
    )
    args = parser.parse_args()
    if args.seeds <= 0:
        raise SystemExit("--seeds must be positive")

    config = CampaignConfig(
        seeds=tuple(range(args.seeds)),
        intensities=tuple(args.intensities),
        horizon=args.horizon,
    )
    records = run_campaign(config)
    write_records_csv(records, args.raw_output)
    recommendation = recommend_intensities(records, args.count)

    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    with args.report_output.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(asdict(recommendation.points[0]).keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for point in recommendation.points:
            writer.writerow(asdict(point))

    selected = ", ".join(f"{value:.2f}" for value in recommendation.intensities)
    print(f"recommended intensities: {selected}")
    print(f"wrote {len(records)} calibration episodes to {args.raw_output}")
    print(f"wrote calibration report to {args.report_output}")


if __name__ == "__main__":
    main()
