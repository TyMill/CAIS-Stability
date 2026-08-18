"""Run the primary reproducible CAIS-Stability article campaign."""

from __future__ import annotations

import argparse
from pathlib import Path

from cais_stability.experiments import CampaignConfig, run_campaign, write_records_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/raw/article_results.csv"))
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--intensities", type=float, nargs="+", default=[0.0, 0.25, 0.50, 0.75, 1.0])
    args = parser.parse_args()
    if args.seeds <= 0:
        raise SystemExit("--seeds must be positive")
    records = run_campaign(CampaignConfig(seeds=tuple(range(args.seeds)), intensities=tuple(args.intensities), horizon=args.horizon))
    output = write_records_csv(records, args.output)
    print(f"wrote {len(records)} episode records to {output}")


if __name__ == "__main__":
    main()
