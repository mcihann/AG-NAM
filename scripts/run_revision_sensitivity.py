from __future__ import annotations

import argparse
from pathlib import Path

from agnam.benchmarking.synthetic_protocol import build_synthetic_schedule
from agnam.revision.sensitivity_runner import (
    run_synthetic_sensitivity_realization,
)


def _select_spec(scenario: str, realization: int):
    scenario = scenario.upper()
    for spec in build_synthetic_schedule():
        if (
            spec.scenario == scenario
            and spec.realization_number == realization
        ):
            return spec
    raise ValueError(
        f"No synthetic realization found for {scenario} realization {realization}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run reviewer-requested AG-NAM sensitivity analysis."
    )
    parser.add_argument("--scenario", choices=["S1", "S2", "S3", "S4"])
    parser.add_argument("--realization", type=int)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--output-root",
        default="results/revision/sensitivity/synthetic",
    )
    args = parser.parse_args()

    if args.all:
        specs = build_synthetic_schedule()
    else:
        if args.scenario is None or args.realization is None:
            parser.error(
                "Use --all or provide both --scenario and --realization."
            )
        specs = (
            _select_spec(args.scenario, args.realization),
        )

    for spec in specs:
        run_synthetic_sensitivity_realization(
            spec,
            output_root=Path(args.output_root),
            force=args.force,
            verbose=True,
        )


if __name__ == "__main__":
    main()
