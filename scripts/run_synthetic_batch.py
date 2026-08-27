from __future__ import annotations

import argparse
from pathlib import Path

from agnam.benchmarking.batch_runner import (
    run_synthetic_batch,
)
from agnam.benchmarking.synthetic_protocol import (
    SyntheticBenchmarkProtocol,
)
from agnam.benchmarking.synthetic_runner import (
    SyntheticModelConfig,
)
from agnam.utils.reproducibility import (
    get_device,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the resumable locked AG-NAM "
            "synthetic benchmark."
        )
    )

    parser.add_argument(
        "--scenario",
        type=str,
        default="S1",
        choices=[
            "S1",
            "S2",
            "S3",
            "S4",
            "ALL",
        ],
        help=(
            "Synthetic scenario to execute. "
            "Use ALL for every scenario."
        ),
    )

    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help=(
            "First human-facing realization number "
            "(inclusive)."
        ),
    )

    parser.add_argument(
        "--end",
        type=int,
        default=20,
        help=(
            "Last human-facing realization number "
            "(inclusive)."
        ),
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default=(
            "results/synthetic"
        ),
        help=(
            "Root directory for synthetic "
            "benchmark checkpoints."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Show which realizations would run "
            "without executing models."
        ),
    )

    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help=(
            "Stop immediately after the first "
            "failed realization."
        ),
    )

    parser.add_argument(
        "--quiet-realizations",
        action="store_true",
        help=(
            "Suppress detailed output inside each "
            "single realization."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    protocol = (
        SyntheticBenchmarkProtocol()
    )

    model_config = (
        SyntheticModelConfig()
    )

    if args.dry_run:
        device = None

    else:
        device = get_device()

        print(
            "Selected device:",
            device,
        )

    run_synthetic_batch(
        protocol=protocol,
        model_config=model_config,
        scenario=args.scenario,
        start=args.start,
        end=args.end,
        results_root=Path(
            args.results_dir
        ),
        device=device,
        dry_run=args.dry_run,
        stop_on_error=(
            args.stop_on_error
        ),
        verbose_realizations=(
            not args.quiet_realizations
        ),
    )


if __name__ == "__main__":
    main()