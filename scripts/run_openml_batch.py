from __future__ import annotations

import argparse
from pathlib import Path

from agnam.benchmarking.openml_batch import (
    run_openml_batch,
)
from agnam.utils.reproducibility import (
    get_device,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the resumable locked "
            "28-task AG-NAM OpenML benchmark."
        )
    )

    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help=(
            "First registry position "
            "(1-based, inclusive)."
        ),
    )

    parser.add_argument(
        "--end",
        type=int,
        default=28,
        help=(
            "Last registry position "
            "(1-based, inclusive)."
        ),
    )

    parser.add_argument(
        "--task-id",
        type=int,
        default=None,
        help=(
            "Run or adopt exactly one "
            "locked OpenML task."
        ),
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default=(
            "results/openml/benchmark"
        ),
    )

    parser.add_argument(
        "--canonical-results-dir",
        type=str,
        default=(
            "results/openml/single"
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    parser.add_argument(
        "--stop-on-error",
        action="store_true",
    )

    parser.add_argument(
        "--quiet-tasks",
        action="store_true",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.dry_run:
        device = None

    else:
        device = (
            get_device()
        )

        print(
            "Selected neural device:",
            device,
        )

    run_openml_batch(
        start=(
            args.start
        ),
        end=(
            args.end
        ),
        task_id=(
            args.task_id
        ),
        results_root=Path(
            args.results_dir
        ),
        canonical_root=Path(
            args
            .canonical_results_dir
        ),
        device=(
            device
        ),
        dry_run=(
            args.dry_run
        ),
        stop_on_error=(
            args.stop_on_error
        ),
        verbose_tasks=(
            not args.quiet_tasks
        ),
    )


if __name__ == "__main__":
    main()