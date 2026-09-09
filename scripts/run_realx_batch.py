from __future__ import annotations

import os

os.environ.setdefault(
    "CUBLAS_WORKSPACE_CONFIG",
    ":4096:8",
)

import argparse

from agnam.benchmarking.realx_batch import (
    run_realx_batch,
    select_run_specs,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run the locked resumable "
            "180-run Real-X benchmark."
        )
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Show the execution plan "
            "without training models."
        ),
    )

    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--task-id",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--realization",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--strength",
        type=str,
        choices=[
            "null",
            "weak",
            "moderate",
            "strong",
        ],
        default=None,
    )

    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help=(
            "One-based first selected run."
        ),
    )

    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help=(
            "One-based last selected run."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-run even validated checkpoints."
        ),
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help=(
            "Continue remaining runs after a failure."
        ),
    )

    return parser


def main():
    parser = build_parser()

    args = parser.parse_args()

    specs = select_run_specs(
        run_id=args.run_id,
        task_id=args.task_id,
        realization=args.realization,
        strength=args.strength,
        start=args.start,
        end=args.end,
    )

    if len(
        specs
    ) == 0:
        raise RuntimeError(
            "No locked Real-X runs match "
            "the supplied filters."
        )

    run_realx_batch(
        specs,
        dry_run=args.dry_run,
        force=args.force,
        continue_on_error=(
            args.continue_on_error
        ),
        verbose=True,
    )


if __name__ == "__main__":
    main()