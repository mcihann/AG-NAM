from __future__ import annotations

import argparse

from agnam.benchmarking.openml_runner import (
    run_single_openml_benchmark,
    save_single_openml_result,
)
from agnam.utils.reproducibility import (
    get_device,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run one canonical locked "
            "AG-NAM OpenML benchmark task."
        )
    )

    parser.add_argument(
        "--task-id",
        type=int,
        default=49,
        help=(
            "Locked OpenML task ID. "
            "Default: 49 (tic-tac-toe)."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    device = get_device()

    result = (
        run_single_openml_benchmark(
            task_id=(
                args.task_id
            ),
            device=(
                device
            ),
            verbose=True,
        )
    )

    record_path, pair_path = (
        save_single_openml_result(
            result
        )
    )

    print(
        "\nSaved canonical OpenML outputs:"
    )

    print(
        record_path
    )

    print(
        pair_path
    )


if __name__ == "__main__":
    main()