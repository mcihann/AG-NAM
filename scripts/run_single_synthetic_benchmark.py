from __future__ import annotations

from pathlib import Path

import pandas as pd

from agnam.benchmarking.synthetic_protocol import (
    SyntheticBenchmarkProtocol,
    build_synthetic_schedule,
    save_records_csv,
)
from agnam.benchmarking.synthetic_runner import (
    SyntheticModelConfig,
    run_single_synthetic_benchmark,
)
from agnam.utils.reproducibility import (
    get_device,
)


def main():
    protocol = (
        SyntheticBenchmarkProtocol()
    )

    model_config = (
        SyntheticModelConfig()
    )

    schedule = (
        build_synthetic_schedule(
            protocol
        )
    )

    # Locked first benchmark realization:
    #
    # scenario = S1
    # realization = 1
    # dataset seed = 10000
    spec = schedule[0]

    device = get_device()

    result = (
        run_single_synthetic_benchmark(
            spec=spec,
            protocol=protocol,
            model_config=model_config,
            device=device,
            verbose=True,
        )
    )

    output_path = (
        Path("results")
        / "synthetic"
        / "S1"
        / "realization_01.csv"
    )

    save_records_csv(
        records=[
            result.record
        ],
        path=output_path,
    )

    pair_path = (
        Path("results")
        / "synthetic"
        / "S1"
        / "realization_01_pairs.csv"
    )

    rows = []

    all_pairs = set(
        result.true_pairs
    )

    all_pairs.update(
        result.selection_pairs
    )

    all_pairs.update(
        result.isr_pairs
    )

    all_pairs.update(
        result.random_pairs
    )

    all_pairs.update(
        result.single_run_pairs
    )

    for pair in sorted(
        all_pairs
    ):
        rows.append(
            {
                "feature_j": (
                    pair[0]
                ),
                "feature_k": (
                    pair[1]
                ),
                "is_true": (
                    pair
                    in result.true_pairs
                ),
                "selection_stable": (
                    pair
                    in result.selection_pairs
                ),
                "isr_retained": (
                    pair
                    in result.isr_pairs
                ),
                "random_control": (
                    pair
                    in result.random_pairs
                ),
                "single_run_top_k": (
                    pair
                    in result.single_run_pairs
                ),
            }
        )

    pair_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        rows
    ).to_csv(
        pair_path,
        index=False,
    )

    print(
        "\nSaved ignored benchmark outputs:"
    )

    print(
        output_path
    )

    print(
        pair_path
    )


if __name__ == "__main__":
    main()