from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from agnam.benchmarking.statistics import (
    StatisticalAnalysisProtocol,
    analyze_paired_comparisons,
    benchmark_completion_table,
    build_diagnostic_summary,
    validate_master_against_schedule,
)
from agnam.benchmarking.synthetic_protocol import (
    SyntheticBenchmarkProtocol,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the locked statistical analysis "
            "for the AG-NAM synthetic benchmark."
        )
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/synthetic",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            "results/synthetic/statistics"
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    results_root = Path(
        args.results_dir
    )

    output_root = Path(
        args.output_dir
    )

    master_path = (
        results_root
        / "master_results.csv"
    )

    if not master_path.exists():
        raise FileNotFoundError(
            f"Master benchmark table not found: "
            f"{master_path}"
        )

    master = pd.read_csv(
        master_path
    )

    benchmark_protocol = (
        SyntheticBenchmarkProtocol()
    )

    statistical_protocol = (
        StatisticalAnalysisProtocol()
    )

    validate_master_against_schedule(
        master,
        protocol=(
            benchmark_protocol
        ),
    )

    completion = (
        benchmark_completion_table(
            master,
            protocol=(
                benchmark_protocol
            ),
        )
    )

    comparisons = (
        analyze_paired_comparisons(
            master,
            benchmark_protocol=(
                benchmark_protocol
            ),
            statistical_protocol=(
                statistical_protocol
            ),
        )
    )

    diagnostics = (
        build_diagnostic_summary(
            master
        )
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    completion_path = (
        output_root
        / "benchmark_completion.csv"
    )

    comparison_path = (
        output_root
        / "paired_comparisons.csv"
    )

    diagnostic_path = (
        output_root
        / "diagnostic_summary.csv"
    )

    completion.to_csv(
        completion_path,
        index=False,
    )

    comparisons.to_csv(
        comparison_path,
        index=False,
    )

    diagnostics.to_csv(
        diagnostic_path,
        index=False,
    )

    print(
        "========================================"
    )

    print(
        "SYNTHETIC STATISTICAL ANALYSIS"
    )

    print(
        "========================================"
    )

    print(
        "\nBENCHMARK COMPLETION"
    )

    print(
        completion.to_string(
            index=False
        )
    )

    benchmark_complete = bool(
        completion[
            "is_complete"
        ].all()
    )

    print(
        "\nPRIMARY DESCRIPTIVE COMPARISONS"
    )

    primary = comparisons[
        comparisons[
            "family"
        ]
        == "primary"
    ]

    for row in (
        primary.itertuples()
    ):
        print(
            f"{row.scenario}: "
            f"AG-NAM - Main NAM AUROC "
            f"mean={row.mean_difference:+.6f}, "
            f"95% bootstrap CI="
            f"[{row.bootstrap_mean_ci_low:+.6f}, "
            f"{row.bootstrap_mean_ci_high:+.6f}], "
            f"median={row.median_difference:+.6f}, "
            f"RBC={row.rank_biserial:+.4f}, "
            f"wins={row.win_rate:.3f}"
        )

    if benchmark_complete:
        print(
            "\nCONFIRMATORY INFERENCE ENABLED"
        )

        print(
            "All 80 locked realizations are complete."
        )

        print(
            "\nPRIMARY HOLM-ADJUSTED RESULTS"
        )

        for row in (
            primary.itertuples()
        ):
            print(
                f"{row.scenario}: "
                f"p_raw={row.p_raw:.6g}, "
                f"p_holm={row.p_holm:.6g}, "
                f"reject={bool(row.reject_holm)}"
            )

        secondary = comparisons[
            comparisons[
                "family"
            ]
            == "secondary"
        ]

        print(
            "\nSECONDARY HOLM-ADJUSTED RESULTS"
        )

        for row in (
            secondary.itertuples()
        ):
            print(
                f"{row.scenario} "
                f"{row.comparison_id}: "
                f"p_raw={row.p_raw:.6g}, "
                f"p_holm={row.p_holm:.6g}, "
                f"reject={bool(row.reject_holm)}"
            )

    else:
        print(
            "\nCONFIRMATORY INFERENCE DEFERRED"
        )

        print(
            "Holm-adjusted primary and secondary "
            "inference will be produced only after "
            "S1-S4 each contain all 20 locked "
            "realizations."
        )

        print(
            "Current paired results are descriptive/"
            "interim and must not be used to alter "
            "the locked AG-NAM method."
        )

    print(
        "\nSaved statistical outputs:"
    )

    print(
        completion_path
    )

    print(
        comparison_path
    )

    print(
        diagnostic_path
    )


if __name__ == "__main__":
    main()