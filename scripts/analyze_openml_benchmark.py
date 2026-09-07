from __future__ import annotations

from pathlib import Path

import pandas as pd

from agnam.benchmarking.openml_statistics import (
    RealWorldStatisticalProtocol,
    analyze_openml_comparisons,
    build_openml_diagnostic_summary,
    openml_completion_table,
    validate_openml_master,
)


def main():
    results_root = (
        Path(
            "results"
        )
        / "openml"
        / "benchmark"
    )

    statistics_root = (
        results_root
        / "statistics"
    )

    master_path = (
        results_root
        / "master_results.csv"
    )

    if not master_path.exists():
        raise FileNotFoundError(
            f"OpenML master results not found: "
            f"{master_path}"
        )

    master = pd.read_csv(
        master_path
    )

    protocol = (
        RealWorldStatisticalProtocol()
    )

    validate_openml_master(
        master
    )

    completion = (
        openml_completion_table(
            master
        )
    )

    comparisons = (
        analyze_openml_comparisons(
            master,
            statistical_protocol=(
                protocol
            ),
        )
    )

    diagnostics = (
        build_openml_diagnostic_summary(
            master
        )
    )

    statistics_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    completion_path = (
        statistics_root
        / "benchmark_completion.csv"
    )

    comparisons_path = (
        statistics_root
        / "paired_comparisons.csv"
    )

    diagnostics_path = (
        statistics_root
        / "diagnostic_summary.csv"
    )

    completion.to_csv(
        completion_path,
        index=False,
    )

    comparisons.to_csv(
        comparisons_path,
        index=False,
    )

    diagnostics.to_csv(
        diagnostics_path,
        index=False,
    )

    completed = int(
        completion[
            "complete"
        ].sum()
    )

    print(
        "========================================"
    )

    print(
        "OPENML STATISTICAL ANALYSIS"
    )

    print(
        "========================================"
    )

    print(
        f"\nBenchmark completion: "
        f"{completed}/28"
    )

    primary = (
        comparisons[
            comparisons[
                "family"
            ]
            == "primary"
        ]
        .iloc[
            0
        ]
    )

    print(
        "\nPRIMARY DESCRIPTIVE COMPARISON"
    )

    print(
        "AG-NAM - Main NAM AUROC"
    )

    print(
        f"Datasets currently available: "
        f"{int(primary['n_datasets'])}"
    )

    print(
        f"Mean difference: "
        f"{primary['mean_difference']:+.6f}"
    )

    print(
        f"Median difference: "
        f"{primary['median_difference']:+.6f}"
    )

    print(
        "95% paired bootstrap CI: "
        f"[{primary['bootstrap_mean_ci_low']:+.6f}, "
        f"{primary['bootstrap_mean_ci_high']:+.6f}]"
    )

    print(
        f"Rank-biserial: "
        f"{primary['rank_biserial']:+.4f}"
    )

    print(
        f"Win rate: "
        f"{primary['win_rate']:.3f}"
    )

    benchmark_complete = (
        completed
        == 28
    )

    if not benchmark_complete:
        print(
            "\nCONFIRMATORY INFERENCE DEFERRED"
        )

        print(
            "Primary and secondary p-values "
            "remain hidden until all 28 locked "
            "OpenML datasets are complete."
        )

        print(
            "Current results are descriptive only "
            "and must not be used to alter the "
            "locked method or baseline protocol."
        )

    else:
        print(
            "\nCONFIRMATORY INFERENCE ENABLED"
        )

        print(
            "All 28 locked OpenML tasks "
            "are complete."
        )

        print(
            "\nPRIMARY RESULT"
        )

        print(
            f"p={primary['p_raw']:.6g}"
        )

        print(
            f"reject="
            f"{bool(primary['reject'])}"
        )

        print(
            "\nSECONDARY HOLM-ADJUSTED RESULTS"
        )

        secondary = (
            comparisons[
                comparisons[
                    "family"
                ]
                == "secondary"
            ]
        )

        for row in (
            secondary.itertuples()
        ):
            print(
                f"{row.comparison_id}: "
                f"p_raw={row.p_raw:.6g}, "
                f"p_holm={row.p_adjusted:.6g}, "
                f"reject={bool(row.reject)}"
            )

    print(
        "\nSaved statistical outputs:"
    )

    print(
        completion_path
    )

    print(
        comparisons_path
    )

    print(
        diagnostics_path
    )


if __name__ == "__main__":
    main()