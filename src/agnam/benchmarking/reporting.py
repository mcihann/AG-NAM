from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from agnam.benchmarking.statistics import (
    StatisticalAnalysisProtocol,
    analyze_paired_comparisons,
    benchmark_completion_table,
    build_diagnostic_summary,
    validate_master_against_schedule,
)
from agnam.benchmarking.synthetic_protocol import (
    SYNTHETIC_SCENARIOS,
    SyntheticBenchmarkProtocol,
)


@dataclass(frozen=True)
class PublicationExportPaths:
    predictive_table: Path
    structure_table: Path
    inference_table: Path
    diagnostic_table: Path
    completion_table: Path
    manifest: Path


def _summary(
    values,
) -> dict[str, float]:
    array = pd.to_numeric(
        pd.Series(
            values
        ),
        errors="coerce",
    ).to_numpy(
        dtype=np.float64
    )

    array = array[
        np.isfinite(
            array
        )
    ]

    if len(
        array
    ) == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "iqr": np.nan,
        }

    q25 = float(
        np.quantile(
            array,
            0.25,
        )
    )

    q75 = float(
        np.quantile(
            array,
            0.75,
        )
    )

    return {
        "n": int(
            len(
                array
            )
        ),
        "mean": float(
            np.mean(
                array
            )
        ),
        "std": (
            float(
                np.std(
                    array,
                    ddof=1,
                )
            )
            if len(
                array
            ) > 1
            else np.nan
        ),
        "median": float(
            np.median(
                array
            )
        ),
        "q25": (
            q25
        ),
        "q75": (
            q75
        ),
        "iqr": float(
            q75
            - q25
        ),
    }


def _metric_summary(
    frame: pd.DataFrame,
    column: str,
) -> dict[str, float]:
    if column not in (
        frame.columns
    ):
        raise ValueError(
            f"Missing metric column: {column}"
        )

    return _summary(
        frame[
            column
        ]
    )


def build_predictive_publication_table(
    master: pd.DataFrame,
    comparisons: pd.DataFrame,
) -> pd.DataFrame:
    """
    One publication-oriented predictive-performance row per scenario.
    """
    rows = []

    for scenario in (
        SYNTHETIC_SCENARIOS
    ):
        frame = (
            master[
                master[
                    "scenario"
                ]
                == scenario
            ]
            .copy()
        )

        if len(
            frame
        ) == 0:
            continue

        main_auroc = (
            _metric_summary(
                frame,
                "main_auroc",
            )
        )

        agnam_auroc = (
            _metric_summary(
                frame,
                "agnam_auroc",
            )
        )

        main_auprc = (
            _metric_summary(
                frame,
                "main_auprc",
            )
        )

        agnam_auprc = (
            _metric_summary(
                frame,
                "agnam_auprc",
            )
        )

        oracle_auroc = (
            _metric_summary(
                frame,
                "oracle_auroc",
            )
        )

        random_auroc = (
            _metric_summary(
                frame,
                "random_pair_auroc",
            )
        )

        no_isr_auroc = (
            _metric_summary(
                frame,
                "no_isr_auroc",
            )
        )

        single_run_auroc = (
            _metric_summary(
                frame,
                "single_run_auroc",
            )
        )

        primary = comparisons[
            (
                comparisons[
                    "scenario"
                ]
                == scenario
            )
            & (
                comparisons[
                    "comparison_id"
                ]
                == "agnam_vs_main_auroc"
            )
        ]

        if len(
            primary
        ) != 1:
            raise RuntimeError(
                "Exactly one primary AUROC comparison "
                f"is required for {scenario}."
            )

        primary_row = (
            primary.iloc[
                0
            ]
        )

        auprc_comparison = comparisons[
            (
                comparisons[
                    "scenario"
                ]
                == scenario
            )
            & (
                comparisons[
                    "comparison_id"
                ]
                == "agnam_vs_main_auprc"
            )
        ]

        if len(
            auprc_comparison
        ) != 1:
            raise RuntimeError(
                "Exactly one AG-NAM vs Main NAM "
                f"AUPRC comparison is required for {scenario}."
            )

        auprc_row = (
            auprc_comparison
            .iloc[
                0
            ]
        )

        rows.append(
            {
                "scenario": (
                    scenario
                ),
                "n_realizations": int(
                    len(
                        frame
                    )
                ),
                "main_auroc_mean": (
                    main_auroc[
                        "mean"
                    ]
                ),
                "main_auroc_std": (
                    main_auroc[
                        "std"
                    ]
                ),
                "agnam_auroc_mean": (
                    agnam_auroc[
                        "mean"
                    ]
                ),
                "agnam_auroc_std": (
                    agnam_auroc[
                        "std"
                    ]
                ),
                "delta_auroc_mean": float(
                    primary_row[
                        "mean_difference"
                    ]
                ),
                "delta_auroc_median": float(
                    primary_row[
                        "median_difference"
                    ]
                ),
                "delta_auroc_ci_low": float(
                    primary_row[
                        "bootstrap_mean_ci_low"
                    ]
                ),
                "delta_auroc_ci_high": float(
                    primary_row[
                        "bootstrap_mean_ci_high"
                    ]
                ),
                "delta_auroc_rbc": float(
                    primary_row[
                        "rank_biserial"
                    ]
                ),
                "delta_auroc_win_rate": float(
                    primary_row[
                        "win_rate"
                    ]
                ),
                "primary_p_raw": float(
                    primary_row[
                        "p_raw"
                    ]
                ),
                "primary_p_holm": float(
                    primary_row[
                        "p_holm"
                    ]
                ),
                "primary_reject_holm": bool(
                    primary_row[
                        "reject_holm"
                    ]
                ),
                "main_auprc_mean": (
                    main_auprc[
                        "mean"
                    ]
                ),
                "main_auprc_std": (
                    main_auprc[
                        "std"
                    ]
                ),
                "agnam_auprc_mean": (
                    agnam_auprc[
                        "mean"
                    ]
                ),
                "agnam_auprc_std": (
                    agnam_auprc[
                        "std"
                    ]
                ),
                "delta_auprc_mean": float(
                    auprc_row[
                        "mean_difference"
                    ]
                ),
                "delta_auprc_ci_low": float(
                    auprc_row[
                        "bootstrap_mean_ci_low"
                    ]
                ),
                "delta_auprc_ci_high": float(
                    auprc_row[
                        "bootstrap_mean_ci_high"
                    ]
                ),
                "delta_auprc_p_holm": float(
                    auprc_row[
                        "p_holm"
                    ]
                ),
                "oracle_auroc_mean": (
                    oracle_auroc[
                        "mean"
                    ]
                ),
                "oracle_auroc_std": (
                    oracle_auroc[
                        "std"
                    ]
                ),
                "random_pair_auroc_mean": (
                    random_auroc[
                        "mean"
                    ]
                ),
                "random_pair_auroc_std": (
                    random_auroc[
                        "std"
                    ]
                ),
                "no_isr_auroc_mean": (
                    no_isr_auroc[
                        "mean"
                    ]
                ),
                "no_isr_auroc_std": (
                    no_isr_auroc[
                        "std"
                    ]
                ),
                "single_run_auroc_mean": (
                    single_run_auroc[
                        "mean"
                    ]
                ),
                "single_run_auroc_std": (
                    single_run_auroc[
                        "std"
                    ]
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_structure_publication_table(
    master: pd.DataFrame,
) -> pd.DataFrame:
    """
    One publication-oriented structure-recovery row per scenario.
    """
    rows = []

    for scenario in (
        SYNTHETIC_SCENARIOS
    ):
        frame = (
            master[
                master[
                    "scenario"
                ]
                == scenario
            ]
            .copy()
        )

        if len(
            frame
        ) == 0:
            continue

        required = (
            "mean_pairwise_jaccard",
            "mean_interaction_auprc",
            "selection_precision",
            "selection_recall",
            "selection_f1",
            "isr_precision",
            "isr_recall",
            "isr_f1",
            "false_positive_reduction",
            "n_selection_stable",
            "n_isr_retained",
            "candidate_k",
        )

        summaries = {
            column: (
                _metric_summary(
                    frame,
                    column,
                )
            )
            for column
            in required
        }

        selection_count = (
            pd.to_numeric(
                frame[
                    "n_selection_stable"
                ],
                errors="coerce",
            )
            .to_numpy(
                dtype=np.float64
            )
        )

        retained_count = (
            pd.to_numeric(
                frame[
                    "n_isr_retained"
                ],
                errors="coerce",
            )
            .to_numpy(
                dtype=np.float64
            )
        )

        candidate_k = (
            pd.to_numeric(
                frame[
                    "candidate_k"
                ],
                errors="coerce",
            )
            .to_numpy(
                dtype=np.float64
            )
        )

        sparsification_selection = (
            np.full(
                len(
                    frame
                ),
                np.nan,
                dtype=np.float64,
            )
        )

        valid_selection = (
            selection_count
            > 0
        )

        sparsification_selection[
            valid_selection
        ] = (
            1.0
            - (
                retained_count[
                    valid_selection
                ]
                / selection_count[
                    valid_selection
                ]
            )
        )

        sparsification_single = (
            np.full(
                len(
                    frame
                ),
                np.nan,
                dtype=np.float64,
            )
        )

        valid_candidate = (
            candidate_k
            > 0
        )

        sparsification_single[
            valid_candidate
        ] = (
            1.0
            - (
                retained_count[
                    valid_candidate
                ]
                / candidate_k[
                    valid_candidate
                ]
            )
        )

        sparsification_selection_summary = (
            _summary(
                sparsification_selection
            )
        )

        sparsification_single_summary = (
            _summary(
                sparsification_single
            )
        )

        rows.append(
            {
                "scenario": (
                    scenario
                ),
                "n_realizations": int(
                    len(
                        frame
                    )
                ),
                "interaction_auprc_mean": (
                    summaries[
                        "mean_interaction_auprc"
                    ][
                        "mean"
                    ]
                ),
                "interaction_auprc_std": (
                    summaries[
                        "mean_interaction_auprc"
                    ][
                        "std"
                    ]
                ),
                "topk_jaccard_mean": (
                    summaries[
                        "mean_pairwise_jaccard"
                    ][
                        "mean"
                    ]
                ),
                "topk_jaccard_std": (
                    summaries[
                        "mean_pairwise_jaccard"
                    ][
                        "std"
                    ]
                ),
                "selection_precision_mean": (
                    summaries[
                        "selection_precision"
                    ][
                        "mean"
                    ]
                ),
                "selection_recall_mean": (
                    summaries[
                        "selection_recall"
                    ][
                        "mean"
                    ]
                ),
                "selection_f1_mean": (
                    summaries[
                        "selection_f1"
                    ][
                        "mean"
                    ]
                ),
                "isr_precision_mean": (
                    summaries[
                        "isr_precision"
                    ][
                        "mean"
                    ]
                ),
                "isr_recall_mean": (
                    summaries[
                        "isr_recall"
                    ][
                        "mean"
                    ]
                ),
                "isr_f1_mean": (
                    summaries[
                        "isr_f1"
                    ][
                        "mean"
                    ]
                ),
                "false_positive_reduction_n": (
                    summaries[
                        "false_positive_reduction"
                    ][
                        "n"
                    ]
                ),
                "false_positive_reduction_mean": (
                    summaries[
                        "false_positive_reduction"
                    ][
                        "mean"
                    ]
                ),
                "false_positive_reduction_std": (
                    summaries[
                        "false_positive_reduction"
                    ][
                        "std"
                    ]
                ),
                "selection_stable_count_mean": (
                    summaries[
                        "n_selection_stable"
                    ][
                        "mean"
                    ]
                ),
                "isr_retained_count_mean": (
                    summaries[
                        "n_isr_retained"
                    ][
                        "mean"
                    ]
                ),
                "sparsification_vs_selection_n": (
                    sparsification_selection_summary[
                        "n"
                    ]
                ),
                "sparsification_vs_selection_mean": (
                    sparsification_selection_summary[
                        "mean"
                    ]
                ),
                "sparsification_vs_single_run_n": (
                    sparsification_single_summary[
                        "n"
                    ]
                ),
                "sparsification_vs_single_run_mean": (
                    sparsification_single_summary[
                        "mean"
                    ]
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def export_synthetic_publication_tables(
    *,
    results_root: str | Path = (
        "results/synthetic"
    ),
    output_root: str | Path = (
        "results/synthetic/publication"
    ),
    benchmark_protocol: (
        SyntheticBenchmarkProtocol
        | None
    ) = None,
    statistical_protocol: (
        StatisticalAnalysisProtocol
        | None
    ) = None,
) -> PublicationExportPaths:
    """
    Export final publication-oriented synthetic benchmark tables.

    Export is permitted only when all 80 locked primary realizations
    are complete.
    """
    if benchmark_protocol is None:
        benchmark_protocol = (
            SyntheticBenchmarkProtocol()
        )

    if statistical_protocol is None:
        statistical_protocol = (
            StatisticalAnalysisProtocol()
        )

    results_root = Path(
        results_root
    )

    output_root = Path(
        output_root
    )

    master_path = (
        results_root
        / "master_results.csv"
    )

    if not master_path.exists():
        raise FileNotFoundError(
            f"Master results not found: {master_path}"
        )

    master = pd.read_csv(
        master_path
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

    if not bool(
        completion[
            "is_complete"
        ].all()
    ):
        raise RuntimeError(
            "Publication-table export requires all "
            "80 locked synthetic realizations."
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

    if comparisons[
        "p_holm"
    ].isna().any():
        raise RuntimeError(
            "Confirmatory Holm-adjusted inference "
            "is incomplete."
        )

    predictive = (
        build_predictive_publication_table(
            master,
            comparisons,
        )
    )

    structure = (
        build_structure_publication_table(
            master
        )
    )

    diagnostics = (
        build_diagnostic_summary(
            master
        )
    )

    inference = (
        comparisons
        .copy()
        .sort_values(
            by=[
                "scenario",
                "family",
                "comparison_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictive_path = (
        output_root
        / "table_synthetic_predictive.csv"
    )

    structure_path = (
        output_root
        / "table_synthetic_structure.csv"
    )

    inference_path = (
        output_root
        / "table_synthetic_inference.csv"
    )

    diagnostic_path = (
        output_root
        / "table_synthetic_diagnostics.csv"
    )

    completion_path = (
        output_root
        / "table_synthetic_completion.csv"
    )

    manifest_path = (
        output_root
        / "synthetic_benchmark_manifest.txt"
    )

    predictive.to_csv(
        predictive_path,
        index=False,
    )

    structure.to_csv(
        structure_path,
        index=False,
    )

    inference.to_csv(
        inference_path,
        index=False,
    )

    diagnostics.to_csv(
        diagnostic_path,
        index=False,
    )

    completion.to_csv(
        completion_path,
        index=False,
    )

    manifest_text = (
        "AG-NAM Synthetic Benchmark Manifest\n"
        "==================================\n"
        "\n"
        "Primary benchmark status: COMPLETE\n"
        "Scenarios: S1, S2, S3, S4\n"
        "Independent realizations per scenario: 20\n"
        "Total locked realizations: 80\n"
        "\n"
        "Primary endpoint: AUROC\n"
        "Primary comparison: AG-NAM vs Main NAM\n"
        "Primary multiplicity control: Holm across 4 scenario tests\n"
        "\n"
        "Secondary family: 20 prespecified tests\n"
        "Secondary multiplicity control: Holm\n"
        "\n"
        "Bootstrap resamples: 10000\n"
        "Bootstrap seed: 20260902\n"
        "Alpha: 0.05\n"
        "\n"
        "The publication tables are derived from the immutable\n"
        "per-realization benchmark checkpoints through the locked\n"
        "statistical-analysis protocol.\n"
    )

    manifest_path.write_text(
        manifest_text,
        encoding="utf-8",
    )

    return PublicationExportPaths(
        predictive_table=(
            predictive_path
        ),
        structure_table=(
            structure_path
        ),
        inference_table=(
            inference_path
        ),
        diagnostic_table=(
            diagnostic_path
        ),
        completion_table=(
            completion_path
        ),
        manifest=(
            manifest_path
        ),
    )