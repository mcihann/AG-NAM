from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from agnam.benchmarking.external_baselines import (
    ExternalBaselineProtocol,
    build_external_baseline_schedule,
)
from agnam.benchmarking.openml_protocol import (
    OPENML_TASKS,
    OpenMLBenchmarkProtocol,
    build_openml_schedule,
)
from agnam.benchmarking.statistics import (
    holm_adjust,
    paired_bootstrap_ci,
    paired_rank_biserial,
)


@dataclass(frozen=True)
class RealWorldStatisticalProtocol:
    """
    Locked inferential protocol for the 28-dataset OpenML benchmark.
    """

    alpha: float = 0.05

    bootstrap_resamples: int = 10_000
    bootstrap_seed: int = 20_260_907

    primary_metric: str = "AUROC"

    def __post_init__(
        self,
    ) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(
                "alpha must lie in (0, 1)."
            )

        if (
            self.bootstrap_resamples
            < 1000
        ):
            raise ValueError(
                "bootstrap_resamples must be "
                "at least 1000."
            )


@dataclass(frozen=True)
class RealWorldComparisonSpec:
    comparison_id: str
    family: str
    metric: str

    model_a_label: str
    model_b_label: str

    model_a_column: str
    model_b_column: str


PRIMARY_COMPARISON = (
    RealWorldComparisonSpec(
        comparison_id=(
            "agnam_vs_main_auroc"
        ),
        family="primary",
        metric="AUROC",
        model_a_label="AG-NAM",
        model_b_label="Main NAM",
        model_a_column="agnam_auroc",
        model_b_column="main_auroc",
    )
)


SECONDARY_COMPARISONS = (
    RealWorldComparisonSpec(
        comparison_id=(
            "agnam_vs_main_auprc"
        ),
        family="secondary",
        metric="AUPRC",
        model_a_label="AG-NAM",
        model_b_label="Main NAM",
        model_a_column="agnam_auprc",
        model_b_column="main_auprc",
    ),
    RealWorldComparisonSpec(
        comparison_id=(
            "agnam_vs_ebm_auroc"
        ),
        family="secondary",
        metric="AUROC",
        model_a_label="AG-NAM",
        model_b_label="EBM",
        model_a_column="agnam_auroc",
        model_b_column="ebm_auroc",
    ),
    RealWorldComparisonSpec(
        comparison_id=(
            "agnam_vs_ebm_auprc"
        ),
        family="secondary",
        metric="AUPRC",
        model_a_label="AG-NAM",
        model_b_label="EBM",
        model_a_column="agnam_auprc",
        model_b_column="ebm_auprc",
    ),
    RealWorldComparisonSpec(
        comparison_id=(
            "agnam_vs_catboost_auroc"
        ),
        family="secondary",
        metric="AUROC",
        model_a_label="AG-NAM",
        model_b_label="CatBoost",
        model_a_column="agnam_auroc",
        model_b_column="catboost_auroc",
    ),
    RealWorldComparisonSpec(
        comparison_id=(
            "agnam_vs_catboost_auprc"
        ),
        family="secondary",
        metric="AUPRC",
        model_a_label="AG-NAM",
        model_b_label="CatBoost",
        model_a_column="agnam_auprc",
        model_b_column="catboost_auprc",
    ),
    RealWorldComparisonSpec(
        comparison_id=(
            "agnam_vs_random_auroc"
        ),
        family="secondary",
        metric="AUROC",
        model_a_label="AG-NAM",
        model_b_label="Random-Pair NAM",
        model_a_column="agnam_auroc",
        model_b_column="random_pair_auroc",
    ),
    RealWorldComparisonSpec(
        comparison_id=(
            "agnam_vs_random_auprc"
        ),
        family="secondary",
        metric="AUPRC",
        model_a_label="AG-NAM",
        model_b_label="Random-Pair NAM",
        model_a_column="agnam_auprc",
        model_b_column="random_pair_auprc",
    ),
    RealWorldComparisonSpec(
        comparison_id=(
            "agnam_vs_single_run_auroc"
        ),
        family="secondary",
        metric="AUROC",
        model_a_label="AG-NAM",
        model_b_label="Single-Run AG-NAM",
        model_a_column="agnam_auroc",
        model_b_column="single_run_auroc",
    ),
    RealWorldComparisonSpec(
        comparison_id=(
            "agnam_vs_single_run_auprc"
        ),
        family="secondary",
        metric="AUPRC",
        model_a_label="AG-NAM",
        model_b_label="Single-Run AG-NAM",
        model_a_column="agnam_auprc",
        model_b_column="single_run_auprc",
    ),
)


def validate_openml_master(
    master: pd.DataFrame,
    *,
    benchmark_protocol: (
        OpenMLBenchmarkProtocol
        | None
    ) = None,
    baseline_protocol: (
        ExternalBaselineProtocol
        | None
    ) = None,
) -> None:
    """
    Validate every available benchmark row against the locked
    28-task schedule.

    Partial benchmark tables are allowed.
    """
    if benchmark_protocol is None:
        benchmark_protocol = (
            OpenMLBenchmarkProtocol()
        )

    if baseline_protocol is None:
        baseline_protocol = (
            ExternalBaselineProtocol()
        )

    if len(
        master
    ) == 0:
        raise ValueError(
            "OpenML master table is empty."
        )

    required_columns = {
        "task_id",
        "task_index",
        "dataset_name",
        "discovery_base_seed",
        "random_pair_seed",
        "final_split_seed",
        "final_model_seed",
        "ebm_seed",
        "catboost_seed",
        "openml_repeat",
        "openml_fold",
        "openml_sample",
        "selection_threshold",
        "isr_threshold",
        "n_discovery_runs",
        "residual_crossfit_folds",
        "ebm_version",
        "catboost_version",
        "benchmark_protocol",
    }

    missing = (
        required_columns
        - set(
            master.columns
        )
    )

    if missing:
        raise ValueError(
            "OpenML master table is missing "
            f"required columns: {sorted(missing)}"
        )

    if master[
        "task_id"
    ].duplicated().any():
        raise ValueError(
            "Duplicate OpenML task IDs detected."
        )

    benchmark_lookup = {
        specification.task_id: (
            specification
        )
        for specification
        in build_openml_schedule(
            benchmark_protocol
        )
    }

    external_lookup = {
        specification.task_id: (
            specification
        )
        for specification
        in build_external_baseline_schedule(
            baseline_protocol
        )
    }

    for row in (
        master.itertuples(
            index=False
        )
    ):
        task_id = int(
            row.task_id
        )

        if task_id not in (
            benchmark_lookup
        ):
            raise ValueError(
                f"Unknown locked OpenML "
                f"task ID: {task_id}"
            )

        benchmark_spec = (
            benchmark_lookup[
                task_id
            ]
        )

        external_spec = (
            external_lookup[
                task_id
            ]
        )

        expected_integers = {
            "task_index": (
                benchmark_spec
                .task_index
            ),
            "discovery_base_seed": (
                benchmark_spec
                .discovery_base_seed
            ),
            "random_pair_seed": (
                benchmark_spec
                .random_pair_seed
            ),
            "final_split_seed": (
                benchmark_spec
                .final_split_seed
            ),
            "final_model_seed": (
                benchmark_spec
                .final_model_seed
            ),
            "ebm_seed": (
                external_spec
                .ebm_seed
            ),
            "catboost_seed": (
                external_spec
                .catboost_seed
            ),
            "openml_repeat": (
                benchmark_protocol
                .repeat
            ),
            "openml_fold": (
                benchmark_protocol
                .fold
            ),
            "openml_sample": (
                benchmark_protocol
                .sample
            ),
            "n_discovery_runs": (
                benchmark_protocol
                .n_discovery_runs
            ),
            "residual_crossfit_folds": (
                benchmark_protocol
                .residual_crossfit_folds
            ),
        }

        for (
            column,
            expected,
        ) in (
            expected_integers
            .items()
        ):
            actual = int(
                getattr(
                    row,
                    column,
                )
            )

            if actual != expected:
                raise ValueError(
                    f"Task {task_id}: "
                    f"{column} mismatch. "
                    f"Expected {expected}, "
                    f"found {actual}."
                )

        if not np.isclose(
            float(
                row.selection_threshold
            ),
            benchmark_protocol
            .selection_threshold,
            atol=1e-12,
            rtol=0.0,
        ):
            raise ValueError(
                f"Task {task_id}: "
                "selection threshold mismatch."
            )

        if not np.isclose(
            float(
                row.isr_threshold
            ),
            benchmark_protocol
            .isr_threshold,
            atol=1e-12,
            rtol=0.0,
        ):
            raise ValueError(
                f"Task {task_id}: "
                "ISR threshold mismatch."
            )

        if str(
            row.ebm_version
        ) != (
            baseline_protocol
            .ebm_version
        ):
            raise ValueError(
                f"Task {task_id}: "
                "EBM version mismatch."
            )

        if str(
            row.catboost_version
        ) != (
            baseline_protocol
            .catboost_version
        ):
            raise ValueError(
                f"Task {task_id}: "
                "CatBoost version mismatch."
            )

        if str(
            row.benchmark_protocol
        ) != (
            "agnam_openml_primary_v1"
        ):
            raise ValueError(
                f"Task {task_id}: "
                "benchmark protocol mismatch."
            )


def openml_completion_table(
    master: pd.DataFrame,
) -> pd.DataFrame:
    """
    Completion table for all 28 locked OpenML tasks.
    """
    completed = set(
        pd.to_numeric(
            master[
                "task_id"
            ],
            errors="raise",
        )
        .astype(
            int
        )
        .tolist()
    )

    rows = []

    for position, task in enumerate(
        OPENML_TASKS,
        start=1,
    ):
        rows.append(
            {
                "registry_position": (
                    position
                ),
                "task_id": (
                    task.task_id
                ),
                "dataset_name": (
                    task.dataset_name
                ),
                "feature_type": (
                    task.feature_type
                ),
                "complete": (
                    task.task_id
                    in completed
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def _paired_wilcoxon(
    differences: np.ndarray,
) -> tuple[
    float,
    float,
    int,
]:
    values = np.asarray(
        differences,
        dtype=np.float64,
    )

    nonzero = values[
        ~np.isclose(
            values,
            0.0,
            atol=1e-12,
            rtol=0.0,
        )
    ]

    if len(
        nonzero
    ) == 0:
        return (
            0.0,
            1.0,
            0,
        )

    result = wilcoxon(
        nonzero,
        alternative="two-sided",
        zero_method="wilcox",
        correction=False,
        method="auto",
    )

    return (
        float(
            result.statistic
        ),
        float(
            result.pvalue
        ),
        int(
            len(
                nonzero
            )
        ),
    )


def compute_dataset_comparison(
    master: pd.DataFrame,
    comparison: RealWorldComparisonSpec,
    *,
    statistical_protocol: (
        RealWorldStatisticalProtocol
        | None
    ) = None,
    bootstrap_seed_offset: int = 0,
) -> dict:
    """
    Compute paired descriptive statistics across datasets.
    """
    if statistical_protocol is None:
        statistical_protocol = (
            RealWorldStatisticalProtocol()
        )

    required = {
        comparison
        .model_a_column,
        comparison
        .model_b_column,
    }

    missing = (
        required
        - set(
            master.columns
        )
    )

    if missing:
        raise ValueError(
            f"Missing model comparison "
            f"columns: {sorted(missing)}"
        )

    frame = (
        master
        .sort_values(
            "task_index"
        )
        .reset_index(
            drop=True
        )
    )

    model_a = pd.to_numeric(
        frame[
            comparison
            .model_a_column
        ],
        errors="raise",
    ).to_numpy(
        dtype=np.float64
    )

    model_b = pd.to_numeric(
        frame[
            comparison
            .model_b_column
        ],
        errors="raise",
    ).to_numpy(
        dtype=np.float64
    )

    if not (
        np.isfinite(
            model_a
        ).all()
        and np.isfinite(
            model_b
        ).all()
    ):
        raise ValueError(
            "Model comparison metrics "
            "must be finite."
        )

    differences = (
        model_a
        - model_b
    )

    ci_low, ci_high = (
        paired_bootstrap_ci(
            differences,
            statistic="mean",
            resamples=(
                statistical_protocol
                .bootstrap_resamples
            ),
            seed=(
                statistical_protocol
                .bootstrap_seed
                + bootstrap_seed_offset
            ),
            alpha=(
                statistical_protocol
                .alpha
            ),
        )
    )

    wilcoxon_statistic, p_raw, effective_n = (
        _paired_wilcoxon(
            differences
        )
    )

    q25 = float(
        np.quantile(
            differences,
            0.25,
        )
    )

    q75 = float(
        np.quantile(
            differences,
            0.75,
        )
    )

    return {
        "family": (
            comparison.family
        ),
        "comparison_id": (
            comparison
            .comparison_id
        ),
        "metric": (
            comparison.metric
        ),
        "model_a": (
            comparison
            .model_a_label
        ),
        "model_b": (
            comparison
            .model_b_label
        ),
        "n_datasets": int(
            len(
                differences
            )
        ),
        "mean_model_a": float(
            np.mean(
                model_a
            )
        ),
        "mean_model_b": float(
            np.mean(
                model_b
            )
        ),
        "mean_difference": float(
            np.mean(
                differences
            )
        ),
        "sd_difference": (
            float(
                np.std(
                    differences,
                    ddof=1,
                )
            )
            if len(
                differences
            ) > 1
            else np.nan
        ),
        "median_difference": float(
            np.median(
                differences
            )
        ),
        "q25_difference": (
            q25
        ),
        "q75_difference": (
            q75
        ),
        "iqr_difference": float(
            q75
            - q25
        ),
        "bootstrap_mean_ci_low": (
            ci_low
        ),
        "bootstrap_mean_ci_high": (
            ci_high
        ),
        "rank_biserial": (
            paired_rank_biserial(
                differences
            )
        ),
        "win_rate": float(
            np.mean(
                differences
                > 1e-12
            )
        ),
        "loss_rate": float(
            np.mean(
                differences
                < -1e-12
            )
        ),
        "tie_rate": float(
            np.mean(
                np.isclose(
                    differences,
                    0.0,
                    atol=1e-12,
                    rtol=0.0,
                )
            )
        ),
        "wilcoxon_statistic": (
            wilcoxon_statistic
        ),
        "wilcoxon_effective_n": (
            effective_n
        ),
        "p_raw_internal": (
            p_raw
        ),
    }


def analyze_openml_comparisons(
    master: pd.DataFrame,
    *,
    benchmark_protocol: (
        OpenMLBenchmarkProtocol
        | None
    ) = None,
    baseline_protocol: (
        ExternalBaselineProtocol
        | None
    ) = None,
    statistical_protocol: (
        RealWorldStatisticalProtocol
        | None
    ) = None,
) -> pd.DataFrame:
    """
    Run the locked real-world comparison framework.

    Confirmatory p-values are exposed only after all 28 tasks are
    complete.

    Primary:
        AG-NAM vs Main NAM, AUROC
        one prespecified test
        no multiplicity adjustment

    Secondary:
        nine prespecified paired tests
        Holm correction across all nine.
    """
    if benchmark_protocol is None:
        benchmark_protocol = (
            OpenMLBenchmarkProtocol()
        )

    if baseline_protocol is None:
        baseline_protocol = (
            ExternalBaselineProtocol()
        )

    if statistical_protocol is None:
        statistical_protocol = (
            RealWorldStatisticalProtocol()
        )

    validate_openml_master(
        master,
        benchmark_protocol=(
            benchmark_protocol
        ),
        baseline_protocol=(
            baseline_protocol
        ),
    )

    completion = (
        openml_completion_table(
            master
        )
    )

    benchmark_complete = bool(
        completion[
            "complete"
        ].all()
    )

    comparisons = (
        (
            PRIMARY_COMPARISON,
        )
        + SECONDARY_COMPARISONS
    )

    rows = []

    for index, comparison in enumerate(
        comparisons
    ):
        row = compute_dataset_comparison(
            master,
            comparison,
            statistical_protocol=(
                statistical_protocol
            ),
            bootstrap_seed_offset=(
                index
            ),
        )

        # Internal p-value is calculated deterministically, but is
        # not exposed as confirmatory inference until 28/28 tasks
        # are complete.
        internal_p = row.pop(
            "p_raw_internal"
        )

        row[
            "p_raw"
        ] = np.nan

        row[
            "p_adjusted"
        ] = np.nan

        row[
            "reject"
        ] = pd.NA

        row[
            "benchmark_complete"
        ] = (
            benchmark_complete
        )

        row[
            "_internal_p"
        ] = (
            internal_p
        )

        rows.append(
            row
        )

    results = pd.DataFrame(
        rows
    )

    results[
        "reject"
    ] = pd.Series(
        pd.array(
            [
                pd.NA
            ]
            * len(
                results
            ),
            dtype="boolean",
        ),
        index=(
            results.index
        ),
    )

    if benchmark_complete:
        primary_mask = (
            results[
                "family"
            ]
            == "primary"
        )

        secondary_mask = (
            results[
                "family"
            ]
            == "secondary"
        )

        primary_p = (
            results.loc[
                primary_mask,
                "_internal_p",
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        if len(
            primary_p
        ) != 1:
            raise RuntimeError(
                "Exactly one primary "
                "real-world comparison is required."
            )

        results.loc[
            primary_mask,
            "p_raw",
        ] = (
            primary_p
        )

        # No multiplicity adjustment is needed for the single
        # primary endpoint. p_adjusted therefore equals p_raw.
        results.loc[
            primary_mask,
            "p_adjusted",
        ] = (
            primary_p
        )

        results.loc[
            primary_mask,
            "reject",
        ] = pd.array(
            primary_p
            < statistical_protocol.alpha,
            dtype="boolean",
        )

        secondary_p = (
            results.loc[
                secondary_mask,
                "_internal_p",
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        if len(
            secondary_p
        ) != 9:
            raise RuntimeError(
                "Exactly nine secondary "
                "real-world comparisons are required."
            )

        secondary_adjusted = (
            holm_adjust(
                secondary_p
            )
        )

        results.loc[
            secondary_mask,
            "p_raw",
        ] = (
            secondary_p
        )

        results.loc[
            secondary_mask,
            "p_adjusted",
        ] = (
            secondary_adjusted
        )

        results.loc[
            secondary_mask,
            "reject",
        ] = pd.array(
            secondary_adjusted
            < statistical_protocol.alpha,
            dtype="boolean",
        )

    return (
        results
        .drop(
            columns=[
                "_internal_p"
            ]
        )
    )


def _summary_values(
    values,
) -> dict:
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
        "q25": q25,
        "q75": q75,
        "iqr": float(
            q75
            - q25
        ),
    }


def build_openml_diagnostic_summary(
    master: pd.DataFrame,
) -> pd.DataFrame:
    """
    Descriptive real-world interaction and prediction diagnostics.

    No ground-truth interaction claim is made.
    """
    required = {
        "feature_type",
        "mean_pairwise_jaccard",
        "n_selection_stable",
        "n_isr_retained",
        "isr_sparsification",
        "agnam_auroc",
        "main_auroc",
        "no_isr_auroc",
        "ebm_auroc",
        "catboost_auroc",
    }

    missing = (
        required
        - set(
            master.columns
        )
    )

    if missing:
        raise ValueError(
            "Missing diagnostic columns: "
            f"{sorted(missing)}"
        )

    rows = []

    groups = [
        (
            "ALL",
            master,
        )
    ]

    for feature_type in (
        "numeric",
        "mixed",
        "categorical",
    ):
        subset = master[
            master[
                "feature_type"
            ]
            == feature_type
        ]

        if len(
            subset
        ) > 0:
            groups.append(
                (
                    feature_type,
                    subset,
                )
            )

    for (
        group_name,
        frame,
    ) in groups:
        diagnostics = {
            "mean_pairwise_jaccard": (
                frame[
                    "mean_pairwise_jaccard"
                ]
            ),
            "n_selection_stable": (
                frame[
                    "n_selection_stable"
                ]
            ),
            "n_isr_retained": (
                frame[
                    "n_isr_retained"
                ]
            ),
            "isr_sparsification": (
                frame[
                    "isr_sparsification"
                ]
            ),
            "agnam_minus_main_auroc": (
                frame[
                    "agnam_auroc"
                ]
                - frame[
                    "main_auroc"
                ]
            ),
            "agnam_minus_no_isr_auroc": (
                frame[
                    "agnam_auroc"
                ]
                - frame[
                    "no_isr_auroc"
                ]
            ),
            "agnam_minus_ebm_auroc": (
                frame[
                    "agnam_auroc"
                ]
                - frame[
                    "ebm_auroc"
                ]
            ),
            "agnam_minus_catboost_auroc": (
                frame[
                    "agnam_auroc"
                ]
                - frame[
                    "catboost_auroc"
                ]
            ),
        }

        for (
            metric,
            values,
        ) in (
            diagnostics.items()
        ):
            summary = (
                _summary_values(
                    values
                )
            )

            rows.append(
                {
                    "group": (
                        group_name
                    ),
                    "metric": (
                        metric
                    ),
                    **summary,
                }
            )

    return pd.DataFrame(
        rows
    )