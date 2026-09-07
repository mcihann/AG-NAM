from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

from agnam.benchmarking.synthetic_protocol import (
    SYNTHETIC_SCENARIOS,
    SyntheticBenchmarkProtocol,
    build_synthetic_schedule,
)


@dataclass(frozen=True)
class StatisticalAnalysisProtocol:
    """
    Locked statistical-analysis protocol for the synthetic benchmark.
    """

    alpha: float = 0.05

    bootstrap_resamples: int = 10_000
    bootstrap_seed: int = 20_260_902

    primary_metric: str = "auroc"

    def __post_init__(
        self,
    ) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(
                "alpha must lie in (0, 1)."
            )

        if self.bootstrap_resamples < 1000:
            raise ValueError(
                "bootstrap_resamples must be at least 1000."
            )


@dataclass(frozen=True)
class ComparisonSpec:
    comparison_id: str
    family: str
    metric: str

    model_a_label: str
    model_b_label: str

    model_a_column: str
    model_b_column: str


PRIMARY_COMPARISONS = (
    ComparisonSpec(
        comparison_id="agnam_vs_main_auroc",
        family="primary",
        metric="AUROC",
        model_a_label="AG-NAM",
        model_b_label="Main NAM",
        model_a_column="agnam_auroc",
        model_b_column="main_auroc",
    ),
)


SECONDARY_COMPARISONS = (
    ComparisonSpec(
        comparison_id="agnam_vs_main_auprc",
        family="secondary",
        metric="AUPRC",
        model_a_label="AG-NAM",
        model_b_label="Main NAM",
        model_a_column="agnam_auprc",
        model_b_column="main_auprc",
    ),
    ComparisonSpec(
        comparison_id="agnam_vs_random_auroc",
        family="secondary",
        metric="AUROC",
        model_a_label="AG-NAM",
        model_b_label="Random-Pair NAM",
        model_a_column="agnam_auroc",
        model_b_column="random_pair_auroc",
    ),
    ComparisonSpec(
        comparison_id="agnam_vs_random_auprc",
        family="secondary",
        metric="AUPRC",
        model_a_label="AG-NAM",
        model_b_label="Random-Pair NAM",
        model_a_column="agnam_auprc",
        model_b_column="random_pair_auprc",
    ),
    ComparisonSpec(
        comparison_id="agnam_vs_single_run_auroc",
        family="secondary",
        metric="AUROC",
        model_a_label="AG-NAM",
        model_b_label="Single-Run AG-NAM",
        model_a_column="agnam_auroc",
        model_b_column="single_run_auroc",
    ),
    ComparisonSpec(
        comparison_id="agnam_vs_single_run_auprc",
        family="secondary",
        metric="AUPRC",
        model_a_label="AG-NAM",
        model_b_label="Single-Run AG-NAM",
        model_a_column="agnam_auprc",
        model_b_column="single_run_auprc",
    ),
)


IDENTITY_COLUMNS = (
    "scenario",
    "realization_index",
    "dataset_seed",
    "outer_split_seed",
    "final_split_seed",
    "discovery_base_seed",
    "random_pair_seed",
    "final_model_seed",
)


def validate_master_against_schedule(
    master: pd.DataFrame,
    *,
    protocol: SyntheticBenchmarkProtocol | None = None,
) -> None:
    """
    Validate benchmark rows against the locked 80-run seed schedule.

    Partial benchmark tables are allowed, but every existing row must
    correspond exactly to its predefined scenario-realization entry.
    """
    if protocol is None:
        protocol = (
            SyntheticBenchmarkProtocol()
        )

    if len(master) == 0:
        raise ValueError(
            "master benchmark table is empty."
        )

    missing_columns = [
        column
        for column in IDENTITY_COLUMNS
        if column not in master.columns
    ]

    if missing_columns:
        raise ValueError(
            "master table is missing identity columns: "
            f"{missing_columns}"
        )

    duplicate_mask = master.duplicated(
        subset=[
            "scenario",
            "realization_index",
        ],
        keep=False,
    )

    if duplicate_mask.any():
        duplicates = master.loc[
            duplicate_mask,
            [
                "scenario",
                "realization_index",
            ],
        ]

        raise ValueError(
            "Duplicate scenario-realization rows detected:\n"
            f"{duplicates.to_string(index=False)}"
        )

    schedule = (
        build_synthetic_schedule(
            protocol
        )
    )

    lookup = {
        (
            specification.scenario,
            specification.realization_index,
        ): specification
        for specification in schedule
    }

    for row in master.itertuples(
        index=False
    ):
        key = (
            str(
                row.scenario
            ),
            int(
                row.realization_index
            ),
        )

        if key not in lookup:
            raise ValueError(
                f"Unknown benchmark realization: {key}"
            )

        specification = (
            lookup[
                key
            ]
        )

        expected = {
            "dataset_seed": (
                specification
                .dataset_seed
            ),
            "outer_split_seed": (
                specification
                .outer_split_seed
            ),
            "final_split_seed": (
                specification
                .final_split_seed
            ),
            "discovery_base_seed": (
                specification
                .discovery_base_seed
            ),
            "random_pair_seed": (
                specification
                .random_pair_seed
            ),
            "final_model_seed": (
                specification
                .final_model_seed
            ),
        }

        for (
            column,
            expected_value,
        ) in expected.items():
            actual_value = int(
                getattr(
                    row,
                    column,
                )
            )

            if (
                actual_value
                != expected_value
            ):
                raise ValueError(
                    f"Seed mismatch for {key}, "
                    f"{column}: expected "
                    f"{expected_value}, found "
                    f"{actual_value}."
                )


def benchmark_completion_table(
    master: pd.DataFrame,
    *,
    protocol: SyntheticBenchmarkProtocol | None = None,
) -> pd.DataFrame:
    """
    Report completion status for every synthetic scenario.
    """
    if protocol is None:
        protocol = (
            SyntheticBenchmarkProtocol()
        )

    rows = []

    for scenario in (
        protocol.scenarios
    ):
        scenario_rows = master[
            master[
                "scenario"
            ]
            == scenario
        ]

        completed_indices = set(
            pd.to_numeric(
                scenario_rows[
                    "realization_index"
                ],
                errors="raise",
            )
            .astype(
                int
            )
            .tolist()
        )

        expected_indices = set(
            range(
                protocol
                .n_realizations
            )
        )

        missing = sorted(
            expected_indices
            - completed_indices
        )

        unexpected = sorted(
            completed_indices
            - expected_indices
        )

        rows.append(
            {
                "scenario": (
                    scenario
                ),
                "completed": len(
                    completed_indices
                    & expected_indices
                ),
                "expected": (
                    protocol
                    .n_realizations
                ),
                "is_complete": (
                    completed_indices
                    == expected_indices
                ),
                "missing_realizations": (
                    "|".join(
                        str(
                            index + 1
                        )
                        for index
                        in missing
                    )
                ),
                "unexpected_indices": (
                    "|".join(
                        str(
                            index
                        )
                        for index
                        in unexpected
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def paired_bootstrap_ci(
    differences: np.ndarray,
    *,
    statistic: str = "mean",
    resamples: int = 10_000,
    seed: int = 20_260_902,
    alpha: float = 0.05,
) -> tuple[
    float,
    float,
]:
    """
    Percentile paired-bootstrap confidence interval.

    Resampling is performed over complete paired realizations.
    """
    values = np.asarray(
        differences,
        dtype=np.float64,
    )

    if values.ndim != 1:
        raise ValueError(
            "differences must be one-dimensional."
        )

    if len(values) == 0:
        raise ValueError(
            "differences cannot be empty."
        )

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            "differences must contain only finite values."
        )

    if statistic not in {
        "mean",
        "median",
    }:
        raise ValueError(
            "statistic must be 'mean' or 'median'."
        )

    if resamples < 1:
        raise ValueError(
            "resamples must be positive."
        )

    rng = (
        np.random.default_rng(
            seed
        )
    )

    sampled_indices = (
        rng.integers(
            0,
            len(values),
            size=(
                resamples,
                len(values),
            ),
        )
    )

    sampled = values[
        sampled_indices
    ]

    if statistic == "mean":
        bootstrap_statistics = (
            sampled.mean(
                axis=1
            )
        )

    else:
        bootstrap_statistics = (
            np.median(
                sampled,
                axis=1,
            )
        )

    lower = float(
        np.quantile(
            bootstrap_statistics,
            alpha / 2.0,
        )
    )

    upper = float(
        np.quantile(
            bootstrap_statistics,
            1.0
            - alpha / 2.0,
        )
    )

    return (
        lower,
        upper,
    )


def paired_rank_biserial(
    differences: np.ndarray,
    *,
    zero_tolerance: float = 1e-12,
) -> float:
    """
    Paired rank-biserial correlation.

    Positive values favor model A when differences are defined as:

        model A - model B
    """
    values = np.asarray(
        differences,
        dtype=np.float64,
    )

    if values.ndim != 1:
        raise ValueError(
            "differences must be one-dimensional."
        )

    nonzero = values[
        ~np.isclose(
            values,
            0.0,
            atol=(
                zero_tolerance
            ),
            rtol=0.0,
        )
    ]

    if len(nonzero) == 0:
        return 0.0

    ranks = rankdata(
        np.abs(
            nonzero
        ),
        method="average",
    )

    positive_rank_sum = float(
        ranks[
            nonzero > 0.0
        ].sum()
    )

    negative_rank_sum = float(
        ranks[
            nonzero < 0.0
        ].sum()
    )

    denominator = (
        positive_rank_sum
        + negative_rank_sum
    )

    if denominator == 0.0:
        return 0.0

    return float(
        (
            positive_rank_sum
            - negative_rank_sum
        )
        / denominator
    )


def holm_adjust(
    p_values: np.ndarray,
) -> np.ndarray:
    """
    Holm step-down family-wise error-rate adjustment.
    """
    values = np.asarray(
        p_values,
        dtype=np.float64,
    )

    if values.ndim != 1:
        raise ValueError(
            "p_values must be one-dimensional."
        )

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            "p_values must be finite."
        )

    if np.any(
        (
            values < 0.0
        )
        | (
            values > 1.0
        )
    ):
        raise ValueError(
            "p_values must lie in [0, 1]."
        )

    number = len(
        values
    )

    if number == 0:
        return (
            values.copy()
        )

    order = np.argsort(
        values
    )

    adjusted = np.empty_like(
        values
    )

    running_maximum = 0.0

    for (
        position,
        index,
    ) in enumerate(
        order
    ):
        multiplier = (
            number
            - position
        )

        candidate = float(
            multiplier
            * values[
                index
            ]
        )

        running_maximum = max(
            running_maximum,
            candidate,
        )

        adjusted[
            index
        ] = min(
            1.0,
            running_maximum,
        )

    return adjusted


def _wilcoxon_two_sided(
    differences: np.ndarray,
    *,
    zero_tolerance: float = 1e-12,
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
            atol=(
                zero_tolerance
            ),
            rtol=0.0,
        )
    ]

    effective_n = len(
        nonzero
    )

    if effective_n == 0:
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
            effective_n
        ),
    )


def compute_paired_comparison(
    frame: pd.DataFrame,
    comparison: ComparisonSpec,
    *,
    scenario: str,
    statistical_protocol: (
        StatisticalAnalysisProtocol
        | None
    ) = None,
    bootstrap_seed_offset: int = 0,
) -> dict:
    """
    Compute one paired model comparison for one scenario.
    """
    if statistical_protocol is None:
        statistical_protocol = (
            StatisticalAnalysisProtocol()
        )

    required = {
        comparison
        .model_a_column,
        comparison
        .model_b_column,
    }

    missing = [
        column
        for column in required
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            f"Missing comparison columns: {missing}"
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

    if len(model_a) != len(
        model_b
    ):
        raise ValueError(
            "Paired model vectors have different lengths."
        )

    if len(model_a) == 0:
        raise ValueError(
            "No paired realizations available."
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
            "Paired model metrics must be finite."
        )

    differences = (
        model_a
        - model_b
    )

    mean_difference = float(
        differences.mean()
    )

    median_difference = float(
        np.median(
            differences
        )
    )

    if len(
        differences
    ) > 1:
        sd_difference = float(
            differences.std(
                ddof=1
            )
        )

    else:
        sd_difference = (
            np.nan
        )

    q25_difference = float(
        np.quantile(
            differences,
            0.25,
        )
    )

    q75_difference = float(
        np.quantile(
            differences,
            0.75,
        )
    )

    (
        ci_mean_low,
        ci_mean_high,
    ) = paired_bootstrap_ci(
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

    (
        ci_median_low,
        ci_median_high,
    ) = paired_bootstrap_ci(
        differences,
        statistic="median",
        resamples=(
            statistical_protocol
            .bootstrap_resamples
        ),
        seed=(
            statistical_protocol
            .bootstrap_seed
            + 100_000
            + bootstrap_seed_offset
        ),
        alpha=(
            statistical_protocol
            .alpha
        ),
    )

    (
        wilcoxon_statistic,
        p_raw,
        effective_n,
    ) = _wilcoxon_two_sided(
        differences
    )

    rank_biserial = (
        paired_rank_biserial(
            differences
        )
    )

    ties = np.isclose(
        differences,
        0.0,
        atol=1e-12,
        rtol=0.0,
    )

    wins = (
        differences
        > 1e-12
    )

    losses = (
        differences
        < -1e-12
    )

    n = len(
        differences
    )

    return {
        "scenario": (
            scenario
        ),
        "family": (
            comparison
            .family
        ),
        "comparison_id": (
            comparison
            .comparison_id
        ),
        "metric": (
            comparison
            .metric
        ),
        "model_a": (
            comparison
            .model_a_label
        ),
        "model_b": (
            comparison
            .model_b_label
        ),
        "n_pairs": int(
            n
        ),
        "wilcoxon_effective_n": (
            effective_n
        ),
        "mean_model_a": float(
            model_a.mean()
        ),
        "mean_model_b": float(
            model_b.mean()
        ),
        "mean_difference": (
            mean_difference
        ),
        "sd_difference": (
            sd_difference
        ),
        "median_difference": (
            median_difference
        ),
        "q25_difference": (
            q25_difference
        ),
        "q75_difference": (
            q75_difference
        ),
        "bootstrap_mean_ci_low": (
            ci_mean_low
        ),
        "bootstrap_mean_ci_high": (
            ci_mean_high
        ),
        "bootstrap_median_ci_low": (
            ci_median_low
        ),
        "bootstrap_median_ci_high": (
            ci_median_high
        ),
        "wilcoxon_statistic": (
            wilcoxon_statistic
        ),
        "p_raw": (
            p_raw
        ),
        "rank_biserial": (
            rank_biserial
        ),
        "win_rate": float(
            wins.sum()
            / n
        ),
        "loss_rate": float(
            losses.sum()
            / n
        ),
        "tie_rate": float(
            ties.sum()
            / n
        ),
        "p_holm": np.nan,
        "reject_holm": pd.NA,
        "family_complete": False,
    }


def analyze_paired_comparisons(
    master: pd.DataFrame,
    *,
    benchmark_protocol: (
        SyntheticBenchmarkProtocol
        | None
    ) = None,
    statistical_protocol: (
        StatisticalAnalysisProtocol
        | None
    ) = None,
) -> pd.DataFrame:
    """
    Run the locked paired-comparison analysis.

    Raw paired statistics may be computed for completed partial
    scenarios.

    Holm-adjusted confirmatory inference is deliberately deferred
    until all four scenarios contain all 20 locked realizations.

    Primary family:
        AG-NAM vs Main NAM, AUROC
        across S1-S4
        => 4 tests, Holm correction.

    Secondary family:
        five predefined comparisons per scenario
        across S1-S4
        => 20 tests, Holm correction.
    """
    if benchmark_protocol is None:
        benchmark_protocol = (
            SyntheticBenchmarkProtocol()
        )

    if statistical_protocol is None:
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

    benchmark_complete = bool(
        completion[
            "is_complete"
        ].all()
    )

    rows = []

    all_comparisons = (
        PRIMARY_COMPARISONS
        + SECONDARY_COMPARISONS
    )

    for (
        scenario_index,
        scenario,
    ) in enumerate(
        benchmark_protocol
        .scenarios
    ):
        scenario_frame = (
            master[
                master[
                    "scenario"
                ]
                == scenario
            ]
            .sort_values(
                "realization_index"
            )
            .reset_index(
                drop=True
            )
        )

        if len(
            scenario_frame
        ) == 0:
            continue

        for (
            comparison_index,
            comparison,
        ) in enumerate(
            all_comparisons
        ):
            seed_offset = (
                scenario_index
                * 1000
                + comparison_index
            )

            rows.append(
                compute_paired_comparison(
                    frame=(
                        scenario_frame
                    ),
                    comparison=(
                        comparison
                    ),
                    scenario=(
                        scenario
                    ),
                    statistical_protocol=(
                        statistical_protocol
                    ),
                    bootstrap_seed_offset=(
                        seed_offset
                    ),
                )
            )

    results = pd.DataFrame(
        rows
    )

    if len(
        results
    ) == 0:
        return results

    # ---------------------------------------------------------
    # Explicit inference dtypes
    # ---------------------------------------------------------
    #
    # p_holm is numeric and may be missing for an incomplete
    # benchmark.
    #
    # reject_holm is a nullable Boolean column. Explicitly using
    # pandas BooleanDtype prevents assigning True/False values into
    # a float column and therefore avoids pandas incompatible-dtype
    # FutureWarnings.
    # ---------------------------------------------------------

    results[
        "p_holm"
    ] = pd.to_numeric(
        results[
            "p_holm"
        ],
        errors="coerce",
    ).astype(
        "float64"
    )

    results[
        "reject_holm"
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

    results[
        "family_complete"
    ] = (
        results[
            "family_complete"
        ]
        .astype(
            bool
        )
    )

    if benchmark_complete:
        for family in (
            "primary",
            "secondary",
        ):
            family_mask = (
                results[
                    "family"
                ]
                == family
            )

            p_values = (
                results.loc[
                    family_mask,
                    "p_raw",
                ]
                .to_numpy(
                    dtype=np.float64
                )
            )

            adjusted = (
                holm_adjust(
                    p_values
                )
            )

            results.loc[
                family_mask,
                "p_holm",
            ] = (
                adjusted
            )

            reject_values = (
                adjusted
                < statistical_protocol
                .alpha
            )

            results.loc[
                family_mask,
                "reject_holm",
            ] = (
                pd.array(
                    reject_values,
                    dtype="boolean",
                )
            )

            results.loc[
                family_mask,
                "family_complete",
            ] = True

    return results


def _summarize_values(
    values: np.ndarray,
) -> dict:
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    if len(values) == 0:
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
            values,
            0.25,
        )
    )

    q75 = float(
        np.quantile(
            values,
            0.75,
        )
    )

    return {
        "n": int(
            len(values)
        ),
        "mean": float(
            values.mean()
        ),
        "std": (
            float(
                values.std(
                    ddof=1
                )
            )
            if len(
                values
            ) > 1
            else np.nan
        ),
        "median": float(
            np.median(
                values
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


def build_diagnostic_summary(
    master: pd.DataFrame,
) -> pd.DataFrame:
    """
    Descriptive, non-confirmatory benchmark diagnostics.

    No hypothesis tests are performed for these quantities.
    """
    required_columns = {
        "scenario",
        "main_auroc",
        "agnam_auroc",
        "oracle_auroc",
        "random_pair_auroc",
        "no_isr_auroc",
        "single_run_auroc",
        "n_selection_stable",
        "n_isr_retained",
        "candidate_k",
    }

    missing = (
        required_columns
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

        main = pd.to_numeric(
            frame[
                "main_auroc"
            ],
            errors="raise",
        ).to_numpy(
            dtype=np.float64
        )

        agnam = pd.to_numeric(
            frame[
                "agnam_auroc"
            ],
            errors="raise",
        ).to_numpy(
            dtype=np.float64
        )

        oracle = pd.to_numeric(
            frame[
                "oracle_auroc"
            ],
            errors="raise",
        ).to_numpy(
            dtype=np.float64
        )

        random_pair = (
            pd.to_numeric(
                frame[
                    "random_pair_auroc"
                ],
                errors="raise",
            )
            .to_numpy(
                dtype=np.float64
            )
        )

        no_isr = pd.to_numeric(
            frame[
                "no_isr_auroc"
            ],
            errors="raise",
        ).to_numpy(
            dtype=np.float64
        )

        single_run = (
            pd.to_numeric(
                frame[
                    "single_run_auroc"
                ],
                errors="raise",
            )
            .to_numpy(
                dtype=np.float64
            )
        )

        selection_count = (
            pd.to_numeric(
                frame[
                    "n_selection_stable"
                ],
                errors="raise",
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
                errors="raise",
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
                errors="raise",
            )
            .to_numpy(
                dtype=np.float64
            )
        )

        oracle_denominator = (
            oracle
            - main
        )

        oracle_capture = np.full(
            len(
                frame
            ),
            np.nan,
            dtype=np.float64,
        )

        valid_oracle = (
            np.abs(
                oracle_denominator
            )
            > 1e-12
        )

        oracle_capture[
            valid_oracle
        ] = (
            (
                agnam[
                    valid_oracle
                ]
                - main[
                    valid_oracle
                ]
            )
            / oracle_denominator[
                valid_oracle
            ]
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

        diagnostics = {
            "oracle_gap_auroc": (
                oracle
                - agnam
            ),
            "oracle_gain_capture": (
                oracle_capture
            ),
            "agnam_minus_random_auroc": (
                agnam
                - random_pair
            ),
            "agnam_minus_no_isr_auroc": (
                agnam
                - no_isr
            ),
            "agnam_minus_single_run_auroc": (
                agnam
                - single_run
            ),
            "isr_sparsification_vs_selection": (
                sparsification_selection
            ),
            "isr_sparsification_vs_single_run": (
                sparsification_single
            ),
        }

        for (
            metric,
            values,
        ) in diagnostics.items():
            summary = (
                _summarize_values(
                    values
                )
            )

            rows.append(
                {
                    "scenario": (
                        scenario
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