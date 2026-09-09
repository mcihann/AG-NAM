from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from agnam.benchmarking.realx_batch import (
    DEFAULT_OUTPUT_ROOT,
    _read_protocol_csv,
)
from agnam.benchmarking.realx_freeze import (
    DEFAULT_MANIFEST_PATH,
    FreezeValidationResult,
    validate_realx_evidence_freeze,
)
from agnam.benchmarking.realx_statistics import (
    ANALYSIS_SCHEMA_VERSION as PRIMARY_ANALYSIS_SCHEMA_VERSION,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    ALPHA,
    DEFAULT_PRIMARY_RESULT_PATH,
    PRIMARY_CONDITION,
    PRIMARY_METRIC,
    PRIMARY_TEST,
)
from agnam.benchmarking.statistics import (
    paired_bootstrap_ci,
)


# =============================================================================
# LOCKED SECONDARY / DESCRIPTIVE SCOPE
# =============================================================================


SECONDARY_ANALYSIS_SCHEMA_VERSION = (
    "realx_secondary_descriptive_v1"
)

EXPECTED_RUNS = 180
EXPECTED_DATASETS = 9
EXPECTED_STRENGTHS = 4
EXPECTED_REALIZATIONS = 5
EXPECTED_DATASET_STRENGTH_ROWS = 36


STRENGTH_ORDER = (
    "null",
    "weak",
    "moderate",
    "strong",
)


DEFAULT_SECONDARY_DIRECTORY = (
    DEFAULT_OUTPUT_ROOT
    / "secondary"
)

DEFAULT_DATASET_STRENGTH_PATH = (
    DEFAULT_SECONDARY_DIRECTORY
    / "dataset_strength_values.csv"
)

DEFAULT_STRENGTH_SUMMARY_PATH = (
    DEFAULT_SECONDARY_DIRECTORY
    / "strength_summary.csv"
)

DEFAULT_PREDICTIVE_COMPARATOR_PATH = (
    DEFAULT_SECONDARY_DIRECTORY
    / "predictive_comparator_summary.csv"
)

DEFAULT_FEATURE_TYPE_SUMMARY_PATH = (
    DEFAULT_SECONDARY_DIRECTORY
    / "feature_type_strength_summary.csv"
)

DEFAULT_SECONDARY_MANIFEST_PATH = (
    DEFAULT_SECONDARY_DIRECTORY
    / "secondary_analysis_manifest.json"
)


STRUCTURE_METRICS = (
    "mean_pairwise_jaccard",
    "mean_interaction_auprc",
    "n_selection_stable",
    "n_isr_retained",
    "isr_sparsification_fraction",
    "selection_precision",
    "selection_recall",
    "selection_f1",
    "isr_precision",
    "isr_recall",
    "isr_f1",
    "false_positive_reduction",
)


BASE_PREDICTIVE_METRICS = (
    "main_auroc",
    "agnam_auroc",
    "delta_auroc",
    "main_auprc",
    "agnam_auprc",
    "delta_auprc",
    "main_balanced_accuracy",
    "agnam_balanced_accuracy",
    "delta_balanced_accuracy",
    "main_f1",
    "agnam_f1",
    "delta_f1",
)


COMPARATOR_DERIVED_METRICS = (
    "agnam_minus_random_pair_auroc",
    "agnam_minus_random_pair_auprc",
    "agnam_minus_no_isr_auroc",
    "agnam_minus_no_isr_auprc",
    "agnam_minus_single_run_auroc",
    "agnam_minus_single_run_auprc",
    "oracle_minus_agnam_auroc",
    "oracle_minus_agnam_auprc",
)


STRENGTH_SUMMARY_METRICS = (
    STRUCTURE_METRICS
    + BASE_PREDICTIVE_METRICS
)


FEATURE_TYPE_SUMMARY_METRICS = (
    "mean_interaction_auprc",
    "isr_sparsification_fraction",
    "delta_auroc",
    "delta_auprc",
)


# =============================================================================
# PRIMARY-ANALYSIS GATE
# =============================================================================


@dataclass(frozen=True)
class PrimaryAnalysisLock:
    frozen_evidence_sha256: str
    reject_null: bool
    p_value: float


def validate_primary_analysis_lock(
    *,
    primary_result_path: str | Path = (
        DEFAULT_PRIMARY_RESULT_PATH
    ),
    expected_frozen_evidence_sha256: str,
) -> PrimaryAnalysisLock:
    """
    Ensure Phase 6F-B was completed against the same frozen evidence.

    The primary result itself is not reinterpreted here. This gate only
    verifies analysis ordering and frozen-evidence identity.
    """
    path = Path(
        primary_result_path
    )

    if not path.exists():
        raise RuntimeError(
            "Phase 6F-B primary confirmatory result is missing."
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        payload = json.load(
            handle
        )

    if str(
        payload.get(
            "analysis_schema_version"
        )
    ) != PRIMARY_ANALYSIS_SCHEMA_VERSION:
        raise RuntimeError(
            "Unexpected primary-analysis schema version."
        )

    if str(
        payload.get(
            "frozen_evidence_sha256"
        )
    ) != str(
        expected_frozen_evidence_sha256
    ):
        raise RuntimeError(
            "Primary analysis and current evidence freeze "
            "do not share the same SHA-256 identity."
        )

    if str(
        payload.get(
            "primary_condition"
        )
    ) != PRIMARY_CONDITION:
        raise RuntimeError(
            "Primary analysis condition differs from the locked plan."
        )

    if str(
        payload.get(
            "primary_metric"
        )
    ) != PRIMARY_METRIC:
        raise RuntimeError(
            "Primary metric differs from the locked plan."
        )

    if str(
        payload.get(
            "primary_test"
        )
    ) != PRIMARY_TEST:
        raise RuntimeError(
            "Primary test differs from the locked plan."
        )

    if int(
        payload.get(
            "n_datasets",
            -1,
        )
    ) != EXPECTED_DATASETS:
        raise RuntimeError(
            "Primary analysis must contain exactly nine datasets."
        )

    if int(
        payload.get(
            "n_realizations_per_dataset",
            -1,
        )
    ) != EXPECTED_REALIZATIONS:
        raise RuntimeError(
            "Primary analysis must use five realizations per dataset."
        )

    return PrimaryAnalysisLock(
        frozen_evidence_sha256=str(
            payload[
                "frozen_evidence_sha256"
            ]
        ),
        reject_null=bool(
            payload[
                "reject_null"
            ]
        ),
        p_value=float(
            payload[
                "p_value"
            ]
        ),
    )


# =============================================================================
# MASTER LAYOUT AUDIT
# =============================================================================


_REQUIRED_MASTER_COLUMNS = {
    "task_id",
    "dataset_name",
    "feature_type",
    "realization",
    "strength_name",
    "interaction_coefficient",

    "mean_pairwise_jaccard",
    "mean_interaction_auprc",

    "n_selection_stable",
    "selection_precision",
    "selection_recall",
    "selection_f1",

    "n_isr_retained",
    "isr_precision",
    "isr_recall",
    "isr_f1",
    "false_positive_reduction",

    "main_auroc",
    "main_auprc",
    "main_balanced_accuracy",
    "main_f1",

    "agnam_auroc",
    "agnam_auprc",
    "agnam_balanced_accuracy",
    "agnam_f1",

    "delta_auroc",
    "delta_auprc",
    "delta_balanced_accuracy",
    "delta_f1",

    "oracle_auroc",
    "oracle_auprc",

    "random_pair_auroc",
    "random_pair_auprc",

    "no_isr_auroc",
    "no_isr_auprc",

    "single_run_auroc",
    "single_run_auprc",
}


def validate_secondary_master_layout(
    master: pd.DataFrame,
) -> None:
    missing = (
        _REQUIRED_MASTER_COLUMNS
        - set(
            master.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Frozen Real-X master table is missing "
            "secondary-analysis columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    if len(
        master
    ) != EXPECTED_RUNS:
        raise RuntimeError(
            "Secondary Real-X analysis requires "
            "exactly 180 frozen runs."
        )

    if (
        master[
            "task_id"
        ]
        .nunique()
        != EXPECTED_DATASETS
    ):
        raise RuntimeError(
            "Secondary analysis requires exactly nine datasets."
        )

    strength_counts = (
        master[
            "strength_name"
        ]
        .astype(
            str
        )
        .value_counts()
        .to_dict()
    )

    expected_strength_counts = {
        strength: 45
        for strength in (
            STRENGTH_ORDER
        )
    }

    if (
        strength_counts
        != expected_strength_counts
    ):
        raise RuntimeError(
            "Strength balance differs from the locked "
            f"45 × 4 design: {strength_counts}"
        )

    for (
        task_id,
        dataset_name,
    ), task_group in master.groupby(
        [
            "task_id",
            "dataset_name",
        ],
        sort=False,
    ):
        if len(
            task_group
        ) != 20:
            raise RuntimeError(
                f"Dataset {dataset_name} must contain 20 runs."
            )

        feature_types = (
            task_group[
                "feature_type"
            ]
            .astype(
                str
            )
            .unique()
        )

        if len(
            feature_types
        ) != 1:
            raise RuntimeError(
                f"Dataset {dataset_name} changed feature type."
            )

        for strength in (
            STRENGTH_ORDER
        ):
            group = task_group.loc[
                task_group[
                    "strength_name"
                ]
                .astype(
                    str
                )
                .eq(
                    strength
                )
            ]

            if len(
                group
            ) != EXPECTED_REALIZATIONS:
                raise RuntimeError(
                    f"Dataset {dataset_name}, strength {strength} "
                    "must contain exactly five realizations."
                )

            observed = set(
                pd.to_numeric(
                    group[
                        "realization"
                    ],
                    errors="raise",
                )
                .astype(
                    int
                )
                .tolist()
            )

            if observed != {
                1,
                2,
                3,
                4,
                5,
            }:
                raise RuntimeError(
                    f"Dataset {dataset_name}, strength {strength} "
                    "does not contain realizations 1-5 exactly."
                )


# =============================================================================
# DERIVED SECONDARY METRICS
# =============================================================================


def add_secondary_derived_metrics(
    master: pd.DataFrame,
) -> pd.DataFrame:
    validate_secondary_master_layout(
        master
    )

    table = master.copy()

    numeric_columns = (
        "n_selection_stable",
        "n_isr_retained",

        "agnam_auroc",
        "agnam_auprc",

        "oracle_auroc",
        "oracle_auprc",

        "random_pair_auroc",
        "random_pair_auprc",

        "no_isr_auroc",
        "no_isr_auprc",

        "single_run_auroc",
        "single_run_auprc",
    )

    for column in numeric_columns:
        table[
            column
        ] = pd.to_numeric(
            table[
                column
            ],
            errors="raise",
        )

    selection = (
        table[
            "n_selection_stable"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    retained = (
        table[
            "n_isr_retained"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    if np.any(
        retained
        > selection
        + 1e-12
    ):
        raise RuntimeError(
            "ISR retained count exceeds the "
            "selection-stable interaction count."
        )

    sparsification = np.full(
        len(
            table
        ),
        np.nan,
        dtype=np.float64,
    )

    defined = (
        selection
        > 0.0
    )

    sparsification[
        defined
    ] = (
        selection[
            defined
        ]
        -
        retained[
            defined
        ]
    ) / selection[
        defined
    ]

    table[
        "isr_sparsification_fraction"
    ] = (
        sparsification
    )

    table[
        "agnam_minus_random_pair_auroc"
    ] = (
        table[
            "agnam_auroc"
        ]
        -
        table[
            "random_pair_auroc"
        ]
    )

    table[
        "agnam_minus_random_pair_auprc"
    ] = (
        table[
            "agnam_auprc"
        ]
        -
        table[
            "random_pair_auprc"
        ]
    )

    table[
        "agnam_minus_no_isr_auroc"
    ] = (
        table[
            "agnam_auroc"
        ]
        -
        table[
            "no_isr_auroc"
        ]
    )

    table[
        "agnam_minus_no_isr_auprc"
    ] = (
        table[
            "agnam_auprc"
        ]
        -
        table[
            "no_isr_auprc"
        ]
    )

    table[
        "agnam_minus_single_run_auroc"
    ] = (
        table[
            "agnam_auroc"
        ]
        -
        table[
            "single_run_auroc"
        ]
    )

    table[
        "agnam_minus_single_run_auprc"
    ] = (
        table[
            "agnam_auprc"
        ]
        -
        table[
            "single_run_auprc"
        ]
    )

    table[
        "oracle_minus_agnam_auroc"
    ] = (
        table[
            "oracle_auroc"
        ]
        -
        table[
            "agnam_auroc"
        ]
    )

    table[
        "oracle_minus_agnam_auprc"
    ] = (
        table[
            "oracle_auprc"
        ]
        -
        table[
            "agnam_auprc"
        ]
    )

    return table


# =============================================================================
# DATASET × STRENGTH AGGREGATION
# =============================================================================


_DATASET_STRENGTH_METRICS = (
    STRUCTURE_METRICS
    + BASE_PREDICTIVE_METRICS
    + (
        "oracle_auroc",
        "oracle_auprc",
        "random_pair_auroc",
        "random_pair_auprc",
        "no_isr_auroc",
        "no_isr_auprc",
        "single_run_auroc",
        "single_run_auprc",
    )
    + COMPARATOR_DERIVED_METRICS
)


def _mean_numeric(
    series: pd.Series,
) -> float:
    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    return float(
        values.mean(
            skipna=True
        )
    )


def build_dataset_strength_table(
    master_with_derived: pd.DataFrame,
) -> pd.DataFrame:
    validate_secondary_master_layout(
        master_with_derived
    )

    required_derived = {
        "isr_sparsification_fraction",
        *COMPARATOR_DERIVED_METRICS,
    }

    missing = (
        required_derived
        - set(
            master_with_derived.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Derived secondary metrics are missing: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    records = []

    grouped = (
        master_with_derived
        .groupby(
            [
                "task_id",
                "dataset_name",
                "feature_type",
                "strength_name",
            ],
            sort=True,
            dropna=False,
        )
    )

    if (
        grouped.ngroups
        != EXPECTED_DATASET_STRENGTH_ROWS
    ):
        raise RuntimeError(
            "Expected exactly 36 dataset-strength groups."
        )

    for (
        task_id,
        dataset_name,
        feature_type,
        strength_name,
    ), group in grouped:
        if len(
            group
        ) != EXPECTED_REALIZATIONS:
            raise RuntimeError(
                f"{dataset_name}/{strength_name} "
                "does not contain exactly five realizations."
            )

        observed_realizations = set(
            pd.to_numeric(
                group[
                    "realization"
                ],
                errors="raise",
            )
            .astype(
                int
            )
            .tolist()
        )

        if observed_realizations != {
            1,
            2,
            3,
            4,
            5,
        }:
            raise RuntimeError(
                f"{dataset_name}/{strength_name} "
                "does not contain realization set 1-5."
            )

        coefficient_values = (
            pd.to_numeric(
                group[
                    "interaction_coefficient"
                ],
                errors="raise",
            )
            .to_numpy(
                dtype=np.float64
            )
        )

        if not np.allclose(
            coefficient_values,
            coefficient_values[
                0
            ],
            atol=1e-12,
            rtol=0.0,
        ):
            raise RuntimeError(
                f"{dataset_name}/{strength_name} "
                "changed interaction coefficient."
            )

        record = {
            "task_id": int(
                task_id
            ),
            "dataset_name": str(
                dataset_name
            ),
            "feature_type": str(
                feature_type
            ),
            "strength_name": str(
                strength_name
            ),
            "interaction_coefficient": float(
                coefficient_values[
                    0
                ]
            ),
            "n_realizations": int(
                len(
                    group
                )
            ),
        }

        for metric in (
            _DATASET_STRENGTH_METRICS
        ):
            record[
                metric
            ] = _mean_numeric(
                group[
                    metric
                ]
            )

        records.append(
            record
        )

    output = pd.DataFrame(
        records
    )

    output[
        "_strength_order"
    ] = output[
        "strength_name"
    ].map(
        {
            name: index
            for (
                index,
                name,
            ) in enumerate(
                STRENGTH_ORDER
            )
        }
    )

    if output[
        "_strength_order"
    ].isna().any():
        raise RuntimeError(
            "Unexpected strength name in dataset-strength table."
        )

    output = (
        output
        .sort_values(
            [
                "task_id",
                "_strength_order",
            ]
        )
        .drop(
            columns=[
                "_strength_order"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    if len(
        output
    ) != EXPECTED_DATASET_STRENGTH_ROWS:
        raise RuntimeError(
            "Dataset-strength output must contain 36 rows."
        )

    return output


# =============================================================================
# GENERIC DESCRIPTIVE SUMMARY
# =============================================================================


def _describe_values(
    values: Sequence[
        float
    ],
) -> dict[str, float | int]:
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    array = array[
        np.isfinite(
            array
        )
    ]

    n = int(
        len(
            array
        )
    )

    if n == 0:
        return {
            "n": 0,
            "mean": float(
                "nan"
            ),
            "std": float(
                "nan"
            ),
            "median": float(
                "nan"
            ),
            "q25": float(
                "nan"
            ),
            "q75": float(
                "nan"
            ),
            "minimum": float(
                "nan"
            ),
            "maximum": float(
                "nan"
            ),
        }

    return {
        "n": n,

        "mean": float(
            np.mean(
                array
            )
        ),

        "std": float(
            np.std(
                array,
                ddof=1,
            )
            if n > 1
            else 0.0
        ),

        "median": float(
            np.median(
                array
            )
        ),

        "q25": float(
            np.quantile(
                array,
                0.25,
            )
        ),

        "q75": float(
            np.quantile(
                array,
                0.75,
            )
        ),

        "minimum": float(
            np.min(
                array
            )
        ),

        "maximum": float(
            np.max(
                array
            )
        ),
    }


def build_metric_summary(
    table: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    metrics: Sequence[str],
) -> pd.DataFrame:
    records = []

    grouped = table.groupby(
        list(
            group_columns
        ),
        sort=True,
        dropna=False,
    )

    for group_key, group in grouped:
        if not isinstance(
            group_key,
            tuple,
        ):
            group_key = (
                group_key,
            )

        group_values = {
            name: value
            for (
                name,
                value,
            ) in zip(
                group_columns,
                group_key,
            )
        }

        for metric in metrics:
            description = _describe_values(
                pd.to_numeric(
                    group[
                        metric
                    ],
                    errors="coerce",
                )
                .to_numpy(
                    dtype=np.float64
                )
            )

            record = dict(
                group_values
            )

            record[
                "metric"
            ] = metric

            record.update(
                description
            )

            records.append(
                record
            )

    return pd.DataFrame(
        records
    )


def build_strength_summary(
    dataset_strength_table: pd.DataFrame,
) -> pd.DataFrame:
    summary = build_metric_summary(
        dataset_strength_table,
        group_columns=(
            "strength_name",
        ),
        metrics=(
            STRENGTH_SUMMARY_METRICS
        ),
    )

    summary[
        "_strength_order"
    ] = summary[
        "strength_name"
    ].map(
        {
            strength: index
            for (
                index,
                strength,
            ) in enumerate(
                STRENGTH_ORDER
            )
        }
    )

    summary = (
        summary
        .sort_values(
            [
                "_strength_order",
                "metric",
            ]
        )
        .drop(
            columns=[
                "_strength_order"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return summary


# =============================================================================
# PREDICTIVE COMPARATOR SUMMARY
# =============================================================================


_COMPARATOR_METRICS = {
    "AG-NAM minus Main NAM": (
        "delta_auroc",
        "delta_auprc",
        "delta_balanced_accuracy",
        "delta_f1",
    ),

    "AG-NAM minus Random-Pair NAM": (
        "agnam_minus_random_pair_auroc",
        "agnam_minus_random_pair_auprc",
    ),

    "AG-NAM minus No-ISR AG-NAM": (
        "agnam_minus_no_isr_auroc",
        "agnam_minus_no_isr_auprc",
    ),

    "AG-NAM minus Single-Run AG-NAM": (
        "agnam_minus_single_run_auroc",
        "agnam_minus_single_run_auprc",
    ),

    "Oracle minus AG-NAM": (
        "oracle_minus_agnam_auroc",
        "oracle_minus_agnam_auprc",
    ),
}


def _sign_counts(
    values: np.ndarray,
    *,
    tolerance: float = 1e-12,
) -> tuple[
    int,
    int,
    int,
]:
    positive = int(
        np.sum(
            values
            > tolerance
        )
    )

    zero = int(
        np.sum(
            np.isclose(
                values,
                0.0,
                atol=tolerance,
                rtol=0.0,
            )
        )
    )

    negative = int(
        np.sum(
            values
            < -tolerance
        )
    )

    return (
        positive,
        zero,
        negative,
    )


def build_predictive_comparator_summary(
    dataset_strength_table: pd.DataFrame,
) -> pd.DataFrame:
    records = []

    for strength in (
        STRENGTH_ORDER
    ):
        strength_table = (
            dataset_strength_table
            .loc[
                dataset_strength_table[
                    "strength_name"
                ]
                .astype(
                    str
                )
                .eq(
                    strength
                )
            ]
            .copy()
        )

        if len(
            strength_table
        ) != EXPECTED_DATASETS:
            raise RuntimeError(
                f"Strength {strength} must contain "
                "exactly nine dataset-level rows."
            )

        for (
            comparator,
            metric_names,
        ) in (
            _COMPARATOR_METRICS
            .items()
        ):
            for metric in metric_names:
                values = pd.to_numeric(
                    strength_table[
                        metric
                    ],
                    errors="raise",
                ).to_numpy(
                    dtype=np.float64
                )

                if not np.isfinite(
                    values
                ).all():
                    raise RuntimeError(
                        f"Predictive comparator {metric} "
                        "contains non-finite dataset values."
                    )

                description = (
                    _describe_values(
                        values
                    )
                )

                (
                    ci_lower,
                    ci_upper,
                ) = paired_bootstrap_ci(
                    values,
                    statistic="mean",
                    resamples=(
                        BOOTSTRAP_RESAMPLES
                    ),
                    seed=(
                        BOOTSTRAP_SEED
                    ),
                    alpha=ALPHA,
                )

                (
                    positive,
                    zero,
                    negative,
                ) = _sign_counts(
                    values
                )

                records.append(
                    {
                        "strength_name": (
                            strength
                        ),

                        "comparator": (
                            comparator
                        ),

                        "metric": metric,

                        **description,

                        "bootstrap_statistic": (
                            "mean"
                        ),

                        "bootstrap_resamples": (
                            BOOTSTRAP_RESAMPLES
                        ),

                        "bootstrap_seed": (
                            BOOTSTRAP_SEED
                        ),

                        "bootstrap_ci_level": (
                            1.0
                            - ALPHA
                        ),

                        "bootstrap_ci_lower": float(
                            ci_lower
                        ),

                        "bootstrap_ci_upper": float(
                            ci_upper
                        ),

                        "positive_dataset_count": (
                            positive
                        ),

                        "zero_dataset_count": (
                            zero
                        ),

                        "negative_dataset_count": (
                            negative
                        ),

                        "inference": (
                            "descriptive_only"
                        ),
                    }
                )

    summary = pd.DataFrame(
        records
    )

    summary[
        "_strength_order"
    ] = summary[
        "strength_name"
    ].map(
        {
            strength: index
            for (
                index,
                strength,
            ) in enumerate(
                STRENGTH_ORDER
            )
        }
    )

    summary = (
        summary
        .sort_values(
            [
                "_strength_order",
                "comparator",
                "metric",
            ]
        )
        .drop(
            columns=[
                "_strength_order"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    forbidden = {
        "p_value",
        "pvalue",
        "reject_null",
        "wilcoxon_statistic",
        "holm_p_value",
    }

    if (
        forbidden
        & set(
            summary.columns
        )
    ):
        raise RuntimeError(
            "Secondary descriptive comparator table "
            "must not contain inferential test outputs."
        )

    return summary


# =============================================================================
# FEATURE-TYPE DESCRIPTIVE SUMMARY
# =============================================================================


def build_feature_type_strength_summary(
    dataset_strength_table: pd.DataFrame,
) -> pd.DataFrame:
    summary = build_metric_summary(
        dataset_strength_table,
        group_columns=(
            "feature_type",
            "strength_name",
        ),
        metrics=(
            FEATURE_TYPE_SUMMARY_METRICS
        ),
    )

    summary[
        "_strength_order"
    ] = summary[
        "strength_name"
    ].map(
        {
            strength: index
            for (
                index,
                strength,
            ) in enumerate(
                STRENGTH_ORDER
            )
        }
    )

    summary = (
        summary
        .sort_values(
            [
                "feature_type",
                "_strength_order",
                "metric",
            ]
        )
        .drop(
            columns=[
                "_strength_order"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return summary


# =============================================================================
# COMPLETE SECONDARY ANALYSIS
# =============================================================================


def analyze_frozen_realx_secondary(
    *,
    output_root: str | Path = (
        DEFAULT_OUTPUT_ROOT
    ),
    freeze_manifest_path: str | Path = (
        DEFAULT_MANIFEST_PATH
    ),
    primary_result_path: str | Path = (
        DEFAULT_PRIMARY_RESULT_PATH
    ),
    repo_root: str | Path = ".",
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    FreezeValidationResult,
    PrimaryAnalysisLock,
]:
    """
    Secondary/descriptive hard gate.

    Frozen evidence is validated before the master result table is
    opened. The completed primary analysis must refer to the identical
    frozen evidence hash.
    """
    freeze_validation = (
        validate_realx_evidence_freeze(
            manifest_path=(
                freeze_manifest_path
            ),
            output_root=(
                output_root
            ),
            repo_root=(
                repo_root
            ),
        )
    )

    if not freeze_validation.valid:
        raise RuntimeError(
            "Frozen Real-X evidence validation failed."
        )

    primary_lock = (
        validate_primary_analysis_lock(
            primary_result_path=(
                primary_result_path
            ),
            expected_frozen_evidence_sha256=(
                freeze_validation
                .aggregate_evidence_sha256
            ),
        )
    )

    master_path = (
        Path(
            output_root
        )
        / "master_results.csv"
    )

    master = _read_protocol_csv(
        master_path
    )

    enriched = (
        add_secondary_derived_metrics(
            master
        )
    )

    dataset_strength = (
        build_dataset_strength_table(
            enriched
        )
    )

    strength_summary = (
        build_strength_summary(
            dataset_strength
        )
    )

    predictive_comparator = (
        build_predictive_comparator_summary(
            dataset_strength
        )
    )

    feature_type_summary = (
        build_feature_type_strength_summary(
            dataset_strength
        )
    )

    return (
        dataset_strength,
        strength_summary,
        predictive_comparator,
        feature_type_summary,
        freeze_validation,
        primary_lock,
    )