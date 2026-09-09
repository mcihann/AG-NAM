from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scipy.stats import wilcoxon

from agnam.benchmarking.realx_batch import (
    DEFAULT_OUTPUT_ROOT,
    _read_protocol_csv,
)
from agnam.benchmarking.realx_freeze import (
    DEFAULT_MANIFEST_PATH,
    FreezeValidationResult,
    validate_realx_evidence_freeze,
)
from agnam.benchmarking.statistics import (
    paired_bootstrap_ci,
)


# =============================================================================
# LOCKED CONFIRMATORY PLAN
# =============================================================================


PRIMARY_CONDITION = "moderate"

PRIMARY_INTERACTION_COEFFICIENT = 1.0

PRIMARY_TEST = (
    "one_sided_wilcoxon_greater"
)

PRIMARY_METRIC = (
    "interaction_auprc_minus_random_prevalence"
)

EXPERIMENTAL_UNIT = "dataset"

REALIZATION_AGGREGATION = (
    "mean_over_5_realizations"
)

ALPHA = 0.05

BOOTSTRAP_RESAMPLES = 10_000

BOOTSTRAP_SEED = 26_090_806

EXPECTED_DATASETS = 9

EXPECTED_REALIZATIONS_PER_DATASET = 5

EXPECTED_PRIMARY_ROWS = (
    EXPECTED_DATASETS
    * EXPECTED_REALIZATIONS_PER_DATASET
)

EXPECTED_MODERATE_TRUE_INTERACTIONS = 3

ANALYSIS_SCHEMA_VERSION = (
    "realx_confirmatory_statistics_v1"
)

DEFAULT_STATISTICS_DIRECTORY = (
    DEFAULT_OUTPUT_ROOT
    / "statistics"
)

DEFAULT_PRIMARY_REALIZATION_PATH = (
    DEFAULT_STATISTICS_DIRECTORY
    / "primary_realization_values.csv"
)

DEFAULT_PRIMARY_DATASET_PATH = (
    DEFAULT_STATISTICS_DIRECTORY
    / "primary_dataset_values.csv"
)

DEFAULT_PRIMARY_RESULT_PATH = (
    DEFAULT_STATISTICS_DIRECTORY
    / "primary_confirmatory_result.json"
)

DEFAULT_ANALYSIS_MANIFEST_PATH = (
    DEFAULT_STATISTICS_DIRECTORY
    / "analysis_manifest.json"
)


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass(frozen=True)
class PrimaryConfirmatoryResult:
    analysis_schema_version: str

    experimental_unit: str

    primary_condition: str
    interaction_coefficient: float

    primary_metric: str
    primary_test: str

    alpha: float

    multiplicity_adjustment: str

    realization_aggregation: str

    n_datasets: int
    n_realizations_per_dataset: int

    wilcoxon_statistic: float
    p_value: float
    n_nonzero_differences: int

    reject_null: bool

    mean_dataset_difference: float
    median_dataset_difference: float

    bootstrap_statistic: str
    bootstrap_resamples: int
    bootstrap_seed: int

    bootstrap_ci_level: float
    bootstrap_ci_lower: float
    bootstrap_ci_upper: float

    positive_dataset_count: int
    zero_dataset_count: int
    negative_dataset_count: int

    min_dataset_difference: float
    max_dataset_difference: float

    frozen_evidence_sha256: str
    phase6e_git_commit: str

    def to_dict(
        self,
    ) -> dict[str, Any]:
        return asdict(
            self
        )


# =============================================================================
# RANDOM-RANKING PREVALENCE
# =============================================================================


def pair_universe_size(
    n_features: int,
) -> int:
    """
    Number of unique unordered non-self feature pairs:

        P = p(p - 1) / 2
    """
    n_features = int(
        n_features
    )

    if n_features < 2:
        raise ValueError(
            "At least two features are required "
            "for an interaction-ranking universe."
        )

    return int(
        n_features
        * (
            n_features - 1
        )
        // 2
    )


def random_ranking_interaction_prevalence(
    *,
    n_features: int,
    n_true_interactions: int,
) -> float:
    """
    Random-ranking AUPRC reference prevalence.

    Interaction AUPRC is computed over the complete pairwise ranking
    universe. The random-ranking reference is therefore the positive
    prevalence in that universe:

        prevalence = m / C(p, 2)
    """
    n_true_interactions = int(
        n_true_interactions
    )

    n_pairs = pair_universe_size(
        n_features
    )

    if n_true_interactions < 0:
        raise ValueError(
            "n_true_interactions cannot be negative."
        )

    if n_true_interactions > n_pairs:
        raise ValueError(
            "n_true_interactions cannot exceed "
            "the pairwise ranking universe."
        )

    return float(
        n_true_interactions
        / n_pairs
    )


# =============================================================================
# CONFIRMATORY WILCOXON
# =============================================================================


def one_sided_wilcoxon_greater(
    differences: np.ndarray,
) -> tuple[
    float,
    float,
    int,
]:
    """
    Locked Real-X primary test.

    Approximate zeros are removed using the same 1e-12 rule already
    used by the existing project Wilcoxon implementation.
    """
    values = np.asarray(
        differences,
        dtype=np.float64,
    )

    if values.ndim != 1:
        raise ValueError(
            "differences must be one-dimensional."
        )

    if len(
        values
    ) == 0:
        raise ValueError(
            "differences cannot be empty."
        )

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            "differences must contain only finite values."
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
        alternative="greater",
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


# =============================================================================
# PRIMARY TABLE CONSTRUCTION
# =============================================================================


_REQUIRED_PRIMARY_COLUMNS = {
    "task_id",
    "dataset_name",
    "feature_type",
    "realization",
    "strength_name",
    "interaction_coefficient",
    "n_features",
    "n_true_interactions",
    "mean_interaction_auprc",
}


def _validate_primary_source_columns(
    master: pd.DataFrame,
) -> None:
    missing = (
        _REQUIRED_PRIMARY_COLUMNS
        - set(
            master.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Frozen master results are missing "
            "primary-analysis columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )


def build_primary_realization_table(
    master: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract the 45 prespecified Moderate-condition realization rows
    and calculate the frozen realization-level primary metric.
    """
    _validate_primary_source_columns(
        master
    )

    primary = (
        master.loc[
            master[
                "strength_name"
            ]
            .astype(
                str
            )
            .eq(
                PRIMARY_CONDITION
            )
        ]
        .copy()
    )

    if len(
        primary
    ) != EXPECTED_PRIMARY_ROWS:
        raise RuntimeError(
            "Primary Moderate condition must contain "
            f"exactly {EXPECTED_PRIMARY_ROWS} rows; "
            f"found {len(primary)}."
        )

    if (
        primary[
            "task_id"
        ]
        .nunique()
        != EXPECTED_DATASETS
    ):
        raise RuntimeError(
            "Primary analysis must contain exactly "
            f"{EXPECTED_DATASETS} datasets."
        )

    coefficient = pd.to_numeric(
        primary[
            "interaction_coefficient"
        ],
        errors="raise",
    ).to_numpy(
        dtype=np.float64
    )

    if not np.allclose(
        coefficient,
        PRIMARY_INTERACTION_COEFFICIENT,
        atol=1e-12,
        rtol=0.0,
    ):
        raise RuntimeError(
            "Primary Moderate rows do not all use lambda=1.0."
        )

    primary[
        "n_features"
    ] = pd.to_numeric(
        primary[
            "n_features"
        ],
        errors="raise",
    ).astype(
        np.int64
    )

    primary[
        "n_true_interactions"
    ] = pd.to_numeric(
        primary[
            "n_true_interactions"
        ],
        errors="raise",
    ).astype(
        np.int64
    )

    if not (
        primary[
            "n_true_interactions"
        ]
        .eq(
            EXPECTED_MODERATE_TRUE_INTERACTIONS
        )
        .all()
    ):
        raise RuntimeError(
            "Every Moderate realization must contain "
            "exactly three active true interactions."
        )

    interaction_auprc = pd.to_numeric(
        primary[
            "mean_interaction_auprc"
        ],
        errors="raise",
    ).to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(
        interaction_auprc
    ).all():
        raise RuntimeError(
            "Primary interaction AUPRC contains "
            "non-finite values."
        )

    if np.any(
        interaction_auprc < 0.0
    ) or np.any(
        interaction_auprc > 1.0
    ):
        raise RuntimeError(
            "Primary interaction AUPRC must lie in [0, 1]."
        )

    n_pairs = []

    random_prevalence = []

    for row in primary.itertuples(
        index=False
    ):
        universe = pair_universe_size(
            int(
                row.n_features
            )
        )

        prevalence = (
            random_ranking_interaction_prevalence(
                n_features=int(
                    row.n_features
                ),
                n_true_interactions=int(
                    row.n_true_interactions
                ),
            )
        )

        n_pairs.append(
            universe
        )

        random_prevalence.append(
            prevalence
        )

    primary[
        "interaction_pair_universe"
    ] = np.asarray(
        n_pairs,
        dtype=np.int64,
    )

    primary[
        "random_ranking_interaction_prevalence"
    ] = np.asarray(
        random_prevalence,
        dtype=np.float64,
    )

    primary[
        "primary_difference"
    ] = (
        interaction_auprc
        -
        primary[
            "random_ranking_interaction_prevalence"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    output_columns = [
        "task_id",
        "dataset_name",
        "feature_type",
        "realization",
        "strength_name",
        "interaction_coefficient",
        "n_features",
        "interaction_pair_universe",
        "n_true_interactions",
        "mean_interaction_auprc",
        "random_ranking_interaction_prevalence",
        "primary_difference",
    ]

    primary = (
        primary[
            output_columns
        ]
        .sort_values(
            [
                "task_id",
                "realization",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return primary


def build_primary_dataset_table(
    realization_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply the prespecified experimental-unit rule:

        five realizations -> arithmetic mean -> one value per dataset
    """
    if len(
        realization_table
    ) != EXPECTED_PRIMARY_ROWS:
        raise RuntimeError(
            "Primary realization table must contain 45 rows."
        )

    records = []

    group_columns = [
        "task_id",
        "dataset_name",
        "feature_type",
    ]

    grouped = realization_table.groupby(
        group_columns,
        sort=True,
        dropna=False,
    )

    if grouped.ngroups != EXPECTED_DATASETS:
        raise RuntimeError(
            "Primary realization table must resolve "
            "to exactly nine dataset groups."
        )

    expected_realizations = set(
        range(
            1,
            EXPECTED_REALIZATIONS_PER_DATASET
            + 1,
        )
    )

    for (
        task_id,
        dataset_name,
        feature_type,
    ), group in grouped:
        if len(
            group
        ) != EXPECTED_REALIZATIONS_PER_DATASET:
            raise RuntimeError(
                f"Dataset {dataset_name} does not contain "
                "exactly five Moderate realizations."
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

        if (
            observed_realizations
            != expected_realizations
        ):
            raise RuntimeError(
                f"Dataset {dataset_name} realization set "
                f"is {sorted(observed_realizations)}, "
                "expected [1, 2, 3, 4, 5]."
            )

        n_feature_values = (
            group[
                "n_features"
            ]
            .astype(
                int
            )
            .unique()
        )

        if len(
            n_feature_values
        ) != 1:
            raise RuntimeError(
                f"Dataset {dataset_name} changed feature count "
                "across realizations."
            )

        pair_universe_values = (
            group[
                "interaction_pair_universe"
            ]
            .astype(
                int
            )
            .unique()
        )

        if len(
            pair_universe_values
        ) != 1:
            raise RuntimeError(
                f"Dataset {dataset_name} changed pair universe "
                "across realizations."
            )

        true_count_values = (
            group[
                "n_true_interactions"
            ]
            .astype(
                int
            )
            .unique()
        )

        if (
            len(
                true_count_values
            ) != 1
            or int(
                true_count_values[
                    0
                ]
            )
            != EXPECTED_MODERATE_TRUE_INTERACTIONS
        ):
            raise RuntimeError(
                f"Dataset {dataset_name} has an invalid "
                "Moderate true-interaction count."
            )

        prevalence_values = (
            group[
                "random_ranking_interaction_prevalence"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        if not np.allclose(
            prevalence_values,
            prevalence_values[
                0
            ],
            atol=1e-15,
            rtol=0.0,
        ):
            raise RuntimeError(
                f"Dataset {dataset_name} changed random-ranking "
                "prevalence across realizations."
            )

        records.append(
            {
                "task_id": int(
                    task_id
                ),
                "dataset_name": str(
                    dataset_name
                ),
                "feature_type": str(
                    feature_type
                ),

                "n_realizations": int(
                    len(
                        group
                    )
                ),

                "n_features": int(
                    n_feature_values[
                        0
                    ]
                ),

                "interaction_pair_universe": int(
                    pair_universe_values[
                        0
                    ]
                ),

                "n_true_interactions": int(
                    true_count_values[
                        0
                    ]
                ),

                "random_ranking_interaction_prevalence": float(
                    prevalence_values[
                        0
                    ]
                ),

                "mean_interaction_auprc": float(
                    group[
                        "mean_interaction_auprc"
                    ]
                    .to_numpy(
                        dtype=np.float64
                    )
                    .mean()
                ),

                "primary_difference": float(
                    group[
                        "primary_difference"
                    ]
                    .to_numpy(
                        dtype=np.float64
                    )
                    .mean()
                ),
            }
        )

    dataset_table = (
        pd.DataFrame(
            records
        )
        .sort_values(
            "task_id"
        )
        .reset_index(
            drop=True
        )
    )

    if len(
        dataset_table
    ) != EXPECTED_DATASETS:
        raise RuntimeError(
            "Primary dataset table must contain exactly nine rows."
        )

    return dataset_table


# =============================================================================
# PRIMARY CONFIRMATORY ANALYSIS
# =============================================================================


def run_primary_confirmatory_analysis(
    dataset_table: pd.DataFrame,
    *,
    frozen_evidence_sha256: str,
    phase6e_git_commit: str,
) -> PrimaryConfirmatoryResult:
    if len(
        dataset_table
    ) != EXPECTED_DATASETS:
        raise RuntimeError(
            "Confirmatory inference requires exactly "
            "nine dataset-level values."
        )

    differences = pd.to_numeric(
        dataset_table[
            "primary_difference"
        ],
        errors="raise",
    ).to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(
        differences
    ).all():
        raise RuntimeError(
            "Dataset-level primary differences "
            "must all be finite."
        )

    (
        statistic,
        p_value,
        n_nonzero,
    ) = one_sided_wilcoxon_greater(
        differences
    )

    (
        ci_lower,
        ci_upper,
    ) = paired_bootstrap_ci(
        differences,
        statistic="mean",
        resamples=(
            BOOTSTRAP_RESAMPLES
        ),
        seed=BOOTSTRAP_SEED,
        alpha=ALPHA,
    )

    tolerance = 1e-12

    positive = int(
        np.sum(
            differences
            > tolerance
        )
    )

    zero = int(
        np.sum(
            np.isclose(
                differences,
                0.0,
                atol=tolerance,
                rtol=0.0,
            )
        )
    )

    negative = int(
        np.sum(
            differences
            < -tolerance
        )
    )

    return PrimaryConfirmatoryResult(
        analysis_schema_version=(
            ANALYSIS_SCHEMA_VERSION
        ),

        experimental_unit=(
            EXPERIMENTAL_UNIT
        ),

        primary_condition=(
            PRIMARY_CONDITION
        ),

        interaction_coefficient=(
            PRIMARY_INTERACTION_COEFFICIENT
        ),

        primary_metric=(
            PRIMARY_METRIC
        ),

        primary_test=(
            PRIMARY_TEST
        ),

        alpha=ALPHA,

        multiplicity_adjustment=(
            "none_single_primary"
        ),

        realization_aggregation=(
            REALIZATION_AGGREGATION
        ),

        n_datasets=(
            EXPECTED_DATASETS
        ),

        n_realizations_per_dataset=(
            EXPECTED_REALIZATIONS_PER_DATASET
        ),

        wilcoxon_statistic=float(
            statistic
        ),

        p_value=float(
            p_value
        ),

        n_nonzero_differences=int(
            n_nonzero
        ),

        reject_null=bool(
            p_value
            < ALPHA
        ),

        mean_dataset_difference=float(
            np.mean(
                differences
            )
        ),

        median_dataset_difference=float(
            np.median(
                differences
            )
        ),

        bootstrap_statistic="mean",

        bootstrap_resamples=(
            BOOTSTRAP_RESAMPLES
        ),

        bootstrap_seed=(
            BOOTSTRAP_SEED
        ),

        bootstrap_ci_level=float(
            1.0
            - ALPHA
        ),

        bootstrap_ci_lower=float(
            ci_lower
        ),

        bootstrap_ci_upper=float(
            ci_upper
        ),

        positive_dataset_count=(
            positive
        ),

        zero_dataset_count=(
            zero
        ),

        negative_dataset_count=(
            negative
        ),

        min_dataset_difference=float(
            np.min(
                differences
            )
        ),

        max_dataset_difference=float(
            np.max(
                differences
            )
        ),

        frozen_evidence_sha256=str(
            frozen_evidence_sha256
        ),

        phase6e_git_commit=str(
            phase6e_git_commit
        ),
    )


# =============================================================================
# FROZEN EVIDENCE GATE
# =============================================================================


def analyze_frozen_realx_primary(
    *,
    output_root: str | Path = (
        DEFAULT_OUTPUT_ROOT
    ),
    manifest_path: str | Path = (
        DEFAULT_MANIFEST_PATH
    ),
    repo_root: str | Path = ".",
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    PrimaryConfirmatoryResult,
    FreezeValidationResult,
]:
    """
    Hard analysis gate.

    No frozen result is opened until the complete Phase 6F-A evidence
    freeze has independently validated.
    """
    freeze_validation = (
        validate_realx_evidence_freeze(
            manifest_path=(
                manifest_path
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

    master_path = (
        Path(
            output_root
        )
        / "master_results.csv"
    )

    master = _read_protocol_csv(
        master_path
    )

    realization_table = (
        build_primary_realization_table(
            master
        )
    )

    dataset_table = (
        build_primary_dataset_table(
            realization_table
        )
    )

    result = (
        run_primary_confirmatory_analysis(
            dataset_table,
            frozen_evidence_sha256=(
                freeze_validation
                .aggregate_evidence_sha256
            ),
            phase6e_git_commit=(
                freeze_validation
                .phase6e_git_commit
            ),
        )
    )

    return (
        realization_table,
        dataset_table,
        result,
        freeze_validation,
    )