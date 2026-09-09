import numpy as np
import pandas as pd
import pytest

from agnam.benchmarking.realx_reporting import (
    EXPECTED_DATASETS,
    build_feature_type_publication_table,
    build_predictive_publication_table,
    build_primary_publication_table,
    build_primary_summary_table,
    build_strength_publication_table,
    extract_figure_series,
    validate_reporting_metadata,
)
from agnam.benchmarking.realx_statistics import (
    ANALYSIS_SCHEMA_VERSION,
    PRIMARY_CONDITION,
    PRIMARY_METRIC,
    PRIMARY_TEST,
)
from agnam.benchmarking.realx_secondary import (
    SECONDARY_ANALYSIS_SCHEMA_VERSION,
    STRENGTH_ORDER,
)


def make_primary_payload(
    evidence_hash="a" * 64,
    commit="b" * 40,
):
    return {
        "analysis_schema_version": (
            ANALYSIS_SCHEMA_VERSION
        ),
        "frozen_evidence_sha256": (
            evidence_hash
        ),
        "phase6e_git_commit": (
            commit
        ),
        "experimental_unit": "dataset",
        "primary_condition": (
            PRIMARY_CONDITION
        ),
        "interaction_coefficient": 1.0,
        "primary_metric": (
            PRIMARY_METRIC
        ),
        "primary_test": (
            PRIMARY_TEST
        ),
        "n_datasets": 9,
        "n_realizations_per_dataset": 5,
        "wilcoxon_statistic": 45.0,
        "p_value": 0.001953125,
        "reject_null": True,
        "mean_dataset_difference": 0.15,
        "median_dataset_difference": 0.04,
        "bootstrap_ci_lower": 0.06,
        "bootstrap_ci_upper": 0.25,
        "positive_dataset_count": 9,
        "zero_dataset_count": 0,
        "negative_dataset_count": 0,
    }


def make_secondary_payload(
    evidence_hash="a" * 64,
):
    return {
        "analysis_schema_version": (
            SECONDARY_ANALYSIS_SCHEMA_VERSION
        ),
        "analysis_status": (
            "SECONDARY_DESCRIPTIVE_COMPLETE"
        ),
        "inference_scope": (
            "descriptive_only"
        ),
        "frozen_evidence_sha256": (
            evidence_hash
        ),
        "dataset_strength_rows": 36,
        "secondary_confirmatory_tests": 0,
    }


def make_primary_dataset():
    records = []

    for index in range(
        EXPECTED_DATASETS
    ):
        records.append(
            {
                "task_id": (
                    100
                    + index
                ),
                "dataset_name": (
                    f"dataset_{index}"
                ),
                "feature_type": (
                    "numeric"
                ),
                "n_features": (
                    10
                    + index
                ),
                "interaction_pair_universe": (
                    45
                    + index
                ),
                "random_ranking_interaction_prevalence": (
                    0.03
                ),
                "mean_interaction_auprc": (
                    0.10
                    + index
                    * 0.01
                ),
                "primary_difference": (
                    0.07
                    + index
                    * 0.01
                ),
            }
        )

    return pd.DataFrame(
        records
    )


def make_strength_summary():
    records = []

    metrics = (
        "mean_interaction_auprc",
        "isr_sparsification_fraction",
        "delta_auroc",
        "delta_auprc",
    )

    for strength_index, strength in enumerate(
        STRENGTH_ORDER
    ):
        for metric in metrics:
            records.append(
                {
                    "strength_name": (
                        strength
                    ),
                    "metric": metric,
                    "n": (
                        0
                        if (
                            strength
                            == "null"
                            and metric
                            == "mean_interaction_auprc"
                        )
                        else 9
                    ),
                    "mean": (
                        np.nan
                        if (
                            strength
                            == "null"
                            and metric
                            == "mean_interaction_auprc"
                        )
                        else (
                            strength_index
                            * 0.01
                        )
                    ),
                    "std": 0.01,
                    "median": 0.01,
                    "q25": 0.00,
                    "q75": 0.02,
                    "minimum": -0.01,
                    "maximum": 0.03,
                }
            )

    return pd.DataFrame(
        records
    )


def make_dataset_strength():
    records = []

    for dataset_index in range(
        9
    ):
        for strength_index, strength in enumerate(
            STRENGTH_ORDER
        ):
            records.append(
                {
                    "task_id": (
                        100
                        + dataset_index
                    ),
                    "dataset_name": (
                        f"dataset_{dataset_index}"
                    ),
                    "feature_type": (
                        "numeric"
                    ),
                    "strength_name": (
                        strength
                    ),
                    "mean_interaction_auprc": (
                        np.nan
                        if strength
                        == "null"
                        else (
                            0.05
                            + strength_index
                            * 0.03
                            + dataset_index
                            * 0.001
                        )
                    ),
                    "isr_sparsification_fraction": (
                        0.8
                        - strength_index
                        * 0.1
                    ),
                    "delta_auroc": (
                        strength_index
                        * 0.01
                    ),
                }
            )

    return pd.DataFrame(
        records
    )


def test_reporting_metadata_accepts_matching_evidence():
    gate = validate_reporting_metadata(
        frozen_evidence_sha256=(
            "a"
            * 64
        ),
        phase6e_git_commit=(
            "b"
            * 40
        ),
        primary_payload=(
            make_primary_payload()
        ),
        secondary_payload=(
            make_secondary_payload()
        ),
    )

    assert gate.primary_reject_null is True


def test_reporting_metadata_rejects_primary_hash_mismatch():
    primary = (
        make_primary_payload(
            evidence_hash=(
                "c"
                * 64
            )
        )
    )

    with pytest.raises(
        RuntimeError
    ):
        validate_reporting_metadata(
            frozen_evidence_sha256=(
                "a"
                * 64
            ),
            phase6e_git_commit=(
                "b"
                * 40
            ),
            primary_payload=(
                primary
            ),
            secondary_payload=(
                make_secondary_payload()
            ),
        )


def test_reporting_metadata_rejects_secondary_hash_mismatch():
    secondary = (
        make_secondary_payload(
            evidence_hash=(
                "c"
                * 64
            )
        )
    )

    with pytest.raises(
        RuntimeError
    ):
        validate_reporting_metadata(
            frozen_evidence_sha256=(
                "a"
                * 64
            ),
            phase6e_git_commit=(
                "b"
                * 40
            ),
            primary_payload=(
                make_primary_payload()
            ),
            secondary_payload=(
                secondary
            ),
        )


def test_reporting_metadata_rejects_secondary_confirmatory_tests():
    secondary = (
        make_secondary_payload()
    )

    secondary[
        "secondary_confirmatory_tests"
    ] = 1

    with pytest.raises(
        RuntimeError
    ):
        validate_reporting_metadata(
            frozen_evidence_sha256=(
                "a"
                * 64
            ),
            phase6e_git_commit=(
                "b"
                * 40
            ),
            primary_payload=(
                make_primary_payload()
            ),
            secondary_payload=(
                secondary
            ),
        )


def test_primary_publication_table_has_nine_rows():
    table = (
        build_primary_publication_table(
            make_primary_dataset()
        )
    )

    assert len(
        table
    ) == 9


def test_primary_publication_table_has_readable_difference_column():
    table = (
        build_primary_publication_table(
            make_primary_dataset()
        )
    )

    assert (
        "AUPRC minus random prevalence"
        in table.columns
    )


def test_primary_summary_table_has_one_row():
    table = (
        build_primary_summary_table(
            make_primary_payload()
        )
    )

    assert len(
        table
    ) == 1

    assert np.isclose(
        table.iloc[
            0
        ][
            "One-sided p-value"
        ],
        0.001953125,
    )


def test_strength_publication_table_has_sixteen_rows():
    table = (
        build_strength_publication_table(
            make_strength_summary()
        )
    )

    assert len(
        table
    ) == 16


def test_strength_publication_table_preserves_null_label():
    table = (
        build_strength_publication_table(
            make_strength_summary()
        )
    )

    assert "null" in set(
        table[
            "Strength"
        ]
    )


def test_predictive_publication_table_rejects_p_values():
    table = pd.DataFrame(
        {
            "strength_name": [
                "moderate"
            ],
            "comparator": [
                "AG-NAM minus Main NAM"
            ],
            "metric": [
                "delta_auroc"
            ],
            "n": [
                9
            ],
            "mean": [
                0.02
            ],
            "median": [
                0.01
            ],
            "bootstrap_ci_lower": [
                0.00
            ],
            "bootstrap_ci_upper": [
                0.04
            ],
            "positive_dataset_count": [
                7
            ],
            "zero_dataset_count": [
                0
            ],
            "negative_dataset_count": [
                2
            ],
            "inference": [
                "descriptive_only"
            ],
            "p_value": [
                0.01
            ],
        }
    )

    with pytest.raises(
        RuntimeError
    ):
        build_predictive_publication_table(
            table
        )


def test_feature_type_publication_table_preserves_regime():
    source = pd.DataFrame(
        {
            "feature_type": [
                "categorical",
                "mixed",
                "numeric",
            ],
            "strength_name": [
                "moderate",
                "moderate",
                "moderate",
            ],
            "metric": [
                "delta_auroc",
                "delta_auroc",
                "delta_auroc",
            ],
            "n": [
                3,
                3,
                3,
            ],
            "mean": [
                0.05,
                0.01,
                0.00,
            ],
            "std": [
                0.01,
                0.01,
                0.01,
            ],
            "median": [
                0.05,
                0.01,
                0.00,
            ],
            "q25": [
                0.04,
                0.00,
                -0.01,
            ],
            "q75": [
                0.06,
                0.02,
                0.01,
            ],
            "minimum": [
                0.03,
                0.00,
                -0.01,
            ],
            "maximum": [
                0.07,
                0.03,
                0.01,
            ],
        }
    )

    output = (
        build_feature_type_publication_table(
            source
        )
    )

    assert set(
        output[
            "Predictor regime"
        ]
    ) == {
        "categorical",
        "mixed",
        "numeric",
    }


def test_extract_figure_series_returns_nine_values_per_strength():
    series = (
        extract_figure_series(
            make_dataset_strength(),
            metric="delta_auroc",
            strengths=(
                "null",
                "weak",
                "moderate",
                "strong",
            ),
        )
    )

    assert all(
        len(
            values
        )
        == 9
        for values in (
            series.values()
        )
    )