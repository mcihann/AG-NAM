import json

import numpy as np
import pandas as pd
import pytest

from agnam.benchmarking.realx_secondary import (
    EXPECTED_DATASET_STRENGTH_ROWS,
    STRENGTH_ORDER,
    add_secondary_derived_metrics,
    build_dataset_strength_table,
    build_feature_type_strength_summary,
    build_predictive_comparator_summary,
    build_strength_summary,
    validate_primary_analysis_lock,
)
from agnam.benchmarking.realx_statistics import (
    ANALYSIS_SCHEMA_VERSION,
    PRIMARY_CONDITION,
    PRIMARY_METRIC,
    PRIMARY_TEST,
)


def make_master():
    records = []

    feature_types = (
        "categorical",
        "mixed",
        "numeric",
    )

    strength_lambda = {
        "null": 0.0,
        "weak": 0.5,
        "moderate": 1.0,
        "strong": 1.5,
    }

    for dataset_index in range(
        9
    ):
        task_id = (
            100
            + dataset_index
        )

        dataset_name = (
            f"dataset_{dataset_index}"
        )

        feature_type = feature_types[
            dataset_index
            % 3
        ]

        for strength_index, strength in enumerate(
            STRENGTH_ORDER
        ):
            for realization in range(
                1,
                6,
            ):
                base = (
                    0.70
                    + dataset_index
                    * 0.002
                    + strength_index
                    * 0.005
                    + realization
                    * 0.0005
                )

                selection_count = (
                    4
                    + (
                        realization
                        % 2
                    )
                )

                retained_count = (
                    selection_count
                    - 2
                )

                mean_interaction_auprc = (
                    np.nan
                    if strength
                    == "null"
                    else (
                        0.10
                        + strength_index
                        * 0.05
                        + dataset_index
                        * 0.001
                    )
                )

                records.append(
                    {
                        "task_id": task_id,
                        "dataset_name": (
                            dataset_name
                        ),
                        "feature_type": (
                            feature_type
                        ),
                        "realization": (
                            realization
                        ),
                        "strength_name": (
                            strength
                        ),
                        "interaction_coefficient": (
                            strength_lambda[
                                strength
                            ]
                        ),

                        "mean_pairwise_jaccard": (
                            0.30
                            + strength_index
                            * 0.02
                        ),

                        "mean_interaction_auprc": (
                            mean_interaction_auprc
                        ),

                        "n_selection_stable": (
                            selection_count
                        ),

                        "selection_precision": (
                            np.nan
                            if strength
                            == "null"
                            else 0.50
                        ),

                        "selection_recall": (
                            np.nan
                            if strength
                            == "null"
                            else 0.60
                        ),

                        "selection_f1": (
                            np.nan
                            if strength
                            == "null"
                            else 0.545
                        ),

                        "n_isr_retained": (
                            retained_count
                        ),

                        "isr_precision": (
                            np.nan
                            if strength
                            == "null"
                            else 0.70
                        ),

                        "isr_recall": (
                            np.nan
                            if strength
                            == "null"
                            else 0.55
                        ),

                        "isr_f1": (
                            np.nan
                            if strength
                            == "null"
                            else 0.615
                        ),

                        "false_positive_reduction": (
                            0.50
                        ),

                        "main_auroc": (
                            base
                        ),

                        "main_auprc": (
                            base
                            - 0.01
                        ),

                        "main_balanced_accuracy": (
                            base
                            - 0.05
                        ),

                        "main_f1": (
                            base
                            - 0.06
                        ),

                        "agnam_auroc": (
                            base
                            + 0.02
                        ),

                        "agnam_auprc": (
                            base
                            + 0.015
                        ),

                        "agnam_balanced_accuracy": (
                            base
                            - 0.03
                        ),

                        "agnam_f1": (
                            base
                            - 0.04
                        ),

                        "delta_auroc": (
                            0.02
                        ),

                        "delta_auprc": (
                            0.025
                        ),

                        "delta_balanced_accuracy": (
                            0.02
                        ),

                        "delta_f1": (
                            0.02
                        ),

                        "oracle_auroc": (
                            base
                            + 0.03
                        ),

                        "oracle_auprc": (
                            base
                            + 0.025
                        ),

                        "random_pair_auroc": (
                            base
                            + 0.005
                        ),

                        "random_pair_auprc": (
                            base
                        ),

                        "no_isr_auroc": (
                            base
                            + 0.015
                        ),

                        "no_isr_auprc": (
                            base
                            + 0.010
                        ),

                        "single_run_auroc": (
                            base
                            + 0.010
                        ),

                        "single_run_auprc": (
                            base
                            + 0.005
                        ),
                    }
                )

    return pd.DataFrame(
        records
    )


def test_derived_metrics_compute_sparsification():
    master = make_master()

    enriched = (
        add_secondary_derived_metrics(
            master
        )
    )

    expected = (
        (
            enriched[
                "n_selection_stable"
            ]
            -
            enriched[
                "n_isr_retained"
            ]
        )
        /
        enriched[
            "n_selection_stable"
        ]
    )

    assert np.allclose(
        enriched[
            "isr_sparsification_fraction"
        ],
        expected,
    )


def test_derived_metrics_compute_predictive_deltas():
    enriched = (
        add_secondary_derived_metrics(
            make_master()
        )
    )

    expected = (
        enriched[
            "agnam_auroc"
        ]
        -
        enriched[
            "random_pair_auroc"
        ]
    )

    assert np.allclose(
        enriched[
            "agnam_minus_random_pair_auroc"
        ],
        expected,
    )


def test_derived_metrics_reject_impossible_isr_count():
    master = make_master()

    master.loc[
        0,
        "n_isr_retained",
    ] = (
        master.loc[
            0,
            "n_selection_stable",
        ]
        + 1
    )

    with pytest.raises(
        RuntimeError
    ):
        add_secondary_derived_metrics(
            master
        )


def test_dataset_strength_table_has_36_rows():
    enriched = (
        add_secondary_derived_metrics(
            make_master()
        )
    )

    table = (
        build_dataset_strength_table(
            enriched
        )
    )

    assert len(
        table
    ) == EXPECTED_DATASET_STRENGTH_ROWS


def test_dataset_strength_uses_mean_over_five():
    enriched = (
        add_secondary_derived_metrics(
            make_master()
        )
    )

    table = (
        build_dataset_strength_table(
            enriched
        )
    )

    first = table.iloc[
        0
    ]

    source = enriched.loc[
        enriched[
            "task_id"
        ].eq(
            first[
                "task_id"
            ]
        )
        &
        enriched[
            "strength_name"
        ].eq(
            first[
                "strength_name"
            ]
        )
    ]

    assert np.isclose(
        first[
            "agnam_auroc"
        ],
        source[
            "agnam_auroc"
        ].mean(),
    )


def test_dataset_strength_rejects_incomplete_design():
    master = (
        make_master()
        .iloc[
            :-1
        ]
        .copy()
    )

    with pytest.raises(
        RuntimeError
    ):
        add_secondary_derived_metrics(
            master
        )


def test_strength_summary_contains_four_strengths():
    table = (
        build_dataset_strength_table(
            add_secondary_derived_metrics(
                make_master()
            )
        )
    )

    summary = (
        build_strength_summary(
            table
        )
    )

    assert set(
        summary[
            "strength_name"
        ]
    ) == set(
        STRENGTH_ORDER
    )


def test_predictive_comparator_summary_has_expected_comparators():
    table = (
        build_dataset_strength_table(
            add_secondary_derived_metrics(
                make_master()
            )
        )
    )

    summary = (
        build_predictive_comparator_summary(
            table
        )
    )

    assert set(
        summary[
            "comparator"
        ]
    ) == {
        "AG-NAM minus Main NAM",
        "AG-NAM minus Random-Pair NAM",
        "AG-NAM minus No-ISR AG-NAM",
        "AG-NAM minus Single-Run AG-NAM",
        "Oracle minus AG-NAM",
    }


def test_predictive_comparator_summary_has_no_p_values():
    table = (
        build_dataset_strength_table(
            add_secondary_derived_metrics(
                make_master()
            )
        )
    )

    summary = (
        build_predictive_comparator_summary(
            table
        )
    )

    forbidden = {
        "p_value",
        "pvalue",
        "reject_null",
        "wilcoxon_statistic",
    }

    assert not (
        forbidden
        & set(
            summary.columns
        )
    )

    assert (
        summary[
            "inference"
        ]
        .eq(
            "descriptive_only"
        )
        .all()
    )


def test_predictive_comparator_bootstrap_is_deterministic():
    table = (
        build_dataset_strength_table(
            add_secondary_derived_metrics(
                make_master()
            )
        )
    )

    first = (
        build_predictive_comparator_summary(
            table
        )
    )

    second = (
        build_predictive_comparator_summary(
            table
        )
    )

    assert np.allclose(
        first[
            "bootstrap_ci_lower"
        ],
        second[
            "bootstrap_ci_lower"
        ],
    )

    assert np.allclose(
        first[
            "bootstrap_ci_upper"
        ],
        second[
            "bootstrap_ci_upper"
        ],
    )


def test_feature_type_summary_contains_three_regimes():
    table = (
        build_dataset_strength_table(
            add_secondary_derived_metrics(
                make_master()
            )
        )
    )

    summary = (
        build_feature_type_strength_summary(
            table
        )
    )

    assert set(
        summary[
            "feature_type"
        ]
    ) == {
        "numeric",
        "mixed",
        "categorical",
    }


def test_predictive_sign_counts_sum_to_n():
    table = (
        build_dataset_strength_table(
            add_secondary_derived_metrics(
                make_master()
            )
        )
    )

    summary = (
        build_predictive_comparator_summary(
            table
        )
    )

    total = (
        summary[
            "positive_dataset_count"
        ]
        +
        summary[
            "zero_dataset_count"
        ]
        +
        summary[
            "negative_dataset_count"
        ]
    )

    assert np.array_equal(
        total.to_numpy(),
        summary[
            "n"
        ].to_numpy(),
    )


def _write_primary_result(
    path,
    *,
    evidence_hash: str,
):
    payload = {
        "analysis_schema_version": (
            ANALYSIS_SCHEMA_VERSION
        ),

        "frozen_evidence_sha256": (
            evidence_hash
        ),

        "primary_condition": (
            PRIMARY_CONDITION
        ),

        "primary_metric": (
            PRIMARY_METRIC
        ),

        "primary_test": (
            PRIMARY_TEST
        ),

        "n_datasets": 9,
        "n_realizations_per_dataset": 5,

        "reject_null": True,
        "p_value": 0.01,
    }

    path.write_text(
        json.dumps(
            payload
        ),
        encoding="utf-8",
    )


def test_primary_analysis_lock_accepts_same_evidence_hash(
    tmp_path,
):
    path = (
        tmp_path
        / "primary.json"
    )

    evidence_hash = (
        "a"
        * 64
    )

    _write_primary_result(
        path,
        evidence_hash=(
            evidence_hash
        ),
    )

    lock = (
        validate_primary_analysis_lock(
            primary_result_path=(
                path
            ),
            expected_frozen_evidence_sha256=(
                evidence_hash
            ),
        )
    )

    assert (
        lock.frozen_evidence_sha256
        == evidence_hash
    )


def test_primary_analysis_lock_rejects_different_evidence_hash(
    tmp_path,
):
    path = (
        tmp_path
        / "primary.json"
    )

    _write_primary_result(
        path,
        evidence_hash=(
            "a"
            * 64
        ),
    )

    with pytest.raises(
        RuntimeError
    ):
        validate_primary_analysis_lock(
            primary_result_path=(
                path
            ),
            expected_frozen_evidence_sha256=(
                "b"
                * 64
            ),
        )