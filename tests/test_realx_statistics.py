import numpy as np
import pandas as pd
import pytest

from agnam.benchmarking.realx_statistics import (
    ALPHA,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    EXPECTED_DATASETS,
    PRIMARY_CONDITION,
    PRIMARY_METRIC,
    PRIMARY_TEST,
    build_primary_dataset_table,
    build_primary_realization_table,
    one_sided_wilcoxon_greater,
    pair_universe_size,
    random_ranking_interaction_prevalence,
    run_primary_confirmatory_analysis,
)


def test_pair_universe_size():
    assert (
        pair_universe_size(
            9
        )
        == 36
    )

    assert (
        pair_universe_size(
            36
        )
        == 630
    )


def test_random_ranking_prevalence():
    observed = (
        random_ranking_interaction_prevalence(
            n_features=9,
            n_true_interactions=3,
        )
    )

    assert np.isclose(
        observed,
        3.0 / 36.0,
    )


def test_random_prevalence_rejects_invalid_true_count():
    with pytest.raises(
        ValueError
    ):
        random_ranking_interaction_prevalence(
            n_features=3,
            n_true_interactions=4,
        )


def test_one_sided_wilcoxon_all_zero():
    (
        statistic,
        p_value,
        n_nonzero,
    ) = one_sided_wilcoxon_greater(
        np.zeros(
            9,
            dtype=np.float64,
        )
    )

    assert statistic == 0.0
    assert p_value == 1.0
    assert n_nonzero == 0


def test_one_sided_wilcoxon_positive_values():
    differences = np.arange(
        1,
        10,
        dtype=np.float64,
    )

    (
        _,
        p_value,
        n_nonzero,
    ) = one_sided_wilcoxon_greater(
        differences
    )

    assert n_nonzero == 9
    assert p_value < 0.05


def make_primary_master():
    records = []

    for dataset_index in range(
        1,
        10,
    ):
        task_id = (
            1000
            + dataset_index
        )

        n_features = (
            5
            + dataset_index
        )

        n_pairs = (
            n_features
            * (
                n_features - 1
            )
            // 2
        )

        prevalence = (
            3.0
            / n_pairs
        )

        for realization in range(
            1,
            6,
        ):
            interaction_auprc = (
                prevalence
                + 0.10
                + (
                    dataset_index
                    * 0.002
                )
                + (
                    realization
                    * 0.001
                )
            )

            records.append(
                {
                    "task_id": task_id,
                    "dataset_name": (
                        f"dataset_{dataset_index}"
                    ),
                    "feature_type": (
                        "numeric"
                    ),
                    "realization": (
                        realization
                    ),
                    "strength_name": (
                        "moderate"
                    ),
                    "interaction_coefficient": (
                        1.0
                    ),
                    "n_features": (
                        n_features
                    ),
                    "n_true_interactions": (
                        3
                    ),
                    "mean_interaction_auprc": (
                        interaction_auprc
                    ),
                }
            )

    return pd.DataFrame(
        records
    )


def test_primary_tables_have_expected_sizes():
    master = (
        make_primary_master()
    )

    realizations = (
        build_primary_realization_table(
            master
        )
    )

    datasets = (
        build_primary_dataset_table(
            realizations
        )
    )

    assert len(
        realizations
    ) == 45

    assert len(
        datasets
    ) == EXPECTED_DATASETS


def test_dataset_aggregation_is_mean_over_five():
    master = (
        make_primary_master()
    )

    realizations = (
        build_primary_realization_table(
            master
        )
    )

    datasets = (
        build_primary_dataset_table(
            realizations
        )
    )

    first_realizations = (
        realizations.loc[
            realizations[
                "task_id"
            ]
            .eq(
                datasets
                .iloc[
                    0
                ][
                    "task_id"
                ]
            ),
            "primary_difference",
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    assert np.isclose(
        datasets.iloc[
            0
        ][
            "primary_difference"
        ],
        np.mean(
            first_realizations
        ),
    )


def test_primary_table_rejects_wrong_realization_set():
    master = (
        make_primary_master()
    )

    first_task = (
        master.iloc[
            0
        ][
            "task_id"
        ]
    )

    indices = master.index[
        master[
            "task_id"
        ]
        .eq(
            first_task
        )
    ].tolist()

    master.loc[
        indices[
            -1
        ],
        "realization",
    ] = 1

    realizations = (
        build_primary_realization_table(
            master
        )
    )

    with pytest.raises(
        RuntimeError
    ):
        build_primary_dataset_table(
            realizations
        )


def test_confirmatory_bootstrap_is_deterministic():
    master = (
        make_primary_master()
    )

    realizations = (
        build_primary_realization_table(
            master
        )
    )

    datasets = (
        build_primary_dataset_table(
            realizations
        )
    )

    first = (
        run_primary_confirmatory_analysis(
            datasets,
            frozen_evidence_sha256=(
                "a" * 64
            ),
            phase6e_git_commit=(
                "b" * 40
            ),
        )
    )

    second = (
        run_primary_confirmatory_analysis(
            datasets,
            frozen_evidence_sha256=(
                "a" * 64
            ),
            phase6e_git_commit=(
                "b" * 40
            ),
        )
    )

    assert (
        first.bootstrap_ci_lower
        == second.bootstrap_ci_lower
    )

    assert (
        first.bootstrap_ci_upper
        == second.bootstrap_ci_upper
    )


def test_confirmatory_result_uses_locked_primary_rules():
    master = (
        make_primary_master()
    )

    realizations = (
        build_primary_realization_table(
            master
        )
    )

    datasets = (
        build_primary_dataset_table(
            realizations
        )
    )

    result = (
        run_primary_confirmatory_analysis(
            datasets,
            frozen_evidence_sha256=(
                "a" * 64
            ),
            phase6e_git_commit=(
                "b" * 40
            ),
        )
    )

    assert (
        result.primary_condition
        == PRIMARY_CONDITION
    )

    assert (
        result.primary_metric
        == PRIMARY_METRIC
    )

    assert (
        result.primary_test
        == PRIMARY_TEST
    )

    assert result.alpha == ALPHA

    assert (
        result.bootstrap_resamples
        == BOOTSTRAP_RESAMPLES
    )

    assert (
        result.bootstrap_seed
        == BOOTSTRAP_SEED
    )

    assert (
        result.multiplicity_adjustment
        == "none_single_primary"
    )