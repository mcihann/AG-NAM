from pathlib import Path

import pytest

from agnam.benchmarking.realx_protocol import (
    LOCKED_REALX_DATASETS,
    LOCKED_REALX_STRENGTHS,
    RealXProtocol,
    build_realx_schedule,
    canonical_realx_config_dict,
    expected_realx_run_count,
    load_and_validate_realx_config,
)


def test_realx_contains_nine_locked_datasets():
    assert len(
        LOCKED_REALX_DATASETS
    ) == 9


def test_realx_feature_type_distribution_is_balanced():
    counts = {}

    for dataset in (
        LOCKED_REALX_DATASETS
    ):
        counts[
            dataset.feature_type
        ] = (
            counts.get(
                dataset.feature_type,
                0,
            )
            + 1
        )

    assert counts == {
        "categorical": 3,
        "mixed": 3,
        "numeric": 3,
    }


def test_realx_locked_task_ids_are_exact():
    assert tuple(
        dataset.task_id
        for dataset in (
            LOCKED_REALX_DATASETS
        )
    ) == (
        49,
        3,
        14952,
        125920,
        31,
        7592,
        3913,
        3902,
        3904,
    )


def test_realx_registry_anchor_rule_is_exact():
    observed = tuple(
        (
            dataset.feature_type,
            dataset.feature_type_rank,
            dataset.selection_anchor,
        )
        for dataset in (
            LOCKED_REALX_DATASETS
        )
    )

    assert observed == (
        (
            "categorical",
            1,
            "first",
        ),
        (
            "categorical",
            2,
            "lower_median",
        ),
        (
            "categorical",
            3,
            "last",
        ),
        (
            "mixed",
            1,
            "first",
        ),
        (
            "mixed",
            5,
            "lower_median",
        ),
        (
            "mixed",
            10,
            "last",
        ),
        (
            "numeric",
            1,
            "first",
        ),
        (
            "numeric",
            8,
            "lower_median",
        ),
        (
            "numeric",
            15,
            "last",
        ),
    )


def test_realx_strengths_are_locked():
    assert tuple(
        (
            strength.name,
            strength.interaction_coefficient,
            strength.active_ground_truth,
        )
        for strength in (
            LOCKED_REALX_STRENGTHS
        )
    ) == (
        (
            "null",
            0.0,
            False,
        ),
        (
            "weak",
            0.5,
            True,
        ),
        (
            "moderate",
            1.0,
            True,
        ),
        (
            "strong",
            1.5,
            True,
        ),
    )


def test_realx_protocol_defaults_are_locked():
    protocol = (
        RealXProtocol()
    )

    assert (
        protocol.n_realizations
        == 5
    )

    assert (
        protocol.max_rows
        == 5000
    )

    assert (
        protocol.n_true_interactions
        == 3
    )

    assert (
        protocol.n_main_effects
        == 3
    )

    assert (
        protocol.discovery_runs
        == 5
    )

    assert (
        protocol.residual_crossfit_folds
        == 5
    )

    assert (
        protocol.selection_threshold
        == 0.60
    )

    assert (
        protocol.isr_threshold
        == 0.60
    )

    assert (
        protocol.primary_strength
        == "moderate"
    )


def test_realx_schedule_contains_180_runs():
    assert (
        expected_realx_run_count()
        == 180
    )

    assert len(
        build_realx_schedule()
    ) == 180


def test_realx_run_ids_are_unique():
    schedule = (
        build_realx_schedule()
    )

    run_ids = [
        run.run_id
        for run in schedule
    ]

    assert len(
        run_ids
    ) == len(
        set(
            run_ids
        )
    )


def test_realx_strengths_share_common_random_numbers():
    schedule = (
        build_realx_schedule()
    )

    subset = [
        run
        for run in schedule
        if (
            run.task_id == 49
            and run.realization == 1
        )
    ]

    assert len(
        subset
    ) == 4

    seed_tuples = {
        (
            run.row_sample_seed,
            run.split_seed,
            run.feature_seed,
            run.surface_seed,
            run.label_seed,
        )
        for run in subset
    }

    assert len(
        seed_tuples
    ) == 1


def test_realx_realizations_use_different_seed_families():
    schedule = (
        build_realx_schedule()
    )

    first = next(
        run
        for run in schedule
        if (
            run.task_id == 49
            and run.realization == 1
            and run.strength_name == "moderate"
        )
    )

    second = next(
        run
        for run in schedule
        if (
            run.task_id == 49
            and run.realization == 2
            and run.strength_name == "moderate"
        )
    )

    assert (
        first.feature_seed
        != second.feature_seed
    )

    assert (
        first.split_seed
        != second.split_seed
    )

    assert (
        first.label_seed
        != second.label_seed
    )


def test_realx_null_has_empty_active_truth():
    schedule = (
        build_realx_schedule()
    )

    null_runs = [
        run
        for run in schedule
        if (
            run.strength_name
            == "null"
        )
    ]

    nonnull_runs = [
        run
        for run in schedule
        if (
            run.strength_name
            != "null"
        )
    ]

    assert all(
        run.active_true_interaction_count
        == 0
        for run in null_runs
    )

    assert all(
        run.active_true_interaction_count
        == 3
        for run in nonnull_runs
    )


def test_realx_yaml_matches_locked_protocol():
    path = Path(
        "configs"
    ) / (
        "realx_semisynthetic_v1.yaml"
    )

    config = (
        load_and_validate_realx_config(
            path
        )
    )

    assert (
        config
        == canonical_realx_config_dict()
    )


def test_realx_invalid_split_is_rejected():
    with pytest.raises(
        ValueError
    ):
        RealXProtocol(
            train_fraction=0.70,
            validation_fraction=0.20,
            test_fraction=0.20,
        )