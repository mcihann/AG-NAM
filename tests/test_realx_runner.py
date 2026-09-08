import numpy as np
import pandas as pd
import pytest

from agnam.benchmarking.realx_protocol import (
    build_realx_schedule,
)
from agnam.benchmarking.realx_runner import (
    RealXModelProtocol,
    build_locked_dev_indices,
    derive_realx_model_seeds,
    get_locked_sanity_run_spec,
    prepare_locked_final_prediction_data,
    realx_candidate_k,
    validate_locked_split,
)


def test_realx_candidate_k_rule_is_locked():
    assert (
        realx_candidate_k(
            1
        )
        == 0
    )

    assert (
        realx_candidate_k(
            2
        )
        == 1
    )

    assert (
        realx_candidate_k(
            9
        )
        == 8
    )

    assert (
        realx_candidate_k(
            20
        )
        == 19
    )

    assert (
        realx_candidate_k(
            21
        )
        == 20
    )

    assert (
        realx_candidate_k(
            100
        )
        == 20
    )


def test_realx_model_protocol_is_locked():
    protocol = (
        RealXModelProtocol()
    )

    assert (
        protocol.isr_reference_size
        == 512
    )

    assert (
        protocol.isr_reference_seed
        == 4026
    )

    assert np.isclose(
        protocol.main_early_stop_fraction,
        0.20,
    )

    assert np.isclose(
        protocol.decomposition_tolerance,
        1e-6,
    )

    assert (
        protocol.publication_eligible
        is False
    )


def test_phase6d_sanity_run_is_prespecified():
    run = (
        get_locked_sanity_run_spec()
    )

    assert (
        run.task_id
        == 49
    )

    assert (
        run.dataset_name
        == "tic-tac-toe"
    )

    assert (
        run.realization
        == 1
    )

    assert (
        run.strength_name
        == "moderate"
    )

    assert np.isclose(
        run.interaction_coefficient,
        1.0,
    )


def test_model_seed_offsets_are_exact():
    run = (
        get_locked_sanity_run_spec()
    )

    seeds = (
        derive_realx_model_seeds(
            run
        )
    )

    assert (
        seeds.discovery_base_seed
        == run.label_seed + 10
    )

    assert (
        seeds.random_pair_seed
        == run.label_seed + 20
    )

    assert (
        seeds.final_model_seed
        == run.label_seed + 30
    )


def test_model_seeds_are_common_across_strengths():
    schedule = (
        build_realx_schedule()
    )

    null_run = next(
        run
        for run in schedule
        if (
            run.task_id == 49
            and run.realization == 1
            and run.strength_name == "null"
        )
    )

    strong_run = next(
        run
        for run in schedule
        if (
            run.task_id == 49
            and run.realization == 1
            and run.strength_name == "strong"
        )
    )

    assert (
        derive_realx_model_seeds(
            null_run
        )
        == derive_realx_model_seeds(
            strong_run
        )
    )


def test_locked_split_accepts_complete_disjoint_partition():
    validate_locked_split(
        n_samples=10,
        train_indices=np.asarray(
            [
                0,
                1,
                2,
                3,
                4,
                5,
            ]
        ),
        validation_indices=np.asarray(
            [
                6,
                7,
            ]
        ),
        test_indices=np.asarray(
            [
                8,
                9,
            ]
        ),
    )


def test_locked_split_rejects_overlap():
    with pytest.raises(
        RuntimeError
    ):
        validate_locked_split(
            n_samples=6,
            train_indices=np.asarray(
                [
                    0,
                    1,
                    2,
                ]
            ),
            validation_indices=np.asarray(
                [
                    2,
                    3,
                ]
            ),
            test_indices=np.asarray(
                [
                    4,
                    5,
                ]
            ),
        )


def test_locked_split_rejects_incomplete_partition():
    with pytest.raises(
        RuntimeError
    ):
        validate_locked_split(
            n_samples=6,
            train_indices=np.asarray(
                [
                    0,
                    1,
                ]
            ),
            validation_indices=np.asarray(
                [
                    2,
                ]
            ),
            test_indices=np.asarray(
                [
                    4,
                    5,
                ]
            ),
        )


def test_discovery_dev_is_train_plus_validation_only():
    train = np.asarray(
        [
            0,
            2,
            4,
            6,
            8,
            10,
        ]
    )

    validation = np.asarray(
        [
            1,
            3,
        ]
    )

    test = np.asarray(
        [
            5,
            7,
            9,
            11,
        ]
    )

    dev = build_locked_dev_indices(
        n_samples=12,
        train_indices=train,
        validation_indices=validation,
        test_indices=test,
    )

    expected = np.sort(
        np.concatenate(
            [
                train,
                validation,
            ]
        )
    )

    assert np.array_equal(
        dev,
        expected,
    )

    assert (
        np.intersect1d(
            dev,
            test,
        ).size
        == 0
    )


def make_prediction_frame(
    n=100,
):
    rng = np.random.default_rng(
        260908
    )

    return pd.DataFrame(
        {
            "num_a": rng.normal(
                size=n
            ),
            "num_b": rng.normal(
                size=n
            ),
            "cat_a": pd.Series(
                rng.choice(
                    [
                        "A",
                        "B",
                        "C",
                    ],
                    size=n,
                ),
                dtype="category",
            ),
            "cat_b": pd.Series(
                rng.choice(
                    [
                        "X",
                        "Y",
                    ],
                    size=n,
                ),
                dtype="category",
            ),
        }
    )


def test_locked_final_prediction_data_preserves_partition_sizes():
    X = make_prediction_frame(
        100
    )

    y = np.asarray(
        [
            index % 2
            for index in range(
                100
            )
        ],
        dtype=np.int64,
    )

    train = np.arange(
        0,
        60,
    )

    validation = np.arange(
        60,
        80,
    )

    test = np.arange(
        80,
        100,
    )

    data = (
        prepare_locked_final_prediction_data(
            X=X,
            y=y,
            train_indices=train,
            validation_indices=(
                validation
            ),
            test_indices=test,
        )
    )

    assert len(
        data.y_train
    ) == 60

    assert len(
        data.y_validation
    ) == 20

    assert len(
        data.y_test
    ) == 20


def test_locked_final_prediction_labels_are_exact():
    X = make_prediction_frame(
        50
    )

    y = np.asarray(
        [
            index % 2
            for index in range(
                50
            )
        ],
        dtype=np.int64,
    )

    train = np.arange(
        0,
        30,
    )

    validation = np.arange(
        30,
        40,
    )

    test = np.arange(
        40,
        50,
    )

    data = (
        prepare_locked_final_prediction_data(
            X=X,
            y=y,
            train_indices=train,
            validation_indices=(
                validation
            ),
            test_indices=test,
        )
    )

    assert np.array_equal(
        data.y_train,
        y[
            train
        ],
    )

    assert np.array_equal(
        data.y_validation,
        y[
            validation
        ],
    )

    assert np.array_equal(
        data.y_test,
        y[
            test
        ],
    )