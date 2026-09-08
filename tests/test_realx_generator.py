import numpy as np
import pandas as pd
import pytest

from agnam.benchmarking.realx_generator import (
    build_realx_blueprint,
    build_realx_split_indices,
    calibrate_logistic_intercept,
    choose_realx_truth_structure,
    encode_realx_states,
    generate_realx_run,
    interaction_marginal_error,
    purify_two_way_surface,
    sample_realx_rows,
)
from agnam.benchmarking.realx_protocol import (
    build_realx_schedule,
)


def make_toy_x(
    n: int = 240,
) -> pd.DataFrame:
    rng = np.random.default_rng(
        12345
    )

    data = {}

    for index in range(
        7
    ):
        data[
            f"num_{index}"
        ] = (
            rng.normal(
                size=n
            )
            + 0.1
            * index
        )

    levels = np.asarray(
        [
            "A",
            "B",
            "C",
            "D",
            "E",
        ],
        dtype=object,
    )

    for index in range(
        7
    ):
        data[
            f"cat_{index}"
        ] = rng.choice(
            levels,
            size=n,
            replace=True,
        )

    frame = pd.DataFrame(
        data
    )

    frame.loc[
        frame.index[
            ::29
        ],
        "num_1",
    ] = np.nan

    frame.loc[
        frame.index[
            ::31
        ],
        "cat_2",
    ] = None

    return frame


def get_runs():
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

    moderate_run = next(
        run
        for run in schedule
        if (
            run.task_id == 49
            and run.realization == 1
            and run.strength_name == "moderate"
        )
    )

    return (
        null_run,
        moderate_run,
    )


def test_row_sampling_is_deterministic():
    X = make_toy_x(
        n=300
    )

    first_X, first_positions = (
        sample_realx_rows(
            X,
            max_rows=100,
            seed=42,
        )
    )

    second_X, second_positions = (
        sample_realx_rows(
            X,
            max_rows=100,
            seed=42,
        )
    )

    assert np.array_equal(
        first_positions,
        second_positions,
    )

    pd.testing.assert_frame_equal(
        first_X,
        second_X,
    )


def test_small_dataset_keeps_all_rows():
    X = make_toy_x(
        n=100
    )

    sampled, positions = (
        sample_realx_rows(
            X,
            max_rows=5000,
            seed=42,
        )
    )

    assert len(
        sampled
    ) == 100

    assert np.array_equal(
        positions,
        np.arange(
            100
        ),
    )


def test_numeric_and_categorical_states_respect_cap():
    X = make_toy_x()

    encoding = (
        encode_realx_states(
            X,
            max_states_per_feature=8,
        )
    )

    assert all(
        count <= 8
        for count in (
            encoding
            .state_counts
            .values()
        )
    )

    assert (
        encoding.feature_kinds[
            "num_0"
        ]
        == "numeric"
    )

    assert (
        encoding.feature_kinds[
            "cat_0"
        ]
        == "categorical"
    )


def test_missing_values_are_explicit_states():
    X = make_toy_x()

    encoding = (
        encode_realx_states(
            X,
            max_states_per_feature=8,
        )
    )

    assert (
        "MISSING"
        in encoding.state_labels[
            "num_1"
        ]
    )

    assert (
        "MISSING"
        in encoding.state_labels[
            "cat_2"
        ]
    )


def test_truth_structure_is_disjoint():
    X = make_toy_x()

    encoding = (
        encode_realx_states(
            X,
            max_states_per_feature=8,
        )
    )

    (
        main_features,
        true_pairs,
        eligible,
    ) = choose_realx_truth_structure(
        encoding,
        feature_seed=123,
        n_main_effects=3,
        n_true_interactions=3,
        min_eligible_features=9,
    )

    used = list(
        main_features
    )

    for pair in (
        true_pairs
    ):
        used.extend(
            pair
        )

    assert len(
        used
    ) == 9

    assert len(
        set(
            used
        )
    ) == 9

    assert len(
        eligible
    ) >= 9


def test_too_few_eligible_features_are_rejected():
    X = pd.DataFrame(
        {
            "a": [
                0,
                1,
            ]
            * 20,
            "b": [
                1,
                0,
            ]
            * 20,
            "c": [
                "x",
                "y",
            ]
            * 20,
        }
    )

    encoding = (
        encode_realx_states(
            X,
            max_states_per_feature=8,
        )
    )

    with pytest.raises(
        RuntimeError
    ):
        choose_realx_truth_structure(
            encoding,
            feature_seed=1,
            n_main_effects=3,
            n_true_interactions=3,
            min_eligible_features=9,
        )


def test_weighted_interaction_purification_has_zero_marginals():
    rng = np.random.default_rng(
        5
    )

    raw = rng.normal(
        size=(
            4,
            5,
        )
    )

    weights = np.asarray(
        [
            [
                8,
                4,
                1,
                3,
                2,
            ],
            [
                2,
                7,
                5,
                1,
                4,
            ],
            [
                1,
                3,
                9,
                2,
                6,
            ],
            [
                4,
                2,
                3,
                8,
                1,
            ],
        ],
        dtype=float,
    )

    purified, error, _ = (
        purify_two_way_surface(
            raw,
            weights,
            tolerance=1e-10,
            max_iterations=1000,
        )
    )

    assert (
        error
        <= 1e-10
    )

    assert (
        interaction_marginal_error(
            purified,
            weights,
        )
        <= 1e-10
    )


def test_logistic_intercept_hits_target_expected_prevalence():
    scores = np.linspace(
        -3.0,
        3.0,
        501,
    )

    intercept = (
        calibrate_logistic_intercept(
            scores,
            target_expected_prevalence=0.50,
        )
    )

    probabilities = (
        1.0
        /
        (
            1.0
            + np.exp(
                -(
                    intercept
                    + scores
                )
            )
        )
    )

    assert np.isclose(
        probabilities.mean(),
        0.50,
        atol=1e-10,
    )


def test_split_is_disjoint_and_complete():
    (
        train,
        validation,
        test,
    ) = build_realx_split_indices(
        100,
        split_seed=42,
        train_fraction=0.60,
        validation_fraction=0.20,
    )

    assert len(
        train
    ) == 60

    assert len(
        validation
    ) == 20

    assert len(
        test
    ) == 20

    assert len(
        set(
            train
        )
        & set(
            validation
        )
    ) == 0

    assert len(
        set(
            train
        )
        & set(
            test
        )
    ) == 0

    assert len(
        set(
            validation
        )
        & set(
            test
        )
    ) == 0

    combined = np.sort(
        np.concatenate(
            [
                train,
                validation,
                test,
            ]
        )
    )

    assert np.array_equal(
        combined,
        np.arange(
            100
        ),
    )


def test_blueprint_composites_have_unit_standard_deviation():
    X = make_toy_x()

    _, moderate_run = (
        get_runs()
    )

    blueprint = (
        build_realx_blueprint(
            X,
            moderate_run,
        )
    )

    assert np.isclose(
        np.std(
            blueprint.main_composite
        ),
        1.0,
        atol=1e-10,
    )

    assert np.isclose(
        np.std(
            blueprint.interaction_composite
        ),
        1.0,
        atol=1e-10,
    )

    assert all(
        error <= 1e-8
        for error in (
            blueprint
            .interaction_marginal_errors
            .values()
        )
    )


def test_strength_conditions_share_same_blueprint_randomness():
    X = make_toy_x()

    null_run, moderate_run = (
        get_runs()
    )

    null_data = (
        generate_realx_run(
            X,
            null_run,
        )
    )

    moderate_data = (
        generate_realx_run(
            X,
            moderate_run,
        )
    )

    assert np.array_equal(
        null_data.source_row_positions,
        moderate_data.source_row_positions,
    )

    assert (
        null_data.state_codes.equals(
            moderate_data.state_codes
        )
    )

    assert (
        null_data.main_features
        == moderate_data.main_features
    )

    assert (
        null_data.template_true_pairs
        == moderate_data.template_true_pairs
    )

    assert np.allclose(
        null_data.main_composite,
        moderate_data.main_composite,
    )

    assert np.allclose(
        null_data.interaction_composite,
        moderate_data.interaction_composite,
    )

    assert np.array_equal(
        null_data.uniform_draws,
        moderate_data.uniform_draws,
    )

    assert np.array_equal(
        null_data.train_indices,
        moderate_data.train_indices,
    )

    assert np.array_equal(
        null_data.validation_indices,
        moderate_data.validation_indices,
    )

    assert np.array_equal(
        null_data.test_indices,
        moderate_data.test_indices,
    )


def test_null_condition_has_zero_active_interaction_signal():
    X = make_toy_x()

    null_run, _ = (
        get_runs()
    )

    generated = (
        generate_realx_run(
            X,
            null_run,
        )
    )

    assert (
        generated.active_true_pairs
        == tuple()
    )

    assert np.allclose(
        generated.interaction_signal,
        0.0,
    )


def test_moderate_condition_has_three_active_true_pairs():
    X = make_toy_x()

    _, moderate_run = (
        get_runs()
    )

    generated = (
        generate_realx_run(
            X,
            moderate_run,
        )
    )

    assert len(
        generated.active_true_pairs
    ) == 3

    assert (
        generated.active_true_pairs
        == generated.template_true_pairs
    )

    assert np.isclose(
        np.std(
            generated.interaction_signal
        ),
        1.0,
        atol=1e-10,
    )


def test_generated_probabilities_and_prevalence_are_valid():
    X = make_toy_x()

    _, moderate_run = (
        get_runs()
    )

    generated = (
        generate_realx_run(
            X,
            moderate_run,
        )
    )

    assert np.all(
        generated.probabilities
        > 0.0
    )

    assert np.all(
        generated.probabilities
        < 1.0
    )

    assert np.isclose(
        generated.expected_prevalence,
        0.50,
        atol=1e-10,
    )

    assert (
        0.0
        <= generated.realized_prevalence
        <= 1.0
    )


def test_generator_does_not_modify_input_dataframe():
    X = make_toy_x()

    original = X.copy(
        deep=True
    )

    _, moderate_run = (
        get_runs()
    )

    _ = generate_realx_run(
        X,
        moderate_run,
    )

    pd.testing.assert_frame_equal(
        X,
        original,
    )