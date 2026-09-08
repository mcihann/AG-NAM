import numpy as np
import pandas as pd

from agnam.benchmarking.realx_generator import (
    choose_realx_truth_structure,
    encode_realx_states,
    interaction_pair_degrees_of_freedom,
    interaction_support_degrees_of_freedom,
)


def test_deterministic_one_to_one_support_has_zero_interaction_df():
    codes_a = np.asarray(
        [
            0,
            0,
            0,
            1,
            1,
            1,
        ],
        dtype=np.int64,
    )

    codes_b = np.asarray(
        [
            0,
            0,
            0,
            1,
            1,
            1,
        ],
        dtype=np.int64,
    )

    df = (
        interaction_support_degrees_of_freedom(
            codes_a,
            codes_b,
            n_states_a=2,
            n_states_b=2,
        )
    )

    assert df == 0


def test_complete_two_by_two_support_has_one_interaction_df():
    codes_a = np.asarray(
        [
            0,
            0,
            1,
            1,
        ],
        dtype=np.int64,
    )

    codes_b = np.asarray(
        [
            0,
            1,
            0,
            1,
        ],
        dtype=np.int64,
    )

    df = (
        interaction_support_degrees_of_freedom(
            codes_a,
            codes_b,
            n_states_a=2,
            n_states_b=2,
        )
    )

    assert df == 1


def test_truth_selection_uses_only_estimable_disjoint_pairs():
    rng = np.random.default_rng(
        20260908
    )

    n = 800

    data = {}

    for index in range(
        12
    ):
        data[
            f"x{index}"
        ] = rng.integers(
            0,
            4,
            size=n,
        ).astype(
            str
        )

    # Add a deliberately deterministic duplicate.
    data[
        "duplicate_x0"
    ] = np.asarray(
        data[
            "x0"
        ],
        dtype=object,
    ).copy()

    X = pd.DataFrame(
        data
    )

    encoding = (
        encode_realx_states(
            X,
            max_states_per_feature=8,
        )
    )

    assert (
        interaction_pair_degrees_of_freedom(
            encoding,
            "x0",
            "duplicate_x0",
        )
        == 0
    )

    (
        main_features,
        true_pairs,
        eligible,
    ) = choose_realx_truth_structure(
        encoding,
        feature_seed=26090831,
        n_main_effects=3,
        n_true_interactions=3,
        min_eligible_features=9,
        minimum_interaction_df=1,
    )

    assert len(
        true_pairs
    ) == 3

    used_pair_features = [
        feature
        for pair in true_pairs
        for feature in pair
    ]

    assert len(
        used_pair_features
    ) == 6

    assert len(
        set(
            used_pair_features
        )
    ) == 6

    assert set(
        main_features
    ).isdisjoint(
        set(
            used_pair_features
        )
    )

    assert len(
        eligible
    ) >= 9

    for (
        feature_a,
        feature_b,
    ) in true_pairs:
        assert (
            interaction_pair_degrees_of_freedom(
                encoding,
                feature_a,
                feature_b,
            )
            >= 1
        )