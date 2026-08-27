import numpy as np
import pytest

from agnam.benchmarking.synthetic_protocol import (
    SyntheticBenchmarkProtocol,
    build_synthetic_schedule,
)
from agnam.benchmarking.synthetic_runner import (
    generate_synthetic_scenario,
    normalize_pairs,
    sample_random_pairs,
)


def test_normalize_pairs_is_order_invariant_and_deduplicated():
    pairs = normalize_pairs(
        [
            ("x4", "x3"),
            ("x3", "x4"),
            ("x6", "x5"),
        ]
    )

    assert pairs == (
        ("x3", "x4"),
        ("x5", "x6"),
    )


def test_random_pairs_are_reproducible():
    features = tuple(
        f"x{index}"
        for index in range(
            1,
            11,
        )
    )

    first = sample_random_pairs(
        feature_names=features,
        n_pairs=7,
        seed=1234,
    )

    second = sample_random_pairs(
        feature_names=features,
        n_pairs=7,
        seed=1234,
    )

    assert first == second

    assert len(
        first
    ) == 7

    assert len(
        set(
            first
        )
    ) == 7


def test_random_pairs_contain_no_self_interactions():
    features = (
        "a",
        "b",
        "c",
        "d",
    )

    pairs = sample_random_pairs(
        feature_names=features,
        n_pairs=6,
        seed=42,
    )

    for first, second in pairs:
        assert first != second


def test_random_pair_count_cannot_exceed_universe():
    with pytest.raises(
        ValueError
    ):
        sample_random_pairs(
            feature_names=(
                "a",
                "b",
                "c",
            ),
            n_pairs=4,
            seed=42,
        )


def test_s1_generator_matches_locked_shape():
    dataset = (
        generate_synthetic_scenario(
            "S1",
            seed=10000,
        )
    )

    assert len(
        dataset.X
    ) == 5000

    assert dataset.X.shape[1] == 20

    assert len(
        dataset.y
    ) == 5000

    assert len(
        dataset.true_interactions
    ) == 3


def test_all_locked_scenarios_can_be_generated():
    protocol = (
        SyntheticBenchmarkProtocol()
    )

    schedule = (
        build_synthetic_schedule(
            protocol
        )
    )

    first_spec_by_scenario = {}

    for specification in schedule:
        first_spec_by_scenario.setdefault(
            specification.scenario,
            specification,
        )

    expected_n = {
        "S1": 5000,
        "S2": 8000,
        "S3": 8000,
        "S4": 10000,
    }

    for scenario, specification in (
        first_spec_by_scenario.items()
    ):
        dataset = (
            generate_synthetic_scenario(
                scenario,
                seed=(
                    specification
                    .dataset_seed
                ),
            )
        )

        assert len(
            dataset.X
        ) == expected_n[
            scenario
        ]

        assert len(
            dataset.y
        ) == expected_n[
            scenario
        ]

        assert len(
            dataset.true_interactions
        ) >= 1


def test_first_primary_schedule_spec_is_locked():
    specification = (
        build_synthetic_schedule()[0]
    )

    assert specification.scenario == "S1"

    assert specification.realization_index == 0

    assert specification.dataset_seed == 10000

    assert specification.outer_split_seed == 20000

    assert specification.final_split_seed == 30000

    assert specification.discovery_base_seed == 40000

    assert specification.discovery_seeds == (
        40000,
        40001,
        40002,
        40003,
        40004,
    )

    assert specification.random_pair_seed == 80000

    assert specification.final_model_seed == 90000