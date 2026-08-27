import numpy as np

from agnam.benchmarking.synthetic_protocol import (
    SYNTHETIC_SCENARIOS,
    SyntheticBenchmarkProtocol,
    SyntheticBenchmarkRecord,
    build_synthetic_schedule,
    candidate_count_from_feature_count,
    compute_false_positive_reduction,
    evaluate_pair_set_recovery,
    records_to_frame,
)


def test_primary_protocol_defaults_are_locked():
    protocol = (
        SyntheticBenchmarkProtocol()
    )

    assert protocol.scenarios == (
        "S1",
        "S2",
        "S3",
        "S4",
    )

    assert protocol.n_realizations == 20

    assert protocol.n_discovery_runs == 5

    assert np.isclose(
        protocol.selection_threshold,
        0.60,
    )

    assert np.isclose(
        protocol.isr_threshold,
        0.60,
    )

    assert protocol.reference_size == 512

    assert protocol.reference_seed == 2026


def test_primary_schedule_contains_80_realizations():
    schedule = (
        build_synthetic_schedule()
    )

    assert len(schedule) == 80

    counts = {
        scenario: 0
        for scenario
        in SYNTHETIC_SCENARIOS
    }

    for specification in schedule:
        counts[
            specification.scenario
        ] += 1

    assert counts == {
        "S1": 20,
        "S2": 20,
        "S3": 20,
        "S4": 20,
    }


def test_schedule_is_deterministic():
    first = (
        build_synthetic_schedule()
    )

    second = (
        build_synthetic_schedule()
    )

    assert first == second


def test_primary_seed_streams_are_unique():
    schedule = (
        build_synthetic_schedule()
    )

    dataset_seeds = [
        item.dataset_seed
        for item in schedule
    ]

    outer_seeds = [
        item.outer_split_seed
        for item in schedule
    ]

    final_seeds = [
        item.final_split_seed
        for item in schedule
    ]

    random_seeds = [
        item.random_pair_seed
        for item in schedule
    ]

    model_seeds = [
        item.final_model_seed
        for item in schedule
    ]

    assert len(
        set(
            dataset_seeds
        )
    ) == 80

    assert len(
        set(
            outer_seeds
        )
    ) == 80

    assert len(
        set(
            final_seeds
        )
    ) == 80

    assert len(
        set(
            random_seeds
        )
    ) == 80

    assert len(
        set(
            model_seeds
        )
    ) == 80


def test_discovery_seed_blocks_do_not_overlap():
    schedule = (
        build_synthetic_schedule()
    )

    all_discovery_seeds = []

    for specification in schedule:
        seeds = (
            specification
            .discovery_seeds
        )

        assert len(
            seeds
        ) == 5

        assert seeds == tuple(
            specification
            .discovery_base_seed
            + index
            for index in range(
                5
            )
        )

        all_discovery_seeds.extend(
            seeds
        )

    assert len(
        all_discovery_seeds
    ) == 400

    assert len(
        set(
            all_discovery_seeds
        )
    ) == 400


def test_candidate_count_matches_locked_rule():
    assert (
        candidate_count_from_feature_count(
            5
        )
        == 5
    )

    assert (
        candidate_count_from_feature_count(
            10
        )
        == 5
    )

    assert (
        candidate_count_from_feature_count(
            20
        )
        == 19
    )

    assert (
        candidate_count_from_feature_count(
            50
        )
        == 20
    )


def test_pair_recovery_perfect():
    truth = (
        ("x3", "x4"),
        ("x5", "x6"),
        ("x8", "x9"),
    )

    predicted = (
        ("x4", "x3"),
        ("x5", "x6"),
        ("x9", "x8"),
    )

    result = (
        evaluate_pair_set_recovery(
            predicted_pairs=predicted,
            true_pairs=truth,
        )
    )

    assert result.true_positive == 3
    assert result.false_positive == 0
    assert result.false_negative == 0

    assert np.isclose(
        result.precision,
        1.0,
    )

    assert np.isclose(
        result.recall,
        1.0,
    )

    assert np.isclose(
        result.f1,
        1.0,
    )


def test_pair_recovery_partial():
    truth = (
        ("x3", "x4"),
        ("x5", "x6"),
        ("x8", "x9"),
    )

    predicted = (
        ("x3", "x4"),
        ("x3", "x5"),
        ("x6", "x9"),
    )

    result = (
        evaluate_pair_set_recovery(
            predicted_pairs=predicted,
            true_pairs=truth,
        )
    )

    assert result.true_positive == 1
    assert result.false_positive == 2
    assert result.false_negative == 2

    assert np.isclose(
        result.precision,
        1.0 / 3.0,
    )

    assert np.isclose(
        result.recall,
        1.0 / 3.0,
    )

    assert np.isclose(
        result.f1,
        1.0 / 3.0,
    )


def test_empty_predicted_set_has_zero_precision_and_recall():
    truth = (
        ("x1", "x2"),
        ("x3", "x4"),
    )

    result = (
        evaluate_pair_set_recovery(
            predicted_pairs=(),
            true_pairs=truth,
        )
    )

    assert result.true_positive == 0
    assert result.false_positive == 0
    assert result.false_negative == 2

    assert np.isclose(
        result.precision,
        0.0,
    )

    assert np.isclose(
        result.recall,
        0.0,
    )

    assert np.isclose(
        result.f1,
        0.0,
    )


def test_false_positive_reduction_and_record_schema():
    selection = (
        evaluate_pair_set_recovery(
            predicted_pairs=(
                ("x1", "x2"),
                ("x1", "x3"),
                ("x2", "x4"),
                ("x3", "x4"),
            ),
            true_pairs=(
                ("x1", "x2"),
                ("x3", "x4"),
            ),
        )
    )

    after_isr = (
        evaluate_pair_set_recovery(
            predicted_pairs=(
                ("x1", "x2"),
                ("x1", "x3"),
                ("x3", "x4"),
            ),
            true_pairs=(
                ("x1", "x2"),
                ("x3", "x4"),
            ),
        )
    )

    reduction = (
        compute_false_positive_reduction(
            selection_recovery=selection,
            isr_recovery=after_isr,
        )
    )

    assert np.isclose(
        reduction,
        0.5,
    )

    record = SyntheticBenchmarkRecord(
        scenario="S1",
        realization_index=0,
        dataset_seed=10000,
        outer_split_seed=20000,
        final_split_seed=30000,
        discovery_base_seed=40000,
        random_pair_seed=80000,
        final_model_seed=90000,
        n_samples=5000,
        n_features=20,
        positive_fraction=0.50,
        n_true_interactions=3,
        candidate_k=19,
    )

    frame = records_to_frame(
        [
            record
        ]
    )

    assert frame.shape[0] == 1

    required_columns = {
        "scenario",
        "realization_index",
        "dataset_seed",
        "main_auroc",
        "agnam_auroc",
        "oracle_auroc",
        "random_pair_auroc",
        "no_isr_auroc",
        "single_run_auroc",
        "isr_precision",
        "isr_recall",
        "runtime_seconds",
    }

    assert required_columns.issubset(
        set(
            frame.columns
        )
    )