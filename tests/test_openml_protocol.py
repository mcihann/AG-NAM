from agnam.benchmarking.openml_protocol import (
    OPENML_TASKS,
    OpenMLBenchmarkProtocol,
    build_openml_schedule,
    openml_registry_frame,
)


def test_openml_registry_contains_28_locked_tasks():
    assert len(
        OPENML_TASKS
    ) == 28

    task_ids = [
        task.task_id
        for task in OPENML_TASKS
    ]

    assert len(
        set(
            task_ids
        )
    ) == 28


def test_openml_feature_type_distribution_is_locked():
    counts = {
        "numeric": 0,
        "mixed": 0,
        "categorical": 0,
    }

    for task in OPENML_TASKS:
        counts[
            task.feature_type
        ] += 1

    assert counts == {
        "numeric": 15,
        "mixed": 10,
        "categorical": 3,
    }


def test_openml_protocol_primary_settings_are_locked():
    protocol = (
        OpenMLBenchmarkProtocol()
    )

    assert protocol.repeat == 0
    assert protocol.fold == 0
    assert protocol.sample == 0

    assert (
        protocol.final_validation_fraction
        == 0.20
    )

    assert (
        protocol.n_discovery_runs
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
        protocol.reference_size
        == 512
    )

    assert (
        protocol.reference_seed
        == 2026
    )

    assert (
        protocol.positive_class_rule
        == "minority_in_outer_development"
    )

    assert (
        protocol.primary_metric
        == "AUROC"
    )


def test_openml_schedule_contains_one_spec_per_task():
    schedule = (
        build_openml_schedule()
    )

    assert len(
        schedule
    ) == 28

    assert tuple(
        item.task_id
        for item in schedule
    ) == tuple(
        task.task_id
        for task in OPENML_TASKS
    )


def test_openml_seed_streams_are_unique():
    schedule = (
        build_openml_schedule()
    )

    assert len(
        {
            item.discovery_base_seed
            for item in schedule
        }
    ) == 28

    assert len(
        {
            item.random_pair_seed
            for item in schedule
        }
    ) == 28

    assert len(
        {
            item.final_split_seed
            for item in schedule
        }
    ) == 28

    assert len(
        {
            item.final_model_seed
            for item in schedule
        }
    ) == 28

    discovery_seeds = [
        seed
        for item in schedule
        for seed in (
            item.discovery_seeds
        )
    ]

    assert len(
        discovery_seeds
    ) == 140

    assert len(
        set(
            discovery_seeds
        )
    ) == 140


def test_first_and_last_openml_tasks_are_locked():
    first = OPENML_TASKS[0]
    last = OPENML_TASKS[-1]

    assert first.task_id == 49
    assert first.dataset_name == "tic-tac-toe"

    assert last.task_id == 3904
    assert last.dataset_name == "jm1"


def test_openml_registry_frame_matches_registry():
    frame = (
        openml_registry_frame()
    )

    assert len(
        frame
    ) == 28

    assert set(
        frame[
            "feature_type"
        ]
    ) == {
        "numeric",
        "mixed",
        "categorical",
    }

    assert (
        frame[
            "expected_n_predictors"
        ]
        > 0
    ).all()