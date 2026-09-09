import pandas as pd

from agnam.benchmarking.realx_batch import (
    REQUIRED_CHECKPOINT_COLUMNS,
    build_batch_plan,
    checkpoint_filename,
    checkpoint_path,
    select_run_specs,
    validate_checkpoint,
    write_checkpoint_atomic,
)
from agnam.benchmarking.realx_benchmark_engine import (
    SCHEMA_VERSION,
)
from agnam.benchmarking.realx_protocol import (
    RealXProtocol,
    build_realx_schedule,
)
from agnam.benchmarking.realx_runner import (
    realx_candidate_k,
)


def test_locked_schedule_contains_180_runs():
    schedule = (
        build_realx_schedule()
    )

    assert len(
        schedule
    ) == 180


def test_locked_schedule_run_ids_are_unique():
    schedule = (
        build_realx_schedule()
    )

    run_ids = [
        item.run_id
        for item in schedule
    ]

    assert len(
        run_ids
    ) == len(
        set(
            run_ids
        )
    )


def test_each_dataset_has_twenty_runs():
    schedule = (
        build_realx_schedule()
    )

    task_ids = sorted(
        {
            item.task_id
            for item in schedule
        }
    )

    assert len(
        task_ids
    ) == 9

    for task_id in task_ids:
        assert sum(
            item.task_id
            == task_id
            for item in schedule
        ) == 20


def test_each_dataset_realization_has_four_strengths():
    schedule = (
        build_realx_schedule()
    )

    combinations = {}

    for item in schedule:
        key = (
            item.task_id,
            item.realization,
        )

        combinations.setdefault(
            key,
            set(),
        ).add(
            item.strength_name
        )

    for strengths in (
        combinations.values()
    ):
        assert strengths == {
            "null",
            "weak",
            "moderate",
            "strong",
        }


def test_checkpoint_filename_is_deterministic():
    run = (
        build_realx_schedule()[
            0
        ]
    )

    assert checkpoint_filename(
        run
    ) == (
        f"task_{run.task_id}_"
        f"r{run.realization:02d}_"
        f"{run.strength_name}.csv"
    )


def test_select_single_run_id():
    schedule = (
        build_realx_schedule()
    )

    target = schedule[
        17
    ]

    selected = (
        select_run_specs(
            run_id=(
                target.run_id
            )
        )
    )

    assert selected == (
        target,
    )


def test_select_task_and_strength():
    selected = (
        select_run_specs(
            task_id=49,
            strength="weak",
        )
    )

    assert len(
        selected
    ) == 5

    assert all(
        item.task_id == 49
        and item.strength_name
        == "weak"
        for item in selected
    )


def test_one_based_window_selection():
    all_runs = (
        select_run_specs()
    )

    selected = (
        select_run_specs(
            start=2,
            end=4,
        )
    )

    assert selected == (
        all_runs[
            1
        ],
        all_runs[
            2
        ],
        all_runs[
            3
        ],
    )


def test_empty_directory_plans_everything(
    tmp_path,
):
    specs = (
        select_run_specs()
    )

    plan = build_batch_plan(
        specs,
        output_root=tmp_path,
    )

    assert len(
        plan
    ) == 180

    assert all(
        item.state
        == "PLANNED"
        for item in plan
    )


def make_valid_record(
    run_spec,
):
    protocol = (
        RealXProtocol()
    )

    record = {
        column: 0
        for column in (
            REQUIRED_CHECKPOINT_COLUMNS
        )
    }

    record.update(
        {
            "schema_version": (
                SCHEMA_VERSION
            ),
            "benchmark_id": (
                protocol
                .benchmark_id
            ),
            "status": "COMPLETE",
            "run_id": (
                run_spec.run_id
            ),
            "task_id": (
                run_spec.task_id
            ),
            "dataset_name": (
                run_spec.dataset_name
            ),
            "feature_type": (
                run_spec.feature_type
            ),
            "realization": (
                run_spec.realization
            ),
            "strength_name": (
                run_spec.strength_name
            ),
            "interaction_coefficient": (
                run_spec
                .interaction_coefficient
            ),
            "n_features": 9,
            "expected_candidate_k": (
                realx_candidate_k(
                    9
                )
            ),
            "observed_candidate_k": (
                realx_candidate_k(
                    9
                )
            ),
            "n_true_interactions": (
                0
                if (
                    run_spec
                    .strength_name
                    == "null"
                )
                else 3
            ),
            "max_decomposition_error": (
                0.0
            ),
            "test_integrity_ok": True,
            "publication_eligible": True,
            "runtime_seconds": 1.0,
        }
    )

    return record


def test_valid_checkpoint_is_accepted(
    tmp_path,
):
    run = (
        build_realx_schedule()[
            0
        ]
    )

    path = checkpoint_path(
        run,
        output_root=tmp_path,
    )

    write_checkpoint_atomic(
        make_valid_record(
            run
        ),
        path,
    )

    valid, reason = (
        validate_checkpoint(
            path,
            run,
        )
    )

    assert valid is True
    assert reason == "valid"


def test_publication_ineligible_checkpoint_is_rejected(
    tmp_path,
):
    run = (
        build_realx_schedule()[
            0
        ]
    )

    record = make_valid_record(
        run
    )

    record[
        "publication_eligible"
    ] = False

    path = checkpoint_path(
        run,
        output_root=tmp_path,
    )

    write_checkpoint_atomic(
        record,
        path,
    )

    valid, reason = (
        validate_checkpoint(
            path,
            run,
        )
    )

    assert valid is False

    assert (
        reason
        == "publication_eligible_false"
    )


def test_candidate_k_mismatch_is_rejected(
    tmp_path,
):
    run = (
        build_realx_schedule()[
            0
        ]
    )

    record = make_valid_record(
        run
    )

    record[
        "observed_candidate_k"
    ] = (
        record[
            "expected_candidate_k"
        ]
        - 1
    )

    path = checkpoint_path(
        run,
        output_root=tmp_path,
    )

    write_checkpoint_atomic(
        record,
        path,
    )

    valid, reason = (
        validate_checkpoint(
            path,
            run,
        )
    )

    assert valid is False

    assert (
        reason
        == "observed_candidate_k_invalid"
    )


def test_atomic_checkpoint_contains_one_row(
    tmp_path,
):
    run = (
        build_realx_schedule()[
            0
        ]
    )

    path = checkpoint_path(
        run,
        output_root=tmp_path,
    )

    write_checkpoint_atomic(
        make_valid_record(
            run
        ),
        path,
    )

    table = pd.read_csv(
        path
    )

    assert len(
        table
    ) == 1