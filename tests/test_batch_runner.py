from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from agnam.benchmarking.batch_runner import (
    build_pair_audit_frame,
    clear_failure,
    realization_pairs_path,
    realization_record_path,
    rebuild_master_table,
    record_failure,
    select_schedule,
    summarize_master_table,
    validate_completed_realization,
)
from agnam.benchmarking.synthetic_protocol import (
    SyntheticBenchmarkProtocol,
    SyntheticBenchmarkRecord,
    build_synthetic_schedule,
)


def first_spec():
    return (
        build_synthetic_schedule()[0]
    )


def make_record(
    spec,
    *,
    main_auroc=0.75,
    agnam_auroc=0.88,
):
    return SyntheticBenchmarkRecord(
        scenario=spec.scenario,
        realization_index=(
            spec.realization_index
        ),
        dataset_seed=(
            spec.dataset_seed
        ),
        outer_split_seed=(
            spec.outer_split_seed
        ),
        final_split_seed=(
            spec.final_split_seed
        ),
        discovery_base_seed=(
            spec.discovery_base_seed
        ),
        random_pair_seed=(
            spec.random_pair_seed
        ),
        final_model_seed=(
            spec.final_model_seed
        ),
        n_samples=5000,
        n_features=20,
        positive_fraction=0.50,
        n_true_interactions=3,
        candidate_k=19,
        main_auroc=main_auroc,
        agnam_auroc=agnam_auroc,
        delta_auroc=(
            agnam_auroc
            - main_auroc
        ),
    )


def test_realization_paths_are_deterministic(
    tmp_path,
):
    spec = first_spec()

    record = (
        realization_record_path(
            tmp_path,
            spec,
        )
    )

    pairs = (
        realization_pairs_path(
            tmp_path,
            spec,
        )
    )

    assert record == (
        tmp_path
        / "S1"
        / "realization_01.csv"
    )

    assert pairs == (
        tmp_path
        / "S1"
        / "realization_01_pairs.csv"
    )


def test_select_schedule_range():
    selected = (
        select_schedule(
            scenario="S1",
            start=2,
            end=4,
        )
    )

    assert len(
        selected
    ) == 3

    assert tuple(
        item.realization_number
        for item in selected
    ) == (
        2,
        3,
        4,
    )

    assert all(
        item.scenario == "S1"
        for item in selected
    )


def test_select_schedule_all_scenarios():
    selected = (
        select_schedule(
            scenario="ALL",
            start=1,
            end=2,
        )
    )

    assert len(
        selected
    ) == 8

    scenario_counts = {}

    for item in selected:
        scenario_counts[
            item.scenario
        ] = (
            scenario_counts.get(
                item.scenario,
                0,
            )
            + 1
        )

    assert scenario_counts == {
        "S1": 2,
        "S2": 2,
        "S3": 2,
        "S4": 2,
    }


def test_invalid_schedule_range_raises():
    protocol = (
        SyntheticBenchmarkProtocol()
    )

    with pytest.raises(
        ValueError
    ):
        select_schedule(
            protocol=protocol,
            scenario="S1",
            start=0,
            end=2,
        )

    with pytest.raises(
        ValueError
    ):
        select_schedule(
            protocol=protocol,
            scenario="S1",
            start=5,
            end=4,
        )

    with pytest.raises(
        ValueError
    ):
        select_schedule(
            protocol=protocol,
            scenario="S1",
            start=1,
            end=21,
        )


def test_missing_checkpoint_is_not_complete(
    tmp_path,
):
    spec = first_spec()

    path = (
        realization_record_path(
            tmp_path,
            spec,
        )
    )

    assert not (
        validate_completed_realization(
            path,
            spec,
        )
    )


def test_valid_checkpoint_is_accepted(
    tmp_path,
):
    spec = first_spec()

    path = (
        realization_record_path(
            tmp_path,
            spec,
        )
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    record = make_record(
        spec
    )

    pd.DataFrame(
        [
            record.__dict__
        ]
    ).to_csv(
        path,
        index=False,
    )

    assert (
        validate_completed_realization(
            path,
            spec,
        )
    )


def test_mismatched_checkpoint_raises(
    tmp_path,
):
    spec = first_spec()

    path = (
        realization_record_path(
            tmp_path,
            spec,
        )
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    record = make_record(
        spec
    )

    row = dict(
        record.__dict__
    )

    row[
        "dataset_seed"
    ] = 999999

    pd.DataFrame(
        [
            row
        ]
    ).to_csv(
        path,
        index=False,
    )

    with pytest.raises(
        RuntimeError
    ):
        validate_completed_realization(
            path,
            spec,
        )


def test_pair_audit_marks_membership():
    result = SimpleNamespace(
        true_pairs=(
            ("x1", "x2"),
            ("x3", "x4"),
        ),
        selection_pairs=(
            ("x1", "x2"),
            ("x1", "x3"),
        ),
        isr_pairs=(
            ("x1", "x2"),
        ),
        random_pairs=(
            ("x2", "x4"),
        ),
        single_run_pairs=(
            ("x1", "x2"),
            ("x3", "x4"),
            ("x1", "x4"),
        ),
    )

    frame = (
        build_pair_audit_frame(
            result
        )
    )

    lookup = {
        (
            row.feature_j,
            row.feature_k,
        ): row
        for row in frame.itertuples()
    }

    assert lookup[
        ("x1", "x2")
    ].is_true

    assert lookup[
        ("x1", "x2")
    ].selection_stable

    assert lookup[
        ("x1", "x2")
    ].isr_retained

    assert lookup[
        ("x2", "x4")
    ].random_control

    assert lookup[
        ("x3", "x4")
    ].single_run_top_k


def test_master_rebuild_excludes_pair_files(
    tmp_path,
):
    schedule = (
        build_synthetic_schedule()
    )

    spec_1 = schedule[0]
    spec_2 = schedule[1]

    for spec, auc in (
        (spec_1, 0.80),
        (spec_2, 0.82),
    ):
        path = (
            realization_record_path(
                tmp_path,
                spec,
            )
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        record = make_record(
            spec,
            agnam_auroc=auc,
        )

        pd.DataFrame(
            [
                record.__dict__
            ]
        ).to_csv(
            path,
            index=False,
        )

        pair_path = (
            realization_pairs_path(
                tmp_path,
                spec,
            )
        )

        pd.DataFrame(
            {
                "feature_j": [
                    "x1"
                ],
                "feature_k": [
                    "x2"
                ],
            }
        ).to_csv(
            pair_path,
            index=False,
        )

    master = (
        rebuild_master_table(
            tmp_path
        )
    )

    assert len(
        master
    ) == 2

    assert tuple(
        master[
            "realization_index"
        ].tolist()
    ) == (
        0,
        1,
    )


def test_scenario_summary_statistics():
    master = pd.DataFrame(
        {
            "scenario": [
                "S1",
                "S1",
            ],
            "main_auroc": [
                0.70,
                0.80,
            ],
            "agnam_auroc": [
                0.85,
                0.89,
            ],
            "delta_auroc": [
                0.15,
                0.09,
            ],
        }
    )

    summary = (
        summarize_master_table(
            master
        )
    )

    row = summary[
        (
            summary[
                "scenario"
            ]
            == "S1"
        )
        & (
            summary[
                "metric"
            ]
            == "agnam_auroc"
        )
    ].iloc[
        0
    ]

    assert int(
        row["n"]
    ) == 2

    assert np.isclose(
        row["mean"],
        0.87,
    )

    assert np.isclose(
        row["median"],
        0.87,
    )

    assert np.isclose(
        row["q25"],
        0.86,
    )

    assert np.isclose(
        row["q75"],
        0.88,
    )

    assert np.isclose(
        row["iqr"],
        0.02,
    )


def test_failure_log_can_be_recorded_and_cleared(
    tmp_path,
):
    spec = first_spec()

    try:
        raise RuntimeError(
            "synthetic test failure"
        )

    except RuntimeError as exception:
        path = record_failure(
            tmp_path,
            spec,
            exception,
        )

    assert path.exists()

    frame = pd.read_csv(
        path
    )

    assert len(
        frame
    ) == 1

    assert (
        frame.iloc[
            0
        ][
            "scenario"
        ]
        == "S1"
    )

    assert (
        frame.iloc[
            0
        ][
            "exception_type"
        ]
        == "RuntimeError"
    )

    clear_failure(
        tmp_path,
        spec,
    )

    cleared = pd.read_csv(
        path
    )

    assert len(
        cleared
    ) == 0