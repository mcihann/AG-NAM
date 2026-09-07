from pathlib import Path

import pandas as pd
import pytest

from agnam.benchmarking.external_baselines import (
    ExternalBaselineProtocol,
    build_external_baseline_schedule,
)
from agnam.benchmarking.openml_batch import (
    adopt_canonical_checkpoint,
    canonical_checkpoint_available,
    enrich_checkpoint_frame,
    rebuild_openml_master,
    rebuild_progress_table,
    select_openml_schedule,
    task_pairs_path,
    task_record_path,
    validate_openml_checkpoint,
)
from agnam.benchmarking.openml_protocol import (
    OpenMLBenchmarkProtocol,
    build_openml_schedule,
)


def first_specs():
    benchmark_protocol = (
        OpenMLBenchmarkProtocol()
    )

    baseline_protocol = (
        ExternalBaselineProtocol()
    )

    execution = (
        build_openml_schedule(
            benchmark_protocol
        )[0]
    )

    external = (
        build_external_baseline_schedule(
            baseline_protocol
        )[0]
    )

    return (
        benchmark_protocol,
        baseline_protocol,
        execution,
        external,
    )


def minimal_task_49_frame():
    return pd.DataFrame(
        [
            {
                "task_id": 49,
                "dataset_id": 50,
                "dataset_name": (
                    "tic-tac-toe"
                ),
                "feature_type": (
                    "categorical"
                ),
                "n_samples": 958,
                "n_features": 9,
                "n_numeric_features": 0,
                "n_categorical_features": 9,
                "main_auroc": 0.98,
                "agnam_auroc": 0.98,
                "ebm_auroc": 0.99,
                "catboost_auroc": 1.0,
            }
        ]
    )


def test_openml_batch_paths_are_deterministic(
    tmp_path,
):
    _, _, execution, _ = (
        first_specs()
    )

    assert (
        task_record_path(
            tmp_path,
            execution,
        )
        == (
            tmp_path
            / "task_49.csv"
        )
    )

    assert (
        task_pairs_path(
            tmp_path,
            execution,
        )
        == (
            tmp_path
            / "task_49_pairs.csv"
        )
    )


def test_select_openml_registry_range():
    selected = (
        select_openml_schedule(
            start=2,
            end=4,
        )
    )

    assert len(
        selected
    ) == 3

    assert tuple(
        specification
        .task_index
        + 1
        for specification
        in selected
    ) == (
        2,
        3,
        4,
    )


def test_select_one_openml_task_id():
    selected = (
        select_openml_schedule(
            task_id=49
        )
    )

    assert len(
        selected
    ) == 1

    assert (
        selected[
            0
        ].task_id
        == 49
    )


def test_unknown_task_id_is_rejected():
    with pytest.raises(
        ValueError
    ):
        select_openml_schedule(
            task_id=99999999
        )


def test_checkpoint_enrichment_adds_locked_metadata():
    (
        benchmark_protocol,
        baseline_protocol,
        execution,
        external,
    ) = first_specs()

    enriched = (
        enrich_checkpoint_frame(
            minimal_task_49_frame(),
            execution,
            external,
            benchmark_protocol=(
                benchmark_protocol
            ),
            baseline_protocol=(
                baseline_protocol
            ),
        )
    )

    row = enriched.iloc[
        0
    ]

    assert int(
        row[
            "discovery_base_seed"
        ]
    ) == 60000

    assert int(
        row[
            "random_pair_seed"
        ]
    ) == 70000

    assert int(
        row[
            "final_model_seed"
        ]
    ) == 80000

    assert int(
        row[
            "ebm_seed"
        ]
    ) == 120000

    assert int(
        row[
            "catboost_seed"
        ]
    ) == 130000

    assert (
        row[
            "benchmark_protocol"
        ]
        == "agnam_openml_primary_v1"
    )


def test_valid_enriched_checkpoint_is_accepted(
    tmp_path,
):
    (
        benchmark_protocol,
        baseline_protocol,
        execution,
        external,
    ) = first_specs()

    frame = (
        enrich_checkpoint_frame(
            minimal_task_49_frame(),
            execution,
            external,
            benchmark_protocol=(
                benchmark_protocol
            ),
            baseline_protocol=(
                baseline_protocol
            ),
        )
    )

    path = (
        task_record_path(
            tmp_path,
            execution,
        )
    )

    frame.to_csv(
        path,
        index=False,
    )

    assert (
        validate_openml_checkpoint(
            path,
            execution,
            external,
            benchmark_protocol=(
                benchmark_protocol
            ),
            baseline_protocol=(
                baseline_protocol
            ),
        )
    )


def test_seed_mismatch_checkpoint_is_rejected(
    tmp_path,
):
    (
        benchmark_protocol,
        baseline_protocol,
        execution,
        external,
    ) = first_specs()

    frame = (
        enrich_checkpoint_frame(
            minimal_task_49_frame(),
            execution,
            external,
            benchmark_protocol=(
                benchmark_protocol
            ),
            baseline_protocol=(
                baseline_protocol
            ),
        )
    )

    frame.loc[
        0,
        "final_model_seed",
    ] = 999999

    path = (
        task_record_path(
            tmp_path,
            execution,
        )
    )

    frame.to_csv(
        path,
        index=False,
    )

    with pytest.raises(
        RuntimeError
    ):
        validate_openml_checkpoint(
            path,
            execution,
            external,
            benchmark_protocol=(
                benchmark_protocol
            ),
            baseline_protocol=(
                baseline_protocol
            ),
        )


def test_canonical_checkpoint_can_be_adopted(
    tmp_path,
):
    (
        benchmark_protocol,
        baseline_protocol,
        execution,
        external,
    ) = first_specs()

    canonical = (
        tmp_path
        / "canonical"
    )

    benchmark = (
        tmp_path
        / "benchmark"
    )

    canonical.mkdir()

    minimal_task_49_frame().to_csv(
        canonical
        / "task_49.csv",
        index=False,
    )

    pd.DataFrame(
        [
            {
                "feature_j": "V1",
                "feature_k": "V2",
                "selection_count": 1,
                "selection_frequency": 0.2,
                "selection_stable": False,
                "isr_retained": False,
                "isr_score": 0.1,
                "random_control": False,
                "single_run_top_k": True,
            }
        ]
    ).to_csv(
        canonical
        / "task_49_pairs.csv",
        index=False,
    )

    assert (
        canonical_checkpoint_available(
            canonical,
            execution,
        )
    )

    adopted = (
        adopt_canonical_checkpoint(
            canonical_root=(
                canonical
            ),
            results_root=(
                benchmark
            ),
            execution_spec=(
                execution
            ),
            external_spec=(
                external
            ),
            benchmark_protocol=(
                benchmark_protocol
            ),
            baseline_protocol=(
                baseline_protocol
            ),
        )
    )

    assert adopted

    assert (
        benchmark
        / "task_49.csv"
    ).exists()

    assert (
        benchmark
        / "task_49_pairs.csv"
    ).exists()

    assert (
        validate_openml_checkpoint(
            benchmark
            / "task_49.csv",
            execution,
            external,
            benchmark_protocol=(
                benchmark_protocol
            ),
            baseline_protocol=(
                baseline_protocol
            ),
        )
    )


def test_master_rebuild_excludes_pair_audit(
    tmp_path,
):
    (
        benchmark_protocol,
        baseline_protocol,
        execution,
        external,
    ) = first_specs()

    frame = (
        enrich_checkpoint_frame(
            minimal_task_49_frame(),
            execution,
            external,
            benchmark_protocol=(
                benchmark_protocol
            ),
            baseline_protocol=(
                baseline_protocol
            ),
        )
    )

    frame.to_csv(
        tmp_path
        / "task_49.csv",
        index=False,
    )

    pd.DataFrame(
        [
            {
                "feature_j": "A",
                "feature_k": "B",
            }
        ]
    ).to_csv(
        tmp_path
        / "task_49_pairs.csv",
        index=False,
    )

    master = (
        rebuild_openml_master(
            tmp_path
        )
    )

    assert len(
        master
    ) == 1

    assert int(
        master.iloc[
            0
        ][
            "task_id"
        ]
    ) == 49


def test_progress_table_has_all_28_locked_tasks(
    tmp_path,
):
    (
        benchmark_protocol,
        baseline_protocol,
        execution,
        external,
    ) = first_specs()

    frame = (
        enrich_checkpoint_frame(
            minimal_task_49_frame(),
            execution,
            external,
            benchmark_protocol=(
                benchmark_protocol
            ),
            baseline_protocol=(
                baseline_protocol
            ),
        )
    )

    frame.to_csv(
        tmp_path
        / "task_49.csv",
        index=False,
    )

    progress = (
        rebuild_progress_table(
            tmp_path,
            benchmark_protocol=(
                benchmark_protocol
            ),
            baseline_protocol=(
                baseline_protocol
            ),
        )
    )

    assert len(
        progress
    ) == 28

    assert int(
        progress[
            "complete"
        ].sum()
    ) == 1

    first = progress.iloc[
        0
    ]

    assert int(
        first[
            "task_id"
        ]
    ) == 49

    assert bool(
        first[
            "complete"
        ]
    )