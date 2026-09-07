import numpy as np
import pandas as pd
import pytest

from agnam.benchmarking.openml_runner import (
    baseline_feature_types,
    categorical_feature_indices,
    get_external_execution_spec,
    get_openml_execution_spec,
    get_openml_task_spec,
    prepare_external_baseline_frame,
)
from agnam.benchmarking.openml_protocol import (
    OPENML_TASKS,
)


def test_first_canonical_openml_task_is_locked():
    task = (
        get_openml_task_spec(
            49
        )
    )

    assert (
        task.task_id
        == 49
    )

    assert (
        task.dataset_name
        == "tic-tac-toe"
    )

    assert (
        task.feature_type
        == "categorical"
    )

    assert (
        task.expected_n_predictors
        == 9
    )


def test_unknown_openml_task_is_rejected():
    with pytest.raises(
        ValueError
    ):
        get_openml_task_spec(
            999999999
        )


def test_openml_execution_spec_is_deterministic():
    first = (
        get_openml_execution_spec(
            49
        )
    )

    second = (
        get_openml_execution_spec(
            49
        )
    )

    assert (
        first
        == second
    )

    assert (
        first.discovery_base_seed
        == 60000
    )

    assert (
        first.random_pair_seed
        == 70000
    )

    assert (
        first.final_split_seed
        == 50000
    )

    assert (
        first.final_model_seed
        == 80000
    )


def test_external_execution_spec_is_locked():
    specification = (
        get_external_execution_spec(
            49
        )
    )

    assert (
        specification.ebm_seed
        == 120000
    )

    assert (
        specification.catboost_seed
        == 130000
    )


def test_feature_type_adapter():
    result = (
        baseline_feature_types(
            [
                False,
                True,
                False,
                True,
            ]
        )
    )

    assert result == (
        "continuous",
        "nominal",
        "continuous",
        "nominal",
    )


def test_categorical_feature_indices():
    indices = (
        categorical_feature_indices(
            [
                False,
                True,
                False,
                True,
            ]
        )
    )

    assert indices == (
        1,
        3,
    )


def test_external_frame_preserves_numeric_missing_values():
    frame = pd.DataFrame(
        {
            "a": [
                1.0,
                np.nan,
                3.0,
            ],
            "b": [
                "x",
                "y",
                "z",
            ],
        }
    )

    prepared = (
        prepare_external_baseline_frame(
            frame,
            [
                False,
                True,
            ],
        )
    )

    assert np.isnan(
        prepared.loc[
            1,
            "a",
        ]
    )

    assert (
        prepared[
            "a"
        ].dtype
        == np.float64
    )


def test_external_frame_encodes_categorical_missing_values():
    frame = pd.DataFrame(
        {
            "category": [
                "A",
                None,
                "B",
            ]
        }
    )

    prepared = (
        prepare_external_baseline_frame(
            frame,
            [
                True,
            ],
        )
    )

    assert (
        prepared.loc[
            1,
            "category",
        ]
        == "__AGNAM_MISSING__"
    )

    assert all(
        isinstance(
            value,
            str,
        )
        for value
        in prepared[
            "category"
        ]
    )


def test_external_frame_rejects_indicator_length_mismatch():
    frame = pd.DataFrame(
        {
            "a": [
                1,
                2,
            ],
            "b": [
                3,
                4,
            ],
        }
    )

    with pytest.raises(
        ValueError
    ):
        prepare_external_baseline_frame(
            frame,
            [
                False,
            ],
        )


def test_locked_registry_task_ids_are_available_to_runner():
    for task in (
        OPENML_TASKS
    ):
        resolved = (
            get_openml_task_spec(
                task.task_id
            )
        )

        assert (
            resolved
            == task
        )