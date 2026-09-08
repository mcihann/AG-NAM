from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from agnam.benchmarking.realx_openml import (
    LoadedRealXOpenML,
    _normalize_name,
    audit_realx_realization,
    load_realx_openml_X,
    validate_realx_feature_type,
)
from agnam.benchmarking.realx_protocol import (
    LOCKED_REALX_DATASETS,
)


class FakeDataset:
    def __init__(
        self,
        *,
        dataset_id,
        name,
        X,
        y,
        categorical_indicator,
        attribute_names,
        target_name,
    ):
        self.dataset_id = dataset_id
        self.name = name
        self._X = X
        self._y = y
        self._categorical_indicator = (
            categorical_indicator
        )
        self._attribute_names = (
            attribute_names
        )
        self.default_target_attribute = (
            target_name
        )

    def get_data(
        self,
        *,
        dataset_format,
        target,
    ):
        assert (
            dataset_format
            == "dataframe"
        )

        assert (
            target
            == self
            .default_target_attribute
        )

        return (
            self._X.copy(),
            self._y.copy(),
            list(
                self
                ._categorical_indicator
            ),
            list(
                self
                ._attribute_names
            ),
        )


class FakeTask:
    def __init__(
        self,
        *,
        dataset,
        target_name,
    ):
        self._dataset = dataset
        self.target_name = target_name

    def get_dataset(
        self,
    ):
        return self._dataset


def make_fake_openml_frame(
    n=200,
):
    rng = np.random.default_rng(
        123
    )

    X = pd.DataFrame(
        {
            "num_1": rng.normal(
                size=n
            ),
            "num_2": rng.normal(
                size=n
            ),
            "cat_1": rng.choice(
                [
                    0,
                    1,
                    2,
                ],
                size=n,
            ),
        }
    )

    y = pd.Series(
        rng.integers(
            0,
            2,
            size=n,
        ),
        name="target",
    )

    return (
        X,
        y,
    )


def test_normalized_dataset_name_accepts_hyphen_underscore_equivalence():
    assert (
        _normalize_name(
            "tic_tac_toe"
        )
        == _normalize_name(
            "tic-tac-toe"
        )
    )


def test_openml_loader_discards_target_and_restores_category(
    monkeypatch,
):
    X, y = (
        make_fake_openml_frame()
    )

    dataset = FakeDataset(
        dataset_id=123,
        name="toy",
        X=X,
        y=y,
        categorical_indicator=[
            False,
            False,
            True,
        ],
        attribute_names=[
            "num_1",
            "num_2",
            "cat_1",
        ],
        target_name="target",
    )

    task = FakeTask(
        dataset=dataset,
        target_name="target",
    )

    monkeypatch.setattr(
        "agnam.benchmarking.realx_openml."
        "openml.tasks.get_task",
        lambda *args, **kwargs: task,
    )

    specification = SimpleNamespace(
        task_id=999,
        dataset_name="toy",
        feature_type="mixed",
    )

    loaded = (
        load_realx_openml_X(
            specification
        )
    )

    assert (
        "target"
        not in loaded.X.columns
    )

    assert (
        str(
            loaded.X[
                "cat_1"
            ].dtype
        )
        == "category"
    )

    assert loaded.numeric_columns == (
        "num_1",
        "num_2",
    )

    assert loaded.categorical_columns == (
        "cat_1",
    )


def test_openml_loader_rejects_dataset_name_mismatch(
    monkeypatch,
):
    X, y = (
        make_fake_openml_frame()
    )

    dataset = FakeDataset(
        dataset_id=123,
        name="wrong-name",
        X=X,
        y=y,
        categorical_indicator=[
            False,
            False,
            True,
        ],
        attribute_names=[
            "num_1",
            "num_2",
            "cat_1",
        ],
        target_name="target",
    )

    task = FakeTask(
        dataset=dataset,
        target_name="target",
    )

    monkeypatch.setattr(
        "agnam.benchmarking.realx_openml."
        "openml.tasks.get_task",
        lambda *args, **kwargs: task,
    )

    specification = SimpleNamespace(
        task_id=999,
        dataset_name="expected-name",
        feature_type="mixed",
    )

    with pytest.raises(
        RuntimeError
    ):
        load_realx_openml_X(
            specification
        )


def test_numeric_feature_type_requires_only_numeric():
    X = pd.DataFrame(
        {
            "a": [
                1.0,
                2.0,
            ],
            "b": [
                3.0,
                4.0,
            ],
        }
    )

    specification = SimpleNamespace(
        feature_type="numeric",
        dataset_name="toy",
    )

    loaded = LoadedRealXOpenML(
        specification=specification,
        X=X,
        openml_dataset_id=1,
        openml_dataset_name="toy",
        target_name="target",
        n_rows=2,
        n_features=2,
        categorical_columns=tuple(),
        numeric_columns=(
            "a",
            "b",
        ),
        n_missing_values=0,
    )

    validate_realx_feature_type(
        loaded
    )


def test_categorical_feature_type_requires_only_categorical():
    X = pd.DataFrame(
        {
            "a": pd.Series(
                [
                    "x",
                    "y",
                ],
                dtype="category",
            ),
            "b": pd.Series(
                [
                    "m",
                    "n",
                ],
                dtype="category",
            ),
        }
    )

    specification = SimpleNamespace(
        feature_type="categorical",
        dataset_name="toy",
    )

    loaded = LoadedRealXOpenML(
        specification=specification,
        X=X,
        openml_dataset_id=1,
        openml_dataset_name="toy",
        target_name="target",
        n_rows=2,
        n_features=2,
        categorical_columns=(
            "a",
            "b",
        ),
        numeric_columns=tuple(),
        n_missing_values=0,
    )

    validate_realx_feature_type(
        loaded
    )


def test_mixed_feature_type_requires_both_kinds():
    X = pd.DataFrame(
        {
            "a": [
                1.0,
                2.0,
            ],
            "b": pd.Series(
                [
                    "x",
                    "y",
                ],
                dtype="category",
            ),
        }
    )

    specification = SimpleNamespace(
        feature_type="mixed",
        dataset_name="toy",
    )

    loaded = LoadedRealXOpenML(
        specification=specification,
        X=X,
        openml_dataset_id=1,
        openml_dataset_name="toy",
        target_name="target",
        n_rows=2,
        n_features=2,
        categorical_columns=(
            "b",
        ),
        numeric_columns=(
            "a",
        ),
        n_missing_values=0,
    )

    validate_realx_feature_type(
        loaded
    )


def test_wrong_locked_feature_type_is_rejected():
    X = pd.DataFrame(
        {
            "a": [
                1.0,
                2.0,
            ],
            "b": [
                3.0,
                4.0,
            ],
        }
    )

    specification = SimpleNamespace(
        feature_type="mixed",
        dataset_name="toy",
    )

    loaded = LoadedRealXOpenML(
        specification=specification,
        X=X,
        openml_dataset_id=1,
        openml_dataset_name="toy",
        target_name="target",
        n_rows=2,
        n_features=2,
        categorical_columns=tuple(),
        numeric_columns=(
            "a",
            "b",
        ),
        n_missing_values=0,
    )

    with pytest.raises(
        RuntimeError
    ):
        validate_realx_feature_type(
            loaded
        )


def make_audit_X(
    n=300,
):
    rng = np.random.default_rng(
        456
    )

    data = {}

    for index in range(
        10
    ):
        data[
            f"x{index}"
        ] = rng.normal(
            size=n
        )

    return pd.DataFrame(
        data
    )


def test_realization_audit_confirms_common_random_numbers():
    X = make_audit_X()

    specification = next(
        dataset
        for dataset in (
            LOCKED_REALX_DATASETS
        )
        if (
            dataset.task_id
            == 3913
        )
    )

    loaded = LoadedRealXOpenML(
        specification=(
            specification
        ),
        X=(
            X
        ),
        openml_dataset_id=999,
        openml_dataset_name=(
            specification
            .dataset_name
        ),
        target_name="discarded",
        n_rows=len(
            X
        ),
        n_features=X.shape[
            1
        ],
        categorical_columns=tuple(),
        numeric_columns=tuple(
            X.columns
            .astype(str)
            .tolist()
        ),
        n_missing_values=0,
    )

    row = (
        audit_realx_realization(
            loaded,
            realization=1,
        )
    )

    assert bool(
        row[
            "common_random_numbers_ok"
        ]
    )

    assert np.isclose(
        row[
            "null_interaction_signal_std"
        ],
        0.0,
    )

    assert np.isclose(
        row[
            "weak_interaction_signal_std"
        ],
        0.5,
    )

    assert np.isclose(
        row[
            "moderate_interaction_signal_std"
        ],
        1.0,
    )

    assert np.isclose(
        row[
            "strong_interaction_signal_std"
        ],
        1.5,
    )


def test_realization_audit_expected_prevalence_is_locked():
    X = make_audit_X()

    specification = next(
        dataset
        for dataset in (
            LOCKED_REALX_DATASETS
        )
        if (
            dataset.task_id
            == 3913
        )
    )

    loaded = LoadedRealXOpenML(
        specification=(
            specification
        ),
        X=X,
        openml_dataset_id=999,
        openml_dataset_name=(
            specification
            .dataset_name
        ),
        target_name="discarded",
        n_rows=len(
            X
        ),
        n_features=X.shape[
            1
        ],
        categorical_columns=tuple(),
        numeric_columns=tuple(
            X.columns
            .astype(str)
            .tolist()
        ),
        n_missing_values=0,
    )

    row = (
        audit_realx_realization(
            loaded,
            realization=1,
        )
    )

    for strength in (
        [
            "null",
            "weak",
            "moderate",
            "strong",
        ]
    ):
        assert np.isclose(
            row[
                f"{strength}_"
                "expected_prevalence"
            ],
            0.50,
            atol=1e-9,
        )


def test_realization_audit_has_three_truth_pairs():
    X = make_audit_X()

    specification = next(
        dataset
        for dataset in (
            LOCKED_REALX_DATASETS
        )
        if (
            dataset.task_id
            == 3913
        )
    )

    loaded = LoadedRealXOpenML(
        specification=(
            specification
        ),
        X=X,
        openml_dataset_id=999,
        openml_dataset_name=(
            specification
            .dataset_name
        ),
        target_name="discarded",
        n_rows=len(
            X
        ),
        n_features=X.shape[
            1
        ],
        categorical_columns=tuple(),
        numeric_columns=tuple(
            X.columns
            .astype(str)
            .tolist()
        ),
        n_missing_values=0,
    )

    row = (
        audit_realx_realization(
            loaded,
            realization=1,
        )
    )

    pairs = (
        row[
            "true_pairs"
        ]
        .split("|")
    )

    assert len(
        pairs
    ) == 3


def test_locked_realx_dataset_ids_remain_nine():
    assert len(
        LOCKED_REALX_DATASETS
    ) == 9

    assert len(
        {
            dataset.task_id
            for dataset in (
                LOCKED_REALX_DATASETS
            )
        }
    ) == 9