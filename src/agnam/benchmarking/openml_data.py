from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import numpy as np
import pandas as pd

from agnam.benchmarking.openml_protocol import (
    OpenMLBenchmarkProtocol,
    OpenMLTaskSpec,
)


@dataclass
class LoadedOpenMLTask:
    """
    Fully loaded and audited-ready OpenML task.

    The raw target is retained so that the original class labels
    remain traceable.

    Binary encoding is defined exclusively from the outer-development
    labels.
    """

    task_spec: OpenMLTaskSpec

    dataset_id: int
    actual_dataset_name: str
    target_name: str

    X: pd.DataFrame
    y_raw: pd.Series

    categorical_indicator: tuple[
        bool,
        ...
    ]

    attribute_names: tuple[
        str,
        ...
    ]

    train_indices: np.ndarray
    test_indices: np.ndarray

    split_repeats: int
    split_folds: int
    split_samples: int

    positive_label: Any
    negative_label: Any

    y_binary: np.ndarray

    @property
    def X_development(
        self,
    ) -> pd.DataFrame:
        return (
            self.X
            .iloc[
                self.train_indices
            ]
            .reset_index(
                drop=True
            )
        )

    @property
    def X_test(
        self,
    ) -> pd.DataFrame:
        return (
            self.X
            .iloc[
                self.test_indices
            ]
            .reset_index(
                drop=True
            )
        )

    @property
    def y_development(
        self,
    ) -> np.ndarray:
        return (
            self.y_binary[
                self.train_indices
            ]
            .copy()
        )

    @property
    def y_test(
        self,
    ) -> np.ndarray:
        return (
            self.y_binary[
                self.test_indices
            ]
            .copy()
        )


@dataclass(frozen=True)
class OpenMLTaskAudit:
    task_id: int

    expected_dataset_name: str
    actual_dataset_name: str

    dataset_id: int
    target_name: str

    n_samples: int
    n_predictors: int

    n_numeric_predictors: int
    n_categorical_predictors: int

    feature_type: str

    n_features_with_missing: int
    missing_fraction: float

    split_repeats: int
    split_folds: int
    split_samples: int

    n_development: int
    n_test: int

    split_disjoint: bool
    split_full_coverage: bool

    full_class_counts: str
    development_class_counts: str
    test_class_counts: str

    positive_label: str
    negative_label: str

    development_positive_count: int
    development_negative_count: int

    test_positive_count: int
    test_negative_count: int

    development_positive_fraction: float
    test_positive_fraction: float

    name_matches_registry: bool
    sample_count_matches_registry: bool
    predictor_count_matches_registry: bool
    numeric_count_matches_registry: bool
    categorical_count_matches_registry: bool
    feature_type_matches_registry: bool

    binary_target: bool
    development_has_both_classes: bool
    test_has_both_classes: bool

    audit_pass: bool


def _as_target_series(
    y,
    *,
    name: str | None = None,
) -> pd.Series:
    """
    Normalize an OpenML target to a one-dimensional pandas Series.
    """
    if isinstance(
        y,
        pd.DataFrame,
    ):
        if y.shape[1] != 1:
            raise ValueError(
                "Only one target column is supported."
            )

        series = y.iloc[
            :,
            0
        ].copy()

    elif isinstance(
        y,
        pd.Series,
    ):
        series = y.copy()

    else:
        array = np.asarray(
            y
        )

        if array.ndim != 1:
            raise ValueError(
                "Target must be one-dimensional."
            )

        series = pd.Series(
            array
        )

    if name is not None:
        series.name = name

    return series.reset_index(
        drop=True
    )


def observed_class_counts(
    y,
) -> dict[
    Any,
    int,
]:
    """
    Return observed target counts only.

    Unused pandas categorical levels are intentionally excluded.
    """
    series = _as_target_series(
        y
    )

    if series.isna().any():
        raise ValueError(
            "Target contains missing values."
        )

    observed = (
        series
        .astype(
            object
        )
        .value_counts(
            dropna=False
        )
    )

    return {
        label: int(
            count
        )
        for label, count
        in observed.items()
    }


def class_count_signature(
    y,
) -> str:
    """
    Stable JSON representation of observed class counts.
    """
    counts = (
        observed_class_counts(
            y
        )
    )

    rows = [
        {
            "label": str(
                label
            ),
            "count": int(
                count
            ),
        }
        for label, count
        in sorted(
            counts.items(),
            key=lambda item: str(
                item[0]
            ),
        )
    ]

    return json.dumps(
        rows,
        ensure_ascii=False,
        separators=(
            ",",
            ":",
        ),
    )


def determine_binary_labels(
    y_development,
) -> tuple[
    Any,
    Any,
]:
    """
    Determine positive and negative labels from outer-development
    data only.

    Positive class:
        minority class in outer development.

    Tie:
        lexicographically smaller string representation.
    """
    counts = (
        observed_class_counts(
            y_development
        )
    )

    if len(
        counts
    ) != 2:
        raise ValueError(
            "Binary benchmark requires exactly "
            "two observed development classes."
        )

    minimum_count = min(
        counts.values()
    )

    positive_candidates = [
        label
        for label, count
        in counts.items()
        if count
        == minimum_count
    ]

    positive_label = sorted(
        positive_candidates,
        key=lambda value: str(
            value
        ),
    )[0]

    negative_candidates = [
        label
        for label
        in counts
        if label
        != positive_label
    ]

    if len(
        negative_candidates
    ) != 1:
        raise RuntimeError(
            "Unable to identify unique negative class."
        )

    negative_label = (
        negative_candidates[
            0
        ]
    )

    return (
        positive_label,
        negative_label,
    )


def encode_binary_target(
    y,
    *,
    positive_label,
) -> np.ndarray:
    """
    Encode the locked positive label as 1 and all other observed
    labels as 0.
    """
    series = _as_target_series(
        y
    )

    if series.isna().any():
        raise ValueError(
            "Target contains missing values."
        )

    values = (
        series
        .eq(
            positive_label
        )
        .astype(
            np.int64
        )
        .to_numpy()
    )

    return values


def validate_split_indices(
    *,
    n_samples: int,
    train_indices,
    test_indices,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Validate one predefined OpenML outer split.
    """
    train = np.asarray(
        train_indices,
        dtype=np.int64,
    )

    test = np.asarray(
        test_indices,
        dtype=np.int64,
    )

    if train.ndim != 1:
        raise ValueError(
            "train_indices must be one-dimensional."
        )

    if test.ndim != 1:
        raise ValueError(
            "test_indices must be one-dimensional."
        )

    if len(
        train
    ) == 0:
        raise ValueError(
            "OpenML development split is empty."
        )

    if len(
        test
    ) == 0:
        raise ValueError(
            "OpenML test split is empty."
        )

    if len(
        np.unique(
            train
        )
    ) != len(
        train
    ):
        raise ValueError(
            "Duplicate indices exist in development split."
        )

    if len(
        np.unique(
            test
        )
    ) != len(
        test
    ):
        raise ValueError(
            "Duplicate indices exist in test split."
        )

    if (
        np.min(
            train
        )
        < 0
        or np.max(
            train
        )
        >= n_samples
    ):
        raise ValueError(
            "Development split contains out-of-range indices."
        )

    if (
        np.min(
            test
        )
        < 0
        or np.max(
            test
        )
        >= n_samples
    ):
        raise ValueError(
            "Test split contains out-of-range indices."
        )

    overlap = np.intersect1d(
        train,
        test,
    )

    if len(
        overlap
    ) != 0:
        raise ValueError(
            "Development and test splits overlap."
        )

    covered = np.union1d(
        train,
        test,
    )

    if len(
        covered
    ) != n_samples:
        raise ValueError(
            "Selected OpenML split does not cover "
            "the complete dataset."
        )

    return (
        train,
        test,
    )


def feature_type_from_indicator(
    categorical_indicator,
) -> str:
    indicator = tuple(
        bool(
            value
        )
        for value
        in categorical_indicator
    )

    n_categorical = sum(
        indicator
    )

    n_numeric = (
        len(
            indicator
        )
        - n_categorical
    )

    if n_categorical == 0:
        return "numeric"

    if n_numeric == 0:
        return "categorical"

    return "mixed"


def load_openml_task(
    task_spec: OpenMLTaskSpec,
    *,
    protocol: OpenMLBenchmarkProtocol | None = None,
) -> LoadedOpenMLTask:
    """
    Download one OpenML task and retrieve the locked predefined
    outer split.

    No predictive model is fitted here.
    """
    if protocol is None:
        protocol = (
            OpenMLBenchmarkProtocol()
        )

    # Lazy import keeps unit tests independent of network access.
    import openml

    task = (
        openml.tasks.get_task(
            task_spec.task_id
        )
    )

    dataset = (
        task.get_dataset()
    )

    (
        X,
        y,
        categorical_indicator,
        attribute_names,
    ) = dataset.get_data(
        target=(
            task.target_name
        ),
        dataset_format="dataframe",
    )

    if not isinstance(
        X,
        pd.DataFrame,
    ):
        X = pd.DataFrame(
            X,
            columns=(
                attribute_names
            ),
        )

    X = X.reset_index(
        drop=True
    )

    y_series = (
        _as_target_series(
            y,
            name=(
                str(
                    task.target_name
                )
            ),
        )
    )

    if len(
        X
    ) != len(
        y_series
    ):
        raise RuntimeError(
            "OpenML predictors and target have "
            "different sample counts."
        )

    categorical_indicator = tuple(
        bool(
            value
        )
        for value
        in categorical_indicator
    )

    attribute_names = tuple(
        str(
            value
        )
        for value
        in attribute_names
    )

    if len(
        categorical_indicator
    ) != X.shape[
        1
    ]:
        raise RuntimeError(
            "Categorical indicator length does not "
            "match predictor count."
        )

    if len(
        attribute_names
    ) != X.shape[
        1
    ]:
        raise RuntimeError(
            "Attribute-name length does not "
            "match predictor count."
        )

    split_dimensions = (
        task.get_split_dimensions()
    )

    if len(
        split_dimensions
    ) != 3:
        raise RuntimeError(
            "Unexpected OpenML split dimensions."
        )

    (
        split_repeats,
        split_folds,
        split_samples,
    ) = (
        int(
            split_dimensions[
                0
            ]
        ),
        int(
            split_dimensions[
                1
            ]
        ),
        int(
            split_dimensions[
                2
            ]
        ),
    )

    if (
        protocol.repeat
        >= split_repeats
    ):
        raise ValueError(
            "Locked repeat index is unavailable "
            "for this OpenML task."
        )

    if (
        protocol.fold
        >= split_folds
    ):
        raise ValueError(
            "Locked fold index is unavailable "
            "for this OpenML task."
        )

    if (
        protocol.sample
        >= split_samples
    ):
        raise ValueError(
            "Locked sample index is unavailable "
            "for this OpenML task."
        )

    (
        train_indices,
        test_indices,
    ) = (
        task
        .get_train_test_split_indices(
            repeat=(
                protocol.repeat
            ),
            fold=(
                protocol.fold
            ),
            sample=(
                protocol.sample
            ),
        )
    )

    (
        train_indices,
        test_indices,
    ) = validate_split_indices(
        n_samples=len(
            X
        ),
        train_indices=(
            train_indices
        ),
        test_indices=(
            test_indices
        ),
    )

    y_development_raw = (
        y_series.iloc[
            train_indices
        ]
        .reset_index(
            drop=True
        )
    )

    positive_label, negative_label = (
        determine_binary_labels(
            y_development_raw
        )
    )

    full_counts = (
        observed_class_counts(
            y_series
        )
    )

    if len(
        full_counts
    ) != 2:
        raise ValueError(
            "OpenML primary registry requires "
            "binary classification tasks."
        )

    y_binary = (
        encode_binary_target(
            y_series,
            positive_label=(
                positive_label
            ),
        )
    )

    return LoadedOpenMLTask(
        task_spec=(
            task_spec
        ),
        dataset_id=int(
            dataset.id
        ),
        actual_dataset_name=str(
            dataset.name
        ),
        target_name=str(
            task.target_name
        ),
        X=(
            X
        ),
        y_raw=(
            y_series
        ),
        categorical_indicator=(
            categorical_indicator
        ),
        attribute_names=(
            attribute_names
        ),
        train_indices=(
            train_indices
        ),
        test_indices=(
            test_indices
        ),
        split_repeats=(
            split_repeats
        ),
        split_folds=(
            split_folds
        ),
        split_samples=(
            split_samples
        ),
        positive_label=(
            positive_label
        ),
        negative_label=(
            negative_label
        ),
        y_binary=(
            y_binary
        ),
    )


def audit_loaded_openml_task(
    loaded: LoadedOpenMLTask,
) -> OpenMLTaskAudit:
    """
    Audit a loaded task against the locked registry.

    No model is trained and no model-derived quantity is inspected.
    """
    spec = (
        loaded.task_spec
    )

    X = (
        loaded.X
    )

    y = (
        loaded.y_raw
    )

    categorical_indicator = (
        loaded
        .categorical_indicator
    )

    n_samples = int(
        len(
            X
        )
    )

    n_predictors = int(
        X.shape[
            1
        ]
    )

    n_categorical = int(
        sum(
            categorical_indicator
        )
    )

    n_numeric = int(
        n_predictors
        - n_categorical
    )

    feature_type = (
        feature_type_from_indicator(
            categorical_indicator
        )
    )

    missing_by_feature = (
        X.isna().sum(
            axis=0
        )
    )

    n_features_with_missing = int(
        (
            missing_by_feature
            > 0
        ).sum()
    )

    if (
        n_samples
        * n_predictors
        == 0
    ):
        missing_fraction = (
            0.0
        )

    else:
        missing_fraction = float(
            X.isna()
            .to_numpy()
            .sum()
            / (
                n_samples
                * n_predictors
            )
        )

    train = (
        loaded.train_indices
    )

    test = (
        loaded.test_indices
    )

    split_disjoint = (
        len(
            np.intersect1d(
                train,
                test,
            )
        )
        == 0
    )

    split_full_coverage = (
        len(
            np.union1d(
                train,
                test,
            )
        )
        == n_samples
    )

    y_development_raw = (
        y.iloc[
            train
        ]
        .reset_index(
            drop=True
        )
    )

    y_test_raw = (
        y.iloc[
            test
        ]
        .reset_index(
            drop=True
        )
    )

    full_counts = (
        observed_class_counts(
            y
        )
    )

    development_counts = (
        observed_class_counts(
            y_development_raw
        )
    )

    test_counts = (
        observed_class_counts(
            y_test_raw
        )
    )

    binary_target = (
        len(
            full_counts
        )
        == 2
    )

    development_has_both_classes = (
        len(
            development_counts
        )
        == 2
    )

    test_has_both_classes = (
        len(
            test_counts
        )
        == 2
    )

    y_development = (
        loaded
        .y_development
    )

    y_test = (
        loaded.y_test
    )

    development_positive_count = int(
        y_development.sum()
    )

    development_negative_count = int(
        len(
            y_development
        )
        - development_positive_count
    )

    test_positive_count = int(
        y_test.sum()
    )

    test_negative_count = int(
        len(
            y_test
        )
        - test_positive_count
    )

    name_matches_registry = (
        loaded
        .actual_dataset_name
        .casefold()
        == spec
        .dataset_name
        .casefold()
    )

    sample_count_matches_registry = (
        n_samples
        == spec.expected_n_samples
    )

    predictor_count_matches_registry = (
        n_predictors
        == spec.expected_n_predictors
    )

    numeric_count_matches_registry = (
        n_numeric
        == spec.expected_n_numeric
    )

    categorical_count_matches_registry = (
        n_categorical
        == spec.expected_n_categorical
    )

    feature_type_matches_registry = (
        feature_type
        == spec.feature_type
    )

    # Dataset name is descriptive metadata.
    # Task ID is the locked identity, therefore a cosmetic name
    # difference is recorded but does not by itself invalidate
    # the benchmark task.
    audit_pass = bool(
        sample_count_matches_registry
        and predictor_count_matches_registry
        and numeric_count_matches_registry
        and categorical_count_matches_registry
        and feature_type_matches_registry
        and binary_target
        and development_has_both_classes
        and test_has_both_classes
        and split_disjoint
        and split_full_coverage
    )

    return OpenMLTaskAudit(
        task_id=(
            spec.task_id
        ),
        expected_dataset_name=(
            spec.dataset_name
        ),
        actual_dataset_name=(
            loaded
            .actual_dataset_name
        ),
        dataset_id=(
            loaded.dataset_id
        ),
        target_name=(
            loaded.target_name
        ),
        n_samples=(
            n_samples
        ),
        n_predictors=(
            n_predictors
        ),
        n_numeric_predictors=(
            n_numeric
        ),
        n_categorical_predictors=(
            n_categorical
        ),
        feature_type=(
            feature_type
        ),
        n_features_with_missing=(
            n_features_with_missing
        ),
        missing_fraction=(
            missing_fraction
        ),
        split_repeats=(
            loaded.split_repeats
        ),
        split_folds=(
            loaded.split_folds
        ),
        split_samples=(
            loaded.split_samples
        ),
        n_development=int(
            len(
                train
            )
        ),
        n_test=int(
            len(
                test
            )
        ),
        split_disjoint=(
            split_disjoint
        ),
        split_full_coverage=(
            split_full_coverage
        ),
        full_class_counts=(
            class_count_signature(
                y
            )
        ),
        development_class_counts=(
            class_count_signature(
                y_development_raw
            )
        ),
        test_class_counts=(
            class_count_signature(
                y_test_raw
            )
        ),
        positive_label=str(
            loaded
            .positive_label
        ),
        negative_label=str(
            loaded
            .negative_label
        ),
        development_positive_count=(
            development_positive_count
        ),
        development_negative_count=(
            development_negative_count
        ),
        test_positive_count=(
            test_positive_count
        ),
        test_negative_count=(
            test_negative_count
        ),
        development_positive_fraction=float(
            development_positive_count
            / len(
                y_development
            )
        ),
        test_positive_fraction=float(
            test_positive_count
            / len(
                y_test
            )
        ),
        name_matches_registry=(
            name_matches_registry
        ),
        sample_count_matches_registry=(
            sample_count_matches_registry
        ),
        predictor_count_matches_registry=(
            predictor_count_matches_registry
        ),
        numeric_count_matches_registry=(
            numeric_count_matches_registry
        ),
        categorical_count_matches_registry=(
            categorical_count_matches_registry
        ),
        feature_type_matches_registry=(
            feature_type_matches_registry
        ),
        binary_target=(
            binary_target
        ),
        development_has_both_classes=(
            development_has_both_classes
        ),
        test_has_both_classes=(
            test_has_both_classes
        ),
        audit_pass=(
            audit_pass
        ),
    )