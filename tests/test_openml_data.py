import numpy as np
import pandas as pd
import pytest

from agnam.benchmarking.openml_data import (
    LoadedOpenMLTask,
    audit_loaded_openml_task,
    class_count_signature,
    determine_binary_labels,
    encode_binary_target,
    feature_type_from_indicator,
    observed_class_counts,
    validate_split_indices,
)
from agnam.benchmarking.openml_protocol import (
    OpenMLTaskSpec,
)


def test_minority_class_becomes_positive():
    y = pd.Series(
        [
            "major",
            "major",
            "major",
            "minor",
        ]
    )

    positive, negative = (
        determine_binary_labels(
            y
        )
    )

    assert positive == "minor"
    assert negative == "major"


def test_tied_classes_use_lexicographic_rule():
    y = pd.Series(
        [
            "B",
            "A",
            "B",
            "A",
        ]
    )

    positive, negative = (
        determine_binary_labels(
            y
        )
    )

    assert positive == "A"
    assert negative == "B"


def test_binary_encoding_uses_locked_positive_label():
    y = pd.Series(
        [
            "no",
            "yes",
            "no",
            "yes",
        ]
    )

    encoded = (
        encode_binary_target(
            y,
            positive_label="yes",
        )
    )

    np.testing.assert_array_equal(
        encoded,
        np.array(
            [
                0,
                1,
                0,
                1,
            ],
            dtype=np.int64,
        ),
    )


def test_missing_target_is_rejected():
    y = pd.Series(
        [
            "A",
            None,
            "B",
        ]
    )

    with pytest.raises(
        ValueError
    ):
        observed_class_counts(
            y
        )


def test_split_validation_accepts_complete_disjoint_split():
    train, test = (
        validate_split_indices(
            n_samples=6,
            train_indices=[
                0,
                1,
                2,
                3,
            ],
            test_indices=[
                4,
                5,
            ],
        )
    )

    np.testing.assert_array_equal(
        train,
        np.array(
            [
                0,
                1,
                2,
                3,
            ]
        ),
    )

    np.testing.assert_array_equal(
        test,
        np.array(
            [
                4,
                5,
            ]
        ),
    )


def test_split_overlap_is_rejected():
    with pytest.raises(
        ValueError
    ):
        validate_split_indices(
            n_samples=5,
            train_indices=[
                0,
                1,
                2,
            ],
            test_indices=[
                2,
                3,
                4,
            ],
        )


def test_feature_type_from_indicator():
    assert (
        feature_type_from_indicator(
            [
                False,
                False,
                False,
            ]
        )
        == "numeric"
    )

    assert (
        feature_type_from_indicator(
            [
                True,
                True,
            ]
        )
        == "categorical"
    )

    assert (
        feature_type_from_indicator(
            [
                False,
                True,
            ]
        )
        == "mixed"
    )


def test_fake_loaded_task_passes_audit():
    spec = OpenMLTaskSpec(
        task_id=123,
        dataset_name="toy",
        feature_type="mixed",
        expected_n_samples=6,
        expected_n_predictors=2,
        expected_n_numeric=1,
        expected_n_categorical=1,
    )

    X = pd.DataFrame(
        {
            "numeric": [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
            ],
            "category": [
                "A",
                "B",
                "A",
                "B",
                "A",
                "B",
            ],
        }
    )

    y_raw = pd.Series(
        [
            "negative",
            "negative",
            "positive",
            "negative",
            "positive",
            "negative",
        ]
    )

    train_indices = np.array(
        [
            0,
            1,
            2,
            3,
        ]
    )

    test_indices = np.array(
        [
            4,
            5,
        ]
    )

    positive_label, negative_label = (
        determine_binary_labels(
            y_raw.iloc[
                train_indices
            ]
        )
    )

    loaded = LoadedOpenMLTask(
        task_spec=spec,
        dataset_id=999,
        actual_dataset_name="toy",
        target_name="class",
        X=X,
        y_raw=y_raw,
        categorical_indicator=(
            False,
            True,
        ),
        attribute_names=(
            "numeric",
            "category",
        ),
        train_indices=(
            train_indices
        ),
        test_indices=(
            test_indices
        ),
        split_repeats=1,
        split_folds=10,
        split_samples=1,
        positive_label=(
            positive_label
        ),
        negative_label=(
            negative_label
        ),
        y_binary=(
            encode_binary_target(
                y_raw,
                positive_label=(
                    positive_label
                ),
            )
        ),
    )

    audit = (
        audit_loaded_openml_task(
            loaded
        )
    )

    assert audit.audit_pass

    assert (
        audit.feature_type
        == "mixed"
    )

    assert (
        audit.n_numeric_predictors
        == 1
    )

    assert (
        audit.n_categorical_predictors
        == 1
    )

    assert (
        audit.n_development
        == 4
    )

    assert (
        audit.n_test
        == 2
    )

    assert (
        audit.development_has_both_classes
    )

    assert (
        audit.test_has_both_classes
    )

    signature = (
        class_count_signature(
            y_raw
        )
    )

    assert (
        '"negative"'
        in signature
    )

    assert (
        '"positive"'
        in signature
    )