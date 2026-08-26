import numpy as np
import pandas as pd
import pytest

from agnam.data.preprocessing import (
    MISSING_CATEGORY_CODE,
    UNKNOWN_CATEGORY_CODE,
    TabularPreprocessor,
)


def make_training_data():
    return pd.DataFrame(
        {
            "age": [20.0, 30.0, np.nan, 50.0],
            "bmi": [20.0, 25.0, 30.0, 35.0],
            "sex": ["F", "M", "F", None],
            "group": ["A", "A", "B", "B"],
        }
    )


def test_fit_transform_shapes():
    X = make_training_data()

    preprocessor = TabularPreprocessor()

    transformed = preprocessor.fit_transform(X)

    assert transformed.numeric.shape == (4, 2)
    assert transformed.numeric_missing.shape == (4, 2)
    assert transformed.categorical.shape == (4, 2)


def test_feature_identity_is_preserved():
    X = make_training_data()

    preprocessor = TabularPreprocessor()
    transformed = preprocessor.fit_transform(X)

    names = [
        spec.name
        for spec in transformed.feature_specs
    ]

    kinds = [
        spec.kind
        for spec in transformed.feature_specs
    ]

    assert names == [
        "age",
        "bmi",
        "sex",
        "group",
    ]

    assert kinds == [
        "numeric",
        "numeric",
        "categorical",
        "categorical",
    ]


def test_numeric_missingness_is_preserved():
    X = make_training_data()

    preprocessor = TabularPreprocessor()
    transformed = preprocessor.fit_transform(X)

    age_index = transformed.numeric_feature_names.index(
        "age"
    )

    assert transformed.numeric_missing[
        2,
        age_index,
    ] == 1.0

    assert transformed.numeric_missing[
        0,
        age_index,
    ] == 0.0


def test_numeric_training_statistics_are_reused():
    train = pd.DataFrame(
        {
            "x": [0.0, 2.0, 4.0],
        }
    )

    test = pd.DataFrame(
        {
            "x": [100.0],
        }
    )

    preprocessor = TabularPreprocessor()

    preprocessor.fit(train)

    transformed = preprocessor.transform(test)

    expected = (
        100.0 - preprocessor.numeric_means_["x"]
    ) / preprocessor.numeric_stds_["x"]

    assert np.isclose(
        transformed.numeric[0, 0],
        expected,
    )


def test_unknown_category_uses_reserved_code():
    train = pd.DataFrame(
        {
            "category": ["A", "B", "A"],
        }
    )

    test = pd.DataFrame(
        {
            "category": ["C"],
        }
    )

    preprocessor = TabularPreprocessor(
        numeric_features=[],
        categorical_features=["category"],
    )

    preprocessor.fit(train)

    transformed = preprocessor.transform(test)

    assert transformed.categorical[0, 0] == (
        UNKNOWN_CATEGORY_CODE
    )


def test_missing_category_uses_reserved_code():
    train = pd.DataFrame(
        {
            "category": ["A", "B", None],
        }
    )

    preprocessor = TabularPreprocessor(
        numeric_features=[],
        categorical_features=["category"],
    )

    transformed = preprocessor.fit_transform(train)

    assert transformed.categorical[
        2,
        0,
    ] == MISSING_CATEGORY_CODE


def test_unknown_and_missing_codes_are_distinct():
    train = pd.DataFrame(
        {
            "category": ["A", "B"],
        }
    )

    test = pd.DataFrame(
        {
            "category": [None, "NEW"],
        }
    )

    preprocessor = TabularPreprocessor(
        numeric_features=[],
        categorical_features=["category"],
    )

    preprocessor.fit(train)

    transformed = preprocessor.transform(test)

    assert transformed.categorical[0, 0] == (
        MISSING_CATEGORY_CODE
    )

    assert transformed.categorical[1, 0] == (
        UNKNOWN_CATEGORY_CODE
    )

    assert MISSING_CATEGORY_CODE != UNKNOWN_CATEGORY_CODE


def test_transform_before_fit_raises():
    X = make_training_data()

    preprocessor = TabularPreprocessor()

    with pytest.raises(RuntimeError):
        preprocessor.transform(X)


def test_column_mismatch_raises():
    X = make_training_data()

    preprocessor = TabularPreprocessor()
    preprocessor.fit(X)

    reordered = X[
        [
            "bmi",
            "age",
            "sex",
            "group",
        ]
    ]

    with pytest.raises(ValueError):
        preprocessor.transform(reordered)


def test_constant_numeric_feature_is_safe():
    X = pd.DataFrame(
        {
            "constant": [5.0, 5.0, 5.0],
        }
    )

    preprocessor = TabularPreprocessor()

    transformed = preprocessor.fit_transform(X)

    assert np.isfinite(
        transformed.numeric
    ).all()

    assert np.allclose(
        transformed.numeric,
        0.0,
    )


def test_explicit_feature_types():
    X = pd.DataFrame(
        {
            "coded_category": [1, 2, 1, 3],
            "continuous": [0.2, 0.7, 1.1, 0.5],
        }
    )

    preprocessor = TabularPreprocessor(
        numeric_features=["continuous"],
        categorical_features=["coded_category"],
    )

    transformed = preprocessor.fit_transform(X)

    assert transformed.numeric_feature_names == (
        "continuous",
    )

    assert transformed.categorical_feature_names == (
        "coded_category",
    )