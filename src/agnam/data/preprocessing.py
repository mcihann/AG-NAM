from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


MISSING_TOKEN = "<MISSING>"
UNKNOWN_TOKEN = "<UNK>"

MISSING_CATEGORY_CODE = 0
UNKNOWN_CATEGORY_CODE = 1
FIRST_KNOWN_CATEGORY_CODE = 2


@dataclass(frozen=True)
class FeatureSpec:
    """Metadata describing one original predictor."""

    name: str
    kind: str
    original_index: int
    transformed_index: int
    n_categories: int | None = None


@dataclass
class TransformedTabular:
    """
    Leakage-safe transformed representation.

    Numeric values and numeric missingness indicators remain associated
    with the same original numerical feature.

    Categorical predictors remain one original feature each and are
    represented by integer category codes.
    """

    numeric: np.ndarray
    numeric_missing: np.ndarray
    categorical: np.ndarray

    numeric_feature_names: tuple[str, ...]
    categorical_feature_names: tuple[str, ...]
    feature_specs: tuple[FeatureSpec, ...]

    def __len__(self) -> int:
        if self.numeric.shape[0] > 0:
            return self.numeric.shape[0]

        return self.categorical.shape[0]


class TabularPreprocessor:
    """
    Leakage-safe preprocessing for AG-NAM.

    Numerical predictors:
        - training-set median imputation
        - explicit missingness indicator
        - training-set mean/std standardization

    Categorical predictors:
        - <MISSING> -> 0
        - <UNK> -> 1
        - categories observed during fitting -> 2, 3, ...

    No statistics or category vocabulary are learned during transform().
    """

    def __init__(
        self,
        numeric_features: Sequence[str] | None = None,
        categorical_features: Sequence[str] | None = None,
    ) -> None:
        self.requested_numeric_features = (
            tuple(numeric_features)
            if numeric_features is not None
            else None
        )

        self.requested_categorical_features = (
            tuple(categorical_features)
            if categorical_features is not None
            else None
        )

        self.numeric_features_: tuple[str, ...] | None = None
        self.categorical_features_: tuple[str, ...] | None = None

        self.numeric_medians_: dict[str, float] = {}
        self.numeric_means_: dict[str, float] = {}
        self.numeric_stds_: dict[str, float] = {}

        self.category_maps_: dict[str, dict[str, int]] = {}

        self.feature_specs_: tuple[FeatureSpec, ...] | None = None
        self.columns_: tuple[str, ...] | None = None

        self.is_fitted_: bool = False

    def _validate_dataframe(self, X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        if X.columns.duplicated().any():
            duplicated = X.columns[X.columns.duplicated()].tolist()
            raise ValueError(
                f"Duplicate feature names are not allowed: {duplicated}"
            )

    def _infer_feature_types(
        self,
        X: pd.DataFrame,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        if (
            self.requested_numeric_features is not None
            or self.requested_categorical_features is not None
        ):
            numeric = self.requested_numeric_features or ()
            categorical = self.requested_categorical_features or ()

            assigned = set(numeric) | set(categorical)
            all_columns = set(X.columns)

            missing_assignments = all_columns - assigned
            unknown_assignments = assigned - all_columns

            if missing_assignments:
                raise ValueError(
                    "Every column must be assigned when explicit feature "
                    f"types are supplied. Unassigned: "
                    f"{sorted(missing_assignments)}"
                )

            if unknown_assignments:
                raise ValueError(
                    f"Unknown feature names: {sorted(unknown_assignments)}"
                )

            overlap = set(numeric) & set(categorical)

            if overlap:
                raise ValueError(
                    f"Features cannot be both numeric and categorical: "
                    f"{sorted(overlap)}"
                )

            return tuple(numeric), tuple(categorical)

        numeric: list[str] = []
        categorical: list[str] = []

        for column in X.columns:
            series = X[column]

            if pd.api.types.is_bool_dtype(series):
                categorical.append(column)

            elif pd.api.types.is_numeric_dtype(series):
                numeric.append(column)

            else:
                categorical.append(column)

        return tuple(numeric), tuple(categorical)

    @staticmethod
    def _categorical_to_string(series: pd.Series) -> pd.Series:
        result = series.astype("object").copy()

        missing_mask = result.isna()

        result = result.astype(str)
        result.loc[missing_mask] = MISSING_TOKEN

        return result

    def fit(
        self,
        X: pd.DataFrame,
    ) -> "TabularPreprocessor":
        self._validate_dataframe(X)

        self.columns_ = tuple(X.columns)

        numeric_features, categorical_features = (
            self._infer_feature_types(X)
        )

        self.numeric_features_ = numeric_features
        self.categorical_features_ = categorical_features

        for feature in numeric_features:
            values = pd.to_numeric(
                X[feature],
                errors="coerce",
            )

            median = float(values.median())

            if not np.isfinite(median):
                raise ValueError(
                    f"Numeric feature '{feature}' contains no finite "
                    "training values."
                )

            imputed = values.fillna(median).to_numpy(
                dtype=np.float64
            )

            mean = float(np.mean(imputed))
            std = float(np.std(imputed, ddof=0))

            if not np.isfinite(std) or std < 1e-12:
                std = 1.0

            self.numeric_medians_[feature] = median
            self.numeric_means_[feature] = mean
            self.numeric_stds_[feature] = std

        for feature in categorical_features:
            values = self._categorical_to_string(X[feature])

            observed = sorted(
                value
                for value in values.unique().tolist()
                if value != MISSING_TOKEN
            )

            mapping = {
                value: index
                for index, value in enumerate(
                    observed,
                    start=FIRST_KNOWN_CATEGORY_CODE,
                )
            }

            self.category_maps_[feature] = mapping

        specs: list[FeatureSpec] = []

        numeric_output_index = 0
        categorical_output_index = 0

        for original_index, feature in enumerate(X.columns):
            if feature in numeric_features:
                specs.append(
                    FeatureSpec(
                        name=feature,
                        kind="numeric",
                        original_index=original_index,
                        transformed_index=numeric_output_index,
                        n_categories=None,
                    )
                )

                numeric_output_index += 1

            else:
                mapping = self.category_maps_[feature]

                # +2 accounts for <MISSING> and <UNK>.
                n_categories = len(mapping) + 2

                specs.append(
                    FeatureSpec(
                        name=feature,
                        kind="categorical",
                        original_index=original_index,
                        transformed_index=categorical_output_index,
                        n_categories=n_categories,
                    )
                )

                categorical_output_index += 1

        self.feature_specs_ = tuple(specs)
        self.is_fitted_ = True

        return self

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError(
                "TabularPreprocessor must be fitted before transform()."
            )

    def _validate_transform_columns(
        self,
        X: pd.DataFrame,
    ) -> None:
        assert self.columns_ is not None

        if tuple(X.columns) != self.columns_:
            raise ValueError(
                "Transform columns must exactly match the fitted "
                "training columns and their order."
            )

    def transform(
        self,
        X: pd.DataFrame,
    ) -> TransformedTabular:
        self._check_fitted()
        self._validate_dataframe(X)
        self._validate_transform_columns(X)

        assert self.numeric_features_ is not None
        assert self.categorical_features_ is not None
        assert self.feature_specs_ is not None

        n = len(X)

        numeric = np.zeros(
            (n, len(self.numeric_features_)),
            dtype=np.float32,
        )

        numeric_missing = np.zeros(
            (n, len(self.numeric_features_)),
            dtype=np.float32,
        )

        categorical = np.zeros(
            (n, len(self.categorical_features_)),
            dtype=np.int64,
        )

        for output_index, feature in enumerate(
            self.numeric_features_
        ):
            values = pd.to_numeric(
                X[feature],
                errors="coerce",
            )

            missing = values.isna().to_numpy()

            median = self.numeric_medians_[feature]
            mean = self.numeric_means_[feature]
            std = self.numeric_stds_[feature]

            imputed = values.fillna(median).to_numpy(
                dtype=np.float64
            )

            standardized = (imputed - mean) / std

            numeric[:, output_index] = standardized.astype(
                np.float32
            )

            numeric_missing[:, output_index] = missing.astype(
                np.float32
            )

        for output_index, feature in enumerate(
            self.categorical_features_
        ):
            values = self._categorical_to_string(X[feature])

            mapping = self.category_maps_[feature]

            codes = np.empty(n, dtype=np.int64)

            for row_index, value in enumerate(values.tolist()):
                if value == MISSING_TOKEN:
                    code = MISSING_CATEGORY_CODE

                else:
                    code = mapping.get(
                        value,
                        UNKNOWN_CATEGORY_CODE,
                    )

                codes[row_index] = code

            categorical[:, output_index] = codes

        return TransformedTabular(
            numeric=numeric,
            numeric_missing=numeric_missing,
            categorical=categorical,
            numeric_feature_names=self.numeric_features_,
            categorical_feature_names=self.categorical_features_,
            feature_specs=self.feature_specs_,
        )

    def fit_transform(
        self,
        X: pd.DataFrame,
    ) -> TransformedTabular:
        return self.fit(X).transform(X)