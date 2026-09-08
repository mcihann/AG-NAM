from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from agnam.benchmarking.realx_protocol import (
    LOCKED_REALX_STRENGTHS,
    RealXProtocol,
    RealXRunSpec,
)


# =============================================================================
# DATA CONTAINERS
# =============================================================================


@dataclass
class RealXStateEncoding:
    state_codes: pd.DataFrame
    state_labels: dict[str, tuple[str, ...]]
    state_counts: dict[str, int]
    feature_kinds: dict[str, str]


@dataclass
class RealXBlueprint:
    """
    Strength-independent latent Real-X construction.

    All four interaction-strength conditions within one
    dataset/realization recreate this blueprint exactly because
    their seed family is shared.
    """

    sampled_X: pd.DataFrame
    source_row_positions: np.ndarray

    state_codes: pd.DataFrame
    state_labels: dict[str, tuple[str, ...]]
    state_counts: dict[str, int]
    feature_kinds: dict[str, str]

    eligible_features: tuple[str, ...]

    main_features: tuple[str, ...]
    true_pairs: tuple[
        tuple[str, str],
        ...,
    ]

    main_effect_tables: dict[
        str,
        np.ndarray,
    ]

    interaction_surfaces: dict[
        tuple[str, str],
        np.ndarray,
    ]

    interaction_marginal_errors: dict[
        tuple[str, str],
        float,
    ]

    main_composite: np.ndarray
    interaction_composite: np.ndarray

    uniform_draws: np.ndarray

    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray


@dataclass
class RealXGeneratedRun:
    run_spec: RealXRunSpec

    X: pd.DataFrame
    source_row_positions: np.ndarray

    y: np.ndarray
    probabilities: np.ndarray
    logits: np.ndarray

    intercept: float

    main_composite: np.ndarray
    interaction_composite: np.ndarray
    interaction_signal: np.ndarray

    eligible_features: tuple[str, ...]
    main_features: tuple[str, ...]

    template_true_pairs: tuple[
        tuple[str, str],
        ...,
    ]

    active_true_pairs: tuple[
        tuple[str, str],
        ...,
    ]

    state_codes: pd.DataFrame
    state_labels: dict[str, tuple[str, ...]]

    interaction_marginal_errors: dict[
        tuple[str, str],
        float,
    ]

    uniform_draws: np.ndarray

    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray

    expected_prevalence: float
    realized_prevalence: float


# =============================================================================
# BASIC NUMERICAL HELPERS
# =============================================================================


def _sigmoid(
    values: np.ndarray,
) -> np.ndarray:
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    output = np.empty_like(
        values
    )

    positive = (
        values
        >= 0.0
    )

    output[
        positive
    ] = (
        1.0
        /
        (
            1.0
            + np.exp(
                -values[
                    positive
                ]
            )
        )
    )

    negative_values = (
        values[
            ~positive
        ]
    )

    exp_values = np.exp(
        negative_values
    )

    output[
        ~positive
    ] = (
        exp_values
        /
        (
            1.0
            + exp_values
        )
    )

    return output


def _standardize_vector(
    values: np.ndarray,
    *,
    name: str,
    tolerance: float = 1e-12,
) -> np.ndarray:
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    centered = (
        values
        - float(
            np.mean(
                values
            )
        )
    )

    standard_deviation = float(
        np.std(
            centered,
            ddof=0,
        )
    )

    if (
        not np.isfinite(
            standard_deviation
        )
        or standard_deviation
        <= tolerance
    ):
        raise RuntimeError(
            f"{name} has degenerate "
            "observed variance."
        )

    return (
        centered
        / standard_deviation
    )


# =============================================================================
# ROW SAMPLING
# =============================================================================


def sample_realx_rows(
    X: pd.DataFrame,
    *,
    max_rows: int,
    seed: int,
) -> tuple[
    pd.DataFrame,
    np.ndarray,
]:
    """
    Deterministically retain all rows when n <= max_rows.

    Otherwise select max_rows real observations without replacement.
    Selected source positions are sorted so original row ordering is
    preserved within the sampled dataset.
    """
    if not isinstance(
        X,
        pd.DataFrame,
    ):
        raise TypeError(
            "X must be a pandas DataFrame."
        )

    if len(
        X
    ) == 0:
        raise ValueError(
            "X must contain at least one row."
        )

    if (
        X.columns.duplicated()
        .any()
    ):
        raise ValueError(
            "X must have unique column names."
        )

    if (
        max_rows
        < 1
    ):
        raise ValueError(
            "max_rows must be positive."
        )

    n_rows = len(
        X
    )

    if (
        n_rows
        <= max_rows
    ):
        positions = np.arange(
            n_rows,
            dtype=np.int64,
        )

    else:
        rng = np.random.default_rng(
            seed
        )

        positions = np.sort(
            rng.choice(
                n_rows,
                size=max_rows,
                replace=False,
            )
        ).astype(
            np.int64
        )

    sampled = (
        X.iloc[
            positions
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    return (
        sampled,
        positions,
    )


# =============================================================================
# STATE ENCODING
# =============================================================================


def _encode_numeric_feature(
    series: pd.Series,
    *,
    max_states: int,
) -> tuple[
    np.ndarray,
    tuple[str, ...],
]:
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    missing_mask = (
        numeric.isna()
        .to_numpy()
    )

    nonmissing = numeric[
        ~numeric.isna()
    ]

    if (
        nonmissing.nunique(
            dropna=True
        )
        < 2
    ):
        nonmissing_codes = np.zeros(
            len(
                nonmissing
            ),
            dtype=np.int64,
        )

        n_numeric_states = 1

    else:
        missing_state_cost = (
            1
            if missing_mask.any()
            else 0
        )

        available_states = (
            max_states
            - missing_state_cost
        )

        if (
            available_states
            < 2
        ):
            raise ValueError(
                "max_states is too small "
                "for numeric encoding with "
                "missing values."
            )

        q = min(
            int(
                nonmissing.nunique(
                    dropna=True
                )
            ),
            available_states,
        )

        quantile_codes = pd.qcut(
            nonmissing,
            q=q,
            labels=False,
            duplicates="drop",
        )

        nonmissing_codes = (
            pd.to_numeric(
                quantile_codes,
                errors="raise",
            )
            .astype(
                int
            )
            .to_numpy()
        )

        n_numeric_states = (
            int(
                np.max(
                    nonmissing_codes
                )
            )
            + 1
        )

    output = np.empty(
        len(
            series
        ),
        dtype=np.int64,
    )

    output[
        ~missing_mask
    ] = (
        nonmissing_codes
    )

    labels = [
        f"Q{index + 1}"
        for index in range(
            n_numeric_states
        )
    ]

    if missing_mask.any():
        missing_code = (
            n_numeric_states
        )

        output[
            missing_mask
        ] = (
            missing_code
        )

        labels.append(
            "MISSING"
        )

    return (
        output,
        tuple(
            labels
        ),
    )


def _encode_categorical_feature(
    series: pd.Series,
    *,
    max_states: int,
) -> tuple[
    np.ndarray,
    tuple[str, ...],
]:
    missing_mask = (
        series.isna()
        .to_numpy()
    )

    nonmissing = (
        series[
            ~series.isna()
        ]
        .astype(str)
    )

    unique_values = sorted(
        nonmissing.unique()
        .tolist()
    )

    missing_state_cost = (
        1
        if missing_mask.any()
        else 0
    )

    available_states = (
        max_states
        - missing_state_cost
    )

    if (
        available_states
        < 1
    ):
        raise ValueError(
            "max_states is too small "
            "for categorical encoding."
        )

    use_other = (
        len(
            unique_values
        )
        > available_states
    )

    if use_other:
        if (
            available_states
            < 2
        ):
            raise ValueError(
                "At least two non-missing "
                "state slots are required "
                "when OTHER is needed."
            )

        counts = (
            nonmissing
            .value_counts()
            .to_dict()
        )

        ordered = sorted(
            counts.items(),
            key=lambda item: (
                -int(
                    item[
                        1
                    ]
                ),
                str(
                    item[
                        0
                    ]
                ),
            ),
        )

        n_kept = (
            available_states
            - 1
        )

        kept_values = [
            str(
                item[
                    0
                ]
            )
            for item in (
                ordered[
                    :n_kept
                ]
            )
        ]

        labels = (
            kept_values
            + [
                "OTHER"
            ]
        )

    else:
        kept_values = (
            unique_values
        )

        labels = list(
            kept_values
        )

    mapping = {
        value: index
        for (
            index,
            value,
        ) in enumerate(
            kept_values
        )
    }

    other_code = (
        labels.index(
            "OTHER"
        )
        if (
            "OTHER"
            in labels
        )
        else None
    )

    output = np.empty(
        len(
            series
        ),
        dtype=np.int64,
    )

    values = series.astype(
        object
    )

    for index in range(
        len(
            series
        )
    ):
        if missing_mask[
            index
        ]:
            continue

        value = str(
            values.iloc[
                index
            ]
        )

        if (
            value
            in mapping
        ):
            output[
                index
            ] = (
                mapping[
                    value
                ]
            )

        else:
            if (
                other_code
                is None
            ):
                raise RuntimeError(
                    "Categorical value was "
                    "not encoded and OTHER "
                    "state is unavailable."
                )

            output[
                index
            ] = (
                other_code
            )

    if missing_mask.any():
        missing_code = len(
            labels
        )

        output[
            missing_mask
        ] = (
            missing_code
        )

        labels.append(
            "MISSING"
        )

    return (
        output,
        tuple(
            labels
        ),
    )


def encode_realx_states(
    X: pd.DataFrame,
    *,
    max_states_per_feature: int,
) -> RealXStateEncoding:
    """
    Construct generation-only finite state representations.

    Numeric:
        empirical quantile states

    Categorical:
        observed levels with deterministic top-frequency retention
        and optional OTHER state

    Missing:
        explicit MISSING state
    """
    if (
        max_states_per_feature
        < 2
    ):
        raise ValueError(
            "max_states_per_feature must "
            "be at least 2."
        )

    columns: dict[
        str,
        np.ndarray,
    ] = {}

    state_labels: dict[
        str,
        tuple[str, ...],
    ] = {}

    state_counts: dict[
        str,
        int,
    ] = {}

    feature_kinds: dict[
        str,
        str,
    ] = {}

    for column in (
        X.columns
    ):
        feature_name = str(
            column
        )

        series = X[
            column
        ]

        is_numeric = (
            pd.api.types
            .is_numeric_dtype(
                series
            )
            and not pd.api.types
            .is_bool_dtype(
                series
            )
        )

        if is_numeric:
            (
                codes,
                labels,
            ) = _encode_numeric_feature(
                series,
                max_states=(
                    max_states_per_feature
                ),
            )

            feature_kind = (
                "numeric"
            )

        else:
            (
                codes,
                labels,
            ) = _encode_categorical_feature(
                series,
                max_states=(
                    max_states_per_feature
                ),
            )

            feature_kind = (
                "categorical"
            )

        columns[
            feature_name
        ] = (
            codes
        )

        state_labels[
            feature_name
        ] = (
            labels
        )

        state_counts[
            feature_name
        ] = len(
            labels
        )

        feature_kinds[
            feature_name
        ] = (
            feature_kind
        )

    state_codes = pd.DataFrame(
        columns,
        index=np.arange(
            len(
                X
            )
        ),
    )

    return RealXStateEncoding(
        state_codes=(
            state_codes
        ),
        state_labels=(
            state_labels
        ),
        state_counts=(
            state_counts
        ),
        feature_kinds=(
            feature_kinds
        ),
    )


# =============================================================================
# TRUTH FEATURE SELECTION
# =============================================================================


def choose_realx_truth_structure(
    encoding: RealXStateEncoding,
    *,
    feature_seed: int,
    n_main_effects: int,
    n_true_interactions: int,
    min_eligible_features: int,
) -> tuple[
    tuple[str, ...],
    tuple[
        tuple[str, str],
        ...,
    ],
    tuple[str, ...],
]:
    eligible = tuple(
        feature
        for feature in (
            encoding
            .state_codes
            .columns
            .tolist()
        )
        if (
            encoding
            .state_counts[
                feature
            ]
            >= 2
        )
    )

    required = (
        n_main_effects
        + 2
        * n_true_interactions
    )

    if (
        len(
            eligible
        )
        < min_eligible_features
        or len(
            eligible
        )
        < required
    ):
        raise RuntimeError(
            "Insufficient eligible Real-X "
            "features. "
            f"Found {len(eligible)}, "
            f"required at least "
            f"{max(min_eligible_features, required)}."
        )

    rng = np.random.default_rng(
        feature_seed
    )

    chosen = rng.choice(
        np.asarray(
            eligible,
            dtype=object,
        ),
        size=required,
        replace=False,
    ).tolist()

    main_features = tuple(
        str(
            value
        )
        for value in (
            chosen[
                :n_main_effects
            ]
        )
    )

    pair_features = [
        str(
            value
        )
        for value in (
            chosen[
                n_main_effects:
            ]
        )
    ]

    true_pairs = tuple(
        (
            pair_features[
                2 * index
            ],
            pair_features[
                2 * index
                + 1
            ],
        )
        for index in range(
            n_true_interactions
        )
    )

    return (
        main_features,
        true_pairs,
        eligible,
    )


# =============================================================================
# MAIN-EFFECT SURFACES
# =============================================================================


def _build_main_effect_table(
    codes: np.ndarray,
    *,
    n_states: int,
    rng: np.random.Generator,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    raw = rng.normal(
        loc=0.0,
        scale=1.0,
        size=n_states,
    )

    weights = np.bincount(
        codes,
        minlength=n_states,
    ).astype(
        np.float64
    )

    total_weight = float(
        np.sum(
            weights
        )
    )

    weighted_mean = float(
        np.sum(
            weights
            * raw
        )
        / total_weight
    )

    table = (
        raw
        - weighted_mean
    )

    observed = table[
        codes
    ]

    observed_std = float(
        np.std(
            observed,
            ddof=0,
        )
    )

    if (
        not np.isfinite(
            observed_std
        )
        or observed_std
        <= 1e-12
    ):
        raise RuntimeError(
            "Generated main-effect "
            "surface is degenerate."
        )

    table = (
        table
        / observed_std
    )

    observed = table[
        codes
    ]

    return (
        table,
        observed,
    )


# =============================================================================
# FUNCTIONAL-ANOVA INTERACTION PURIFICATION
# =============================================================================


def _joint_count_matrix(
    codes_a: np.ndarray,
    codes_b: np.ndarray,
    *,
    n_states_a: int,
    n_states_b: int,
) -> np.ndarray:
    counts = np.zeros(
        (
            n_states_a,
            n_states_b,
        ),
        dtype=np.float64,
    )

    np.add.at(
        counts,
        (
            codes_a,
            codes_b,
        ),
        1.0,
    )

    return counts


def interaction_marginal_error(
    surface: np.ndarray,
    joint_weights: np.ndarray,
) -> float:
    """
    Maximum absolute empirical weighted row/column marginal mean.
    """
    surface = np.asarray(
        surface,
        dtype=np.float64,
    )

    weights = np.asarray(
        joint_weights,
        dtype=np.float64,
    )

    row_mass = np.sum(
        weights,
        axis=1,
    )

    column_mass = np.sum(
        weights,
        axis=0,
    )

    row_error = 0.0
    column_error = 0.0

    valid_rows = (
        row_mass
        > 0.0
    )

    valid_columns = (
        column_mass
        > 0.0
    )

    if valid_rows.any():
        row_means = (
            np.sum(
                weights
                * surface,
                axis=1,
            )[
                valid_rows
            ]
            / row_mass[
                valid_rows
            ]
        )

        row_error = float(
            np.max(
                np.abs(
                    row_means
                )
            )
        )

    if valid_columns.any():
        column_means = (
            np.sum(
                weights
                * surface,
                axis=0,
            )[
                valid_columns
            ]
            / column_mass[
                valid_columns
            ]
        )

        column_error = float(
            np.max(
                np.abs(
                    column_means
                )
            )
        )

    return max(
        row_error,
        column_error,
    )


def purify_two_way_surface(
    raw_surface: np.ndarray,
    joint_weights: np.ndarray,
    *,
    tolerance: float,
    max_iterations: int,
) -> tuple[
    np.ndarray,
    float,
    int,
]:
    """
    Alternating empirical weighted row/column projections.

    At convergence, the retained pair surface has approximately zero
    empirical conditional marginal means along both axes.
    """
    surface = np.asarray(
        raw_surface,
        dtype=np.float64,
    ).copy()

    weights = np.asarray(
        joint_weights,
        dtype=np.float64,
    )

    if (
        surface.shape
        != weights.shape
    ):
        raise ValueError(
            "Surface and joint-weight "
            "shapes must match."
        )

    if (
        np.sum(
            weights
        )
        <= 0.0
    ):
        raise ValueError(
            "Joint weights must contain "
            "positive total mass."
        )

    row_mass = np.sum(
        weights,
        axis=1,
    )

    column_mass = np.sum(
        weights,
        axis=0,
    )

    valid_rows = (
        row_mass
        > 0.0
    )

    valid_columns = (
        column_mass
        > 0.0
    )

    for iteration in range(
        1,
        max_iterations
        + 1,
    ):
        row_means = np.zeros(
            surface.shape[
                0
            ],
            dtype=np.float64,
        )

        row_means[
            valid_rows
        ] = (
            np.sum(
                weights
                * surface,
                axis=1,
            )[
                valid_rows
            ]
            / row_mass[
                valid_rows
            ]
        )

        surface = (
            surface
            - row_means[
                :,
                None,
            ]
        )

        column_means = np.zeros(
            surface.shape[
                1
            ],
            dtype=np.float64,
        )

        column_means[
            valid_columns
        ] = (
            np.sum(
                weights
                * surface,
                axis=0,
            )[
                valid_columns
            ]
            / column_mass[
                valid_columns
            ]
        )

        surface = (
            surface
            - column_means[
                None,
                :,
            ]
        )

        error = (
            interaction_marginal_error(
                surface,
                weights,
            )
        )

        if (
            error
            <= tolerance
        ):
            return (
                surface,
                error,
                iteration,
            )

    error = interaction_marginal_error(
        surface,
        weights,
    )

    raise RuntimeError(
        "Interaction purification did "
        "not converge. "
        f"Final marginal error: {error:.3e}"
    )


def _build_interaction_surface(
    codes_a: np.ndarray,
    codes_b: np.ndarray,
    *,
    n_states_a: int,
    n_states_b: int,
    rng: np.random.Generator,
    tolerance: float,
    max_iterations: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    float,
]:
    raw = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(
            n_states_a,
            n_states_b,
        ),
    )

    joint_counts = (
        _joint_count_matrix(
            codes_a,
            codes_b,
            n_states_a=(
                n_states_a
            ),
            n_states_b=(
                n_states_b
            ),
        )
    )

    (
        purified,
        marginal_error,
        _,
    ) = purify_two_way_surface(
        raw,
        joint_counts,
        tolerance=(
            tolerance
        ),
        max_iterations=(
            max_iterations
        ),
    )

    observed = purified[
        codes_a,
        codes_b,
    ]

    observed_std = float(
        np.std(
            observed,
            ddof=0,
        )
    )

    if (
        not np.isfinite(
            observed_std
        )
        or observed_std
        <= 1e-12
    ):
        raise RuntimeError(
            "Purified interaction "
            "surface is degenerate on "
            "the observed real-X support."
        )

    purified = (
        purified
        / observed_std
    )

    observed = purified[
        codes_a,
        codes_b,
    ]

    scaled_error = (
        interaction_marginal_error(
            purified,
            joint_counts,
        )
    )

    return (
        purified,
        observed,
        scaled_error,
    )


# =============================================================================
# SPLIT + PREVALENCE CALIBRATION
# =============================================================================


def build_realx_split_indices(
    n_rows: int,
    *,
    split_seed: int,
    train_fraction: float,
    validation_fraction: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    if (
        n_rows
        < 5
    ):
        raise ValueError(
            "At least five rows are "
            "required."
        )

    rng = np.random.default_rng(
        split_seed
    )

    permutation = rng.permutation(
        n_rows
    )

    n_train = int(
        np.floor(
            train_fraction
            * n_rows
        )
    )

    n_validation = int(
        np.floor(
            validation_fraction
            * n_rows
        )
    )

    n_test = (
        n_rows
        - n_train
        - n_validation
    )

    if min(
        n_train,
        n_validation,
        n_test,
    ) < 1:
        raise RuntimeError(
            "Real-X split produced an "
            "empty partition."
        )

    train_indices = np.sort(
        permutation[
            :n_train
        ]
    ).astype(
        np.int64
    )

    validation_indices = np.sort(
        permutation[
            n_train:
            n_train
            + n_validation
        ]
    ).astype(
        np.int64
    )

    test_indices = np.sort(
        permutation[
            n_train
            + n_validation:
        ]
    ).astype(
        np.int64
    )

    return (
        train_indices,
        validation_indices,
        test_indices,
    )


def calibrate_logistic_intercept(
    linear_predictor_without_intercept: np.ndarray,
    *,
    target_expected_prevalence: float,
    tolerance: float = 1e-12,
    max_iterations: int = 200,
) -> float:
    scores = np.asarray(
        linear_predictor_without_intercept,
        dtype=np.float64,
    )

    if not (
        0.0
        < target_expected_prevalence
        < 1.0
    ):
        raise ValueError(
            "Target prevalence must lie "
            "strictly in (0, 1)."
        )

    lower = -40.0
    upper = 40.0

    for _ in range(
        max_iterations
    ):
        midpoint = (
            lower
            + upper
        ) / 2.0

        prevalence = float(
            np.mean(
                _sigmoid(
                    midpoint
                    + scores
                )
            )
        )

        if abs(
            prevalence
            - target_expected_prevalence
        ) <= tolerance:
            return float(
                midpoint
            )

        if (
            prevalence
            < target_expected_prevalence
        ):
            lower = (
                midpoint
            )

        else:
            upper = (
                midpoint
            )

    return float(
        (
            lower
            + upper
        )
        / 2.0
    )


# =============================================================================
# BLUEPRINT
# =============================================================================


def build_realx_blueprint(
    X: pd.DataFrame,
    run_spec: RealXRunSpec,
    *,
    protocol: RealXProtocol | None = None,
) -> RealXBlueprint:
    if protocol is None:
        protocol = (
            RealXProtocol()
        )

    (
        sampled_X,
        source_positions,
    ) = sample_realx_rows(
        X,
        max_rows=(
            protocol.max_rows
        ),
        seed=(
            run_spec
            .row_sample_seed
        ),
    )

    encoding = (
        encode_realx_states(
            sampled_X,
            max_states_per_feature=(
                protocol
                .max_states_per_feature
            ),
        )
    )

    (
        main_features,
        true_pairs,
        eligible_features,
    ) = choose_realx_truth_structure(
        encoding,
        feature_seed=(
            run_spec.feature_seed
        ),
        n_main_effects=(
            protocol.n_main_effects
        ),
        n_true_interactions=(
            protocol
            .n_true_interactions
        ),
        min_eligible_features=(
            protocol
            .min_eligible_features
        ),
    )

    surface_rng = np.random.default_rng(
        run_spec.surface_seed
    )

    main_effect_tables = {}

    main_contributions = []

    for feature in (
        main_features
    ):
        codes = (
            encoding
            .state_codes[
                feature
            ]
            .to_numpy(
                dtype=np.int64
            )
        )

        (
            table,
            contribution,
        ) = _build_main_effect_table(
            codes,
            n_states=(
                encoding
                .state_counts[
                    feature
                ]
            ),
            rng=(
                surface_rng
            ),
        )

        main_effect_tables[
            feature
        ] = (
            table
        )

        main_contributions.append(
            contribution
        )

    raw_main_composite = np.sum(
        np.vstack(
            main_contributions
        ),
        axis=0,
    )

    main_composite = (
        _standardize_vector(
            raw_main_composite,
            name=(
                "main-effect composite"
            ),
        )
    )

    interaction_surfaces = {}

    interaction_marginal_errors = {}

    pair_contributions = []

    for (
        feature_a,
        feature_b,
    ) in true_pairs:
        codes_a = (
            encoding
            .state_codes[
                feature_a
            ]
            .to_numpy(
                dtype=np.int64
            )
        )

        codes_b = (
            encoding
            .state_codes[
                feature_b
            ]
            .to_numpy(
                dtype=np.int64
            )
        )

        (
            surface,
            contribution,
            marginal_error,
        ) = _build_interaction_surface(
            codes_a,
            codes_b,
            n_states_a=(
                encoding
                .state_counts[
                    feature_a
                ]
            ),
            n_states_b=(
                encoding
                .state_counts[
                    feature_b
                ]
            ),
            rng=(
                surface_rng
            ),
            tolerance=(
                protocol
                .purification_tolerance
            ),
            max_iterations=(
                protocol
                .purification_max_iterations
            ),
        )

        pair = (
            feature_a,
            feature_b,
        )

        interaction_surfaces[
            pair
        ] = (
            surface
        )

        interaction_marginal_errors[
            pair
        ] = (
            marginal_error
        )

        pair_contributions.append(
            contribution
        )

    raw_interaction_composite = (
        np.sum(
            np.vstack(
                pair_contributions
            ),
            axis=0,
        )
    )

    interaction_composite = (
        _standardize_vector(
            raw_interaction_composite,
            name=(
                "interaction composite"
            ),
        )
    )

    uniform_rng = np.random.default_rng(
        run_spec.label_seed
    )

    uniform_draws = uniform_rng.random(
        len(
            sampled_X
        )
    )

    (
        train_indices,
        validation_indices,
        test_indices,
    ) = build_realx_split_indices(
        len(
            sampled_X
        ),
        split_seed=(
            run_spec.split_seed
        ),
        train_fraction=(
            protocol.train_fraction
        ),
        validation_fraction=(
            protocol
            .validation_fraction
        ),
    )

    return RealXBlueprint(
        sampled_X=(
            sampled_X
        ),
        source_row_positions=(
            source_positions
        ),
        state_codes=(
            encoding.state_codes
        ),
        state_labels=(
            encoding.state_labels
        ),
        state_counts=(
            encoding.state_counts
        ),
        feature_kinds=(
            encoding.feature_kinds
        ),
        eligible_features=(
            eligible_features
        ),
        main_features=(
            main_features
        ),
        true_pairs=(
            true_pairs
        ),
        main_effect_tables=(
            main_effect_tables
        ),
        interaction_surfaces=(
            interaction_surfaces
        ),
        interaction_marginal_errors=(
            interaction_marginal_errors
        ),
        main_composite=(
            main_composite
        ),
        interaction_composite=(
            interaction_composite
        ),
        uniform_draws=(
            uniform_draws
        ),
        train_indices=(
            train_indices
        ),
        validation_indices=(
            validation_indices
        ),
        test_indices=(
            test_indices
        ),
    )


# =============================================================================
# RUN GENERATION
# =============================================================================


def _locked_strength_lookup() -> dict[
    str,
    Any,
]:
    return {
        strength.name: (
            strength
        )
        for strength in (
            LOCKED_REALX_STRENGTHS
        )
    }


def validate_realx_run_strength(
    run_spec: RealXRunSpec,
) -> None:
    lookup = (
        _locked_strength_lookup()
    )

    if (
        run_spec.strength_name
        not in lookup
    ):
        raise RuntimeError(
            "Unknown Real-X strength: "
            f"{run_spec.strength_name}"
        )

    expected = lookup[
        run_spec.strength_name
    ]

    if not np.isclose(
        run_spec.interaction_coefficient,
        expected.interaction_coefficient,
        atol=1e-12,
        rtol=0.0,
    ):
        raise RuntimeError(
            "Run interaction coefficient "
            "does not match locked strength."
        )

    if (
        bool(
            run_spec.active_ground_truth
        )
        != bool(
            expected.active_ground_truth
        )
    ):
        raise RuntimeError(
            "Run active-ground-truth flag "
            "does not match locked strength."
        )


def generate_realx_run(
    X: pd.DataFrame,
    run_spec: RealXRunSpec,
    *,
    protocol: RealXProtocol | None = None,
) -> RealXGeneratedRun:
    """
    Generate one semi-synthetic binary outcome from real X.

    Original outcome labels are neither accepted nor used.
    """
    if protocol is None:
        protocol = (
            RealXProtocol()
        )

    validate_realx_run_strength(
        run_spec
    )

    blueprint = (
        build_realx_blueprint(
            X,
            run_spec,
            protocol=(
                protocol
            ),
        )
    )

    interaction_signal = (
        float(
            run_spec
            .interaction_coefficient
        )
        * blueprint
        .interaction_composite
    )

    linear_without_intercept = (
        float(
            protocol
            .main_effect_coefficient
        )
        * blueprint
        .main_composite
        + interaction_signal
    )

    intercept = (
        calibrate_logistic_intercept(
            linear_without_intercept,
            target_expected_prevalence=(
                protocol
                .target_expected_prevalence
            ),
        )
    )

    logits = (
        intercept
        + linear_without_intercept
    )

    probabilities = (
        _sigmoid(
            logits
        )
    )

    labels = (
        blueprint
        .uniform_draws
        < probabilities
    ).astype(
        np.int64
    )

    if (
        run_spec.active_ground_truth
    ):
        active_true_pairs = (
            blueprint.true_pairs
        )

    else:
        active_true_pairs = tuple()

    return RealXGeneratedRun(
        run_spec=(
            run_spec
        ),
        X=(
            blueprint.sampled_X
        ),
        source_row_positions=(
            blueprint
            .source_row_positions
        ),
        y=(
            labels
        ),
        probabilities=(
            probabilities
        ),
        logits=(
            logits
        ),
        intercept=(
            intercept
        ),
        main_composite=(
            blueprint.main_composite
        ),
        interaction_composite=(
            blueprint
            .interaction_composite
        ),
        interaction_signal=(
            interaction_signal
        ),
        eligible_features=(
            blueprint
            .eligible_features
        ),
        main_features=(
            blueprint
            .main_features
        ),
        template_true_pairs=(
            blueprint.true_pairs
        ),
        active_true_pairs=(
            active_true_pairs
        ),
        state_codes=(
            blueprint.state_codes
        ),
        state_labels=(
            blueprint.state_labels
        ),
        interaction_marginal_errors=(
            blueprint
            .interaction_marginal_errors
        ),
        uniform_draws=(
            blueprint.uniform_draws
        ),
        train_indices=(
            blueprint.train_indices
        ),
        validation_indices=(
            blueprint
            .validation_indices
        ),
        test_indices=(
            blueprint.test_indices
        ),
        expected_prevalence=float(
            np.mean(
                probabilities
            )
        ),
        realized_prevalence=float(
            np.mean(
                labels
            )
        ),
    )