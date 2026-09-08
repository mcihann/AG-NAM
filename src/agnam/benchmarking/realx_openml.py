from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import openml
import pandas as pd

from agnam.benchmarking.realx_generator import (
    RealXGeneratedRun,
    generate_realx_run,
)
from agnam.benchmarking.realx_protocol import (
    LOCKED_REALX_DATASETS,
    LOCKED_REALX_STRENGTHS,
    RealXDatasetSpec,
    RealXProtocol,
    RealXRunSpec,
    build_realx_schedule,
)


# =============================================================================
# DATA CONTAINERS
# =============================================================================


@dataclass
class LoadedRealXOpenML:
    specification: RealXDatasetSpec

    X: pd.DataFrame

    openml_dataset_id: int
    openml_dataset_name: str
    target_name: str

    n_rows: int
    n_features: int

    categorical_columns: tuple[str, ...]
    numeric_columns: tuple[str, ...]

    n_missing_values: int


@dataclass
class RealXDatasetAuditResult:
    task_id: int
    dataset_name: str
    feature_type: str

    original_rows: int
    original_features: int

    n_numeric_features: int
    n_categorical_features: int
    n_missing_values: int

    realization_rows: pd.DataFrame


# =============================================================================
# HELPERS
# =============================================================================


def _normalize_name(
    value: str,
) -> str:
    return (
        str(value)
        .strip()
        .casefold()
        .replace("_", "-")
    )


def _hash_integer_array(
    values: np.ndarray,
) -> str:
    array = np.asarray(
        values,
        dtype=np.int64,
    )

    return sha256(
        array.tobytes()
    ).hexdigest()


def _hash_truth_structure(
    main_features: tuple[str, ...],
    true_pairs: tuple[
        tuple[str, str],
        ...,
    ],
) -> str:
    payload = {
        "main_features": list(
            main_features
        ),
        "true_pairs": [
            [
                str(
                    pair[
                        0
                    ]
                ),
                str(
                    pair[
                        1
                    ]
                ),
            ]
            for pair in (
                true_pairs
            )
        ],
    }

    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(
            ",",
            ":",
        ),
    ).encode(
        "utf-8"
    )

    return sha256(
        encoded
    ).hexdigest()


def _serialize_features(
    features: tuple[str, ...],
) -> str:
    return "|".join(
        str(
            feature
        )
        for feature in (
            features
        )
    )


def _serialize_pairs(
    pairs: tuple[
        tuple[str, str],
        ...,
    ],
) -> str:
    return "|".join(
        (
            f"{pair[0]}::{pair[1]}"
        )
        for pair in (
            pairs
        )
    )


def _split_has_both_classes(
    generated: RealXGeneratedRun,
    indices: np.ndarray,
) -> bool:
    labels = generated.y[
        indices
    ]

    return (
        np.unique(
            labels
        ).size
        == 2
    )


def _same_generated_blueprint(
    first: RealXGeneratedRun,
    second: RealXGeneratedRun,
) -> dict[str, bool]:
    return {
        "same_sampled_rows": (
            np.array_equal(
                first.source_row_positions,
                second.source_row_positions,
            )
        ),
        "same_X": (
            first.X.equals(
                second.X
            )
        ),
        "same_state_encoding": (
            first.state_codes.equals(
                second.state_codes
            )
        ),
        "same_main_features": (
            first.main_features
            == second.main_features
        ),
        "same_truth_templates": (
            first.template_true_pairs
            == second.template_true_pairs
        ),
        "same_main_composite": (
            np.allclose(
                first.main_composite,
                second.main_composite,
                atol=0.0,
                rtol=0.0,
            )
        ),
        "same_interaction_composite": (
            np.allclose(
                first.interaction_composite,
                second.interaction_composite,
                atol=0.0,
                rtol=0.0,
            )
        ),
        "same_uniform_draws": (
            np.array_equal(
                first.uniform_draws,
                second.uniform_draws,
            )
        ),
        "same_train_split": (
            np.array_equal(
                first.train_indices,
                second.train_indices,
            )
        ),
        "same_validation_split": (
            np.array_equal(
                first.validation_indices,
                second.validation_indices,
            )
        ),
        "same_test_split": (
            np.array_equal(
                first.test_indices,
                second.test_indices,
            )
        ),
    }


# =============================================================================
# OPENML LOADING
# =============================================================================


def load_realx_openml_X(
    specification: RealXDatasetSpec,
) -> LoadedRealXOpenML:
    """
    Load raw predictor values for one locked OpenML task.

    Original outcome labels are obtained only because the OpenML API
    returns them together with X. They are immediately discarded and
    are never returned by this function.

    OpenML categorical metadata is explicitly restored into pandas
    categorical dtypes so integer-coded nominal predictors are not
    silently treated as continuous variables by the Real-X generator.
    """
    task = openml.tasks.get_task(
        specification.task_id,
        download_data=True,
    )

    dataset = task.get_dataset()

    target_name = getattr(
        task,
        "target_name",
        None,
    )

    if target_name is None:
        target_name = getattr(
            dataset,
            "default_target_attribute",
            None,
        )

    if target_name is None:
        raise RuntimeError(
            "Unable to determine target "
            f"for OpenML task "
            f"{specification.task_id}."
        )

    (
        X,
        y,
        categorical_indicator,
        attribute_names,
    ) = dataset.get_data(
        dataset_format="dataframe",
        target=target_name,
    )

    # The original outcome is deliberately destroyed here.
    del y

    if not isinstance(
        X,
        pd.DataFrame,
    ):
        raise RuntimeError(
            "OpenML did not return a "
            "pandas DataFrame."
        )

    X = (
        X.copy()
        .reset_index(
            drop=True
        )
    )

    if (
        X.columns.duplicated()
        .any()
    ):
        raise RuntimeError(
            "OpenML predictor frame contains "
            "duplicate column names."
        )

    if (
        target_name
        in X.columns
    ):
        raise RuntimeError(
            "Target leakage detected: "
            f"{target_name} is present in X."
        )

    if (
        len(
            categorical_indicator
        )
        != len(
            attribute_names
        )
    ):
        raise RuntimeError(
            "OpenML categorical metadata "
            "length mismatch."
        )

    if (
        len(
            attribute_names
        )
        != X.shape[
            1
        ]
    ):
        raise RuntimeError(
            "OpenML feature metadata does "
            "not match returned X."
        )

    expected_columns = [
        str(
            column
        )
        for column in (
            attribute_names
        )
    ]

    observed_columns = [
        str(
            column
        )
        for column in (
            X.columns
        )
    ]

    if (
        expected_columns
        != observed_columns
    ):
        raise RuntimeError(
            "OpenML attribute ordering "
            "does not match X columns."
        )

    categorical_columns = []

    numeric_columns = []

    for (
        column,
        is_categorical,
    ) in zip(
        X.columns,
        categorical_indicator,
    ):
        feature_name = str(
            column
        )

        if bool(
            is_categorical
        ):
            X[
                column
            ] = X[
                column
            ].astype(
                "category"
            )

            categorical_columns.append(
                feature_name
            )

        else:
            X[
                column
            ] = pd.to_numeric(
                X[
                    column
                ],
                errors="coerce",
            )

            numeric_columns.append(
                feature_name
            )

    observed_name = str(
        dataset.name
    )

    if (
        _normalize_name(
            observed_name
        )
        != _normalize_name(
            specification.dataset_name
        )
    ):
        raise RuntimeError(
            "Locked dataset-name mismatch. "
            f"Expected "
            f"{specification.dataset_name}, "
            f"OpenML returned "
            f"{observed_name}."
        )

    n_rows = int(
        X.shape[
            0
        ]
    )

    n_features = int(
        X.shape[
            1
        ]
    )

    if (
        n_rows
        < 1
        or n_features
        < 1
    ):
        raise RuntimeError(
            "OpenML returned an empty "
            "predictor frame."
        )

    missing_count = int(
        X.isna()
        .sum()
        .sum()
    )

    return LoadedRealXOpenML(
        specification=(
            specification
        ),
        X=(
            X
        ),
        openml_dataset_id=int(
            dataset.dataset_id
        ),
        openml_dataset_name=(
            observed_name
        ),
        target_name=str(
            target_name
        ),
        n_rows=(
            n_rows
        ),
        n_features=(
            n_features
        ),
        categorical_columns=tuple(
            categorical_columns
        ),
        numeric_columns=tuple(
            numeric_columns
        ),
        n_missing_values=(
            missing_count
        ),
    )


# =============================================================================
# DATASET-TYPE AUDIT
# =============================================================================


def validate_realx_feature_type(
    loaded: LoadedRealXOpenML,
) -> None:
    """
    Confirm that the locked broad feature-type stratum is consistent
    with OpenML's own feature metadata.
    """
    expected = (
        loaded
        .specification
        .feature_type
    )

    n_numeric = len(
        loaded.numeric_columns
    )

    n_categorical = len(
        loaded.categorical_columns
    )

    if (
        expected
        == "numeric"
    ):
        valid = (
            n_numeric
            > 0
            and n_categorical
            == 0
        )

    elif (
        expected
        == "categorical"
    ):
        valid = (
            n_categorical
            > 0
            and n_numeric
            == 0
        )

    elif (
        expected
        == "mixed"
    ):
        valid = (
            n_numeric
            > 0
            and n_categorical
            > 0
        )

    else:
        raise RuntimeError(
            "Unknown locked feature type: "
            f"{expected}"
        )

    if not valid:
        raise RuntimeError(
            "Locked feature-type audit "
            "failed for "
            f"{loaded.specification.dataset_name}. "
            f"Expected {expected}; "
            f"OpenML metadata yielded "
            f"{n_numeric} numeric and "
            f"{n_categorical} categorical "
            "predictors."
        )


# =============================================================================
# REALIZATION AUDIT
# =============================================================================


def _runs_for_dataset_realization(
    schedule: tuple[
        RealXRunSpec,
        ...,
    ],
    *,
    task_id: int,
    realization: int,
) -> tuple[
    RealXRunSpec,
    ...,
]:
    runs = tuple(
        run
        for run in (
            schedule
        )
        if (
            run.task_id
            == task_id
            and run.realization
            == realization
        )
    )

    if len(
        runs
    ) != len(
        LOCKED_REALX_STRENGTHS
    ):
        raise RuntimeError(
            "Expected exactly four "
            "strength runs for task "
            f"{task_id}, realization "
            f"{realization}; found "
            f"{len(runs)}."
        )

    strength_order = {
        strength.name: index
        for (
            index,
            strength,
        ) in enumerate(
            LOCKED_REALX_STRENGTHS
        )
    }

    return tuple(
        sorted(
            runs,
            key=lambda run: (
                strength_order[
                    run.strength_name
                ]
            ),
        )
    )


def audit_realx_realization(
    loaded: LoadedRealXOpenML,
    *,
    realization: int,
    schedule: tuple[
        RealXRunSpec,
        ...,
    ] | None = None,
    protocol: RealXProtocol | None = None,
) -> dict:
    if protocol is None:
        protocol = (
            RealXProtocol()
        )

    if schedule is None:
        schedule = (
            build_realx_schedule(
                protocol
            )
        )

    run_specs = (
        _runs_for_dataset_realization(
            schedule,
            task_id=(
                loaded
                .specification
                .task_id
            ),
            realization=(
                realization
            ),
        )
    )

    generated_by_strength = {}

    for run_spec in (
        run_specs
    ):
        generated_by_strength[
            run_spec.strength_name
        ] = (
            generate_realx_run(
                loaded.X,
                run_spec,
                protocol=(
                    protocol
                ),
            )
        )

    reference = (
        generated_by_strength[
            "null"
        ]
    )

    invariance_flags = []

    for strength in (
        LOCKED_REALX_STRENGTHS
    ):
        generated = (
            generated_by_strength[
                strength.name
            ]
        )

        flags = (
            _same_generated_blueprint(
                reference,
                generated,
            )
        )

        invariance_flags.extend(
            flags.values()
        )

    all_common_random_numbers = all(
        invariance_flags
    )

    if not (
        all_common_random_numbers
    ):
        raise RuntimeError(
            "Common-random-number invariance "
            "failed for "
            f"{loaded.specification.dataset_name}, "
            f"realization {realization}."
        )

    max_purification_error = max(
        reference
        .interaction_marginal_errors
        .values()
    )

    # Purification converges to the locked 1e-10 threshold before
    # global component standardization. The post-standardization
    # audit allows only a small floating-point margin.
    numerical_audit_tolerance = max(
        1e-8,
        protocol
        .purification_tolerance
        * 100.0,
    )

    if (
        max_purification_error
        > numerical_audit_tolerance
    ):
        raise RuntimeError(
            "Post-standardization interaction "
            "marginal error exceeds numerical "
            "audit tolerance for "
            f"{loaded.specification.dataset_name}, "
            f"realization {realization}: "
            f"{max_purification_error:.3e}"
        )

    if not np.isclose(
        np.std(
            reference.main_composite
        ),
        1.0,
        atol=1e-10,
        rtol=0.0,
    ):
        raise RuntimeError(
            "Main composite is not "
            "unit-standardized."
        )

    if not np.isclose(
        np.std(
            reference.interaction_composite
        ),
        1.0,
        atol=1e-10,
        rtol=0.0,
    ):
        raise RuntimeError(
            "Interaction composite is not "
            "unit-standardized."
        )

    expected_signal_stds = {
        "null": 0.0,
        "weak": 0.5,
        "moderate": 1.0,
        "strong": 1.5,
    }

    for (
        strength_name,
        expected_std,
    ) in (
        expected_signal_stds.items()
    ):
        observed_std = float(
            np.std(
                generated_by_strength[
                    strength_name
                ]
                .interaction_signal,
                ddof=0,
            )
        )

        if not np.isclose(
            observed_std,
            expected_std,
            atol=1e-10,
            rtol=0.0,
        ):
            raise RuntimeError(
                "Unexpected interaction-signal "
                "standard deviation for "
                f"{strength_name}: "
                f"{observed_std:.12f}"
            )

    class_audit = {}

    for strength in (
        LOCKED_REALX_STRENGTHS
    ):
        generated = (
            generated_by_strength[
                strength.name
            ]
        )

        class_audit[
            f"{strength.name}_train_two_classes"
        ] = (
            _split_has_both_classes(
                generated,
                generated.train_indices,
            )
        )

        class_audit[
            f"{strength.name}_validation_two_classes"
        ] = (
            _split_has_both_classes(
                generated,
                generated.validation_indices,
            )
        )

        class_audit[
            f"{strength.name}_test_two_classes"
        ] = (
            _split_has_both_classes(
                generated,
                generated.test_indices,
            )
        )

    if not all(
        class_audit.values()
    ):
        failing = [
            key
            for (
                key,
                value,
            ) in (
                class_audit.items()
            )
            if not value
        ]

        raise RuntimeError(
            "A generated split contains only "
            "one class for "
            f"{loaded.specification.dataset_name}, "
            f"realization {realization}: "
            f"{failing}"
        )

    expected_prevalence_errors = {}

    for strength in (
        LOCKED_REALX_STRENGTHS
    ):
        generated = (
            generated_by_strength[
                strength.name
            ]
        )

        expected_prevalence_errors[
            strength.name
        ] = abs(
            generated.expected_prevalence
            - protocol
            .target_expected_prevalence
        )

    if max(
        expected_prevalence_errors.values()
    ) > 1e-9:
        raise RuntimeError(
            "Expected prevalence calibration "
            "failed."
        )

    state_counts = [
        len(
            reference
            .state_labels[
                feature
            ]
        )
        for feature in (
            reference
            .state_labels
        )
    ]

    row = {
        "task_id": (
            loaded
            .specification
            .task_id
        ),
        "dataset_name": (
            loaded
            .specification
            .dataset_name
        ),
        "feature_type": (
            loaded
            .specification
            .feature_type
        ),
        "realization": (
            realization
        ),
        "original_rows": (
            loaded.n_rows
        ),
        "original_features": (
            loaded.n_features
        ),
        "sampled_rows": int(
            len(
                reference.X
            )
        ),
        "eligible_features": int(
            len(
                reference
                .eligible_features
            )
        ),
        "n_numeric_features": int(
            len(
                loaded
                .numeric_columns
            )
        ),
        "n_categorical_features": int(
            len(
                loaded
                .categorical_columns
            )
        ),
        "n_missing_values_original_X": (
            loaded
            .n_missing_values
        ),
        "main_features": (
            _serialize_features(
                reference
                .main_features
            )
        ),
        "true_pairs": (
            _serialize_pairs(
                reference
                .template_true_pairs
            )
        ),
        "truth_structure_hash": (
            _hash_truth_structure(
                reference.main_features,
                reference.template_true_pairs,
            )
        ),
        "source_row_hash": (
            _hash_integer_array(
                reference
                .source_row_positions
            )
        ),
        "min_state_count": int(
            min(
                state_counts
            )
        ),
        "max_state_count": int(
            max(
                state_counts
            )
        ),
        "mean_state_count": float(
            np.mean(
                state_counts
            )
        ),
        "max_purification_error": float(
            max_purification_error
        ),
        "main_composite_std": float(
            np.std(
                reference
                .main_composite,
                ddof=0,
            )
        ),
        "interaction_composite_std": float(
            np.std(
                reference
                .interaction_composite,
                ddof=0,
            )
        ),
        "train_size": int(
            len(
                reference
                .train_indices
            )
        ),
        "validation_size": int(
            len(
                reference
                .validation_indices
            )
        ),
        "test_size": int(
            len(
                reference
                .test_indices
            )
        ),
        "common_random_numbers_ok": bool(
            all_common_random_numbers
        ),
    }

    for strength in (
        LOCKED_REALX_STRENGTHS
    ):
        generated = (
            generated_by_strength[
                strength.name
            ]
        )

        row[
            (
                f"{strength.name}_"
                "expected_prevalence"
            )
        ] = float(
            generated
            .expected_prevalence
        )

        row[
            (
                f"{strength.name}_"
                "realized_prevalence"
            )
        ] = float(
            generated
            .realized_prevalence
        )

        row[
            (
                f"{strength.name}_"
                "interaction_signal_std"
            )
        ] = float(
            np.std(
                generated
                .interaction_signal,
                ddof=0,
            )
        )

    row.update(
        class_audit
    )

    return row


# =============================================================================
# FULL NINE-DATASET AUDIT
# =============================================================================


def audit_realx_dataset(
    loaded: LoadedRealXOpenML,
    *,
    protocol: RealXProtocol | None = None,
    schedule: tuple[
        RealXRunSpec,
        ...,
    ] | None = None,
) -> RealXDatasetAuditResult:
    if protocol is None:
        protocol = (
            RealXProtocol()
        )

    if schedule is None:
        schedule = (
            build_realx_schedule(
                protocol
            )
        )

    validate_realx_feature_type(
        loaded
    )

    rows = []

    for realization in range(
        1,
        protocol.n_realizations
        + 1,
    ):
        rows.append(
            audit_realx_realization(
                loaded,
                realization=(
                    realization
                ),
                schedule=(
                    schedule
                ),
                protocol=(
                    protocol
                ),
            )
        )

    frame = pd.DataFrame(
        rows
    )

    return RealXDatasetAuditResult(
        task_id=(
            loaded
            .specification
            .task_id
        ),
        dataset_name=(
            loaded
            .specification
            .dataset_name
        ),
        feature_type=(
            loaded
            .specification
            .feature_type
        ),
        original_rows=(
            loaded.n_rows
        ),
        original_features=(
            loaded.n_features
        ),
        n_numeric_features=(
            len(
                loaded.numeric_columns
            )
        ),
        n_categorical_features=(
            len(
                loaded
                .categorical_columns
            )
        ),
        n_missing_values=(
            loaded
            .n_missing_values
        ),
        realization_rows=(
            frame
        ),
    )


def audit_all_locked_realx_datasets(
    *,
    output_path: str | Path = (
        "results/realx/audit/"
        "realx_generator_audit.csv"
    ),
    protocol: RealXProtocol | None = None,
) -> pd.DataFrame:
    if protocol is None:
        protocol = (
            RealXProtocol()
        )

    schedule = (
        build_realx_schedule(
            protocol
        )
    )

    all_frames = []

    for (
        dataset_number,
        specification,
    ) in enumerate(
        LOCKED_REALX_DATASETS,
        start=1,
    ):
        print(
            f"[{dataset_number}/"
            f"{len(LOCKED_REALX_DATASETS)}] "
            f"Loading task "
            f"{specification.task_id} — "
            f"{specification.dataset_name}"
        )

        loaded = (
            load_realx_openml_X(
                specification
            )
        )

        print(
            "    X:",
            f"{loaded.n_rows} rows × "
            f"{loaded.n_features} features",
        )

        print(
            "    metadata:",
            f"{len(loaded.numeric_columns)} numeric, "
            f"{len(loaded.categorical_columns)} categorical, "
            f"{loaded.n_missing_values} missing values",
        )

        result = (
            audit_realx_dataset(
                loaded,
                protocol=(
                    protocol
                ),
                schedule=(
                    schedule
                ),
            )
        )

        all_frames.append(
            result.realization_rows
        )

        print(
            "    realizations:",
            len(
                result.realization_rows
            ),
            "PASS",
        )

    audit_table = pd.concat(
        all_frames,
        axis=0,
        ignore_index=True,
    )

    expected_rows = (
        len(
            LOCKED_REALX_DATASETS
        )
        * protocol.n_realizations
    )

    if len(
        audit_table
    ) != expected_rows:
        raise RuntimeError(
            "Expected "
            f"{expected_rows} dataset-realization "
            "audit rows, found "
            f"{len(audit_table)}."
        )

    if not (
        audit_table[
            "common_random_numbers_ok"
        ]
        .astype(bool)
        .all()
    ):
        raise RuntimeError(
            "At least one Real-X audit row "
            "failed common-random-number "
            "verification."
        )

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit_table.to_csv(
        output_path,
        index=False,
    )

    return audit_table