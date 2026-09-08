from __future__ import annotations

import os

# Must be defined before CUDA/cuBLAS is initialized.
os.environ.setdefault(
    "CUBLAS_WORKSPACE_CONFIG",
    ":4096:8",
)

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch

import agnam.benchmarking.synthetic_runner as sr

from agnam.benchmarking.realx_discovery import (
    run_realx_reproducible_interaction_discovery,
)
from agnam.benchmarking.realx_generator import (
    RealXGeneratedRun,
    generate_realx_run,
)
from agnam.benchmarking.realx_openml import (
    load_realx_openml_X,
    validate_realx_feature_type,
)
from agnam.benchmarking.realx_protocol import (
    LOCKED_REALX_DATASETS,
    RealXProtocol,
    RealXRunSpec,
    build_realx_schedule,
)


# =============================================================================
# PHASE 6D MODEL-SIDE LOCK
# =============================================================================


@dataclass(frozen=True)
class RealXModelProtocol:
    isr_reference_size: int = 512
    isr_reference_seed: int = 4026

    main_early_stop_fraction: float = 0.20

    decomposition_tolerance: float = 1e-6

    sanity_task_id: int = 49
    sanity_realization: int = 1
    sanity_strength: str = "moderate"

    publication_eligible: bool = False

    def __post_init__(
        self,
    ) -> None:
        if self.isr_reference_size != 512:
            raise ValueError(
                "Real-X ISR reference size is locked to 512."
            )

        if self.isr_reference_seed != 4026:
            raise ValueError(
                "Real-X ISR reference seed is locked to 4026."
            )

        if not np.isclose(
            self.main_early_stop_fraction,
            0.20,
            atol=1e-12,
            rtol=0.0,
        ):
            raise ValueError(
                "Real-X discovery early-stop fraction "
                "is locked to 0.20."
            )

        if not np.isclose(
            self.decomposition_tolerance,
            1e-6,
            atol=0.0,
            rtol=0.0,
        ):
            raise ValueError(
                "Real-X decomposition tolerance "
                "is locked to 1e-6."
            )

        if self.sanity_task_id != 49:
            raise ValueError(
                "Phase 6D task is locked to OpenML task 49."
            )

        if self.sanity_realization != 1:
            raise ValueError(
                "Phase 6D realization is locked to 1."
            )

        if self.sanity_strength != "moderate":
            raise ValueError(
                "Phase 6D strength is locked to moderate."
            )

        if self.publication_eligible:
            raise ValueError(
                "Phase 6D output cannot be publication eligible."
            )


@dataclass(frozen=True)
class RealXModelSeeds:
    discovery_base_seed: int
    random_pair_seed: int
    final_model_seed: int


@dataclass
class RealXPreparedIntegrationData:
    generated: RealXGeneratedRun

    dev_indices: np.ndarray

    X_dev: pd.DataFrame
    y_dev: np.ndarray

    final_data: Any


@dataclass
class RealXIntegrationSanityResult:
    task_id: int
    dataset_name: str
    realization: int

    strength_name: str
    interaction_coefficient: float

    n_samples: int
    n_features: int

    train_size: int
    validation_size: int
    test_size: int
    discovery_dev_size: int

    discovery_base_seed: int
    random_pair_seed: int
    final_model_seed: int

    expected_candidate_k: int
    observed_candidate_k: int

    true_pairs: tuple[
        tuple[str, str],
        ...,
    ]

    selection_pairs: tuple[
        tuple[str, str],
        ...,
    ]

    isr_pairs: tuple[
        tuple[str, str],
        ...,
    ]

    oracle_pairs: tuple[
        tuple[str, str],
        ...,
    ]

    random_pairs: tuple[
        tuple[str, str],
        ...,
    ]

    no_isr_pairs: tuple[
        tuple[str, str],
        ...,
    ]

    single_run_pairs: tuple[
        tuple[str, str],
        ...,
    ]

    mean_pairwise_jaccard: float
    mean_interaction_auprc: float

    selection_precision: float
    selection_recall: float
    selection_f1: float

    isr_precision: float
    isr_recall: float
    isr_f1: float

    false_positive_reduction: float

    main_auroc: float
    main_auprc: float
    main_balanced_accuracy: float
    main_f1: float

    agnam_auroc: float
    agnam_auprc: float
    agnam_balanced_accuracy: float
    agnam_f1: float

    oracle_auroc: float
    oracle_auprc: float

    random_pair_auroc: float
    random_pair_auprc: float

    no_isr_auroc: float
    no_isr_auprc: float

    single_run_auroc: float
    single_run_auprc: float

    delta_auroc: float
    delta_auprc: float
    delta_balanced_accuracy: float
    delta_f1: float

    max_decomposition_error: float

    test_integrity_ok: bool

    publication_eligible: bool

    runtime_seconds: float

    def to_record(
        self,
    ) -> dict[str, Any]:
        def serialize_pairs(
            pairs,
        ) -> str:
            return "|".join(
                f"{first}::{second}"
                for (
                    first,
                    second,
                ) in pairs
            )

        return {
            "task_id": self.task_id,
            "dataset_name": self.dataset_name,
            "realization": self.realization,
            "strength_name": self.strength_name,
            "interaction_coefficient": (
                self.interaction_coefficient
            ),
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "train_size": self.train_size,
            "validation_size": (
                self.validation_size
            ),
            "test_size": self.test_size,
            "discovery_dev_size": (
                self.discovery_dev_size
            ),
            "discovery_base_seed": (
                self.discovery_base_seed
            ),
            "random_pair_seed": (
                self.random_pair_seed
            ),
            "final_model_seed": (
                self.final_model_seed
            ),
            "expected_candidate_k": (
                self.expected_candidate_k
            ),
            "observed_candidate_k": (
                self.observed_candidate_k
            ),
            "true_pairs": serialize_pairs(
                self.true_pairs
            ),
            "selection_pairs": serialize_pairs(
                self.selection_pairs
            ),
            "isr_pairs": serialize_pairs(
                self.isr_pairs
            ),
            "oracle_pairs": serialize_pairs(
                self.oracle_pairs
            ),
            "random_pairs": serialize_pairs(
                self.random_pairs
            ),
            "no_isr_pairs": serialize_pairs(
                self.no_isr_pairs
            ),
            "single_run_pairs": serialize_pairs(
                self.single_run_pairs
            ),
            "mean_pairwise_jaccard": (
                self.mean_pairwise_jaccard
            ),
            "mean_interaction_auprc": (
                self.mean_interaction_auprc
            ),
            "selection_precision": (
                self.selection_precision
            ),
            "selection_recall": (
                self.selection_recall
            ),
            "selection_f1": (
                self.selection_f1
            ),
            "isr_precision": (
                self.isr_precision
            ),
            "isr_recall": (
                self.isr_recall
            ),
            "isr_f1": self.isr_f1,
            "false_positive_reduction": (
                self.false_positive_reduction
            ),
            "main_auroc": self.main_auroc,
            "main_auprc": self.main_auprc,
            "main_balanced_accuracy": (
                self.main_balanced_accuracy
            ),
            "main_f1": self.main_f1,
            "agnam_auroc": (
                self.agnam_auroc
            ),
            "agnam_auprc": (
                self.agnam_auprc
            ),
            "agnam_balanced_accuracy": (
                self.agnam_balanced_accuracy
            ),
            "agnam_f1": (
                self.agnam_f1
            ),
            "oracle_auroc": (
                self.oracle_auroc
            ),
            "oracle_auprc": (
                self.oracle_auprc
            ),
            "random_pair_auroc": (
                self.random_pair_auroc
            ),
            "random_pair_auprc": (
                self.random_pair_auprc
            ),
            "no_isr_auroc": (
                self.no_isr_auroc
            ),
            "no_isr_auprc": (
                self.no_isr_auprc
            ),
            "single_run_auroc": (
                self.single_run_auroc
            ),
            "single_run_auprc": (
                self.single_run_auprc
            ),
            "delta_auroc": (
                self.delta_auroc
            ),
            "delta_auprc": (
                self.delta_auprc
            ),
            "delta_balanced_accuracy": (
                self.delta_balanced_accuracy
            ),
            "delta_f1": (
                self.delta_f1
            ),
            "max_decomposition_error": (
                self.max_decomposition_error
            ),
            "test_integrity_ok": (
                self.test_integrity_ok
            ),
            "publication_eligible": (
                self.publication_eligible
            ),
            "runtime_seconds": (
                self.runtime_seconds
            ),
        }


# =============================================================================
# LOCKED RUN RESOLUTION
# =============================================================================


def get_locked_sanity_run_spec(
    *,
    realx_protocol: RealXProtocol | None = None,
    model_protocol: RealXModelProtocol | None = None,
) -> RealXRunSpec:
    if realx_protocol is None:
        realx_protocol = (
            RealXProtocol()
        )

    if model_protocol is None:
        model_protocol = (
            RealXModelProtocol()
        )

    schedule = (
        build_realx_schedule(
            realx_protocol
        )
    )

    matches = tuple(
        run
        for run in schedule
        if (
            run.task_id
            == model_protocol.sanity_task_id
            and run.realization
            == model_protocol.sanity_realization
            and run.strength_name
            == model_protocol.sanity_strength
        )
    )

    if len(
        matches
    ) != 1:
        raise RuntimeError(
            "Unable to uniquely resolve "
            "the locked Phase 6D run."
        )

    return matches[
        0
    ]


def get_dataset_specification(
    task_id: int,
):
    matches = tuple(
        item
        for item in (
            LOCKED_REALX_DATASETS
        )
        if (
            item.task_id
            == task_id
        )
    )

    if len(
        matches
    ) != 1:
        raise RuntimeError(
            "Unable to uniquely resolve "
            f"locked OpenML task {task_id}."
        )

    return matches[
        0
    ]


# =============================================================================
# MODEL SEEDS
# =============================================================================


def derive_realx_model_seeds(
    run_spec: RealXRunSpec,
) -> RealXModelSeeds:
    """
    Extend the locked dataset/realization seed family.

    The same model-side seeds are used across all four strength
    conditions of the same dataset-realization.
    """
    return RealXModelSeeds(
        discovery_base_seed=(
            int(
                run_spec.label_seed
            )
            + 10
        ),
        random_pair_seed=(
            int(
                run_spec.label_seed
            )
            + 20
        ),
        final_model_seed=(
            int(
                run_spec.label_seed
            )
            + 30
        ),
    )


# =============================================================================
# REAL-X CANDIDATE RULE
# =============================================================================


def realx_candidate_k(
    n_features: int,
) -> int:
    """
    Locked Real-X rule:

        K = min(p - 1, 20)
    """
    if n_features < 2:
        return 0

    return int(
        min(
            n_features - 1,
            20,
        )
    )


# =============================================================================
# LOCKED SPLIT AUDIT
# =============================================================================


def validate_locked_split(
    *,
    n_samples: int,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    test_indices: np.ndarray,
) -> None:
    train_indices = np.asarray(
        train_indices,
        dtype=np.int64,
    )

    validation_indices = np.asarray(
        validation_indices,
        dtype=np.int64,
    )

    test_indices = np.asarray(
        test_indices,
        dtype=np.int64,
    )

    groups = (
        train_indices,
        validation_indices,
        test_indices,
    )

    for group in groups:
        if len(
            np.unique(
                group
            )
        ) != len(
            group
        ):
            raise RuntimeError(
                "Locked split contains duplicate indices."
            )

        if (
            np.any(
                group < 0
            )
            or np.any(
                group >= n_samples
            )
        ):
            raise RuntimeError(
                "Locked split contains out-of-range indices."
            )

    if np.intersect1d(
        train_indices,
        validation_indices,
    ).size:
        raise RuntimeError(
            "Train and validation partitions overlap."
        )

    if np.intersect1d(
        train_indices,
        test_indices,
    ).size:
        raise RuntimeError(
            "Train and test partitions overlap."
        )

    if np.intersect1d(
        validation_indices,
        test_indices,
    ).size:
        raise RuntimeError(
            "Validation and test partitions overlap."
        )

    combined = np.sort(
        np.concatenate(
            [
                train_indices,
                validation_indices,
                test_indices,
            ]
        )
    )

    expected = np.arange(
        n_samples,
        dtype=np.int64,
    )

    if not np.array_equal(
        combined,
        expected,
    ):
        raise RuntimeError(
            "Locked split is not a complete "
            "partition of the Real-X dataset."
        )


def build_locked_dev_indices(
    *,
    n_samples: int,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    test_indices: np.ndarray,
) -> np.ndarray:
    validate_locked_split(
        n_samples=n_samples,
        train_indices=train_indices,
        validation_indices=(
            validation_indices
        ),
        test_indices=test_indices,
    )

    dev_indices = np.sort(
        np.concatenate(
            [
                np.asarray(
                    train_indices,
                    dtype=np.int64,
                ),
                np.asarray(
                    validation_indices,
                    dtype=np.int64,
                ),
            ]
        )
    )

    if np.intersect1d(
        dev_indices,
        np.asarray(
            test_indices,
            dtype=np.int64,
        ),
    ).size:
        raise RuntimeError(
            "Real-X test samples leaked "
            "into the discovery pool."
        )

    return dev_indices


# =============================================================================
# FINAL TRAIN / VALIDATION / TEST DATA
# =============================================================================


def prepare_locked_final_prediction_data(
    *,
    X: pd.DataFrame,
    y: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    test_indices: np.ndarray,
):
    y = np.asarray(
        y
    )

    if len(
        X
    ) != len(
        y
    ):
        raise ValueError(
            "X and y lengths do not match."
        )

    validate_locked_split(
        n_samples=len(
            X
        ),
        train_indices=train_indices,
        validation_indices=(
            validation_indices
        ),
        test_indices=test_indices,
    )

    X_train = (
        X.iloc[
            train_indices
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    X_validation = (
        X.iloc[
            validation_indices
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    X_test = (
        X.iloc[
            test_indices
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    y_train = np.asarray(
        y[
            train_indices
        ]
    )

    y_validation = np.asarray(
        y[
            validation_indices
        ]
    )

    y_test = np.asarray(
        y[
            test_indices
        ]
    )

    preprocessor = (
        sr.TabularPreprocessor()
    )

    train_data = (
        preprocessor
        .fit_transform(
            X_train
        )
    )

    validation_data = (
        preprocessor
        .transform(
            X_validation
        )
    )

    test_data = (
        preprocessor
        .transform(
            X_test
        )
    )

    return sr.FinalPredictionData(
        train_data=train_data,
        validation_data=(
            validation_data
        ),
        test_data=test_data,
        y_train=y_train,
        y_validation=(
            y_validation
        ),
        y_test=y_test,
    )


def prepare_realx_integration_data(
    generated: RealXGeneratedRun,
) -> RealXPreparedIntegrationData:
    dev_indices = (
        build_locked_dev_indices(
            n_samples=len(
                generated.X
            ),
            train_indices=(
                generated.train_indices
            ),
            validation_indices=(
                generated.validation_indices
            ),
            test_indices=(
                generated.test_indices
            ),
        )
    )

    X_dev = (
        generated.X
        .iloc[
            dev_indices
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    y_dev = np.asarray(
        generated.y[
            dev_indices
        ]
    )

    final_data = (
        prepare_locked_final_prediction_data(
            X=generated.X,
            y=generated.y,
            train_indices=(
                generated.train_indices
            ),
            validation_indices=(
                generated.validation_indices
            ),
            test_indices=(
                generated.test_indices
            ),
        )
    )

    return RealXPreparedIntegrationData(
        generated=generated,
        dev_indices=dev_indices,
        X_dev=X_dev,
        y_dev=y_dev,
        final_data=final_data,
    )


# =============================================================================
# CONFIG ADAPTERS
# =============================================================================


def _build_realx_discovery_configs(
    *,
    seeds: RealXModelSeeds,
    model_config,
):
    specification_adapter = (
        SimpleNamespace(
            discovery_base_seed=(
                seeds.discovery_base_seed
            )
        )
    )

    return sr._build_discovery_configs(
        spec=(
            specification_adapter
        ),
        config=model_config,
    )


def _build_realx_isr_config(
    *,
    realx_protocol: RealXProtocol,
    model_protocol: RealXModelProtocol,
    model_config,
):
    protocol_adapter = (
        SimpleNamespace(
            reference_size=(
                model_protocol
                .isr_reference_size
            ),
            reference_seed=(
                model_protocol
                .isr_reference_seed
            ),
            isr_threshold=(
                realx_protocol
                .isr_threshold
            ),
        )
    )

    return sr._build_isr_config(
        protocol=(
            protocol_adapter
        ),
        config=model_config,
    )


# =============================================================================
# PHASE 6D
# =============================================================================


def run_realx_integration_sanity(
    *,
    realx_protocol: RealXProtocol | None = None,
    model_protocol: RealXModelProtocol | None = None,
    model_config=None,
    device: torch.device | None = None,
    verbose: bool = True,
    output_path: str | Path = (
        "results/realx/integration_sanity/"
        "task_49_r01_moderate.csv"
    ),
) -> RealXIntegrationSanityResult:
    if realx_protocol is None:
        realx_protocol = (
            RealXProtocol()
        )

    if model_protocol is None:
        model_protocol = (
            RealXModelProtocol()
        )

    if model_config is None:
        model_config = (
            sr.SyntheticModelConfig()
        )

    if device is None:
        from agnam.utils.reproducibility import (
            get_device,
        )

        device = get_device()

    start_time = perf_counter()

    run_spec = (
        get_locked_sanity_run_spec(
            realx_protocol=(
                realx_protocol
            ),
            model_protocol=(
                model_protocol
            ),
        )
    )

    dataset_spec = (
        get_dataset_specification(
            run_spec.task_id
        )
    )

    loaded = (
        load_realx_openml_X(
            dataset_spec
        )
    )

    validate_realx_feature_type(
        loaded
    )

    generated = (
        generate_realx_run(
            loaded.X,
            run_spec,
            protocol=(
                realx_protocol
            ),
        )
    )

    prepared = (
        prepare_realx_integration_data(
            generated
        )
    )

    seeds = derive_realx_model_seeds(
        run_spec
    )

    true_pairs = sr.normalize_pairs(
        generated.active_true_pairs
    )

    if len(
        true_pairs
    ) != (
        realx_protocol
        .n_true_interactions
    ):
        raise RuntimeError(
            "Moderate Phase 6D run must "
            "contain exactly three active "
            "true interactions."
        )

    expected_candidate_k = (
        realx_candidate_k(
            generated
            .X
            .shape[
                1
            ]
        )
    )

    test_X_snapshot = (
        generated.X
        .iloc[
            generated.test_indices
        ]
        .copy(
            deep=True
        )
        .reset_index(
            drop=True
        )
    )

    test_y_snapshot = (
        np.asarray(
            generated.y[
                generated.test_indices
            ]
        )
        .copy()
    )

    if verbose:
        print(
            "=============================================="
        )

        print(
            "REAL-X / AG-NAM INTEGRATION SANITY"
        )

        print(
            "=============================================="
        )

        print(
            "INTEGRATION ONLY — NOT BENCHMARK EVIDENCE"
        )

        print()

        print(
            f"Task: {run_spec.task_id}"
        )

        print(
            f"Dataset: {run_spec.dataset_name}"
        )

        print(
            f"Realization: {run_spec.realization}"
        )

        print(
            f"Strength: {run_spec.strength_name}"
        )

        print(
            "Interaction coefficient:",
            f"{run_spec.interaction_coefficient:.1f}",
        )

        print(
            f"Device: {device}"
        )

        print()

        print(
            "LOCKED SPLIT"
        )

        print(
            "  train:",
            len(
                generated.train_indices
            ),
        )

        print(
            "  validation:",
            len(
                generated.validation_indices
            ),
        )

        print(
            "  untouched test:",
            len(
                generated.test_indices
            ),
        )

        print(
            "  discovery development:",
            len(
                prepared.dev_indices
            ),
        )

        print()

        print(
            "REAL-X CANDIDATE RULE"
        )

        print(
            "  p:",
            generated
            .X
            .shape[
                1
            ],
        )

        print(
            "  K:",
            expected_candidate_k,
        )

    # =========================================================
    # DISCOVERY
    # =========================================================

    if verbose:
        print(
            "\n=== DISCOVERY ==="
        )

    (
        nam_discovery_config,
        proposer_config,
    ) = (
        _build_realx_discovery_configs(
            seeds=seeds,
            model_config=(
                model_config
            ),
        )
    )

    discovery = (
        run_realx_reproducible_interaction_discovery(
            X=prepared.X_dev,
            y=prepared.y_dev,
            candidate_k=(
                expected_candidate_k
            ),
            n_runs=(
                realx_protocol
                .discovery_runs
            ),
            base_seed=(
                seeds.discovery_base_seed
            ),
            selection_threshold=(
                realx_protocol
                .selection_threshold
            ),
            crossfit_n_splits=(
                realx_protocol
                .residual_crossfit_folds
            ),
            main_early_stop_fraction=(
                model_protocol
                .main_early_stop_fraction
            ),
            nam_hidden_dim=(
                model_config
                .nam_hidden_dim
            ),
            nam_depth=(
                model_config
                .nam_depth
            ),
            nam_dropout=(
                model_config
                .nam_dropout
            ),
            categorical_embedding_dim=(
                model_config
                .categorical_embedding_dim
            ),
            proposer_d_model=(
                model_config
                .proposer_d_model
            ),
            proposer_n_heads=(
                model_config
                .proposer_n_heads
            ),
            proposer_n_layers=(
                model_config
                .proposer_n_layers
            ),
            proposer_dropout=(
                model_config
                .proposer_dropout
            ),
            nam_config=(
                nam_discovery_config
            ),
            proposer_config=(
                proposer_config
            ),
            device=device,
        )
    )

    observed_k_values = {
        int(
            run.k
        )
        for run in discovery.runs
    }

    if observed_k_values != {
        expected_candidate_k
    }:
        raise RuntimeError(
            "Real-X candidate-K adapter failed. "
            f"Expected only "
            f"{expected_candidate_k}, "
            f"observed "
            f"{sorted(observed_k_values)}."
        )

    observed_candidate_k = (
        expected_candidate_k
    )

    selection_pairs = (
        sr.normalize_pairs(
            (
                (
                    item.feature_j,
                    item.feature_k,
                )
                for item in (
                    discovery
                    .selection
                    .accepted_interactions
                )
            )
        )
    )

    selection_recovery = (
        sr.evaluate_pair_set_recovery(
            predicted_pairs=(
                selection_pairs
            ),
            true_pairs=true_pairs,
        )
    )

    ranking_auprcs = (
        sr._interaction_ranking_auprcs(
            discovery=discovery,
            true_pairs=true_pairs,
        )
    )

    # =========================================================
    # ISR
    # =========================================================

    if verbose:
        print(
            "\n=== ISR ==="
        )

    isr_config = (
        _build_realx_isr_config(
            realx_protocol=(
                realx_protocol
            ),
            model_protocol=(
                model_protocol
            ),
            model_config=(
                model_config
            ),
        )
    )

    isr_result = (
        sr.evaluate_isr_for_discovery(
            X=prepared.X_dev,
            discovery=discovery,
            config=isr_config,
            device=device,
            verbose=verbose,
        )
    )

    isr_pairs = (
        sr.normalize_pairs(
            (
                (
                    item.feature_j,
                    item.feature_k,
                )
                for item in (
                    isr_result
                    .retained_interactions
                )
            )
        )
    )

    if not set(
        isr_pairs
    ).issubset(
        set(
            selection_pairs
        )
    ):
        raise RuntimeError(
            "ISR returned an interaction "
            "outside the selection-stable set."
        )

    isr_recovery = (
        sr.evaluate_pair_set_recovery(
            predicted_pairs=(
                isr_pairs
            ),
            true_pairs=true_pairs,
        )
    )

    false_positive_reduction = (
        sr.compute_false_positive_reduction(
            selection_recovery=(
                selection_recovery
            ),
            isr_recovery=(
                isr_recovery
            ),
        )
    )

    # =========================================================
    # CONTROLS
    # =========================================================

    oracle_pairs = true_pairs

    random_pairs = (
        sr.sample_random_pairs(
            feature_names=tuple(
                generated.X.columns
            ),
            n_pairs=len(
                isr_pairs
            ),
            seed=(
                seeds.random_pair_seed
            ),
        )
    )

    # Do not depend on any cached Top-K representation.
    first_run_ranked = (
        discovery
        .runs[
            0
        ]
        .score_result
        .ranked_interactions[
            :expected_candidate_k
        ]
    )

    single_run_pairs = (
        sr.normalize_pairs(
            (
                (
                    item.feature_j,
                    item.feature_k,
                )
                for item in (
                    first_run_ranked
                )
            )
        )
    )

    no_isr_pairs = (
        selection_pairs
    )

    # =========================================================
    # FINAL PREDICTIVE MODELS
    # =========================================================

    if verbose:
        print(
            "\n=== FINAL PREDICTIVE MODELS ==="
        )

        print(
            "1/6 Main-effect NAM"
        )

    main = sr._train_main_nam(
        data=prepared.final_data,
        seed=(
            seeds.final_model_seed
        ),
        config=model_config,
        device=device,
    )

    if verbose:
        print(
            "2/6 Full AG-NAM"
        )

    (
        full,
        max_decomposition_error,
    ) = (
        sr._train_interaction_model(
            data=prepared.final_data,
            interaction_pairs=(
                isr_pairs
            ),
            seed=(
                seeds.final_model_seed
            ),
            config=model_config,
            device=device,
        )
    )

    if (
        max_decomposition_error
        > model_protocol
        .decomposition_tolerance
    ):
        raise RuntimeError(
            "Final AG-NAM exact-decomposition "
            "audit failed. "
            f"Maximum error: "
            f"{max_decomposition_error:.3e}"
        )

    if verbose:
        print(
            "3/6 Oracle Interaction NAM"
        )

    oracle, _ = (
        sr._train_interaction_model(
            data=prepared.final_data,
            interaction_pairs=(
                oracle_pairs
            ),
            seed=(
                seeds.final_model_seed
            ),
            config=model_config,
            device=device,
        )
    )

    if verbose:
        print(
            "4/6 Random-Pair NAM"
        )

    random_model, _ = (
        sr._train_interaction_model(
            data=prepared.final_data,
            interaction_pairs=(
                random_pairs
            ),
            seed=(
                seeds.final_model_seed
            ),
            config=model_config,
            device=device,
        )
    )

    if verbose:
        print(
            "5/6 AG-NAM without ISR"
        )

    no_isr, _ = (
        sr._train_interaction_model(
            data=prepared.final_data,
            interaction_pairs=(
                no_isr_pairs
            ),
            seed=(
                seeds.final_model_seed
            ),
            config=model_config,
            device=device,
        )
    )

    if verbose:
        print(
            "6/6 Single-Run AG-NAM"
        )

    single_run, _ = (
        sr._train_interaction_model(
            data=prepared.final_data,
            interaction_pairs=(
                single_run_pairs
            ),
            seed=(
                seeds.final_model_seed
            ),
            config=model_config,
            device=device,
        )
    )

    # =========================================================
    # TEST INTEGRITY
    # =========================================================

    current_test_X = (
        generated.X
        .iloc[
            generated.test_indices
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    current_test_y = np.asarray(
        generated.y[
            generated.test_indices
        ]
    )

    test_integrity_ok = bool(
        test_X_snapshot.equals(
            current_test_X
        )
        and np.array_equal(
            test_y_snapshot,
            current_test_y,
        )
        and np.array_equal(
            prepared
            .final_data
            .y_test,
            test_y_snapshot,
        )
    )

    if not test_integrity_ok:
        raise RuntimeError(
            "Untouched Real-X test "
            "integrity audit failed."
        )

    runtime_seconds = float(
        perf_counter()
        - start_time
    )

    result = (
        RealXIntegrationSanityResult(
            task_id=(
                run_spec.task_id
            ),
            dataset_name=(
                run_spec.dataset_name
            ),
            realization=(
                run_spec.realization
            ),
            strength_name=(
                run_spec.strength_name
            ),
            interaction_coefficient=float(
                run_spec
                .interaction_coefficient
            ),

            n_samples=int(
                len(
                    generated.X
                )
            ),
            n_features=int(
                generated
                .X
                .shape[
                    1
                ]
            ),

            train_size=int(
                len(
                    generated.train_indices
                )
            ),
            validation_size=int(
                len(
                    generated.validation_indices
                )
            ),
            test_size=int(
                len(
                    generated.test_indices
                )
            ),
            discovery_dev_size=int(
                len(
                    prepared.dev_indices
                )
            ),

            discovery_base_seed=(
                seeds.discovery_base_seed
            ),
            random_pair_seed=(
                seeds.random_pair_seed
            ),
            final_model_seed=(
                seeds.final_model_seed
            ),

            expected_candidate_k=(
                expected_candidate_k
            ),
            observed_candidate_k=(
                observed_candidate_k
            ),

            true_pairs=true_pairs,
            selection_pairs=(
                selection_pairs
            ),
            isr_pairs=isr_pairs,
            oracle_pairs=(
                oracle_pairs
            ),
            random_pairs=(
                random_pairs
            ),
            no_isr_pairs=(
                no_isr_pairs
            ),
            single_run_pairs=(
                single_run_pairs
            ),

            mean_pairwise_jaccard=float(
                discovery
                .selection
                .mean_pairwise_jaccard
            ),

            mean_interaction_auprc=float(
                np.mean(
                    ranking_auprcs
                )
            ),

            selection_precision=float(
                selection_recovery
                .precision
            ),
            selection_recall=float(
                selection_recovery
                .recall
            ),
            selection_f1=float(
                selection_recovery
                .f1
            ),

            isr_precision=float(
                isr_recovery
                .precision
            ),
            isr_recall=float(
                isr_recovery
                .recall
            ),
            isr_f1=float(
                isr_recovery
                .f1
            ),

            false_positive_reduction=float(
                false_positive_reduction
            ),

            main_auroc=float(
                main.auroc
            ),
            main_auprc=float(
                main.auprc
            ),
            main_balanced_accuracy=float(
                main
                .balanced_accuracy
            ),
            main_f1=float(
                main.f1
            ),

            agnam_auroc=float(
                full.auroc
            ),
            agnam_auprc=float(
                full.auprc
            ),
            agnam_balanced_accuracy=float(
                full
                .balanced_accuracy
            ),
            agnam_f1=float(
                full.f1
            ),

            oracle_auroc=float(
                oracle.auroc
            ),
            oracle_auprc=float(
                oracle.auprc
            ),

            random_pair_auroc=float(
                random_model.auroc
            ),
            random_pair_auprc=float(
                random_model.auprc
            ),

            no_isr_auroc=float(
                no_isr.auroc
            ),
            no_isr_auprc=float(
                no_isr.auprc
            ),

            single_run_auroc=float(
                single_run.auroc
            ),
            single_run_auprc=float(
                single_run.auprc
            ),

            delta_auroc=float(
                full.auroc
                - main.auroc
            ),
            delta_auprc=float(
                full.auprc
                - main.auprc
            ),
            delta_balanced_accuracy=float(
                full.balanced_accuracy
                - main.balanced_accuracy
            ),
            delta_f1=float(
                full.f1
                - main.f1
            ),

            max_decomposition_error=float(
                max_decomposition_error
            ),

            test_integrity_ok=(
                test_integrity_ok
            ),

            publication_eligible=False,

            runtime_seconds=(
                runtime_seconds
            ),
        )
    )

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        [
            result.to_record()
        ]
    ).to_csv(
        output_path,
        index=False,
    )

    if verbose:
        print()

        print(
            "=============================================="
        )

        print(
            "PHASE 6D INTEGRATION SUMMARY"
        )

        print(
            "=============================================="
        )

        print(
            "Candidate K:",
            result.observed_candidate_k,
        )

        print()

        print(
            "TRUE INTERACTIONS"
        )

        for pair in result.true_pairs:
            print(
                " ",
                pair,
            )

        print()

        print(
            "SELECTION-STABLE"
        )

        for pair in result.selection_pairs:
            print(
                " ",
                pair,
            )

        print()

        print(
            "ISR-RETAINED"
        )

        for pair in result.isr_pairs:
            print(
                " ",
                pair,
            )

        print()

        print(
            "STRUCTURE RECOVERY"
        )

        print(
            "  Mean interaction AUPRC:",
            f"{result.mean_interaction_auprc:.4f}",
        )

        print(
            "  Selection precision/recall:",
            f"{result.selection_precision:.4f}",
            "/",
            f"{result.selection_recall:.4f}",
        )

        print(
            "  ISR precision/recall:",
            f"{result.isr_precision:.4f}",
            "/",
            f"{result.isr_recall:.4f}",
        )

        print()

        print(
            "PREDICTIVE TEST AUROC"
        )

        print(
            "  Main NAM:",
            f"{result.main_auroc:.4f}",
        )

        print(
            "  Full AG-NAM:",
            f"{result.agnam_auroc:.4f}",
        )

        print(
            "  Oracle:",
            f"{result.oracle_auroc:.4f}",
        )

        print(
            "  Random Pair:",
            f"{result.random_pair_auroc:.4f}",
        )

        print(
            "  No ISR:",
            f"{result.no_isr_auroc:.4f}",
        )

        print(
            "  Single Run:",
            f"{result.single_run_auroc:.4f}",
        )

        print()

        print(
            "Exact decomposition error:",
            f"{result.max_decomposition_error:.3e}",
        )

        print(
            "Untouched test integrity:",
            result.test_integrity_ok,
        )

        print(
            "Publication eligible:",
            result.publication_eligible,
        )

        print(
            "Runtime (s):",
            f"{result.runtime_seconds:.1f}",
        )

        print()

        print(
            "Saved ignored integration output:"
        )

        print(
            str(
                output_path
            )
        )

        print()

        print(
            "STATUS: REAL-X / AG-NAM "
            "INTEGRATION SANITY PASSED"
        )

        print(
            "WARNING: THIS RUN IS NOT "
            "BENCHMARK EVIDENCE"
        )

    return result