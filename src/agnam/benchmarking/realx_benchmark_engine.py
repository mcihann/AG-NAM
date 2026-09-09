from __future__ import annotations

import os

os.environ.setdefault(
    "CUBLAS_WORKSPACE_CONFIG",
    ":4096:8",
)

from dataclasses import dataclass
from time import perf_counter
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

import agnam.benchmarking.synthetic_runner as sr

from agnam.benchmarking.realx_discovery import (
    run_realx_reproducible_interaction_discovery,
)
from agnam.benchmarking.realx_generator import (
    generate_realx_run,
)
from agnam.benchmarking.realx_openml import (
    load_realx_openml_X,
    validate_realx_feature_type,
)
from agnam.benchmarking.realx_protocol import (
    RealXProtocol,
    RealXRunSpec,
    build_realx_schedule,
)
from agnam.benchmarking.realx_runner import (
    derive_realx_model_seeds,
    get_dataset_specification,
    prepare_realx_integration_data,
    realx_candidate_k,
)


SCHEMA_VERSION = "realx_benchmark_v1"


@dataclass(frozen=True)
class RealXBenchmarkModelProtocol:
    isr_reference_size: int = 512
    isr_reference_seed: int = 4026

    main_early_stop_fraction: float = 0.20
    decomposition_tolerance: float = 1e-6

    def __post_init__(
        self,
    ) -> None:
        if self.isr_reference_size != 512:
            raise ValueError(
                "ISR reference size is locked to 512."
            )

        if self.isr_reference_seed != 4026:
            raise ValueError(
                "ISR reference seed is locked to 4026."
            )

        if not np.isclose(
            self.main_early_stop_fraction,
            0.20,
            atol=1e-12,
            rtol=0.0,
        ):
            raise ValueError(
                "Discovery early-stop fraction is locked to 0.20."
            )

        if not np.isclose(
            self.decomposition_tolerance,
            1e-6,
            atol=0.0,
            rtol=0.0,
        ):
            raise ValueError(
                "Decomposition tolerance is locked to 1e-6."
            )


@dataclass
class RealXBenchmarkResult:
    schema_version: str
    benchmark_id: str
    status: str

    run_id: str

    task_id: int
    dataset_name: str
    feature_type: str

    realization: int
    strength_name: str
    interaction_coefficient: float

    n_samples: int
    n_features: int

    positive_fraction: float
    expected_prevalence: float

    train_size: int
    validation_size: int
    test_size: int
    discovery_dev_size: int

    discovery_base_seed: int
    random_pair_seed: int
    final_model_seed: int

    n_true_interactions: int

    expected_candidate_k: int
    observed_candidate_k: int

    template_true_pairs: tuple[
        tuple[str, str],
        ...,
    ]

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
    std_interaction_auprc: float

    n_selection_stable: int

    selection_precision: float
    selection_recall: float
    selection_f1: float

    n_isr_retained: int

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

    delta_auroc: float
    delta_auprc: float
    delta_balanced_accuracy: float
    delta_f1: float

    oracle_auroc: float
    oracle_auprc: float

    random_pair_auroc: float
    random_pair_auprc: float

    no_isr_auroc: float
    no_isr_auprc: float

    single_run_auroc: float
    single_run_auprc: float

    max_decomposition_error: float

    test_integrity_ok: bool
    publication_eligible: bool

    runtime_seconds: float

    @staticmethod
    def _serialize_pairs(
        pairs,
    ) -> str:
        return "|".join(
            f"{first}::{second}"
            for (
                first,
                second,
            ) in pairs
        )

    def to_record(
        self,
    ) -> dict[str, Any]:
        record = dict(
            vars(
                self
            )
        )

        for name in (
            "template_true_pairs",
            "true_pairs",
            "selection_pairs",
            "isr_pairs",
            "oracle_pairs",
            "random_pairs",
            "no_isr_pairs",
            "single_run_pairs",
        ):
            record[
                name
            ] = self._serialize_pairs(
                getattr(
                    self,
                    name,
                )
            )

        return record


def _verify_locked_run_spec(
    run_spec: RealXRunSpec,
    protocol: RealXProtocol,
) -> None:
    schedule = build_realx_schedule(
        protocol
    )

    matches = [
        item
        for item in schedule
        if item.run_id
        == run_spec.run_id
    ]

    if len(
        matches
    ) != 1:
        raise RuntimeError(
            f"Run {run_spec.run_id} is not present "
            "exactly once in the locked schedule."
        )

    if matches[
        0
    ] != run_spec:
        raise RuntimeError(
            "Requested run specification differs "
            "from the locked schedule."
        )


def _build_discovery_configs(
    *,
    discovery_seed: int,
    model_config,
):
    adapter = SimpleNamespace(
        discovery_base_seed=(
            discovery_seed
        )
    )

    return sr._build_discovery_configs(
        spec=adapter,
        config=model_config,
    )


def _build_isr_config(
    *,
    realx_protocol: RealXProtocol,
    benchmark_protocol: RealXBenchmarkModelProtocol,
    model_config,
):
    adapter = SimpleNamespace(
        reference_size=(
            benchmark_protocol
            .isr_reference_size
        ),
        reference_seed=(
            benchmark_protocol
            .isr_reference_seed
        ),
        isr_threshold=(
            realx_protocol
            .isr_threshold
        ),
    )

    return sr._build_isr_config(
        protocol=adapter,
        config=model_config,
    )


def _recovery_metrics(
    predicted_pairs,
    true_pairs,
):
    predicted_pairs = set(
        predicted_pairs
    )

    true_pairs = set(
        true_pairs
    )

    true_positive = len(
        predicted_pairs
        & true_pairs
    )

    false_positive = len(
        predicted_pairs
        - true_pairs
    )

    false_negative = len(
        true_pairs
        - predicted_pairs
    )

    if len(
        true_pairs
    ) == 0:
        return {
            "precision": float(
                "nan"
            ),
            "recall": float(
                "nan"
            ),
            "f1": float(
                "nan"
            ),
            "true_positive": (
                true_positive
            ),
            "false_positive": (
                false_positive
            ),
            "false_negative": (
                false_negative
            ),
        }

    precision_denominator = (
        true_positive
        + false_positive
    )

    precision = (
        true_positive
        / precision_denominator
        if precision_denominator
        > 0
        else 0.0
    )

    recall = (
        true_positive
        / len(
            true_pairs
        )
    )

    f1 = (
        2.0
        * precision
        * recall
        / (
            precision
            + recall
        )
        if (
            precision
            + recall
        )
        > 0.0
        else 0.0
    )

    return {
        "precision": float(
            precision
        ),
        "recall": float(
            recall
        ),
        "f1": float(
            f1
        ),
        "true_positive": (
            true_positive
        ),
        "false_positive": (
            false_positive
        ),
        "false_negative": (
            false_negative
        ),
    }


def _false_positive_reduction(
    selection_metrics,
    isr_metrics,
) -> float:
    selection_false_positive = int(
        selection_metrics[
            "false_positive"
        ]
    )

    isr_false_positive = int(
        isr_metrics[
            "false_positive"
        ]
    )

    if selection_false_positive == 0:
        return float(
            "nan"
        )

    if (
        isr_false_positive
        > selection_false_positive
    ):
        raise RuntimeError(
            "ISR false-positive count exceeds "
            "selection-stage false-positive count."
        )

    return float(
        (
            selection_false_positive
            - isr_false_positive
        )
        / selection_false_positive
    )


def run_realx_benchmark_spec(
    run_spec: RealXRunSpec,
    *,
    realx_protocol: RealXProtocol | None = None,
    benchmark_protocol: RealXBenchmarkModelProtocol | None = None,
    model_config=None,
    device: torch.device | None = None,
    verbose: bool = True,
) -> RealXBenchmarkResult:
    if realx_protocol is None:
        realx_protocol = (
            RealXProtocol()
        )

    if benchmark_protocol is None:
        benchmark_protocol = (
            RealXBenchmarkModelProtocol()
        )

    if model_config is None:
        model_config = (
            sr.SyntheticModelConfig()
        )

    _verify_locked_run_spec(
        run_spec,
        realx_protocol,
    )

    if device is None:
        from agnam.utils.reproducibility import (
            get_device,
        )

        device = get_device()

    start_time = perf_counter()

    dataset_spec = (
        get_dataset_specification(
            run_spec.task_id
        )
    )

    loaded = load_realx_openml_X(
        dataset_spec
    )

    validate_realx_feature_type(
        loaded
    )

    generated = generate_realx_run(
        loaded.X,
        run_spec,
        protocol=realx_protocol,
    )

    prepared = (
        prepare_realx_integration_data(
            generated
        )
    )

    model_seeds = (
        derive_realx_model_seeds(
            run_spec
        )
    )

    template_true_pairs = (
        sr.normalize_pairs(
            generated
            .template_true_pairs
        )
    )

    true_pairs = sr.normalize_pairs(
        generated.active_true_pairs
    )

    expected_candidate_k = (
        realx_candidate_k(
            generated.X.shape[
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
            "LOCKED REAL-X BENCHMARK RUN"
        )

        print(
            "=============================================="
        )

        print(
            f"Run ID: {run_spec.run_id}"
        )

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
            run_spec.interaction_coefficient,
        )

        print(
            f"Device: {device}"
        )

        print(
            f"Candidate K: {expected_candidate_k}"
        )

    (
        nam_discovery_config,
        proposer_config,
    ) = _build_discovery_configs(
        discovery_seed=(
            model_seeds
            .discovery_base_seed
        ),
        model_config=model_config,
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
                model_seeds
                .discovery_base_seed
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
                benchmark_protocol
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
        for run in (
            discovery.runs
        )
    }

    if observed_k_values != {
        expected_candidate_k
    }:
        raise RuntimeError(
            "Locked candidate-K audit failed. "
            f"Expected {expected_candidate_k}, "
            f"observed {sorted(observed_k_values)}."
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

    ranking_auprcs = []

    if len(
        true_pairs
    ) > 0:
        ranking_auprcs = (
            sr._interaction_ranking_auprcs(
                discovery=discovery,
                true_pairs=true_pairs,
            )
        )

    selection_metrics = (
        _recovery_metrics(
            selection_pairs,
            true_pairs,
        )
    )

    isr_config = (
        _build_isr_config(
            realx_protocol=(
                realx_protocol
            ),
            benchmark_protocol=(
                benchmark_protocol
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
            "ISR retained a pair outside "
            "the selection-stable set."
        )

    isr_metrics = (
        _recovery_metrics(
            isr_pairs,
            true_pairs,
        )
    )

    false_positive_reduction = (
        _false_positive_reduction(
            selection_metrics,
            isr_metrics,
        )
    )

    oracle_pairs = (
        true_pairs
    )

    random_pairs = (
        sr.sample_random_pairs(
            feature_names=tuple(
                generated.X.columns
            ),
            n_pairs=len(
                isr_pairs
            ),
            seed=(
                model_seeds
                .random_pair_seed
            ),
        )
    )

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
            model_seeds
            .final_model_seed
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
        full_error,
    ) = sr._train_interaction_model(
        data=prepared.final_data,
        interaction_pairs=isr_pairs,
        seed=(
            model_seeds
            .final_model_seed
        ),
        config=model_config,
        device=device,
    )

    if verbose:
        print(
            "3/6 Oracle Interaction NAM"
        )

    (
        oracle,
        oracle_error,
    ) = sr._train_interaction_model(
        data=prepared.final_data,
        interaction_pairs=(
            oracle_pairs
        ),
        seed=(
            model_seeds
            .final_model_seed
        ),
        config=model_config,
        device=device,
    )

    if verbose:
        print(
            "4/6 Random-Pair NAM"
        )

    (
        random_model,
        random_error,
    ) = sr._train_interaction_model(
        data=prepared.final_data,
        interaction_pairs=(
            random_pairs
        ),
        seed=(
            model_seeds
            .final_model_seed
        ),
        config=model_config,
        device=device,
    )

    if verbose:
        print(
            "5/6 AG-NAM without ISR"
        )

    (
        no_isr,
        no_isr_error,
    ) = sr._train_interaction_model(
        data=prepared.final_data,
        interaction_pairs=(
            no_isr_pairs
        ),
        seed=(
            model_seeds
            .final_model_seed
        ),
        config=model_config,
        device=device,
    )

    if verbose:
        print(
            "6/6 Single-Run AG-NAM"
        )

    (
        single_run,
        single_run_error,
    ) = sr._train_interaction_model(
        data=prepared.final_data,
        interaction_pairs=(
            single_run_pairs
        ),
        seed=(
            model_seeds
            .final_model_seed
        ),
        config=model_config,
        device=device,
    )

    max_decomposition_error = float(
        max(
            full_error,
            oracle_error,
            random_error,
            no_isr_error,
            single_run_error,
        )
    )

    if (
        max_decomposition_error
        > benchmark_protocol
        .decomposition_tolerance
    ):
        raise RuntimeError(
            "Exact decomposition audit failed. "
            f"Maximum error: "
            f"{max_decomposition_error:.3e}"
        )

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
            "Untouched test integrity audit failed."
        )

    runtime_seconds = float(
        perf_counter()
        - start_time
    )

    if len(
        ranking_auprcs
    ) > 0:
        mean_interaction_auprc = float(
            np.mean(
                ranking_auprcs
            )
        )

        std_interaction_auprc = float(
            np.std(
                ranking_auprcs,
                ddof=1,
            )
            if len(
                ranking_auprcs
            ) > 1
            else 0.0
        )

    else:
        mean_interaction_auprc = float(
            "nan"
        )

        std_interaction_auprc = float(
            "nan"
        )

    result = RealXBenchmarkResult(
        schema_version=(
            SCHEMA_VERSION
        ),
        benchmark_id=(
            realx_protocol
            .benchmark_id
        ),
        status="COMPLETE",

        run_id=run_spec.run_id,

        task_id=run_spec.task_id,
        dataset_name=(
            run_spec.dataset_name
        ),
        feature_type=(
            run_spec.feature_type
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
            generated.X.shape[
                1
            ]
        ),

        positive_fraction=float(
            np.mean(
                generated.y
            )
        ),
        expected_prevalence=float(
            generated.expected_prevalence
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
            model_seeds
            .discovery_base_seed
        ),
        random_pair_seed=(
            model_seeds
            .random_pair_seed
        ),
        final_model_seed=(
            model_seeds
            .final_model_seed
        ),

        n_true_interactions=int(
            len(
                true_pairs
            )
        ),

        expected_candidate_k=(
            expected_candidate_k
        ),
        observed_candidate_k=(
            observed_candidate_k
        ),

        template_true_pairs=(
            template_true_pairs
        ),
        true_pairs=true_pairs,
        selection_pairs=(
            selection_pairs
        ),
        isr_pairs=isr_pairs,
        oracle_pairs=oracle_pairs,
        random_pairs=random_pairs,
        no_isr_pairs=no_isr_pairs,
        single_run_pairs=(
            single_run_pairs
        ),

        mean_pairwise_jaccard=float(
            discovery
            .selection
            .mean_pairwise_jaccard
        ),

        mean_interaction_auprc=(
            mean_interaction_auprc
        ),
        std_interaction_auprc=(
            std_interaction_auprc
        ),

        n_selection_stable=int(
            len(
                selection_pairs
            )
        ),

        selection_precision=float(
            selection_metrics[
                "precision"
            ]
        ),
        selection_recall=float(
            selection_metrics[
                "recall"
            ]
        ),
        selection_f1=float(
            selection_metrics[
                "f1"
            ]
        ),

        n_isr_retained=int(
            len(
                isr_pairs
            )
        ),

        isr_precision=float(
            isr_metrics[
                "precision"
            ]
        ),
        isr_recall=float(
            isr_metrics[
                "recall"
            ]
        ),
        isr_f1=float(
            isr_metrics[
                "f1"
            ]
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
            main.balanced_accuracy
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
            full.balanced_accuracy
        ),
        agnam_f1=float(
            full.f1
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

        max_decomposition_error=(
            max_decomposition_error
        ),

        test_integrity_ok=True,
        publication_eligible=True,

        runtime_seconds=(
            runtime_seconds
        ),
    )

    if verbose:
        print(
            "\nRUN COMPLETE"
        )

        print(
            f"Main AUROC: "
            f"{result.main_auroc:.4f}"
        )

        print(
            f"AG-NAM AUROC: "
            f"{result.agnam_auroc:.4f}"
        )

        print(
            f"Delta AUROC: "
            f"{result.delta_auroc:+.4f}"
        )

        print(
            f"ISR retained: "
            f"{result.n_isr_retained}"
        )

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
            f"Runtime: "
            f"{result.runtime_seconds:.1f} s"
        )

    return result