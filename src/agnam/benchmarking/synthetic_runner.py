from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from agnam.benchmarking.synthetic_protocol import (
    SyntheticBenchmarkProtocol,
    SyntheticBenchmarkRecord,
    SyntheticRealizationSpec,
    candidate_count_from_feature_count,
    compute_false_positive_reduction,
    evaluate_pair_set_recovery,
)
from agnam.data.preprocessing import TabularPreprocessor
from agnam.data.synthetic import (
    generate_s1,
    generate_s2,
    generate_s3,
    generate_s4,
)
from agnam.interpretation.interaction_metrics import (
    canonical_pair,
    evaluate_interaction_ranking,
)
from agnam.models.agnam import AGNAM
from agnam.models.main_effect_nam import MainEffectNAM
from agnam.training.agnam_trainer import (
    AGNAMTrainingConfig,
    compute_agnam_binary_metrics,
    predict_agnam_probabilities,
    train_agnam,
)
from agnam.training.discovery import (
    ReproducibleDiscoveryResult,
    run_reproducible_interaction_discovery,
)
from agnam.training.isr import (
    ISRConfig,
    ISRDiscoveryResult,
    evaluate_isr_for_discovery,
)
from agnam.training.nam_trainer import (
    NAMTrainingConfig,
    compute_binary_metrics,
    predict_probabilities,
    train_main_effect_nam,
)
from agnam.training.proposer_trainer import (
    ProposerTrainingConfig,
)
from agnam.utils.reproducibility import (
    get_device,
    seed_everything,
)


Pair = tuple[str, str]


@dataclass(frozen=True)
class SyntheticModelConfig:
    """
    Locked model/training configuration for primary synthetic runs.
    """

    nam_hidden_dim: int = 64
    nam_depth: int = 2
    nam_dropout: float = 0.10
    categorical_embedding_dim: int = 16

    proposer_d_model: int = 64
    proposer_n_heads: int = 4
    proposer_n_layers: int = 2
    proposer_dropout: float = 0.10

    interaction_embedding_dim: int = 16
    interaction_hidden_dim: int = 64
    interaction_depth: int = 2
    interaction_dropout: float = 0.10

    discovery_learning_rate: float = 1e-3
    discovery_weight_decay: float = 1e-5
    discovery_batch_size: int = 256

    nam_discovery_max_epochs: int = 100
    nam_discovery_patience: int = 15

    proposer_max_epochs: int = 200
    proposer_patience: int = 20

    pairwise_max_epochs: int = 200
    pairwise_patience: int = 20

    final_learning_rate: float = 1e-3
    final_weight_decay: float = 1e-5
    final_batch_size: int = 256
    final_max_epochs: int = 300
    final_patience: int = 30


@dataclass
class FinalPredictionData:
    train_data: object
    validation_data: object
    test_data: object

    y_train: np.ndarray
    y_validation: np.ndarray
    y_test: np.ndarray


@dataclass(frozen=True)
class ModelEvaluation:
    auroc: float
    auprc: float
    balanced_accuracy: float
    f1: float

    best_epoch: int


@dataclass
class SingleSyntheticBenchmarkResult:
    record: SyntheticBenchmarkRecord

    true_pairs: tuple[Pair, ...]

    selection_pairs: tuple[Pair, ...]
    isr_pairs: tuple[Pair, ...]
    oracle_pairs: tuple[Pair, ...]
    random_pairs: tuple[Pair, ...]
    single_run_pairs: tuple[Pair, ...]

    discovery: ReproducibleDiscoveryResult
    isr: ISRDiscoveryResult


def generate_synthetic_scenario(
    scenario: str,
    *,
    seed: int,
):
    """
    Generate one locked synthetic scenario realization.
    """
    scenario = scenario.upper()

    if scenario == "S1":
        return generate_s1(
            seed=seed,
            n=5000,
        )

    if scenario == "S2":
        return generate_s2(
            seed=seed,
            n=8000,
        )

    if scenario == "S3":
        return generate_s3(
            seed=seed,
            n=8000,
        )

    if scenario == "S4":
        return generate_s4(
            seed=seed,
            n=10000,
        )

    raise ValueError(
        f"Unknown synthetic scenario: {scenario}"
    )


def normalize_pairs(
    pairs,
) -> tuple[Pair, ...]:
    """
    Canonicalize and deterministically order feature pairs.
    """
    normalized = {
        canonical_pair(
            first,
            second,
        )
        for first, second
        in pairs
    }

    return tuple(
        sorted(
            normalized
        )
    )


def sample_random_pairs(
    feature_names: tuple[str, ...],
    *,
    n_pairs: int,
    seed: int,
) -> tuple[Pair, ...]:
    """
    Uniformly sample unique non-self feature pairs.

    Ground-truth interaction information is deliberately not used.
    """
    all_pairs = tuple(
        (
            feature_names[first],
            feature_names[second],
        )
        for first, second
        in combinations(
            range(
                len(feature_names)
            ),
            2,
        )
    )

    if n_pairs < 0:
        raise ValueError(
            "n_pairs cannot be negative."
        )

    if n_pairs > len(
        all_pairs
    ):
        raise ValueError(
            "n_pairs exceeds the number of available feature pairs."
        )

    if n_pairs == 0:
        return ()

    rng = np.random.default_rng(
        seed
    )

    selected_indices = rng.choice(
        len(all_pairs),
        size=n_pairs,
        replace=False,
    )

    selected = [
        all_pairs[
            int(index)
        ]
        for index in selected_indices
    ]

    return normalize_pairs(
        selected
    )


def _build_discovery_configs(
    spec: SyntheticRealizationSpec,
    config: SyntheticModelConfig,
):
    nam_config = NAMTrainingConfig(
        learning_rate=(
            config.discovery_learning_rate
        ),
        weight_decay=(
            config.discovery_weight_decay
        ),
        batch_size=(
            config.discovery_batch_size
        ),
        max_epochs=(
            config.nam_discovery_max_epochs
        ),
        patience=(
            config.nam_discovery_patience
        ),
        seed=(
            spec.discovery_base_seed
        ),
        deterministic=True,
    )

    proposer_config = (
        ProposerTrainingConfig(
            learning_rate=(
                config.discovery_learning_rate
            ),
            weight_decay=(
                config.discovery_weight_decay
            ),
            batch_size=(
                config.discovery_batch_size
            ),
            max_epochs=(
                config.proposer_max_epochs
            ),
            patience=(
                config.proposer_patience
            ),
            seed=(
                spec.discovery_base_seed
            ),
            deterministic=True,
        )
    )

    return (
        nam_config,
        proposer_config,
    )


def _build_isr_config(
    protocol: SyntheticBenchmarkProtocol,
    config: SyntheticModelConfig,
) -> ISRConfig:
    return ISRConfig(
        reference_size=(
            protocol.reference_size
        ),
        reference_seed=(
            protocol.reference_seed
        ),
        isr_threshold=(
            protocol.isr_threshold
        ),
        feature_embedding_dim=(
            config.interaction_embedding_dim
        ),
        hidden_dim=(
            config.interaction_hidden_dim
        ),
        depth=(
            config.interaction_depth
        ),
        dropout=(
            config.interaction_dropout
        ),
        learning_rate=(
            config.discovery_learning_rate
        ),
        weight_decay=(
            config.discovery_weight_decay
        ),
        batch_size=(
            config.discovery_batch_size
        ),
        max_epochs=(
            config.pairwise_max_epochs
        ),
        patience=(
            config.pairwise_patience
        ),
        min_delta=1e-5,
        surface_batch_size=8192,
    )


def _prepare_final_prediction_data(
    X_outer_dev: pd.DataFrame,
    y_outer_dev: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    *,
    final_split_seed: int,
) -> FinalPredictionData:
    indices = np.arange(
        len(
            X_outer_dev
        )
    )

    train_idx, validation_idx = (
        train_test_split(
            indices,
            test_size=0.25,
            random_state=(
                final_split_seed
            ),
            stratify=(
                y_outer_dev
            ),
        )
    )

    X_train = (
        X_outer_dev
        .iloc[
            train_idx
        ]
        .reset_index(
            drop=True
        )
    )

    X_validation = (
        X_outer_dev
        .iloc[
            validation_idx
        ]
        .reset_index(
            drop=True
        )
    )

    y_train = (
        y_outer_dev[
            train_idx
        ]
    )

    y_validation = (
        y_outer_dev[
            validation_idx
        ]
    )

    preprocessor = (
        TabularPreprocessor()
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

    return FinalPredictionData(
        train_data=train_data,
        validation_data=(
            validation_data
        ),
        test_data=test_data,
        y_train=(
            y_train
        ),
        y_validation=(
            y_validation
        ),
        y_test=(
            np.asarray(
                y_test
            )
        ),
    )


def _train_main_nam(
    data: FinalPredictionData,
    *,
    seed: int,
    config: SyntheticModelConfig,
    device: torch.device,
) -> ModelEvaluation:
    seed_everything(
        seed
    )

    model = MainEffectNAM(
        feature_specs=(
            data
            .train_data
            .feature_specs
        ),
        hidden_dim=(
            config.nam_hidden_dim
        ),
        depth=(
            config.nam_depth
        ),
        dropout=(
            config.nam_dropout
        ),
        categorical_embedding_dim=(
            config
            .categorical_embedding_dim
        ),
    )

    training_config = (
        NAMTrainingConfig(
            learning_rate=(
                config.final_learning_rate
            ),
            weight_decay=(
                config.final_weight_decay
            ),
            batch_size=(
                config.final_batch_size
            ),
            max_epochs=(
                config.final_max_epochs
            ),
            patience=(
                config.final_patience
            ),
            seed=seed,
            deterministic=True,
        )
    )

    training = (
        train_main_effect_nam(
            model=model,
            train_data=(
                data.train_data
            ),
            train_y=(
                data.y_train
            ),
            val_data=(
                data.validation_data
            ),
            val_y=(
                data.y_validation
            ),
            config=(
                training_config
            ),
            device=device,
        )
    )

    probabilities = (
        predict_probabilities(
            model=model,
            transformed=(
                data.test_data
            ),
            device=device,
        )
    )

    metrics = (
        compute_binary_metrics(
            y_true=data.y_test,
            probabilities=(
                probabilities
            ),
        )
    )

    return ModelEvaluation(
        auroc=float(
            metrics["auroc"]
        ),
        auprc=float(
            metrics["auprc"]
        ),
        balanced_accuracy=float(
            metrics[
                "balanced_accuracy"
            ]
        ),
        f1=float(
            metrics["f1"]
        ),
        best_epoch=int(
            training.best_epoch
        ),
    )


def _train_interaction_model(
    data: FinalPredictionData,
    interaction_pairs: tuple[
        Pair,
        ...
    ],
    *,
    seed: int,
    config: SyntheticModelConfig,
    device: torch.device,
) -> tuple[
    ModelEvaluation,
    float,
]:
    """
    Freshly train one final AG-NAM-style classifier.
    """
    seed_everything(
        seed
    )

    model = AGNAM(
        feature_specs=(
            data
            .train_data
            .feature_specs
        ),
        interaction_pairs=(
            interaction_pairs
        ),
        main_hidden_dim=(
            config.nam_hidden_dim
        ),
        main_depth=(
            config.nam_depth
        ),
        main_dropout=(
            config.nam_dropout
        ),
        categorical_embedding_dim=(
            config
            .categorical_embedding_dim
        ),
        interaction_embedding_dim=(
            config
            .interaction_embedding_dim
        ),
        interaction_hidden_dim=(
            config
            .interaction_hidden_dim
        ),
        interaction_depth=(
            config
            .interaction_depth
        ),
        interaction_dropout=(
            config
            .interaction_dropout
        ),
    )

    training_config = (
        AGNAMTrainingConfig(
            learning_rate=(
                config.final_learning_rate
            ),
            weight_decay=(
                config.final_weight_decay
            ),
            batch_size=(
                config.final_batch_size
            ),
            max_epochs=(
                config.final_max_epochs
            ),
            patience=(
                config.final_patience
            ),
            seed=seed,
            deterministic=True,
        )
    )

    training = train_agnam(
        model=model,
        train_data=(
            data.train_data
        ),
        train_y=(
            data.y_train
        ),
        val_data=(
            data.validation_data
        ),
        val_y=(
            data.y_validation
        ),
        config=(
            training_config
        ),
        device=device,
    )

    probabilities = (
        predict_agnam_probabilities(
            model=model,
            transformed=(
                data.test_data
            ),
            device=device,
        )
    )

    metrics = (
        compute_agnam_binary_metrics(
            y_true=(
                data.y_test
            ),
            probabilities=(
                probabilities
            ),
        )
    )

    # ---------------------------------------------------------
    # Exact additive decomposition audit on untouched test
    # ---------------------------------------------------------

    model.eval()

    with torch.no_grad():
        numeric = torch.from_numpy(
            data
            .test_data
            .numeric
        ).to(
            device
        )

        missing = torch.from_numpy(
            data
            .test_data
            .numeric_missing
        ).to(
            device
        )

        categorical = torch.from_numpy(
            data
            .test_data
            .categorical
        ).to(
            device
        )

        output = model(
            numeric=numeric,
            numeric_missing=missing,
            categorical=categorical,
        )

        reconstructed = (
            output.baseline
            + output
            .main_contributions
            .sum(
                dim=1
            )
            + output
            .interaction_contributions
            .sum(
                dim=1
            )
        )

        maximum_error = float(
            torch.max(
                torch.abs(
                    output.logits
                    - reconstructed
                )
            )
            .cpu()
            .item()
        )

    evaluation = ModelEvaluation(
        auroc=float(
            metrics["auroc"]
        ),
        auprc=float(
            metrics["auprc"]
        ),
        balanced_accuracy=float(
            metrics[
                "balanced_accuracy"
            ]
        ),
        f1=float(
            metrics["f1"]
        ),
        best_epoch=int(
            training.best_epoch
        ),
    )

    return (
        evaluation,
        maximum_error,
    )


def _interaction_ranking_auprcs(
    discovery: ReproducibleDiscoveryResult,
    true_pairs: tuple[
        Pair,
        ...
    ],
) -> np.ndarray:
    values = []

    for run in (
        discovery.runs
    ):
        metrics = (
            evaluate_interaction_ranking(
                ranked_interactions=(
                    run
                    .score_result
                    .ranked_interactions
                ),
                true_interactions=(
                    true_pairs
                ),
                k=(
                    run.k
                ),
            )
        )

        values.append(
            metrics.interaction_auprc
        )

    return np.asarray(
        values,
        dtype=np.float64,
    )


def run_single_synthetic_benchmark(
    spec: SyntheticRealizationSpec,
    *,
    protocol: SyntheticBenchmarkProtocol | None = None,
    model_config: SyntheticModelConfig | None = None,
    device: torch.device | None = None,
    verbose: bool = True,
) -> SingleSyntheticBenchmarkResult:
    """
    Run one complete synthetic benchmark realization.

    This is the canonical single-realization engine used later by
    the 80-realization batch runner.
    """
    if protocol is None:
        protocol = (
            SyntheticBenchmarkProtocol()
        )

    if model_config is None:
        model_config = (
            SyntheticModelConfig()
        )

    if device is None:
        from agnam.utils.reproducibility import get_device

        device = get_device()

    start_time = perf_counter()

    # =========================================================
    # DATASET
    # =========================================================

    dataset = (
        generate_synthetic_scenario(
            spec.scenario,
            seed=(
                spec.dataset_seed
            ),
        )
    )

    X = dataset.X

    y = (
        dataset.y
        .to_numpy()
    )

    true_pairs = normalize_pairs(
        dataset.true_interactions
    )

    if verbose:
        print(
            "========================================"
        )

        print(
            "SYNTHETIC BENCHMARK REALIZATION"
        )

        print(
            "========================================"
        )

        print(
            f"Scenario: "
            f"{spec.scenario}"
        )

        print(
            f"Realization: "
            f"{spec.realization_number}"
        )

        print(
            f"Dataset seed: "
            f"{spec.dataset_seed}"
        )

        print(
            f"Device: "
            f"{device}"
        )

    # =========================================================
    # OUTER SPLIT
    # =========================================================

    all_indices = np.arange(
        len(X)
    )

    outer_dev_idx, test_idx = (
        train_test_split(
            all_indices,
            test_size=(
                protocol
                .outer_test_fraction
            ),
            random_state=(
                spec
                .outer_split_seed
            ),
            stratify=y,
        )
    )

    X_outer_dev = (
        X.iloc[
            outer_dev_idx
        ]
        .reset_index(
            drop=True
        )
    )

    y_outer_dev = (
        y[
            outer_dev_idx
        ]
    )

    X_test = (
        X.iloc[
            test_idx
        ]
        .reset_index(
            drop=True
        )
    )

    y_test = (
        y[
            test_idx
        ]
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
    ) = _build_discovery_configs(
        spec=spec,
        config=model_config,
    )

    discovery = (
        run_reproducible_interaction_discovery(
            X=X_outer_dev,
            y=y_outer_dev,
            n_runs=(
                protocol
                .n_discovery_runs
            ),
            base_seed=(
                spec
                .discovery_base_seed
            ),
            selection_threshold=(
                protocol
                .selection_threshold
            ),
            crossfit_n_splits=(
                protocol
                .residual_crossfit_folds
            ),
            main_early_stop_fraction=0.20,
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

    selection_pairs = normalize_pairs(
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

    selection_recovery = (
        evaluate_pair_set_recovery(
            predicted_pairs=(
                selection_pairs
            ),
            true_pairs=(
                true_pairs
            ),
        )
    )

    ranking_auprcs = (
        _interaction_ranking_auprcs(
            discovery=(
                discovery
            ),
            true_pairs=(
                true_pairs
            ),
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
        _build_isr_config(
            protocol=protocol,
            config=model_config,
        )
    )

    isr_result = (
        evaluate_isr_for_discovery(
            X=X_outer_dev,
            discovery=discovery,
            config=isr_config,
            device=device,
            verbose=verbose,
        )
    )

    isr_pairs = normalize_pairs(
        (
            (
                item.feature_j,
                item.feature_k,
            )
            for item
            in isr_result
            .retained_interactions
        )
    )

    isr_recovery = (
        evaluate_pair_set_recovery(
            predicted_pairs=(
                isr_pairs
            ),
            true_pairs=(
                true_pairs
            ),
        )
    )

    false_positive_reduction = (
        compute_false_positive_reduction(
            selection_recovery=(
                selection_recovery
            ),
            isr_recovery=(
                isr_recovery
            ),
        )
    )

    # =========================================================
    # CONTROL INTERACTION SETS
    # =========================================================

    oracle_pairs = (
        true_pairs
    )

    feature_names = tuple(
        X.columns
    )

    random_pairs = (
        sample_random_pairs(
            feature_names=(
                feature_names
            ),
            n_pairs=len(
                isr_pairs
            ),
            seed=(
                spec
                .random_pair_seed
            ),
        )
    )

    single_run_pairs = normalize_pairs(
        discovery
        .runs[0]
        .top_k_pairs
    )

    # no-ISR uses selection-stable set directly.
    no_isr_pairs = (
        selection_pairs
    )

    # =========================================================
    # FINAL PREDICTION DATA
    # =========================================================

    final_data = (
        _prepare_final_prediction_data(
            X_outer_dev=(
                X_outer_dev
            ),
            y_outer_dev=(
                y_outer_dev
            ),
            X_test=(
                X_test
            ),
            y_test=(
                y_test
            ),
            final_split_seed=(
                spec
                .final_split_seed
            ),
        )
    )

    # =========================================================
    # PREDICTIVE MODELS
    # =========================================================

    if verbose:
        print(
            "\n=== FINAL PREDICTIVE MODELS ==="
        )

    if verbose:
        print(
            "1/6 Main-effect NAM"
        )

    main = _train_main_nam(
        data=final_data,
        seed=(
            spec
            .final_model_seed
        ),
        config=model_config,
        device=device,
    )

    if verbose:
        print(
            "2/6 Full AG-NAM"
        )

    full, max_decomposition_error = (
        _train_interaction_model(
            data=final_data,
            interaction_pairs=(
                isr_pairs
            ),
            seed=(
                spec
                .final_model_seed
            ),
            config=model_config,
            device=device,
        )
    )

    if verbose:
        print(
            "3/6 Oracle Interaction NAM"
        )

    oracle, _ = (
        _train_interaction_model(
            data=final_data,
            interaction_pairs=(
                oracle_pairs
            ),
            seed=(
                spec
                .final_model_seed
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
        _train_interaction_model(
            data=final_data,
            interaction_pairs=(
                random_pairs
            ),
            seed=(
                spec
                .final_model_seed
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
        _train_interaction_model(
            data=final_data,
            interaction_pairs=(
                no_isr_pairs
            ),
            seed=(
                spec
                .final_model_seed
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
        _train_interaction_model(
            data=final_data,
            interaction_pairs=(
                single_run_pairs
            ),
            seed=(
                spec
                .final_model_seed
            ),
            config=model_config,
            device=device,
        )
    )

    runtime_seconds = float(
        perf_counter()
        - start_time
    )

    # =========================================================
    # RECORD
    # =========================================================

    candidate_k = (
        candidate_count_from_feature_count(
            n_features=(
                X.shape[1]
            ),
            protocol=(
                protocol
            ),
        )
    )

    record = (
        SyntheticBenchmarkRecord(
            scenario=(
                spec.scenario
            ),
            realization_index=(
                spec.realization_index
            ),
            dataset_seed=(
                spec.dataset_seed
            ),
            outer_split_seed=(
                spec.outer_split_seed
            ),
            final_split_seed=(
                spec.final_split_seed
            ),
            discovery_base_seed=(
                spec.discovery_base_seed
            ),
            random_pair_seed=(
                spec.random_pair_seed
            ),
            final_model_seed=(
                spec.final_model_seed
            ),
            n_samples=int(
                len(X)
            ),
            n_features=int(
                X.shape[1]
            ),
            positive_fraction=float(
                np.mean(
                    y
                )
            ),
            n_true_interactions=int(
                len(
                    true_pairs
                )
            ),
            candidate_k=int(
                candidate_k
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
            std_interaction_auprc=float(
                np.std(
                    ranking_auprcs,
                    ddof=1,
                )
                if len(
                    ranking_auprcs
                ) > 1
                else 0.0
            ),
            n_selection_stable=int(
                len(
                    selection_pairs
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
            n_isr_retained=int(
                len(
                    isr_pairs
                )
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
            delta_auroc=float(
                full.auroc
                - main.auroc
            ),
            delta_auprc=float(
                full.auprc
                - main.auprc
            ),
            delta_balanced_accuracy=float(
                full
                .balanced_accuracy
                - main
                .balanced_accuracy
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
            max_decomposition_error=float(
                max_decomposition_error
            ),
            runtime_seconds=float(
                runtime_seconds
            ),
        )
    )

    if verbose:
        print(
            "\n========================================"
        )

        print(
            "REALIZATION SUMMARY"
        )

        print(
            "========================================"
        )

        print(
            f"True interactions: "
            f"{true_pairs}"
        )

        print(
            f"Selection-stable: "
            f"{selection_pairs}"
        )

        print(
            f"ISR-retained S*: "
            f"{isr_pairs}"
        )

        print(
            f"Random control: "
            f"{random_pairs}"
        )

        print(
            "\nSTRUCTURE RECOVERY"
        )

        print(
            f"Mean interaction AUPRC: "
            f"{record.mean_interaction_auprc:.4f}"
        )

        print(
            f"Selection precision/recall: "
            f"{record.selection_precision:.4f} / "
            f"{record.selection_recall:.4f}"
        )

        print(
            f"ISR precision/recall: "
            f"{record.isr_precision:.4f} / "
            f"{record.isr_recall:.4f}"
        )

        print(
            f"False-positive reduction: "
            f"{record.false_positive_reduction:.4f}"
        )

        print(
            "\nPREDICTIVE TEST AUROC"
        )

        print(
            f"Main NAM:       "
            f"{main.auroc:.4f}"
        )

        print(
            f"Full AG-NAM:    "
            f"{full.auroc:.4f}"
        )

        print(
            f"Oracle:         "
            f"{oracle.auroc:.4f}"
        )

        print(
            f"Random pairs:   "
            f"{random_model.auroc:.4f}"
        )

        print(
            f"No ISR:         "
            f"{no_isr.auroc:.4f}"
        )

        print(
            f"Single run:     "
            f"{single_run.auroc:.4f}"
        )

        print(
            f"\nDelta AG-NAM - NAM: "
            f"{record.delta_auroc:+.4f}"
        )

        print(
            f"Runtime (s): "
            f"{runtime_seconds:.1f}"
        )

    return SingleSyntheticBenchmarkResult(
        record=record,
        true_pairs=(
            true_pairs
        ),
        selection_pairs=(
            selection_pairs
        ),
        isr_pairs=(
            isr_pairs
        ),
        oracle_pairs=(
            oracle_pairs
        ),
        random_pairs=(
            random_pairs
        ),
        single_run_pairs=(
            single_run_pairs
        ),
        discovery=(
            discovery
        ),
        isr=(
            isr_result
        ),
    )