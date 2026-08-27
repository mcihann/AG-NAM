from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from agnam.data.preprocessing import TabularPreprocessor
from agnam.interpretation.interaction_scoring import (
    InteractionScoreResult,
    candidate_count,
    compute_interaction_scores,
)
from agnam.interpretation.selection_reproducibility import (
    Pair,
    SelectionReproducibilityResult,
    summarize_selection_reproducibility,
)
from agnam.models.attention_proposer import (
    ResidualAttentionProposer,
)
from agnam.training.nam_trainer import (
    NAMTrainingConfig,
)
from agnam.training.proposer_trainer import (
    ProposerTrainingConfig,
    ProposerTrainingResult,
    train_residual_attention_proposer,
)
from agnam.training.residual_targets import (
    generate_train_validation_residual_targets,
)
from agnam.utils.reproducibility import seed_everything


@dataclass
class InteractionDiscoveryRun:
    run_index: int
    seed: int

    train_indices: np.ndarray
    validation_indices: np.ndarray
    scoring_indices: np.ndarray

    k: int

    score_result: InteractionScoreResult
    proposer_training_result: ProposerTrainingResult

    @property
    def top_k_pairs(
        self,
    ) -> tuple[Pair, ...]:
        return tuple(
            tuple(
                sorted(
                    (
                        item.feature_j,
                        item.feature_k,
                    )
                )
            )
            for item in (
                self.score_result
                .ranked_interactions[:self.k]
            )
        )


@dataclass
class ReproducibleDiscoveryResult:
    runs: tuple[
        InteractionDiscoveryRun,
        ...
    ]

    selection: SelectionReproducibilityResult


def _with_seed_nam_config(
    config: NAMTrainingConfig,
    seed: int,
) -> NAMTrainingConfig:
    return NAMTrainingConfig(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        batch_size=config.batch_size,
        max_epochs=config.max_epochs,
        patience=config.patience,
        min_delta=config.min_delta,
        seed=seed,
        deterministic=config.deterministic,
    )


def _with_seed_proposer_config(
    config: ProposerTrainingConfig,
    seed: int,
) -> ProposerTrainingConfig:
    return ProposerTrainingConfig(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        batch_size=config.batch_size,
        max_epochs=config.max_epochs,
        patience=config.patience,
        min_delta=config.min_delta,
        seed=seed,
        deterministic=config.deterministic,
    )


def run_single_interaction_discovery(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    run_index: int,
    seed: int,
    numeric_features: Sequence[str] | None = None,
    categorical_features: Sequence[str] | None = None,
    crossfit_n_splits: int = 5,
    main_early_stop_fraction: float = 0.20,
    nam_hidden_dim: int = 64,
    nam_depth: int = 2,
    nam_dropout: float = 0.10,
    categorical_embedding_dim: int = 16,
    proposer_d_model: int = 64,
    proposer_n_heads: int = 4,
    proposer_n_layers: int = 2,
    proposer_dropout: float = 0.10,
    nam_config: NAMTrainingConfig | None = None,
    proposer_config: ProposerTrainingConfig | None = None,
    device: torch.device | None = None,
) -> InteractionDiscoveryRun:
    """
    Run one leakage-safe interaction-discovery resampling.

    Partition:
        60% proposer training
        20% proposer validation
        20% untouched interaction-scoring holdout
    """
    if not isinstance(
        X,
        pd.DataFrame,
    ):
        raise TypeError(
            "X must be a pandas DataFrame."
        )

    y = np.asarray(
        y,
        dtype=np.int64,
    )

    if len(X) != len(y):
        raise ValueError(
            "X and y lengths must match."
        )

    if not np.all(
        np.isin(y, [0, 1])
    ):
        raise ValueError(
            "y must contain only 0 and 1."
        )

    if nam_config is None:
        nam_config = NAMTrainingConfig(
            seed=seed,
        )

    if proposer_config is None:
        proposer_config = (
            ProposerTrainingConfig(
                seed=seed,
            )
        )

    nam_config = _with_seed_nam_config(
        nam_config,
        seed,
    )

    proposer_config = (
        _with_seed_proposer_config(
            proposer_config,
            seed,
        )
    )

    seed_everything(
        seed,
        deterministic=(
            proposer_config.deterministic
        ),
    )

    all_indices = np.arange(
        len(X)
    )

    development_indices, scoring_indices = (
        train_test_split(
            all_indices,
            test_size=0.20,
            random_state=seed,
            stratify=y,
        )
    )

    train_indices, validation_indices = (
        train_test_split(
            development_indices,
            test_size=0.25,
            random_state=seed + 1,
            stratify=y[
                development_indices
            ],
        )
    )

    X_train = (
        X.iloc[
            train_indices
        ]
        .reset_index(
            drop=True
        )
    )

    X_validation = (
        X.iloc[
            validation_indices
        ]
        .reset_index(
            drop=True
        )
    )

    X_scoring = (
        X.iloc[
            scoring_indices
        ]
        .reset_index(
            drop=True
        )
    )

    y_train = y[
        train_indices
    ]

    y_validation = y[
        validation_indices
    ]

    residual_targets = (
        generate_train_validation_residual_targets(
            X_train=X_train,
            y_train=y_train,
            X_validation=X_validation,
            y_validation=y_validation,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            crossfit_n_splits=(
                crossfit_n_splits
            ),
            main_early_stop_fraction=(
                main_early_stop_fraction
            ),
            hidden_dim=nam_hidden_dim,
            depth=nam_depth,
            dropout=nam_dropout,
            categorical_embedding_dim=(
                categorical_embedding_dim
            ),
            training_config=nam_config,
            seed=seed,
            device=device,
        )
    )

    preprocessor = TabularPreprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    train_data = (
        preprocessor.fit_transform(
            X_train
        )
    )

    validation_data = (
        preprocessor.transform(
            X_validation
        )
    )

    scoring_data = (
        preprocessor.transform(
            X_scoring
        )
    )

    seed_everything(
        seed,
        deterministic=(
            proposer_config.deterministic
        ),
    )

    proposer = (
        ResidualAttentionProposer(
            feature_specs=(
                train_data.feature_specs
            ),
            d_model=proposer_d_model,
            n_heads=proposer_n_heads,
            n_layers=proposer_n_layers,
            dropout=proposer_dropout,
        )
    )

    training_result = (
        train_residual_attention_proposer(
            model=proposer,
            train_data=train_data,
            train_residuals=(
                residual_targets
                .train_residuals
            ),
            val_data=validation_data,
            val_residuals=(
                residual_targets
                .validation_residuals
            ),
            config=proposer_config,
            device=device,
        )
    )

    score_result = (
        compute_interaction_scores(
            model=proposer,
            transformed=scoring_data,
            batch_size=max(
                proposer_config.batch_size,
                256,
            ),
            device=device,
        )
    )

    k = candidate_count(
        proposer.n_features
    )

    return InteractionDiscoveryRun(
        run_index=run_index,
        seed=seed,
        train_indices=(
            train_indices.copy()
        ),
        validation_indices=(
            validation_indices.copy()
        ),
        scoring_indices=(
            scoring_indices.copy()
        ),
        k=k,
        score_result=score_result,
        proposer_training_result=(
            training_result
        ),
    )


def run_reproducible_interaction_discovery(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    n_runs: int = 5,
    base_seed: int = 42,
    selection_threshold: float = 0.60,
    numeric_features: Sequence[str] | None = None,
    categorical_features: Sequence[str] | None = None,
    crossfit_n_splits: int = 5,
    main_early_stop_fraction: float = 0.20,
    nam_hidden_dim: int = 64,
    nam_depth: int = 2,
    nam_dropout: float = 0.10,
    categorical_embedding_dim: int = 16,
    proposer_d_model: int = 64,
    proposer_n_heads: int = 4,
    proposer_n_layers: int = 2,
    proposer_dropout: float = 0.10,
    nam_config: NAMTrainingConfig | None = None,
    proposer_config: ProposerTrainingConfig | None = None,
    device: torch.device | None = None,
) -> ReproducibleDiscoveryResult:
    if n_runs < 1:
        raise ValueError(
            "n_runs must be at least 1."
        )

    runs: list[
        InteractionDiscoveryRun
    ] = []

    # Predefined seed schedule:
    # base_seed, base_seed + 1, ...
    for run_index in range(
        n_runs
    ):
        run_seed = (
            base_seed + run_index
        )

        result = (
            run_single_interaction_discovery(
                X=X,
                y=y,
                run_index=run_index,
                seed=run_seed,
                numeric_features=(
                    numeric_features
                ),
                categorical_features=(
                    categorical_features
                ),
                crossfit_n_splits=(
                    crossfit_n_splits
                ),
                main_early_stop_fraction=(
                    main_early_stop_fraction
                ),
                nam_hidden_dim=(
                    nam_hidden_dim
                ),
                nam_depth=nam_depth,
                nam_dropout=nam_dropout,
                categorical_embedding_dim=(
                    categorical_embedding_dim
                ),
                proposer_d_model=(
                    proposer_d_model
                ),
                proposer_n_heads=(
                    proposer_n_heads
                ),
                proposer_n_layers=(
                    proposer_n_layers
                ),
                proposer_dropout=(
                    proposer_dropout
                ),
                nam_config=nam_config,
                proposer_config=(
                    proposer_config
                ),
                device=device,
            )
        )

        runs.append(
            result
        )

    k_values = {
        run.k
        for run in runs
    }

    if len(k_values) != 1:
        raise RuntimeError(
            "All discovery runs must use the same K."
        )

    k = next(
        iter(k_values)
    )

    rankings = tuple(
        run.score_result.ranked_interactions
        for run in runs
    )

    selection = (
        summarize_selection_reproducibility(
            rankings=rankings,
            k=k,
            threshold=(
                selection_threshold
            ),
        )
    )

    return ReproducibleDiscoveryResult(
        runs=tuple(
            runs
        ),
        selection=selection,
    )