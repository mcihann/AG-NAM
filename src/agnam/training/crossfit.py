from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
)

from agnam.data.preprocessing import TabularPreprocessor
from agnam.models.main_effect_nam import MainEffectNAM
from agnam.training.nam_trainer import (
    NAMTrainingConfig,
    compute_binary_metrics,
    predict_probabilities,
    train_main_effect_nam,
)
from agnam.training.residuals import (
    bernoulli_pseudo_residual_numpy,
)
from agnam.utils.reproducibility import seed_everything


@dataclass
class CrossFitFoldResult:
    fold: int
    holdout_indices: np.ndarray
    best_epoch: int
    holdout_metrics: dict[str, float]


@dataclass
class CrossFittedResidualResult:
    oof_probabilities: np.ndarray
    residuals: np.ndarray
    fold_assignment: np.ndarray
    fold_results: tuple[CrossFitFoldResult, ...]
    overall_metrics: dict[str, float]


def generate_cross_fitted_residuals(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    numeric_features: Sequence[str] | None = None,
    categorical_features: Sequence[str] | None = None,
    n_splits: int = 5,
    early_stop_fraction: float = 0.20,
    hidden_dim: int = 64,
    depth: int = 2,
    dropout: float = 0.10,
    categorical_embedding_dim: int = 16,
    training_config: NAMTrainingConfig | None = None,
    seed: int = 42,
    device: torch.device | None = None,
) -> CrossFittedResidualResult:
    """
    Generate leakage-safe out-of-fold probabilities and pseudo-residuals.

    Every observation receives exactly one probability estimate from a
    MainEffectNAM that was not trained or early-stopped on that observation.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    y = np.asarray(y, dtype=np.int64)

    if len(X) != len(y):
        raise ValueError(
            "X and y must contain the same number of observations."
        )

    if not np.all(np.isin(y, [0, 1])):
        raise ValueError(
            "y must contain only binary labels 0 and 1."
        )

    if n_splits < 2:
        raise ValueError(
            "n_splits must be at least 2."
        )

    if not 0.0 < early_stop_fraction < 0.5:
        raise ValueError(
            "early_stop_fraction must lie between 0 and 0.5."
        )

    if training_config is None:
        training_config = NAMTrainingConfig(
            seed=seed,
        )

    n_samples = len(X)

    oof_probabilities = np.full(
        n_samples,
        np.nan,
        dtype=np.float64,
    )

    fold_assignment = np.full(
        n_samples,
        -1,
        dtype=np.int64,
    )

    assignment_count = np.zeros(
        n_samples,
        dtype=np.int64,
    )

    fold_results: list[CrossFitFoldResult] = []

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    for fold, (
        development_indices,
        holdout_indices,
    ) in enumerate(
        splitter.split(X, y)
    ):
        fold_seed = seed + fold

        seed_everything(
            fold_seed,
            deterministic=training_config.deterministic,
        )

        development_y = y[
            development_indices
        ]

        (
            subtrain_indices,
            early_stop_indices,
        ) = train_test_split(
            development_indices,
            test_size=early_stop_fraction,
            random_state=fold_seed,
            stratify=development_y,
        )

        X_subtrain = X.iloc[
            subtrain_indices
        ].reset_index(drop=True)

        X_early_stop = X.iloc[
            early_stop_indices
        ].reset_index(drop=True)

        X_holdout = X.iloc[
            holdout_indices
        ].reset_index(drop=True)

        y_subtrain = y[
            subtrain_indices
        ]

        y_early_stop = y[
            early_stop_indices
        ]

        y_holdout = y[
            holdout_indices
        ]

        # CRITICAL:
        # Preprocessing is fitted ONLY on the actual subtraining data.
        preprocessor = TabularPreprocessor(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )

        subtrain_data = preprocessor.fit_transform(
            X_subtrain
        )

        early_stop_data = preprocessor.transform(
            X_early_stop
        )

        holdout_data = preprocessor.transform(
            X_holdout
        )

        # Seed BEFORE constructing the model so parameter initialization
        # itself is deterministic.
        seed_everything(
            fold_seed,
            deterministic=training_config.deterministic,
        )

        model = MainEffectNAM(
            feature_specs=subtrain_data.feature_specs,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
            categorical_embedding_dim=(
                categorical_embedding_dim
            ),
        )

        fold_training_config = NAMTrainingConfig(
            learning_rate=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
            batch_size=training_config.batch_size,
            max_epochs=training_config.max_epochs,
            patience=training_config.patience,
            min_delta=training_config.min_delta,
            seed=fold_seed,
            deterministic=training_config.deterministic,
        )

        training_result = train_main_effect_nam(
            model=model,
            train_data=subtrain_data,
            train_y=y_subtrain,
            val_data=early_stop_data,
            val_y=y_early_stop,
            config=fold_training_config,
            device=device,
        )

        holdout_probabilities = predict_probabilities(
            model=model,
            transformed=holdout_data,
            batch_size=max(
                training_config.batch_size,
                1024,
            ),
            device=device,
        )

        if np.any(
            assignment_count[holdout_indices] != 0
        ):
            raise RuntimeError(
                "At least one observation received more than "
                "one out-of-fold prediction."
            )

        oof_probabilities[
            holdout_indices
        ] = holdout_probabilities

        fold_assignment[
            holdout_indices
        ] = fold

        assignment_count[
            holdout_indices
        ] += 1

        holdout_metrics = compute_binary_metrics(
            y_true=y_holdout,
            probabilities=holdout_probabilities,
        )

        fold_results.append(
            CrossFitFoldResult(
                fold=fold,
                holdout_indices=holdout_indices.copy(),
                best_epoch=training_result.best_epoch,
                holdout_metrics=holdout_metrics,
            )
        )

    if not np.all(
        assignment_count == 1
    ):
        raise RuntimeError(
            "Every observation must receive exactly one "
            "out-of-fold prediction."
        )

    if not np.isfinite(
        oof_probabilities
    ).all():
        raise RuntimeError(
            "Out-of-fold probabilities contain missing "
            "or non-finite values."
        )

    residuals = (
        bernoulli_pseudo_residual_numpy(
            y_true=y,
            probabilities=oof_probabilities,
        )
    )

    overall_metrics = compute_binary_metrics(
        y_true=y,
        probabilities=oof_probabilities,
    )

    return CrossFittedResidualResult(
        oof_probabilities=oof_probabilities,
        residuals=residuals,
        fold_assignment=fold_assignment,
        fold_results=tuple(fold_results),
        overall_metrics=overall_metrics,
    )