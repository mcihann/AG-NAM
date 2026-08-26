from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from agnam.data.preprocessing import TabularPreprocessor
from agnam.models.main_effect_nam import MainEffectNAM
from agnam.training.crossfit import (
    CrossFittedResidualResult,
    generate_cross_fitted_residuals,
)
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
class TrainValidationResidualTargets:
    train_residuals: np.ndarray
    validation_residuals: np.ndarray

    train_oof_probabilities: np.ndarray
    validation_probabilities: np.ndarray

    train_crossfit: CrossFittedResidualResult

    validation_main_metrics: dict[str, float]


def generate_train_validation_residual_targets(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_validation: pd.DataFrame,
    y_validation: np.ndarray,
    *,
    numeric_features: Sequence[str] | None = None,
    categorical_features: Sequence[str] | None = None,
    crossfit_n_splits: int = 5,
    main_early_stop_fraction: float = 0.20,
    hidden_dim: int = 64,
    depth: int = 2,
    dropout: float = 0.10,
    categorical_embedding_dim: int = 16,
    training_config: NAMTrainingConfig | None = None,
    seed: int = 42,
    device: torch.device | None = None,
) -> TrainValidationResidualTargets:
    """
    Generate leakage-safe residual targets for proposer training
    and proposer validation.

    Training residuals:
        obtained exclusively through cross-fitting within X_train.

    Validation residuals:
        obtained from a MainEffectNAM trained exclusively on X_train.
        X_validation is never used to construct training residuals.
    """

    if not isinstance(X_train, pd.DataFrame):
        raise TypeError(
            "X_train must be a pandas DataFrame."
        )

    if not isinstance(X_validation, pd.DataFrame):
        raise TypeError(
            "X_validation must be a pandas DataFrame."
        )

    if tuple(X_train.columns) != tuple(
        X_validation.columns
    ):
        raise ValueError(
            "Training and validation columns must match exactly."
        )

    y_train = np.asarray(
        y_train,
        dtype=np.int64,
    )

    y_validation = np.asarray(
        y_validation,
        dtype=np.int64,
    )

    if len(X_train) != len(y_train):
        raise ValueError(
            "X_train and y_train lengths must match."
        )

    if len(X_validation) != len(y_validation):
        raise ValueError(
            "X_validation and y_validation lengths must match."
        )

    if not np.all(
        np.isin(y_train, [0, 1])
    ):
        raise ValueError(
            "y_train must contain only 0 and 1."
        )

    if not np.all(
        np.isin(y_validation, [0, 1])
    ):
        raise ValueError(
            "y_validation must contain only 0 and 1."
        )

    if training_config is None:
        training_config = NAMTrainingConfig(
            seed=seed,
        )

    # ---------------------------------------------------------
    # 1. TRAINING RESIDUALS
    # ---------------------------------------------------------
    # Generated ONLY inside proposer-training data.
    train_crossfit = generate_cross_fitted_residuals(
        X=X_train,
        y=y_train,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        n_splits=crossfit_n_splits,
        early_stop_fraction=main_early_stop_fraction,
        hidden_dim=hidden_dim,
        depth=depth,
        dropout=dropout,
        categorical_embedding_dim=(
            categorical_embedding_dim
        ),
        training_config=training_config,
        seed=seed,
        device=device,
    )

    # ---------------------------------------------------------
    # 2. VALIDATION RESIDUALS
    # ---------------------------------------------------------
    # Fit a fresh MainEffectNAM using ONLY proposer-training data.
    #
    # X_validation is used only after training, for prediction.
    # ---------------------------------------------------------

    indices = np.arange(
        len(X_train)
    )

    (
        subtrain_indices,
        early_stop_indices,
    ) = train_test_split(
        indices,
        test_size=main_early_stop_fraction,
        random_state=seed + 10_000,
        stratify=y_train,
    )

    X_subtrain = X_train.iloc[
        subtrain_indices
    ].reset_index(drop=True)

    X_early_stop = X_train.iloc[
        early_stop_indices
    ].reset_index(drop=True)

    X_validation_local = (
        X_validation.reset_index(
            drop=True
        )
    )

    y_subtrain = y_train[
        subtrain_indices
    ]

    y_early_stop = y_train[
        early_stop_indices
    ]

    preprocessor = TabularPreprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    subtrain_data = (
        preprocessor.fit_transform(
            X_subtrain
        )
    )

    early_stop_data = (
        preprocessor.transform(
            X_early_stop
        )
    )

    validation_data = (
        preprocessor.transform(
            X_validation_local
        )
    )

    validation_seed = (
        seed + 10_000
    )

    seed_everything(
        validation_seed,
        deterministic=(
            training_config.deterministic
        ),
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

    validation_training_config = (
        NAMTrainingConfig(
            learning_rate=(
                training_config.learning_rate
            ),
            weight_decay=(
                training_config.weight_decay
            ),
            batch_size=(
                training_config.batch_size
            ),
            max_epochs=(
                training_config.max_epochs
            ),
            patience=(
                training_config.patience
            ),
            min_delta=(
                training_config.min_delta
            ),
            seed=validation_seed,
            deterministic=(
                training_config.deterministic
            ),
        )
    )

    train_main_effect_nam(
        model=model,
        train_data=subtrain_data,
        train_y=y_subtrain,
        val_data=early_stop_data,
        val_y=y_early_stop,
        config=validation_training_config,
        device=device,
    )

    validation_probabilities = (
        predict_probabilities(
            model=model,
            transformed=validation_data,
            batch_size=max(
                training_config.batch_size,
                1024,
            ),
            device=device,
        )
    )

    validation_residuals = (
        bernoulli_pseudo_residual_numpy(
            y_true=y_validation,
            probabilities=(
                validation_probabilities
            ),
        )
    )

    validation_main_metrics = (
        compute_binary_metrics(
            y_true=y_validation,
            probabilities=(
                validation_probabilities
            ),
        )
    )

    return TrainValidationResidualTargets(
        train_residuals=(
            train_crossfit.residuals
        ),
        validation_residuals=(
            validation_residuals
        ),
        train_oof_probabilities=(
            train_crossfit.oof_probabilities
        ),
        validation_probabilities=(
            validation_probabilities
        ),
        train_crossfit=train_crossfit,
        validation_main_metrics=(
            validation_main_metrics
        ),
    )