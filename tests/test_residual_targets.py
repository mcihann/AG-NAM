import numpy as np
import torch
from sklearn.model_selection import train_test_split

from agnam.data.synthetic import generate_s1
from agnam.training.nam_trainer import (
    NAMTrainingConfig,
)
from agnam.training.residual_targets import (
    generate_train_validation_residual_targets,
)


def tiny_config(seed=42):
    return NAMTrainingConfig(
        learning_rate=1e-3,
        weight_decay=1e-5,
        batch_size=128,
        max_epochs=3,
        patience=3,
        seed=seed,
        deterministic=True,
    )


def prepare_data():
    dataset = generate_s1(
        seed=42,
        n=500,
    )

    train_idx, val_idx = train_test_split(
        np.arange(len(dataset.X)),
        test_size=0.20,
        random_state=42,
        stratify=dataset.y,
    )

    return (
        dataset.X.iloc[
            train_idx
        ].reset_index(drop=True),
        dataset.y.iloc[
            train_idx
        ].to_numpy(),
        dataset.X.iloc[
            val_idx
        ].reset_index(drop=True),
        dataset.y.iloc[
            val_idx
        ].to_numpy(),
    )


def test_residual_target_shapes_and_ranges():
    (
        X_train,
        y_train,
        X_val,
        y_val,
    ) = prepare_data()

    result = (
        generate_train_validation_residual_targets(
            X_train=X_train,
            y_train=y_train,
            X_validation=X_val,
            y_validation=y_val,
            crossfit_n_splits=3,
            hidden_dim=8,
            depth=1,
            dropout=0.0,
            training_config=tiny_config(),
            seed=42,
            device=torch.device("cpu"),
        )
    )

    assert result.train_residuals.shape == (
        len(X_train),
    )

    assert (
        result.validation_residuals.shape
        == (len(X_val),)
    )

    assert np.isfinite(
        result.train_residuals
    ).all()

    assert np.isfinite(
        result.validation_residuals
    ).all()

    assert np.all(
        result.train_residuals >= -1.0
    )

    assert np.all(
        result.train_residuals <= 1.0
    )

    assert np.all(
        result.validation_residuals >= -1.0
    )

    assert np.all(
        result.validation_residuals <= 1.0
    )


def test_validation_data_do_not_change_training_residuals():
    (
        X_train,
        y_train,
        X_val,
        y_val,
    ) = prepare_data()

    first = (
        generate_train_validation_residual_targets(
            X_train=X_train,
            y_train=y_train,
            X_validation=X_val,
            y_validation=y_val,
            crossfit_n_splits=3,
            hidden_dim=8,
            depth=1,
            dropout=0.0,
            training_config=tiny_config(),
            seed=123,
            device=torch.device("cpu"),
        )
    )

    X_val_modified = X_val.copy()

    X_val_modified.iloc[
        :,
        :
    ] = (
        X_val_modified.to_numpy()
        + 1000.0
    )

    second = (
        generate_train_validation_residual_targets(
            X_train=X_train,
            y_train=y_train,
            X_validation=X_val_modified,
            y_validation=y_val,
            crossfit_n_splits=3,
            hidden_dim=8,
            depth=1,
            dropout=0.0,
            training_config=tiny_config(),
            seed=123,
            device=torch.device("cpu"),
        )
    )

    np.testing.assert_allclose(
        first.train_residuals,
        second.train_residuals,
        atol=1e-7,
        rtol=1e-6,
    )

    np.testing.assert_allclose(
        first.train_oof_probabilities,
        second.train_oof_probabilities,
        atol=1e-7,
        rtol=1e-6,
    )