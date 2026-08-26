import numpy as np
import torch

from agnam.data.synthetic import generate_s1
from agnam.training.crossfit import (
    generate_cross_fitted_residuals,
)
from agnam.training.nam_trainer import (
    NAMTrainingConfig,
)


def small_training_config(seed=42):
    return NAMTrainingConfig(
        learning_rate=1e-3,
        weight_decay=1e-5,
        batch_size=128,
        max_epochs=4,
        patience=4,
        seed=seed,
        deterministic=True,
    )


def test_crossfit_covers_every_observation_once():
    dataset = generate_s1(
        seed=42,
        n=600,
    )

    result = generate_cross_fitted_residuals(
        X=dataset.X,
        y=dataset.y.to_numpy(),
        n_splits=3,
        hidden_dim=8,
        depth=1,
        dropout=0.0,
        training_config=small_training_config(),
        seed=42,
        device=torch.device("cpu"),
    )

    assert len(
        result.oof_probabilities
    ) == 600

    assert len(
        result.fold_assignment
    ) == 600

    assert np.all(
        result.fold_assignment >= 0
    )

    assert set(
        result.fold_assignment
    ) == {0, 1, 2}


def test_holdout_folds_are_disjoint_and_complete():
    dataset = generate_s1(
        seed=42,
        n=600,
    )

    result = generate_cross_fitted_residuals(
        X=dataset.X,
        y=dataset.y.to_numpy(),
        n_splits=3,
        hidden_dim=8,
        depth=1,
        dropout=0.0,
        training_config=small_training_config(),
        seed=42,
        device=torch.device("cpu"),
    )

    all_indices = np.concatenate(
        [
            fold.holdout_indices
            for fold in result.fold_results
        ]
    )

    assert len(
        np.unique(all_indices)
    ) == 600

    np.testing.assert_array_equal(
        np.sort(all_indices),
        np.arange(600),
    )


def test_crossfit_probabilities_and_residuals_are_valid():
    dataset = generate_s1(
        seed=42,
        n=600,
    )

    result = generate_cross_fitted_residuals(
        X=dataset.X,
        y=dataset.y.to_numpy(),
        n_splits=3,
        hidden_dim=8,
        depth=1,
        dropout=0.0,
        training_config=small_training_config(),
        seed=42,
        device=torch.device("cpu"),
    )

    assert np.isfinite(
        result.oof_probabilities
    ).all()

    assert np.all(
        result.oof_probabilities >= 0.0
    )

    assert np.all(
        result.oof_probabilities <= 1.0
    )

    assert np.isfinite(
        result.residuals
    ).all()

    assert np.all(
        result.residuals >= -1.0
    )

    assert np.all(
        result.residuals <= 1.0
    )


def test_crossfit_is_reproducible():
    dataset = generate_s1(
        seed=123,
        n=500,
    )

    kwargs = dict(
        X=dataset.X,
        y=dataset.y.to_numpy(),
        n_splits=3,
        hidden_dim=8,
        depth=1,
        dropout=0.0,
        training_config=small_training_config(
            seed=123
        ),
        seed=123,
        device=torch.device("cpu"),
    )

    first = generate_cross_fitted_residuals(
        **kwargs
    )

    second = generate_cross_fitted_residuals(
        **kwargs
    )

    np.testing.assert_allclose(
        first.oof_probabilities,
        second.oof_probabilities,
        atol=1e-7,
        rtol=1e-6,
    )

    np.testing.assert_allclose(
        first.residuals,
        second.residuals,
        atol=1e-7,
        rtol=1e-6,
    )

    np.testing.assert_array_equal(
        first.fold_assignment,
        second.fold_assignment,
    )