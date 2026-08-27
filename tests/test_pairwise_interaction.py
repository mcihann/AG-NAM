import numpy as np
import pandas as pd
import torch

from agnam.data.preprocessing import (
    TabularPreprocessor,
)
from agnam.models.pairwise_interaction import (
    PairwiseResidualNetwork,
)
from agnam.training.pairwise_trainer import (
    PairwiseTrainingConfig,
    predict_pairwise_residuals,
    train_pairwise_residual_network,
)


def make_pair_data():
    return pd.DataFrame(
        {
            "x1": np.linspace(
                -1.0,
                1.0,
                100,
            ),
            "x2": np.linspace(
                1.0,
                -1.0,
                100,
            ),
        }
    )


def test_pairwise_output_shape():
    X = make_pair_data()

    preprocessor = (
        TabularPreprocessor()
    )

    transformed = (
        preprocessor.fit_transform(X)
    )

    model = (
        PairwiseResidualNetwork(
            feature_specs=(
                transformed
                .feature_specs
            ),
            feature_embedding_dim=8,
            hidden_dim=16,
            depth=1,
            dropout=0.0,
        )
    )

    numeric = torch.from_numpy(
        transformed.numeric
    )

    missing = torch.from_numpy(
        transformed.numeric_missing
    )

    categorical = torch.from_numpy(
        transformed.categorical
    )

    prediction = model(
        numeric=numeric,
        numeric_missing=missing,
        categorical=categorical,
    )

    assert prediction.shape == (
        100,
    )


def test_pairwise_training_runs():
    X = make_pair_data()

    target = (
        X["x1"].to_numpy()
        * X["x2"].to_numpy()
    ).astype(
        np.float32
    )

    preprocessor = (
        TabularPreprocessor()
    )

    transformed = (
        preprocessor.fit_transform(X)
    )

    model = (
        PairwiseResidualNetwork(
            feature_specs=(
                transformed
                .feature_specs
            ),
            feature_embedding_dim=8,
            hidden_dim=16,
            depth=1,
            dropout=0.0,
        )
    )

    config = (
        PairwiseTrainingConfig(
            learning_rate=1e-3,
            batch_size=64,
            max_epochs=3,
            patience=3,
            seed=42,
        )
    )

    result = (
        train_pairwise_residual_network(
            model=model,
            train_data=transformed,
            train_residuals=target,
            val_data=transformed,
            val_residuals=target,
            config=config,
            device=torch.device(
                "cpu"
            ),
        )
    )

    assert result.best_epoch >= 0

    prediction = (
        predict_pairwise_residuals(
            model=model,
            transformed=transformed,
            device=torch.device(
                "cpu"
            ),
        )
    )

    assert prediction.shape == (
        100,
    )

    assert np.isfinite(
        prediction
    ).all()