import numpy as np
import torch
from sklearn.model_selection import train_test_split

from agnam.data.preprocessing import TabularPreprocessor
from agnam.data.synthetic import generate_s1
from agnam.models.attention_proposer import ResidualAttentionProposer
from agnam.training.proposer_trainer import (
    ProposerTrainingConfig,
    compute_residual_metrics,
    predict_residuals,
    train_residual_attention_proposer,
)


def prepare_small_problem():
    dataset = generate_s1(
        seed=42,
        n=600,
    )

    residual_target = (
        dataset.y.to_numpy().astype(np.float32)
        - 0.5
    )

    train_idx, val_idx = train_test_split(
        np.arange(len(dataset.X)),
        test_size=0.25,
        random_state=42,
        stratify=dataset.y,
    )

    X_train = dataset.X.iloc[
        train_idx
    ].reset_index(drop=True)

    X_val = dataset.X.iloc[
        val_idx
    ].reset_index(drop=True)

    r_train = residual_target[
        train_idx
    ]

    r_val = residual_target[
        val_idx
    ]

    preprocessor = TabularPreprocessor()

    train_data = (
        preprocessor.fit_transform(
            X_train
        )
    )

    val_data = (
        preprocessor.transform(
            X_val
        )
    )

    return (
        train_data,
        r_train,
        val_data,
        r_val,
    )


def test_compute_residual_metrics_perfect_prediction():
    target = np.array(
        [-0.5, 0.2, 0.7, -0.1]
    )

    metrics = compute_residual_metrics(
        residual_true=target,
        residual_pred=target.copy(),
    )

    assert np.isclose(
        metrics["mse"],
        0.0,
    )

    assert np.isclose(
        metrics["r2"],
        1.0,
    )

    assert np.isclose(
        metrics["correlation"],
        1.0,
    )


def test_predict_residuals_shape():
    (
        train_data,
        _,
        val_data,
        _,
    ) = prepare_small_problem()

    model = ResidualAttentionProposer(
        feature_specs=train_data.feature_specs,
        d_model=16,
        n_heads=4,
        n_layers=1,
        dropout=0.0,
    )

    predictions = predict_residuals(
        model=model,
        transformed=val_data,
        device=torch.device("cpu"),
    )

    assert predictions.shape == (
        len(val_data),
    )

    assert np.isfinite(
        predictions
    ).all()


def test_training_runs_and_returns_finite_metrics():
    (
        train_data,
        r_train,
        val_data,
        r_val,
    ) = prepare_small_problem()

    model = ResidualAttentionProposer(
        feature_specs=train_data.feature_specs,
        d_model=16,
        n_heads=4,
        n_layers=1,
        dropout=0.0,
    )

    config = ProposerTrainingConfig(
        learning_rate=1e-3,
        weight_decay=1e-5,
        batch_size=128,
        max_epochs=5,
        patience=5,
        seed=42,
    )

    result = train_residual_attention_proposer(
        model=model,
        train_data=train_data,
        train_residuals=r_train,
        val_data=val_data,
        val_residuals=r_val,
        config=config,
        device=torch.device("cpu"),
    )

    assert result.best_epoch >= 0
    assert np.isfinite(
        result.best_validation_mse
    )

    for value in result.metrics.values():
        assert (
            np.isfinite(value)
            or np.isnan(value)
        )


def test_training_is_reproducible_on_cpu():
    (
        train_data,
        r_train,
        val_data,
        r_val,
    ) = prepare_small_problem()

    model_1 = ResidualAttentionProposer(
        feature_specs=train_data.feature_specs,
        d_model=16,
        n_heads=4,
        n_layers=1,
        dropout=0.0,
    )

    model_2 = ResidualAttentionProposer(
        feature_specs=train_data.feature_specs,
        d_model=16,
        n_heads=4,
        n_layers=1,
        dropout=0.0,
    )

    model_2.load_state_dict(
        model_1.state_dict()
    )

    config = ProposerTrainingConfig(
        learning_rate=1e-3,
        weight_decay=1e-5,
        batch_size=128,
        max_epochs=4,
        patience=4,
        seed=123,
        deterministic=True,
    )

    result_1 = train_residual_attention_proposer(
        model=model_1,
        train_data=train_data,
        train_residuals=r_train,
        val_data=val_data,
        val_residuals=r_val,
        config=config,
        device=torch.device("cpu"),
    )

    result_2 = train_residual_attention_proposer(
        model=model_2,
        train_data=train_data,
        train_residuals=r_train,
        val_data=val_data,
        val_residuals=r_val,
        config=config,
        device=torch.device("cpu"),
    )

    np.testing.assert_allclose(
        result_1.history["train_mse"],
        result_2.history["train_mse"],
        atol=1e-7,
        rtol=1e-6,
    )

    np.testing.assert_allclose(
        result_1.history["val_mse"],
        result_2.history["val_mse"],
        atol=1e-7,
        rtol=1e-6,
    )


def test_training_target_length_mismatch_raises():
    (
        train_data,
        r_train,
        val_data,
        r_val,
    ) = prepare_small_problem()

    model = ResidualAttentionProposer(
        feature_specs=train_data.feature_specs,
        d_model=16,
        n_heads=4,
        n_layers=1,
        dropout=0.0,
    )

    config = ProposerTrainingConfig(
        max_epochs=2,
        patience=2,
    )

    try:
        train_residual_attention_proposer(
            model=model,
            train_data=train_data,
            train_residuals=r_train[:-1],
            val_data=val_data,
            val_residuals=r_val,
            config=config,
            device=torch.device("cpu"),
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Length mismatch must raise ValueError."
        )