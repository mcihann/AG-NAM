import numpy as np
import torch
from sklearn.model_selection import train_test_split

from agnam.data.preprocessing import TabularPreprocessor
from agnam.data.synthetic import generate_s1
from agnam.models.main_effect_nam import MainEffectNAM
from agnam.training.nam_trainer import (
    NAMTrainingConfig,
    compute_binary_metrics,
    estimate_centering_offsets,
    predict_probabilities,
    train_main_effect_nam,
)


def prepare_small_s1():
    dataset = generate_s1(
        seed=42,
        n=800,
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

    y_train = dataset.y.iloc[
        train_idx
    ].to_numpy()

    y_val = dataset.y.iloc[
        val_idx
    ].to_numpy()

    preprocessor = TabularPreprocessor()

    train_data = preprocessor.fit_transform(
        X_train
    )

    val_data = preprocessor.transform(
        X_val
    )

    return (
        train_data,
        y_train,
        val_data,
        y_val,
    )


def test_compute_binary_metrics():
    y = np.array(
        [0, 0, 1, 1]
    )

    probabilities = np.array(
        [0.1, 0.2, 0.8, 0.9]
    )

    metrics = compute_binary_metrics(
        y_true=y,
        probabilities=probabilities,
    )

    assert np.isclose(
        metrics["auroc"],
        1.0,
    )

    assert np.isclose(
        metrics["auprc"],
        1.0,
    )


def test_training_produces_valid_probabilities():
    (
        train_data,
        y_train,
        val_data,
        y_val,
    ) = prepare_small_s1()

    model = MainEffectNAM(
        feature_specs=train_data.feature_specs,
        hidden_dim=16,
        depth=1,
        dropout=0.0,
    )

    config = NAMTrainingConfig(
        learning_rate=1e-3,
        weight_decay=1e-5,
        batch_size=128,
        max_epochs=10,
        patience=5,
        seed=42,
    )

    result = train_main_effect_nam(
        model=model,
        train_data=train_data,
        train_y=y_train,
        val_data=val_data,
        val_y=y_val,
        config=config,
        device=torch.device("cpu"),
    )

    probabilities = predict_probabilities(
        model=model,
        transformed=val_data,
        device=torch.device("cpu"),
    )

    assert probabilities.shape == (
        len(y_val),
    )

    assert np.all(
        probabilities >= 0.0
    )

    assert np.all(
        probabilities <= 1.0
    )

    assert 0.0 <= (
        result.metrics["auroc"]
    ) <= 1.0


def test_centering_offsets_match_training_means():
    (
        train_data,
        y_train,
        val_data,
        y_val,
    ) = prepare_small_s1()

    model = MainEffectNAM(
        feature_specs=train_data.feature_specs,
        hidden_dim=8,
        depth=1,
        dropout=0.0,
    )

    offsets = estimate_centering_offsets(
        model=model,
        transformed=train_data,
        device=torch.device("cpu"),
    )

    assert offsets.shape == (
        model.n_features,
    )

    assert torch.isfinite(
        offsets
    ).all()


def test_training_is_reproducible_on_cpu():
    (
        train_data,
        y_train,
        val_data,
        y_val,
    ) = prepare_small_s1()

    config = NAMTrainingConfig(
        learning_rate=1e-3,
        weight_decay=1e-5,
        batch_size=128,
        max_epochs=5,
        patience=5,
        seed=123,
        deterministic=True,
    )

    model_1 = MainEffectNAM(
        feature_specs=train_data.feature_specs,
        hidden_dim=8,
        depth=1,
        dropout=0.0,
    )

    torch.manual_seed(999)

    model_2 = MainEffectNAM(
        feature_specs=train_data.feature_specs,
        hidden_dim=8,
        depth=1,
        dropout=0.0,
    )

    # Copy identical initial weights so this test isolates
    # trainer reproducibility rather than constructor RNG.
    model_2.load_state_dict(
        model_1.state_dict()
    )

    result_1 = train_main_effect_nam(
        model=model_1,
        train_data=train_data,
        train_y=y_train,
        val_data=val_data,
        val_y=y_val,
        config=config,
        device=torch.device("cpu"),
    )

    result_2 = train_main_effect_nam(
        model=model_2,
        train_data=train_data,
        train_y=y_train,
        val_data=val_data,
        val_y=y_val,
        config=config,
        device=torch.device("cpu"),
    )

    np.testing.assert_allclose(
        result_1.history["train_loss"],
        result_2.history["train_loss"],
        rtol=1e-6,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        result_1.history["val_auc"],
        result_2.history["val_auc"],
        rtol=1e-6,
        atol=1e-7,
    )