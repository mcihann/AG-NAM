from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split

from agnam.data.preprocessing import TabularPreprocessor
from agnam.data.synthetic import generate_s1
from agnam.models.main_effect_nam import MainEffectNAM
from agnam.training.nam_trainer import (
    NAMTrainingConfig,
    compute_binary_metrics,
    predict_probabilities,
    train_main_effect_nam,
)
from agnam.utils.reproducibility import get_device


def main():
    dataset = generate_s1(
        seed=42,
        n=5000,
    )

    indices = np.arange(
        len(dataset.X)
    )

    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.20,
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

    model = MainEffectNAM(
        feature_specs=train_data.feature_specs,
        hidden_dim=64,
        depth=2,
        dropout=0.10,
        categorical_embedding_dim=16,
    )

    config = NAMTrainingConfig(
        learning_rate=1e-3,
        weight_decay=1e-5,
        batch_size=256,
        max_epochs=300,
        patience=30,
        seed=42,
    )

    device = get_device()

    print("Device:", device)

    result = train_main_effect_nam(
        model=model,
        train_data=train_data,
        train_y=y_train,
        val_data=val_data,
        val_y=y_val,
        config=config,
        device=device,
    )

    probabilities = predict_probabilities(
        model=model,
        transformed=val_data,
        device=device,
    )

    metrics = compute_binary_metrics(
        y_true=y_val,
        probabilities=probabilities,
    )

    print("\n=== MAIN-EFFECT NAM / S1 ===")
    print(
        f"Best epoch: "
        f"{result.best_epoch + 1}"
    )

    print(
        f"Validation AUROC: "
        f"{metrics['auroc']:.4f}"
    )

    print(
        f"Validation AUPRC: "
        f"{metrics['auprc']:.4f}"
    )

    print(
        f"Balanced Accuracy: "
        f"{metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"F1: "
        f"{metrics['f1']:.4f}"
    )

    print(
        "\nExpected behavior:"
        "\nMain-effect NAM should learn the true univariate effects,"
        "\nbut should remain limited because S1 also contains"
        "\nthree genuine pairwise interactions."
    )


if __name__ == "__main__":
    main()