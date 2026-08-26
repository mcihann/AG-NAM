from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split

from agnam.data.preprocessing import TabularPreprocessor
from agnam.data.synthetic import generate_s1
from agnam.models.attention_proposer import (
    ResidualAttentionProposer,
)
from agnam.training.nam_trainer import (
    NAMTrainingConfig,
)
from agnam.training.proposer_trainer import (
    ProposerTrainingConfig,
    compute_residual_metrics,
    predict_residuals,
    train_residual_attention_proposer,
)
from agnam.training.residual_targets import (
    generate_train_validation_residual_targets,
)
from agnam.utils.reproducibility import (
    get_device,
    seed_everything,
)


def main():
    seed = 42

    seed_everything(seed)

    dataset = generate_s1(
        seed=seed,
        n=5000,
    )

    indices = np.arange(
        len(dataset.X)
    )

    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=seed,
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

    device = get_device()

    print(
        "Device:",
        device,
    )

    nam_config = NAMTrainingConfig(
        learning_rate=1e-3,
        weight_decay=1e-5,
        batch_size=256,
        max_epochs=100,
        patience=15,
        seed=seed,
        deterministic=True,
    )

    print(
        "\nGenerating leakage-safe residual targets..."
    )

    residual_targets = (
        generate_train_validation_residual_targets(
            X_train=X_train,
            y_train=y_train,
            X_validation=X_val,
            y_validation=y_val,
            crossfit_n_splits=5,
            main_early_stop_fraction=0.20,
            hidden_dim=64,
            depth=2,
            dropout=0.10,
            categorical_embedding_dim=16,
            training_config=nam_config,
            seed=seed,
            device=device,
        )
    )

    # Proposer preprocessing is independently fitted
    # on proposer-training observations only.
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

    seed_everything(seed)

    proposer = ResidualAttentionProposer(
        feature_specs=(
            train_data.feature_specs
        ),
        d_model=64,
        n_heads=4,
        n_layers=2,
        dropout=0.10,
    )

    proposer_config = (
        ProposerTrainingConfig(
            learning_rate=1e-3,
            weight_decay=1e-5,
            batch_size=256,
            max_epochs=200,
            patience=20,
            seed=seed,
            deterministic=True,
        )
    )

    result = (
        train_residual_attention_proposer(
            model=proposer,
            train_data=train_data,
            train_residuals=(
                residual_targets.train_residuals
            ),
            val_data=val_data,
            val_residuals=(
                residual_targets.validation_residuals
            ),
            config=proposer_config,
            device=device,
        )
    )

    predictions = predict_residuals(
        model=proposer,
        transformed=val_data,
        device=device,
    )

    metrics = compute_residual_metrics(
        residual_true=(
            residual_targets.validation_residuals
        ),
        residual_pred=predictions,
    )

    print(
        "\n=== CLEAN RESIDUAL ATTENTION PROPOSER / S1 ==="
    )

    print(
        f"Best epoch: "
        f"{result.best_epoch + 1}"
    )

    print(
        f"Validation MSE: "
        f"{metrics['mse']:.6f}"
    )

    print(
        f"Zero-baseline MSE: "
        f"{metrics['zero_baseline_mse']:.6f}"
    )

    print(
        f"MSE improvement: "
        f"{metrics['mse_improvement_over_zero']:.6f}"
    )

    improvement_percent = (
        100.0
        * metrics[
            "mse_improvement_over_zero"
        ]
        / metrics[
            "zero_baseline_mse"
        ]
    )

    print(
        f"MSE improvement (%): "
        f"{improvement_percent:.2f}"
    )

    print(
        f"Validation R2: "
        f"{metrics['r2']:.4f}"
    )

    print(
        f"Residual correlation: "
        f"{metrics['correlation']:.4f}"
    )

    print(
        "\nMain-effect NAM validation AUROC: "
        f"{residual_targets.validation_main_metrics['auroc']:.4f}"
    )


if __name__ == "__main__":
    main()