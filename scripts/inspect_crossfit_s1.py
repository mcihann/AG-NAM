from agnam.data.synthetic import generate_s1
from agnam.training.crossfit import (
    generate_cross_fitted_residuals,
)
from agnam.training.nam_trainer import (
    NAMTrainingConfig,
)
from agnam.utils.reproducibility import get_device


def main():
    dataset = generate_s1(
        seed=42,
        n=5000,
    )

    config = NAMTrainingConfig(
        learning_rate=1e-3,
        weight_decay=1e-5,
        batch_size=256,
        max_epochs=100,
        patience=15,
        seed=42,
        deterministic=True,
    )

    device = get_device()

    print("Device:", device)

    result = generate_cross_fitted_residuals(
        X=dataset.X,
        y=dataset.y.to_numpy(),
        n_splits=5,
        early_stop_fraction=0.20,
        hidden_dim=64,
        depth=2,
        dropout=0.10,
        training_config=config,
        seed=42,
        device=device,
    )

    print("\n=== CROSS-FITTED MAIN NAM / S1 ===")

    print(
        f"OOF AUROC: "
        f"{result.overall_metrics['auroc']:.4f}"
    )

    print(
        f"OOF AUPRC: "
        f"{result.overall_metrics['auprc']:.4f}"
    )

    print(
        f"Residual mean: "
        f"{result.residuals.mean():.6f}"
    )

    print(
        f"Residual std: "
        f"{result.residuals.std():.6f}"
    )

    print(
        f"Residual min: "
        f"{result.residuals.min():.6f}"
    )

    print(
        f"Residual max: "
        f"{result.residuals.max():.6f}"
    )

    print("\nPer-fold results:")

    for fold in result.fold_results:
        print(
            f"Fold {fold.fold + 1}: "
            f"AUROC={fold.holdout_metrics['auroc']:.4f}, "
            f"best_epoch={fold.best_epoch + 1}"
        )


if __name__ == "__main__":
    main()