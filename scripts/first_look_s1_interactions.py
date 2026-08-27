from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from agnam.data.preprocessing import TabularPreprocessor
from agnam.data.synthetic import generate_s1
from agnam.interpretation.interaction_metrics import (
    canonical_pair,
    evaluate_interaction_ranking,
)
from agnam.interpretation.interaction_scoring import (
    candidate_count,
    compute_interaction_scores,
)
from agnam.models.attention_proposer import (
    ResidualAttentionProposer,
)
from agnam.training.nam_trainer import (
    NAMTrainingConfig,
)
from agnam.training.proposer_trainer import (
    ProposerTrainingConfig,
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

    y = dataset.y.to_numpy()
    all_indices = np.arange(
        len(dataset.X)
    )

    # -------------------------------------------------
    # 60% train / 20% validation / 20% scoring holdout
    # -------------------------------------------------

    development_idx, scoring_idx = (
        train_test_split(
            all_indices,
            test_size=0.20,
            random_state=seed,
            stratify=y,
        )
    )

    train_idx, validation_idx = (
        train_test_split(
            development_idx,
            test_size=0.25,
            random_state=seed + 1,
            stratify=y[
                development_idx
            ],
        )
    )

    X_train = (
        dataset.X
        .iloc[train_idx]
        .reset_index(drop=True)
    )

    X_validation = (
        dataset.X
        .iloc[validation_idx]
        .reset_index(drop=True)
    )

    X_scoring = (
        dataset.X
        .iloc[scoring_idx]
        .reset_index(drop=True)
    )

    y_train = y[train_idx]
    y_validation = y[
        validation_idx
    ]

    print(
        "=== S1 FIRST-LOOK PROTOCOL ==="
    )

    print(
        f"Train:      {len(X_train)}"
    )
    print(
        f"Validation: {len(X_validation)}"
    )
    print(
        f"Scoring:    {len(X_scoring)}"
    )

    device = get_device()

    print(
        f"Device:     {device}"
    )

    # -------------------------------------------------
    # Leakage-safe residual targets
    # -------------------------------------------------

    nam_config = NAMTrainingConfig(
        learning_rate=1e-3,
        weight_decay=1e-5,
        batch_size=256,
        max_epochs=100,
        patience=15,
        seed=seed,
        deterministic=True,
    )

    residual_targets = (
        generate_train_validation_residual_targets(
            X_train=X_train,
            y_train=y_train,
            X_validation=X_validation,
            y_validation=y_validation,
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

    # -------------------------------------------------
    # Proposer-specific preprocessing
    # fitted on proposer training only
    # -------------------------------------------------

    preprocessor = TabularPreprocessor()

    train_data = (
        preprocessor.fit_transform(
            X_train
        )
    )

    validation_data = (
        preprocessor.transform(
            X_validation
        )
    )

    scoring_data = (
        preprocessor.transform(
            X_scoring
        )
    )

    # -------------------------------------------------
    # Residual attention proposer
    # -------------------------------------------------

    seed_everything(seed)

    proposer = (
        ResidualAttentionProposer(
            feature_specs=(
                train_data.feature_specs
            ),
            d_model=64,
            n_heads=4,
            n_layers=2,
            dropout=0.10,
        )
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

    training_result = (
        train_residual_attention_proposer(
            model=proposer,
            train_data=train_data,
            train_residuals=(
                residual_targets
                .train_residuals
            ),
            val_data=validation_data,
            val_residuals=(
                residual_targets
                .validation_residuals
            ),
            config=proposer_config,
            device=device,
        )
    )

    # -------------------------------------------------
    # Interaction attribution on untouched holdout
    # -------------------------------------------------

    score_result = (
        compute_interaction_scores(
            model=proposer,
            transformed=scoring_data,
            batch_size=256,
            device=device,
        )
    )

    k = candidate_count(
        proposer.n_features
    )

    metrics = (
        evaluate_interaction_ranking(
            ranked_interactions=(
                score_result
                .ranked_interactions
            ),
            true_interactions=(
                dataset.true_interactions
            ),
            k=k,
        )
    )

    true_pairs = {
        canonical_pair(a, b)
        for a, b
        in dataset.true_interactions
    }

    print(
        "\n=== PROPOSER VALIDATION ==="
    )

    print(
        f"Best epoch: "
        f"{training_result.best_epoch + 1}"
    )

    print(
        f"Validation MSE: "
        f"{training_result.metrics['mse']:.6f}"
    )

    print(
        "\n=== TOP-K INTERACTION CANDIDATES ==="
    )

    print(
        f"K = {k}\n"
    )

    rows = []

    for item in (
        score_result
        .ranked_interactions[:k]
    ):
        pair = canonical_pair(
            item.feature_j,
            item.feature_k,
        )

        is_true = (
            pair in true_pairs
        )

        marker = (
            "TRUE"
            if is_true
            else ""
        )

        print(
            f"{item.rank:>3}. "
            f"{item.feature_j:<4} x "
            f"{item.feature_k:<4} "
            f"score={item.score:.8f} "
            f"{marker}"
        )

        rows.append(
            {
                "rank": item.rank,
                "feature_j": (
                    item.feature_j
                ),
                "feature_k": (
                    item.feature_k
                ),
                "score": item.score,
                "is_true_interaction": (
                    is_true
                ),
            }
        )

    print(
        "\n=== TRUE INTERACTION RANKS ==="
    )

    for pair, rank in (
        metrics.true_ranks.items()
    ):
        print(
            f"{pair[0]} x {pair[1]}: "
            f"rank {rank}"
        )

    print(
        "\n=== RECOVERY METRICS ==="
    )

    print(
        f"Interaction AUPRC: "
        f"{metrics.interaction_auprc:.4f}"
    )

    print(
        f"Precision@{k}: "
        f"{metrics.precision_at_k:.4f}"
    )

    print(
        f"Recall@{k}: "
        f"{metrics.recall_at_k:.4f}"
    )

    print(
        f"NDCG@{k}: "
        f"{metrics.ndcg_at_k:.4f}"
    )

    print(
        f"Exact recovery@{k}: "
        f"{metrics.exact_recovery_at_k:.0f}"
    )

    print(
        f"FDR@{k}: "
        f"{metrics.false_discovery_rate_at_k:.4f}"
    )

    output_dir = Path(
        "results"
    )

    output_dir.mkdir(
        exist_ok=True
    )

    output_path = (
        output_dir
        / "first_look_s1_interaction_ranking.csv"
    )

    pd.DataFrame(
        rows
    ).to_csv(
        output_path,
        index=False,
    )

    print(
        "\nSaved ignored result:"
    )
    print(
        output_path
    )


if __name__ == "__main__":
    main()