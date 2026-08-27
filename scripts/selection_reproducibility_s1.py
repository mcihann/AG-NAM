from __future__ import annotations

from pathlib import Path

import pandas as pd

from agnam.data.synthetic import generate_s1
from agnam.interpretation.interaction_metrics import canonical_pair
from agnam.training.discovery import (
    run_reproducible_interaction_discovery,
)
from agnam.training.nam_trainer import (
    NAMTrainingConfig,
)
from agnam.training.proposer_trainer import (
    ProposerTrainingConfig,
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

    device = get_device()

    print(
        "=== S1 SELECTION REPRODUCIBILITY ==="
    )

    print(
        f"Device: {device}"
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
        run_reproducible_interaction_discovery(
            X=dataset.X,
            y=dataset.y.to_numpy(),
            n_runs=5,
            base_seed=42,
            selection_threshold=0.60,
            crossfit_n_splits=5,
            main_early_stop_fraction=0.20,
            nam_hidden_dim=64,
            nam_depth=2,
            nam_dropout=0.10,
            categorical_embedding_dim=16,
            proposer_d_model=64,
            proposer_n_heads=4,
            proposer_n_layers=2,
            proposer_dropout=0.10,
            nam_config=nam_config,
            proposer_config=proposer_config,
            device=device,
        )
    )

    selection = result.selection

    print(
        "\n=== RUN SUMMARY ==="
    )

    for run in result.runs:
        print(
            f"Run {run.run_index + 1} "
            f"(seed={run.seed}): "
            f"best_epoch="
            f"{run.proposer_training_result.best_epoch + 1}, "
            f"val_MSE="
            f"{run.proposer_training_result.metrics['mse']:.6f}"
        )

    print(
        "\nK:",
        selection.k,
    )

    print(
        "B:",
        selection.n_runs,
    )

    print(
        "Selection threshold:",
        selection.threshold,
    )

    print(
        "Mean pairwise Top-K Jaccard:",
        f"{selection.mean_pairwise_jaccard:.4f}",
    )

    print(
        "\n=== ACCEPTED INTERACTIONS "
        "(pi >= 0.60) ==="
    )

    for item in (
        selection.accepted_interactions
    ):
        print(
            f"{item.feature_j:<4} x "
            f"{item.feature_k:<4} "
            f"selected={item.selection_count}/5 "
            f"pi={item.selection_probability:.2f} "
            f"mean_rank={item.mean_rank:.2f} "
            f"median_rank={item.median_rank:.1f} "
            f"ranks={item.ranks}"
        )

    true_pairs = {
        canonical_pair(a, b)
        for a, b
        in dataset.true_interactions
    }

    lookup = {
        canonical_pair(
            item.feature_j,
            item.feature_k,
        ): item
        for item in selection.interactions
    }

    print(
        "\n=== TRUE INTERACTION "
        "REPRODUCIBILITY ==="
    )

    for pair in sorted(
        true_pairs
    ):
        item = lookup[pair]

        print(
            f"{pair[0]} x {pair[1]}: "
            f"{item.selection_count}/5, "
            f"pi={item.selection_probability:.2f}, "
            f"mean_rank={item.mean_rank:.2f}, "
            f"ranks={item.ranks}, "
            f"accepted={item.accepted}"
        )

    rows = []

    for item in (
        selection.interactions
    ):
        pair = canonical_pair(
            item.feature_j,
            item.feature_k,
        )

        rows.append(
            {
                "feature_j": (
                    item.feature_j
                ),
                "feature_k": (
                    item.feature_k
                ),
                "selection_count": (
                    item.selection_count
                ),
                "selection_probability": (
                    item.selection_probability
                ),
                "mean_rank": (
                    item.mean_rank
                ),
                "median_rank": (
                    item.median_rank
                ),
                "ranks": "|".join(
                    str(rank)
                    for rank
                    in item.ranks
                ),
                "accepted": (
                    item.accepted
                ),
                "is_true_interaction": (
                    pair in true_pairs
                ),
            }
        )

    output_path = Path(
        "results"
    ) / (
        "s1_selection_reproducibility.csv"
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