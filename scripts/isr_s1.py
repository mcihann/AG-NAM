from __future__ import annotations

from pathlib import Path

import pandas as pd

from agnam.data.synthetic import (
    generate_s1,
)
from agnam.interpretation.interaction_metrics import (
    canonical_pair,
)
from agnam.training.discovery import (
    run_reproducible_interaction_discovery,
)
from agnam.training.isr import (
    ISRConfig,
    evaluate_isr_for_discovery,
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
        "=== S1 FULL STABILITY AUDIT ==="
    )

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

    print(
        "\nRunning B=5 interaction discovery..."
    )

    discovery = (
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
            proposer_config=(
                proposer_config
            ),
            device=device,
        )
    )

    print(
        "\nSelection-stable interactions:",
        len(
            discovery
            .selection
            .accepted_interactions
        ),
    )

    isr_config = ISRConfig(
        reference_size=512,
        reference_seed=2026,
        isr_threshold=0.60,
        feature_embedding_dim=16,
        hidden_dim=64,
        depth=2,
        dropout=0.10,
        learning_rate=1e-3,
        weight_decay=1e-5,
        batch_size=256,
        max_epochs=200,
        patience=20,
        min_delta=1e-5,
        surface_batch_size=8192,
    )

    result = (
        evaluate_isr_for_discovery(
            X=dataset.X,
            discovery=discovery,
            config=isr_config,
            device=device,
            verbose=True,
        )
    )

    true_pairs = {
        canonical_pair(a, b)
        for a, b
        in dataset.true_interactions
    }

    print(
        "\n========================================"
    )

    print(
        "FINAL ISR SUMMARY"
    )

    print(
        "========================================"
    )

    print(
        f"Candidates after pi filter: "
        f"{len(result.interactions)}"
    )

    print(
        f"Retained after ISR: "
        f"{len(result.retained_interactions)}"
    )

    print(
        "\nALL PI-STABLE CANDIDATES"
    )

    rows = []

    for item in (
        result.interactions
    ):
        pair = canonical_pair(
            item.feature_j,
            item.feature_k,
        )

        is_true = (
            pair in true_pairs
        )

        print(
            f"{item.feature_j:<4} x "
            f"{item.feature_k:<4} "
            f"pi="
            f"{item.selection_probability:.2f} "
            f"ISR="
            f"{item.isr:.4f} "
            f"{'RETAIN' if item.final_accepted else 'REJECT':<6} "
            f"{'TRUE' if is_true else ''}"
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
                    item
                    .selection_probability
                ),
                "isr": (
                    item.isr
                ),
                "final_accepted": (
                    item.final_accepted
                ),
                "is_true_interaction": (
                    is_true
                ),
                "pairwise_correlations": (
                    "|".join(
                        f"{value:.6f}"
                        for value
                        in item
                        .pairwise_correlations
                    )
                ),
            }
        )

    retained_pairs = {
        canonical_pair(
            item.feature_j,
            item.feature_k,
        )
        for item
        in result.retained_interactions
    }

    true_retained = (
        retained_pairs
        & true_pairs
    )

    false_retained = (
        retained_pairs
        - true_pairs
    )

    missed_true = (
        true_pairs
        - retained_pairs
    )

    print(
        "\nTRUE INTERACTIONS RETAINED:"
    )

    for pair in sorted(
        true_retained
    ):
        print(
            pair
        )

    print(
        "\nFALSE INTERACTIONS RETAINED:"
    )

    for pair in sorted(
        false_retained
    ):
        print(
            pair
        )

    print(
        "\nTRUE INTERACTIONS MISSED:"
    )

    for pair in sorted(
        missed_true
    ):
        print(
            pair
        )

    precision = (
        len(true_retained)
        / len(retained_pairs)
        if retained_pairs
        else 0.0
    )

    recall = (
        len(true_retained)
        / len(true_pairs)
    )

    print(
        "\nFINAL STABILITY FILTER METRICS"
    )

    print(
        f"Precision: "
        f"{precision:.4f}"
    )

    print(
        f"Recall: "
        f"{recall:.4f}"
    )

    output_path = (
        Path("results")
        / "s1_isr_results.csv"
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