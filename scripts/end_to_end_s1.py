from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from agnam.data.preprocessing import TabularPreprocessor
from agnam.data.synthetic import generate_s1
from agnam.models.agnam import AGNAM
from agnam.models.main_effect_nam import MainEffectNAM
from agnam.training.agnam_trainer import (
    AGNAMTrainingConfig,
    compute_agnam_binary_metrics,
    predict_agnam_probabilities,
    train_agnam,
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
    compute_binary_metrics,
    predict_probabilities,
    train_main_effect_nam,
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

    seed_everything(
        seed
    )

    device = get_device()

    print(
        "========================================"
    )
    print(
        "AG-NAM END-TO-END S1 SANITY AUDIT"
    )
    print(
        "========================================"
    )

    print(
        f"Device: {device}"
    )

    # =========================================================
    # DATA
    # =========================================================

    dataset = generate_s1(
        seed=seed,
        n=5000,
    )

    X = dataset.X
    y = dataset.y.to_numpy()

    all_indices = np.arange(
        len(X)
    )

    outer_dev_idx, test_idx = (
        train_test_split(
            all_indices,
            test_size=0.20,
            random_state=seed,
            stratify=y,
        )
    )

    X_outer_dev = (
        X.iloc[
            outer_dev_idx
        ]
        .reset_index(
            drop=True
        )
    )

    y_outer_dev = y[
        outer_dev_idx
    ]

    X_test = (
        X.iloc[
            test_idx
        ]
        .reset_index(
            drop=True
        )
    )

    y_test = y[
        test_idx
    ]

    print(
        "\n=== OUTER SPLIT ==="
    )

    print(
        f"Outer development: "
        f"{len(X_outer_dev)}"
    )

    print(
        f"Untouched test:    "
        f"{len(X_test)}"
    )

    # =========================================================
    # DISCOVERY
    # =========================================================

    print(
        "\n=== INTERACTION DISCOVERY ==="
    )

    nam_discovery_config = (
        NAMTrainingConfig(
            learning_rate=1e-3,
            weight_decay=1e-5,
            batch_size=256,
            max_epochs=100,
            patience=15,
            seed=seed,
            deterministic=True,
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

    discovery = (
        run_reproducible_interaction_discovery(
            X=X_outer_dev,
            y=y_outer_dev,
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
            nam_config=nam_discovery_config,
            proposer_config=proposer_config,
            device=device,
        )
    )

    print(
        "Selection-stable candidates:",
        len(
            discovery
            .selection
            .accepted_interactions
        ),
    )

    print(
        "Mean pairwise Top-K Jaccard:",
        f"{discovery.selection.mean_pairwise_jaccard:.4f}",
    )

    # =========================================================
    # ISR
    # =========================================================

    print(
        "\n=== ISR FILTER ==="
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

    isr_result = (
        evaluate_isr_for_discovery(
            X=X_outer_dev,
            discovery=discovery,
            config=isr_config,
            device=device,
            verbose=True,
        )
    )

    retained_pairs = tuple(
        (
            item.feature_j,
            item.feature_k,
        )
        for item
        in isr_result.retained_interactions
    )

    print(
        "\n=== FINAL DISCOVERED INTERACTION SET S* ==="
    )

    print(
        f"|S*| = {len(retained_pairs)}"
    )

    if len(
        retained_pairs
    ) == 0:
        print(
            "No interactions retained."
        )
    else:
        for index, pair in enumerate(
            retained_pairs,
            start=1,
        ):
            print(
                f"{index:>2}. "
                f"{pair[0]} x {pair[1]}"
            )

    # =========================================================
    # FINAL TRAIN / VALIDATION SPLIT
    # =========================================================

    development_indices = np.arange(
        len(
            X_outer_dev
        )
    )

    final_train_idx, final_val_idx = (
        train_test_split(
            development_indices,
            test_size=0.25,
            random_state=seed + 100,
            stratify=y_outer_dev,
        )
    )

    X_final_train = (
        X_outer_dev.iloc[
            final_train_idx
        ]
        .reset_index(
            drop=True
        )
    )

    y_final_train = y_outer_dev[
        final_train_idx
    ]

    X_final_val = (
        X_outer_dev.iloc[
            final_val_idx
        ]
        .reset_index(
            drop=True
        )
    )

    y_final_val = y_outer_dev[
        final_val_idx
    ]

    print(
        "\n=== FINAL PREDICTION SPLIT ==="
    )

    print(
        f"Final train: "
        f"{len(X_final_train)}"
    )

    print(
        f"Final validation: "
        f"{len(X_final_val)}"
    )

    print(
        f"Untouched test: "
        f"{len(X_test)}"
    )

    # =========================================================
    # FINAL PREPROCESSING
    # =========================================================

    preprocessor = (
        TabularPreprocessor()
    )

    final_train_data = (
        preprocessor
        .fit_transform(
            X_final_train
        )
    )

    final_val_data = (
        preprocessor
        .transform(
            X_final_val
        )
    )

    test_data = (
        preprocessor
        .transform(
            X_test
        )
    )

    # =========================================================
    # MAIN-EFFECT NAM BASELINE
    # =========================================================

    print(
        "\n=== TRAINING MAIN-EFFECT NAM ==="
    )

    final_seed = (
        seed + 50_000
    )

    seed_everything(
        final_seed
    )

    main_model = MainEffectNAM(
        feature_specs=(
            final_train_data
            .feature_specs
        ),
        hidden_dim=64,
        depth=2,
        dropout=0.10,
        categorical_embedding_dim=16,
    )

    main_config = (
        NAMTrainingConfig(
            learning_rate=1e-3,
            weight_decay=1e-5,
            batch_size=256,
            max_epochs=300,
            patience=30,
            seed=final_seed,
            deterministic=True,
        )
    )

    main_training = (
        train_main_effect_nam(
            model=main_model,
            train_data=(
                final_train_data
            ),
            train_y=(
                y_final_train
            ),
            val_data=(
                final_val_data
            ),
            val_y=(
                y_final_val
            ),
            config=main_config,
            device=device,
        )
    )

    main_test_probabilities = (
        predict_probabilities(
            model=main_model,
            transformed=test_data,
            device=device,
        )
    )

    main_test_metrics = (
        compute_binary_metrics(
            y_true=y_test,
            probabilities=(
                main_test_probabilities
            ),
        )
    )

    # =========================================================
    # FINAL AG-NAM
    # =========================================================

    print(
        "\n=== TRAINING FINAL AG-NAM ==="
    )

    # Same seed before construction means the main-effect
    # portion starts from the same RNG state used for the
    # standalone MainEffectNAM.
    seed_everything(
        final_seed
    )

    agnam_model = AGNAM(
        feature_specs=(
            final_train_data
            .feature_specs
        ),
        interaction_pairs=(
            retained_pairs
        ),
        main_hidden_dim=64,
        main_depth=2,
        main_dropout=0.10,
        categorical_embedding_dim=16,
        interaction_embedding_dim=16,
        interaction_hidden_dim=64,
        interaction_depth=2,
        interaction_dropout=0.10,
    )

    agnam_config = (
        AGNAMTrainingConfig(
            learning_rate=1e-3,
            weight_decay=1e-5,
            batch_size=256,
            max_epochs=300,
            patience=30,
            seed=final_seed,
            deterministic=True,
        )
    )

    agnam_training = train_agnam(
        model=agnam_model,
        train_data=(
            final_train_data
        ),
        train_y=y_final_train,
        val_data=(
            final_val_data
        ),
        val_y=y_final_val,
        config=agnam_config,
        device=device,
    )

    agnam_test_probabilities = (
        predict_agnam_probabilities(
            model=agnam_model,
            transformed=test_data,
            device=device,
        )
    )

    agnam_test_metrics = (
        compute_agnam_binary_metrics(
            y_true=y_test,
            probabilities=(
                agnam_test_probabilities
            ),
        )
    )

    # =========================================================
    # EXACT DECOMPOSITION AUDIT
    # =========================================================

    agnam_model.eval()

    with torch.no_grad():
        numeric = torch.from_numpy(
            test_data.numeric
        ).to(
            device
        )

        missing = torch.from_numpy(
            test_data
            .numeric_missing
        ).to(
            device
        )

        categorical = torch.from_numpy(
            test_data.categorical
        ).to(
            device
        )

        decomposition_output = (
            agnam_model(
                numeric=numeric,
                numeric_missing=missing,
                categorical=categorical,
            )
        )

        reconstructed_logits = (
            decomposition_output.baseline
            + decomposition_output
            .main_contributions
            .sum(dim=1)
            + decomposition_output
            .interaction_contributions
            .sum(dim=1)
        )

        maximum_decomposition_error = (
            torch.max(
                torch.abs(
                    decomposition_output
                    .logits
                    - reconstructed_logits
                )
            )
            .cpu()
            .item()
        )

    # =========================================================
    # GROUND-TRUTH DIAGNOSTIC
    # =========================================================

    true_pairs = {
        tuple(
            sorted(pair)
        )
        for pair
        in dataset.true_interactions
    }

    discovered_pairs = {
        tuple(
            sorted(pair)
        )
        for pair
        in retained_pairs
    }

    true_retained = (
        true_pairs
        & discovered_pairs
    )

    false_retained = (
        discovered_pairs
        - true_pairs
    )

    missed_true = (
        true_pairs
        - discovered_pairs
    )

    # =========================================================
    # FINAL REPORT
    # =========================================================

    print(
        "\n========================================"
    )

    print(
        "END-TO-END TEST RESULTS"
    )

    print(
        "========================================"
    )

    print(
        "\nMAIN-EFFECT NAM"
    )

    print(
        f"Best epoch: "
        f"{main_training.best_epoch + 1}"
    )

    print(
        f"Test AUROC: "
        f"{main_test_metrics['auroc']:.4f}"
    )

    print(
        f"Test AUPRC: "
        f"{main_test_metrics['auprc']:.4f}"
    )

    print(
        f"Balanced Accuracy: "
        f"{main_test_metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"F1: "
        f"{main_test_metrics['f1']:.4f}"
    )

    print(
        "\nFINAL AG-NAM"
    )

    print(
        f"Best epoch: "
        f"{agnam_training.best_epoch + 1}"
    )

    print(
        f"Test AUROC: "
        f"{agnam_test_metrics['auroc']:.4f}"
    )

    print(
        f"Test AUPRC: "
        f"{agnam_test_metrics['auprc']:.4f}"
    )

    print(
        f"Balanced Accuracy: "
        f"{agnam_test_metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"F1: "
        f"{agnam_test_metrics['f1']:.4f}"
    )

    print(
        "\nDELTA AG-NAM - MAIN NAM"
    )

    print(
        f"Delta AUROC: "
        f"{agnam_test_metrics['auroc'] - main_test_metrics['auroc']:+.4f}"
    )

    print(
        f"Delta AUPRC: "
        f"{agnam_test_metrics['auprc'] - main_test_metrics['auprc']:+.4f}"
    )

    print(
        f"Delta Balanced Accuracy: "
        f"{agnam_test_metrics['balanced_accuracy'] - main_test_metrics['balanced_accuracy']:+.4f}"
    )

    print(
        f"Delta F1: "
        f"{agnam_test_metrics['f1'] - main_test_metrics['f1']:+.4f}"
    )

    print(
        "\nDISCOVERY DIAGNOSTIC"
    )

    print(
        f"True retained: "
        f"{len(true_retained)}/"
        f"{len(true_pairs)}"
    )

    print(
        "True retained pairs:",
        sorted(
            true_retained
        ),
    )

    print(
        "False retained pairs:",
        sorted(
            false_retained
        ),
    )

    print(
        "Missed true pairs:",
        sorted(
            missed_true
        ),
    )

    print(
        "\nEXACT DECOMPOSITION AUDIT"
    )

    print(
        "Maximum absolute logit reconstruction error:",
        f"{maximum_decomposition_error:.12e}",
    )

    # =========================================================
    # SAVE
    # =========================================================

    result_row = {
        "dataset": "S1",
        "seed": seed,
        "n": len(X),
        "n_retained_interactions": (
            len(
                retained_pairs
            )
        ),
        "main_test_auroc": (
            main_test_metrics[
                "auroc"
            ]
        ),
        "main_test_auprc": (
            main_test_metrics[
                "auprc"
            ]
        ),
        "main_test_balanced_accuracy": (
            main_test_metrics[
                "balanced_accuracy"
            ]
        ),
        "main_test_f1": (
            main_test_metrics[
                "f1"
            ]
        ),
        "agnam_test_auroc": (
            agnam_test_metrics[
                "auroc"
            ]
        ),
        "agnam_test_auprc": (
            agnam_test_metrics[
                "auprc"
            ]
        ),
        "agnam_test_balanced_accuracy": (
            agnam_test_metrics[
                "balanced_accuracy"
            ]
        ),
        "agnam_test_f1": (
            agnam_test_metrics[
                "f1"
            ]
        ),
        "delta_auroc": (
            agnam_test_metrics[
                "auroc"
            ]
            - main_test_metrics[
                "auroc"
            ]
        ),
        "delta_auprc": (
            agnam_test_metrics[
                "auprc"
            ]
            - main_test_metrics[
                "auprc"
            ]
        ),
        "true_interactions_retained": (
            len(
                true_retained
            )
        ),
        "false_interactions_retained": (
            len(
                false_retained
            )
        ),
        "true_interactions_missed": (
            len(
                missed_true
            )
        ),
        "max_decomposition_error": (
            maximum_decomposition_error
        ),
        "retained_pairs": "|".join(
            f"{a}*{b}"
            for a, b
            in retained_pairs
        ),
    }

    output_path = (
        Path(
            "results"
        )
        / "s1_end_to_end_sanity.csv"
    )

    pd.DataFrame(
        [
            result_row
        ]
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