from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from agnam.data.preprocessing import (
    TabularPreprocessor,
)
from agnam.data.synthetic import (
    generate_s1,
)
from agnam.interpretation.final_decomposition import (
    compute_purified_agnam_decomposition,
)
from agnam.interpretation.surface_reproducibility import (
    evaluate_pairwise_raw_grid,
    purify_interaction_grid,
)
from agnam.models.agnam import (
    AGNAM,
)
from agnam.training.agnam_trainer import (
    AGNAMTrainingConfig,
    compute_agnam_binary_metrics,
    train_agnam,
)
from agnam.training.discovery import (
    run_reproducible_interaction_discovery,
)
from agnam.training.isr import (
    ISRConfig,
    evaluate_isr_for_discovery,
    sample_reference_indices,
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

    seed_everything(
        seed
    )

    device = get_device()

    print(
        "============================================"
    )

    print(
        "AG-NAM PURIFIED END-TO-END S1 AUDIT"
    )

    print(
        "============================================"
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

    y_outer_dev = (
        y[
            outer_dev_idx
        ]
    )

    X_test = (
        X.iloc[
            test_idx
        ]
        .reset_index(
            drop=True
        )
    )

    y_test = (
        y[
            test_idx
        ]
    )

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
    # LOCKED DISCOVERY
    # =========================================================

    print(
        "\n=== LOCKED INTERACTION DISCOVERY ==="
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
            nam_config=(
                nam_discovery_config
            ),
            proposer_config=(
                proposer_config
            ),
            device=device,
        )
    )

    print(
        f"Selection-stable candidates: "
        f"{len(discovery.selection.accepted_interactions)}"
    )

    print(
        "Mean pairwise Top-K Jaccard: "
        f"{discovery.selection.mean_pairwise_jaccard:.4f}"
    )

    # =========================================================
    # LOCKED ISR
    # =========================================================

    print(
        "\n=== LOCKED ISR FILTER ==="
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
            verbose=False,
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
        f"Candidates after pi filter: "
        f"{len(isr_result.interactions)}"
    )

    print(
        f"Retained after ISR: "
        f"{len(retained_pairs)}"
    )

    print(
        "\n=== FINAL INTERACTION SET S* ==="
    )

    print(
        f"|S*| = {len(retained_pairs)}"
    )

    for index, (
        feature_j,
        feature_k,
    ) in enumerate(
        retained_pairs,
        start=1,
    ):
        print(
            f"{index:>2}. "
            f"{feature_j} x {feature_k}"
        )

    # =========================================================
    # FINAL MODEL SPLIT
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

    y_final_train = (
        y_outer_dev[
            final_train_idx
        ]
    )

    X_final_val = (
        X_outer_dev.iloc[
            final_val_idx
        ]
        .reset_index(
            drop=True
        )
    )

    y_final_val = (
        y_outer_dev[
            final_val_idx
        ]
    )

    print(
        "\n=== FINAL MODEL SPLIT ==="
    )

    print(
        f"Final training:   "
        f"{len(X_final_train)}"
    )

    print(
        f"Final validation: "
        f"{len(X_final_val)}"
    )

    print(
        f"Untouched test:   "
        f"{len(X_test)}"
    )

    # =========================================================
    # FINAL PREPROCESSING
    # =========================================================

    preprocessor = (
        TabularPreprocessor()
    )

    final_train_data = (
        preprocessor.fit_transform(
            X_final_train
        )
    )

    final_val_data = (
        preprocessor.transform(
            X_final_val
        )
    )

    test_data = (
        preprocessor.transform(
            X_test
        )
    )

    # =========================================================
    # PURIFICATION REFERENCE
    # =========================================================

    reference_indices = (
        sample_reference_indices(
            n_samples=(
                len(X_final_train)
            ),
            reference_size=512,
            seed=2026,
        )
    )

    X_reference = (
        X_final_train.iloc[
            reference_indices
        ]
        .reset_index(
            drop=True
        )
    )

    reference_data = (
        preprocessor.transform(
            X_reference
        )
    )

    print(
        "\n=== PURIFICATION REFERENCE ==="
    )

    print(
        f"Reference support: "
        f"{len(reference_data)}"
    )

    print(
        "Reference source: final training subset only"
    )

    print(
        "Reference seed: 2026"
    )

    # =========================================================
    # FRESH FINAL AG-NAM
    # =========================================================

    print(
        "\n=== TRAINING FINAL AG-NAM ==="
    )

    final_seed = (
        seed + 50_000
    )

    seed_everything(
        final_seed
    )

    model = AGNAM(
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

    training_config = (
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

    training_result = train_agnam(
        model=model,
        train_data=final_train_data,
        train_y=y_final_train,
        val_data=final_val_data,
        val_y=y_final_val,
        config=training_config,
        device=device,
    )

    print(
        f"Best epoch: "
        f"{training_result.best_epoch + 1}"
    )

    # =========================================================
    # RAW TEST OUTPUT
    # =========================================================

    model.eval()

    with torch.no_grad():
        numeric = torch.from_numpy(
            test_data.numeric
        ).to(
            device
        )

        missing = torch.from_numpy(
            test_data.numeric_missing
        ).to(
            device
        )

        categorical = torch.from_numpy(
            test_data.categorical
        ).to(
            device
        )

        raw_output = model(
            numeric=numeric,
            numeric_missing=missing,
            categorical=categorical,
        )

        raw_logits = (
            raw_output.logits
            .detach()
            .cpu()
        )

        raw_probabilities = (
            torch.sigmoid(
                raw_logits
            )
            .numpy()
        )

    raw_metrics = (
        compute_agnam_binary_metrics(
            y_true=y_test,
            probabilities=(
                raw_probabilities
            ),
        )
    )

    # =========================================================
    # PURIFIED TEST DECOMPOSITION
    # =========================================================

    print(
        "\n=== FUNCTIONAL-ANOVA PURIFICATION ==="
    )

    purified = (
        compute_purified_agnam_decomposition(
            model=model,
            transformed=test_data,
            reference_data=(
                reference_data
            ),
            query_batch_size=256,
            surface_batch_size=8192,
            device=device,
        )
    )

    purified_probabilities = (
        torch.sigmoid(
            purified.logits
        )
        .numpy()
    )

    purified_metrics = (
        compute_agnam_binary_metrics(
            y_true=y_test,
            probabilities=(
                purified_probabilities
            ),
        )
    )

    # =========================================================
    # INVARIANCE AUDITS
    # =========================================================

    max_logit_difference = float(
        torch.max(
            torch.abs(
                raw_logits
                - purified.logits
            )
        ).item()
    )

    max_probability_difference = float(
        np.max(
            np.abs(
                raw_probabilities
                - purified_probabilities
            )
        )
    )

    purified_reconstruction = (
        purified.baseline
        + purified
        .main_contributions
        .sum(dim=1)
        + purified
        .interaction_contributions
        .sum(dim=1)
    )

    max_additive_reconstruction_error = float(
        torch.max(
            torch.abs(
                purified.logits
                - purified_reconstruction
            )
        ).item()
    )

    # =========================================================
    # ZERO-MARGINAL IDENTIFIABILITY AUDIT
    # =========================================================

    interaction_audit_rows = []

    global_max_row_marginal = 0.0
    global_max_column_marginal = 0.0
    global_max_grand_mean = 0.0

    for interaction_index, (
        pair,
        interaction_network,
    ) in enumerate(
        zip(
            model.interaction_pairs,
            model.interaction_networks,
        ),
        start=1,
    ):
        raw_reference_grid = (
            evaluate_pairwise_raw_grid(
                model=interaction_network,
                reference_data=(
                    reference_data
                ),
                batch_size=8192,
                device=device,
            )
        )

        purified_reference_grid = (
            purify_interaction_grid(
                raw_reference_grid
            )
        )

        row_marginals = (
            purified_reference_grid
            .mean(
                axis=1
            )
        )

        column_marginals = (
            purified_reference_grid
            .mean(
                axis=0
            )
        )

        purified_grand_mean = float(
            purified_reference_grid.mean()
        )

        max_row_marginal = float(
            np.max(
                np.abs(
                    row_marginals
                )
            )
        )

        max_column_marginal = float(
            np.max(
                np.abs(
                    column_marginals
                )
            )
        )

        absolute_grand_mean = float(
            abs(
                purified_grand_mean
            )
        )

        global_max_row_marginal = max(
            global_max_row_marginal,
            max_row_marginal,
        )

        global_max_column_marginal = max(
            global_max_column_marginal,
            max_column_marginal,
        )

        global_max_grand_mean = max(
            global_max_grand_mean,
            absolute_grand_mean,
        )

        interaction_audit_rows.append(
            {
                "interaction_index": (
                    interaction_index
                ),
                "feature_j": (
                    pair[0]
                ),
                "feature_k": (
                    pair[1]
                ),
                "max_abs_row_marginal": (
                    max_row_marginal
                ),
                "max_abs_column_marginal": (
                    max_column_marginal
                ),
                "abs_grand_mean": (
                    absolute_grand_mean
                ),
            }
        )

    # =========================================================
    # REPORT
    # =========================================================

    print(
        "\n============================================"
    )

    print(
        "PURIFIED END-TO-END AUDIT RESULTS"
    )

    print(
        "============================================"
    )

    print(
        "\nRAW FINAL AG-NAM"
    )

    print(
        f"AUROC: "
        f"{raw_metrics['auroc']:.6f}"
    )

    print(
        f"AUPRC: "
        f"{raw_metrics['auprc']:.6f}"
    )

    print(
        f"Balanced Accuracy: "
        f"{raw_metrics['balanced_accuracy']:.6f}"
    )

    print(
        f"F1: "
        f"{raw_metrics['f1']:.6f}"
    )

    print(
        "\nPURIFIED FINAL AG-NAM"
    )

    print(
        f"AUROC: "
        f"{purified_metrics['auroc']:.6f}"
    )

    print(
        f"AUPRC: "
        f"{purified_metrics['auprc']:.6f}"
    )

    print(
        f"Balanced Accuracy: "
        f"{purified_metrics['balanced_accuracy']:.6f}"
    )

    print(
        f"F1: "
        f"{purified_metrics['f1']:.6f}"
    )

    print(
        "\nPREDICTIVE INVARIANCE"
    )

    print(
        "Delta AUROC: "
        f"{purified_metrics['auroc'] - raw_metrics['auroc']:+.12e}"
    )

    print(
        "Delta AUPRC: "
        f"{purified_metrics['auprc'] - raw_metrics['auprrc']:+.12e}"
        if "auprrc" in raw_metrics
        else (
            "Delta AUPRC: "
            f"{purified_metrics['auprc'] - raw_metrics['auprc']:+.12e}"
        )
    )

    print(
        "Delta Balanced Accuracy: "
        f"{purified_metrics['balanced_accuracy'] - raw_metrics['balanced_accuracy']:+.12e}"
    )

    print(
        "Delta F1: "
        f"{purified_metrics['f1'] - raw_metrics['f1']:+.12e}"
    )

    print(
        "Maximum raw-vs-purified logit difference: "
        f"{max_logit_difference:.12e}"
    )

    print(
        "Maximum raw-vs-purified probability difference: "
        f"{max_probability_difference:.12e}"
    )

    print(
        "\nPURIFIED ADDITIVE IDENTITY"
    )

    print(
        "Maximum additive reconstruction error: "
        f"{max_additive_reconstruction_error:.12e}"
    )

    print(
        "Internal purification invariance error: "
        f"{purified.max_logit_invariance_error:.12e}"
    )

    print(
        "\nFUNCTIONAL-ANOVA ZERO-MARGINAL AUDIT"
    )

    for row in (
        interaction_audit_rows
    ):
        print(
            f"{row['feature_j']:<4} x "
            f"{row['feature_k']:<4} "
            f"row="
            f"{row['max_abs_row_marginal']:.3e} "
            f"column="
            f"{row['max_abs_column_marginal']:.3e} "
            f"grand="
            f"{row['abs_grand_mean']:.3e}"
        )

    print(
        "\nGLOBAL ZERO-MARGINAL ERRORS"
    )

    print(
        "Maximum absolute row marginal: "
        f"{global_max_row_marginal:.12e}"
    )

    print(
        "Maximum absolute column marginal: "
        f"{global_max_column_marginal:.12e}"
    )

    print(
        "Maximum absolute purified grand mean: "
        f"{global_max_grand_mean:.12e}"
    )

    # =========================================================
    # SAVE
    # =========================================================

    output_dir = Path(
        "results"
    )

    output_dir.mkdir(
        exist_ok=True
    )

    summary_path = (
        output_dir
        / "s1_purified_end_to_end_audit.csv"
    )

    summary_row = {
        "dataset": "S1",
        "seed": seed,
        "n": len(X),
        "n_retained_interactions": (
            len(
                retained_pairs
            )
        ),
        "reference_size": (
            len(
                reference_data
            )
        ),
        "raw_auroc": (
            raw_metrics[
                "auroc"
            ]
        ),
        "purified_auroc": (
            purified_metrics[
                "auroc"
            ]
        ),
        "raw_auprc": (
            raw_metrics[
                "auprc"
            ]
        ),
        "purified_auprc": (
            purified_metrics[
                "auprc"
            ]
        ),
        "raw_balanced_accuracy": (
            raw_metrics[
                "balanced_accuracy"
            ]
        ),
        "purified_balanced_accuracy": (
            purified_metrics[
                "balanced_accuracy"
            ]
        ),
        "raw_f1": (
            raw_metrics[
                "f1"
            ]
        ),
        "purified_f1": (
            purified_metrics[
                "f1"
            ]
        ),
        "max_logit_difference": (
            max_logit_difference
        ),
        "max_probability_difference": (
            max_probability_difference
        ),
        "max_additive_reconstruction_error": (
            max_additive_reconstruction_error
        ),
        "max_internal_purification_error": (
            purified
            .max_logit_invariance_error
        ),
        "max_row_marginal_error": (
            global_max_row_marginal
        ),
        "max_column_marginal_error": (
            global_max_column_marginal
        ),
        "max_purified_grand_mean": (
            global_max_grand_mean
        ),
        "retained_pairs": "|".join(
            f"{first}*{second}"
            for first, second
            in retained_pairs
        ),
    }

    pd.DataFrame(
        [
            summary_row
        ]
    ).to_csv(
        summary_path,
        index=False,
    )

    interaction_path = (
        output_dir
        / "s1_purified_interaction_marginal_audit.csv"
    )

    pd.DataFrame(
        interaction_audit_rows
    ).to_csv(
        interaction_path,
        index=False,
    )

    print(
        "\nSaved ignored results:"
    )

    print(
        summary_path
    )

    print(
        interaction_path
    )


if __name__ == "__main__":
    main()