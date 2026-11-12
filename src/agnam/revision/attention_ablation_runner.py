from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from agnam.benchmarking.synthetic_protocol import (
    SyntheticBenchmarkProtocol,
    SyntheticRealizationSpec,
    candidate_count_from_feature_count,
    evaluate_pair_set_recovery,
)
from agnam.benchmarking.synthetic_runner import (
    SyntheticModelConfig,
    generate_synthetic_scenario,
)
from agnam.data.preprocessing import TabularPreprocessor
from agnam.interpretation.interaction_metrics import (
    canonical_pair,
    evaluate_interaction_ranking,
)
from agnam.interpretation.interaction_scoring import RankedInteraction
from agnam.interpretation.selection_reproducibility import (
    summarize_selection_reproducibility,
)
from agnam.models.attention_proposer import ResidualAttentionProposer
from agnam.revision.attention_ablation import (
    ATTENTION_SCORE_MODES,
    AttentionScoreAblationResult,
    AttentionScoreMode,
    compute_attention_score_ablation,
)
from agnam.training.nam_trainer import NAMTrainingConfig
from agnam.training.proposer_trainer import (
    ProposerTrainingConfig,
    train_residual_attention_proposer,
)
from agnam.training.residual_targets import (
    generate_train_validation_residual_targets,
)
from agnam.utils.reproducibility import get_device, seed_everything


Pair = tuple[str, str]


@dataclass(frozen=True)
class AttentionAblationRealizationResult:
    """Reviewer-requested attention-score ablation outputs."""

    run_metrics: pd.DataFrame
    summary: pd.DataFrame
    rankings: pd.DataFrame


def _canonicalize_pairs(pairs) -> tuple[Pair, ...]:
    return tuple(
        sorted(
            {
                canonical_pair(first, second)
                for first, second in pairs
            }
        )
    )


def _f1_from_precision_recall(precision: float, recall: float) -> float:
    if precision + recall == 0.0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def _train_and_score_one_run(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    run_index: int,
    seed: int,
    model_config: SyntheticModelConfig,
    device: torch.device,
) -> AttentionScoreAblationResult:
    """
    Reproduce one locked AG-NAM discovery run, but compute three
    attention-derived ranking rules from the same fitted proposer.

    The original discovery implementation is intentionally not modified.
    """
    y = np.asarray(y, dtype=np.int64)

    all_indices = np.arange(len(X))

    development_indices, scoring_indices = train_test_split(
        all_indices,
        test_size=0.20,
        random_state=seed,
        stratify=y,
    )

    train_indices, validation_indices = train_test_split(
        development_indices,
        test_size=0.25,
        random_state=seed + 1,
        stratify=y[development_indices],
    )

    X_train = X.iloc[train_indices].reset_index(drop=True)
    X_validation = X.iloc[validation_indices].reset_index(drop=True)
    X_scoring = X.iloc[scoring_indices].reset_index(drop=True)

    y_train = y[train_indices]
    y_validation = y[validation_indices]

    nam_config = NAMTrainingConfig(
        learning_rate=model_config.discovery_learning_rate,
        weight_decay=model_config.discovery_weight_decay,
        batch_size=model_config.discovery_batch_size,
        max_epochs=model_config.nam_discovery_max_epochs,
        patience=model_config.nam_discovery_patience,
        seed=seed,
        deterministic=True,
    )

    proposer_config = ProposerTrainingConfig(
        learning_rate=model_config.discovery_learning_rate,
        weight_decay=model_config.discovery_weight_decay,
        batch_size=model_config.discovery_batch_size,
        max_epochs=model_config.proposer_max_epochs,
        patience=model_config.proposer_patience,
        seed=seed,
        deterministic=True,
    )

    residual_targets = generate_train_validation_residual_targets(
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        crossfit_n_splits=5,
        main_early_stop_fraction=0.20,
        hidden_dim=model_config.nam_hidden_dim,
        depth=model_config.nam_depth,
        dropout=model_config.nam_dropout,
        categorical_embedding_dim=model_config.categorical_embedding_dim,
        training_config=nam_config,
        seed=seed,
        device=device,
    )

    preprocessor = TabularPreprocessor()
    train_data = preprocessor.fit_transform(X_train)
    validation_data = preprocessor.transform(X_validation)
    scoring_data = preprocessor.transform(X_scoring)

    seed_everything(seed, deterministic=True)

    proposer = ResidualAttentionProposer(
        feature_specs=train_data.feature_specs,
        d_model=model_config.proposer_d_model,
        n_heads=model_config.proposer_n_heads,
        n_layers=model_config.proposer_n_layers,
        dropout=model_config.proposer_dropout,
    )

    train_residual_attention_proposer(
        model=proposer,
        train_data=train_data,
        train_residuals=residual_targets.train_residuals,
        val_data=validation_data,
        val_residuals=residual_targets.validation_residuals,
        config=proposer_config,
        device=device,
    )

    return compute_attention_score_ablation(
        model=proposer,
        transformed=scoring_data,
        batch_size=max(proposer_config.batch_size, 256),
        device=device,
    )


def summarize_attention_ablation_rankings(
    rankings_by_mode: Mapping[
        AttentionScoreMode,
        tuple[tuple[RankedInteraction, ...], ...],
    ],
    *,
    true_pairs: tuple[Pair, ...],
    k: int,
    selection_threshold: float = 0.60,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build run-level, realization-level, and full-ranking outputs."""
    true_pairs = _canonicalize_pairs(true_pairs)

    run_rows: list[dict] = []
    summary_rows: list[dict] = []
    ranking_rows: list[dict] = []

    expected_n_runs: int | None = None

    for mode in ATTENTION_SCORE_MODES:
        rankings = rankings_by_mode[mode]

        if expected_n_runs is None:
            expected_n_runs = len(rankings)
        elif len(rankings) != expected_n_runs:
            raise ValueError("All score modes must contain the same number of runs.")

        if len(rankings) == 0:
            raise ValueError("At least one ranking run is required.")

        mode_run_rows: list[dict] = []

        for run_index, ranking in enumerate(rankings):
            metrics = evaluate_interaction_ranking(
                ranked_interactions=ranking,
                true_interactions=true_pairs,
                k=k,
            )

            f1_at_k = _f1_from_precision_recall(
                metrics.precision_at_k,
                metrics.recall_at_k,
            )

            row = {
                "mode": mode,
                "run_index": run_index,
                "k": metrics.k,
                "interaction_auprc": metrics.interaction_auprc,
                "precision_at_k": metrics.precision_at_k,
                "recall_at_k": metrics.recall_at_k,
                "f1_at_k": f1_at_k,
                "ndcg_at_k": metrics.ndcg_at_k,
                "false_discovery_rate_at_k": metrics.false_discovery_rate_at_k,
                "exact_recovery_at_k": metrics.exact_recovery_at_k,
            }
            run_rows.append(row)
            mode_run_rows.append(row)

            true_pair_set = set(true_pairs)
            for item in ranking:
                pair = canonical_pair(item.feature_j, item.feature_k)
                ranking_rows.append(
                    {
                        "mode": mode,
                        "run_index": run_index,
                        "feature_j": pair[0],
                        "feature_k": pair[1],
                        "rank": item.rank,
                        "score": float(item.score),
                        "is_true": pair in true_pair_set,
                        "in_top_k": item.rank <= k,
                    }
                )

        reproducibility = summarize_selection_reproducibility(
            rankings=rankings,
            k=k,
            threshold=selection_threshold,
        )

        stable_pairs = tuple(
            (item.feature_j, item.feature_k)
            for item in reproducibility.accepted_interactions
        )
        stable_recovery = evaluate_pair_set_recovery(
            predicted_pairs=stable_pairs,
            true_pairs=true_pairs,
        )

        metric_frame = pd.DataFrame(mode_run_rows)

        summary_rows.append(
            {
                "mode": mode,
                "n_runs": len(rankings),
                "k": k,
                "selection_threshold": selection_threshold,
                "interaction_auprc_mean": float(metric_frame["interaction_auprc"].mean()),
                "interaction_auprc_sd": float(metric_frame["interaction_auprc"].std(ddof=1))
                if len(metric_frame) > 1
                else 0.0,
                "precision_at_k_mean": float(metric_frame["precision_at_k"].mean()),
                "recall_at_k_mean": float(metric_frame["recall_at_k"].mean()),
                "f1_at_k_mean": float(metric_frame["f1_at_k"].mean()),
                "ndcg_at_k_mean": float(metric_frame["ndcg_at_k"].mean()),
                "mean_pairwise_jaccard": reproducibility.mean_pairwise_jaccard,
                "selection_stable_count": len(stable_pairs),
                "selection_stable_tp": stable_recovery.true_positive,
                "selection_stable_fp": stable_recovery.false_positive,
                "selection_stable_fn": stable_recovery.false_negative,
                "selection_stable_precision": stable_recovery.precision,
                "selection_stable_recall": stable_recovery.recall,
                "selection_stable_f1": stable_recovery.f1,
            }
        )

    return (
        pd.DataFrame(run_rows),
        pd.DataFrame(summary_rows),
        pd.DataFrame(ranking_rows),
    )


def run_synthetic_attention_ablation_realization(
    spec: SyntheticRealizationSpec,
    *,
    protocol: SyntheticBenchmarkProtocol | None = None,
    model_config: SyntheticModelConfig | None = None,
    device: torch.device | None = None,
    verbose: bool = True,
) -> AttentionAblationRealizationResult:
    """Run the reviewer-requested attention scoring ablation."""
    if protocol is None:
        protocol = SyntheticBenchmarkProtocol()
    if model_config is None:
        model_config = SyntheticModelConfig()
    if device is None:
        device = get_device()

    dataset = generate_synthetic_scenario(
        spec.scenario,
        seed=spec.dataset_seed,
    )

    X = dataset.X
    y = dataset.y.to_numpy(dtype=np.int64)
    true_pairs = _canonicalize_pairs(dataset.true_interactions)

    all_indices = np.arange(len(X))
    outer_dev_idx, _ = train_test_split(
        all_indices,
        test_size=protocol.outer_test_fraction,
        random_state=spec.outer_split_seed,
        stratify=y,
    )

    X_outer_dev = X.iloc[outer_dev_idx].reset_index(drop=True)
    y_outer_dev = y[outer_dev_idx]

    k = candidate_count_from_feature_count(
        X.shape[1],
        protocol=protocol,
    )

    rankings_by_mode: dict[
        AttentionScoreMode,
        list[tuple[RankedInteraction, ...]],
    ] = {
        mode: [] for mode in ATTENTION_SCORE_MODES
    }

    if verbose:
        print("========================================")
        print("REVISION ATTENTION-SCORE ABLATION")
        print("========================================")
        print(f"Scenario: {spec.scenario}")
        print(f"Realization: {spec.realization_number}")
        print(f"Dataset seed: {spec.dataset_seed}")
        print(f"Discovery base seed: {spec.discovery_base_seed}")
        print(f"K: {k}")
        print(f"Device: {device}")

    for run_index, seed in enumerate(spec.discovery_seeds):
        if verbose:
            print(
                f"\nRun {run_index + 1}/{len(spec.discovery_seeds)} "
                f"(seed={seed})"
            )

        result = _train_and_score_one_run(
            X=X_outer_dev,
            y=y_outer_dev,
            run_index=run_index,
            seed=seed,
            model_config=model_config,
            device=device,
        )

        for mode in ATTENTION_SCORE_MODES:
            rankings_by_mode[mode].append(
                result.by_mode(mode).ranked_interactions
            )

    frozen_rankings = {
        mode: tuple(rankings)
        for mode, rankings in rankings_by_mode.items()
    }

    run_metrics, summary, rankings = summarize_attention_ablation_rankings(
        frozen_rankings,
        true_pairs=true_pairs,
        k=k,
        selection_threshold=protocol.selection_threshold,
    )

    metadata = {
        "scenario": spec.scenario,
        "realization_index": spec.realization_index,
        "realization_number": spec.realization_number,
        "dataset_seed": spec.dataset_seed,
        "outer_split_seed": spec.outer_split_seed,
        "discovery_base_seed": spec.discovery_base_seed,
        "n_discovery_runs": protocol.n_discovery_runs,
        "n_samples": len(X),
        "n_features": X.shape[1],
        "n_true_interactions": len(true_pairs),
    }

    for key, value in metadata.items():
        run_metrics[key] = value
        summary[key] = value
        rankings[key] = value

    run_metrics["run_seed"] = run_metrics["run_index"].map(
        lambda index: spec.discovery_base_seed + int(index)
    )
    rankings["run_seed"] = rankings["run_index"].map(
        lambda index: spec.discovery_base_seed + int(index)
    )

    run_column_order = [
        "scenario",
        "realization_index",
        "realization_number",
        "dataset_seed",
        "outer_split_seed",
        "discovery_base_seed",
        "n_discovery_runs",
        "run_index",
        "run_seed",
        "mode",
        "n_samples",
        "n_features",
        "n_true_interactions",
        "k",
        "interaction_auprc",
        "precision_at_k",
        "recall_at_k",
        "f1_at_k",
        "ndcg_at_k",
        "false_discovery_rate_at_k",
        "exact_recovery_at_k",
    ]

    summary_column_order = [
        "scenario",
        "realization_index",
        "realization_number",
        "dataset_seed",
        "outer_split_seed",
        "discovery_base_seed",
        "n_discovery_runs",
        "mode",
        "n_samples",
        "n_features",
        "n_true_interactions",
        "k",
        "selection_threshold",
        "interaction_auprc_mean",
        "interaction_auprc_sd",
        "precision_at_k_mean",
        "recall_at_k_mean",
        "f1_at_k_mean",
        "ndcg_at_k_mean",
        "mean_pairwise_jaccard",
        "selection_stable_count",
        "selection_stable_tp",
        "selection_stable_fp",
        "selection_stable_fn",
        "selection_stable_precision",
        "selection_stable_recall",
        "selection_stable_f1",
    ]

    ranking_column_order = [
        "scenario",
        "realization_index",
        "realization_number",
        "dataset_seed",
        "outer_split_seed",
        "discovery_base_seed",
        "n_discovery_runs",
        "run_index",
        "run_seed",
        "mode",
        "n_samples",
        "n_features",
        "n_true_interactions",
        "feature_j",
        "feature_k",
        "rank",
        "score",
        "is_true",
        "in_top_k",
    ]

    return AttentionAblationRealizationResult(
        run_metrics=run_metrics[run_column_order].copy(),
        summary=summary[summary_column_order].copy(),
        rankings=rankings[ranking_column_order].copy(),
    )
