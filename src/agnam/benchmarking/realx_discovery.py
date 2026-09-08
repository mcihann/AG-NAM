from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import pandas as pd
import numpy as np
import torch

import agnam.training.discovery as discovery


def validate_realx_candidate_k(
    candidate_k: int,
    *,
    available_ranked_pairs: int,
) -> int:
    """
    Validate the externally locked Real-X candidate count.

    Real-X defines K independently from the canonical synthetic
    discovery rule. The proposer still produces the complete
    interaction ranking; only the number of top-ranked candidates
    entering reproducibility selection is overridden.
    """
    candidate_k = int(
        candidate_k
    )

    available_ranked_pairs = int(
        available_ranked_pairs
    )

    if candidate_k < 1:
        raise ValueError(
            "Real-X candidate_k must be at least 1."
        )

    if available_ranked_pairs < 1:
        raise ValueError(
            "Interaction ranking contains no available pairs."
        )

    if candidate_k > available_ranked_pairs:
        raise ValueError(
            "Real-X candidate_k exceeds the number "
            "of available ranked feature pairs. "
            f"K={candidate_k}, "
            f"available={available_ranked_pairs}."
        )

    return candidate_k


def override_discovery_candidate_k(
    base_result,
    *,
    candidate_k: int,
    selection_threshold: float,
):
    """
    Replace the canonical synthetic Top-K value after complete
    interaction scoring and recompute reproducibility selection.

    No residual target, proposer model, attention score, ranking,
    train/validation/scoring split, or random seed is changed.

    This is mathematically equivalent to supplying K to the
    canonical discovery engine because K is used only after
    compute_interaction_scores() has produced the complete ranking.
    """
    runs = tuple(
        base_result.runs
    )

    if len(
        runs
    ) == 0:
        raise RuntimeError(
            "Discovery returned no repeated runs."
        )

    ranked_pair_counts = tuple(
        len(
            run
            .score_result
            .ranked_interactions
        )
        for run in runs
    )

    if len(
        set(
            ranked_pair_counts
        )
    ) != 1:
        raise RuntimeError(
            "Repeated discovery runs produced "
            "different ranking lengths."
        )

    candidate_k = (
        validate_realx_candidate_k(
            candidate_k,
            available_ranked_pairs=(
                ranked_pair_counts[
                    0
                ]
            ),
        )
    )

    updated_runs = tuple(
        replace(
            run,
            k=candidate_k,
        )
        for run in runs
    )

    if not all(
        int(
            run.k
        )
        == candidate_k
        for run in updated_runs
    ):
        raise RuntimeError(
            "Unable to apply the locked "
            "Real-X candidate K."
        )

    rankings = tuple(
        run
        .score_result
        .ranked_interactions
        for run in updated_runs
    )

    selection = (
        discovery
        .summarize_selection_reproducibility(
            rankings=rankings,
            k=candidate_k,
            threshold=(
                selection_threshold
            ),
        )
    )

    return type(
        base_result
    )(
        runs=updated_runs,
        selection=selection,
    )


def run_realx_reproducible_interaction_discovery(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    candidate_k: int,
    n_runs: int = 5,
    base_seed: int = 42,
    selection_threshold: float = 0.60,
    numeric_features: Sequence[str] | None = None,
    categorical_features: Sequence[str] | None = None,
    crossfit_n_splits: int = 5,
    main_early_stop_fraction: float = 0.20,
    nam_hidden_dim: int = 64,
    nam_depth: int = 2,
    nam_dropout: float = 0.10,
    categorical_embedding_dim: int = 16,
    proposer_d_model: int = 64,
    proposer_n_heads: int = 4,
    proposer_n_layers: int = 2,
    proposer_dropout: float = 0.10,
    nam_config=None,
    proposer_config=None,
    device: torch.device | None = None,
):
    """
    Run the canonical AG-NAM discovery pipeline and apply the
    prespecified Real-X candidate-count rule to its complete
    interaction rankings.

    The canonical discovery implementation is deliberately left
    unchanged so all existing synthetic benchmarks preserve their
    original K rule.
    """
    base_result = (
        discovery
        .run_reproducible_interaction_discovery(
            X=X,
            y=y,
            n_runs=n_runs,
            base_seed=base_seed,
            selection_threshold=(
                selection_threshold
            ),
            numeric_features=(
                numeric_features
            ),
            categorical_features=(
                categorical_features
            ),
            crossfit_n_splits=(
                crossfit_n_splits
            ),
            main_early_stop_fraction=(
                main_early_stop_fraction
            ),
            nam_hidden_dim=(
                nam_hidden_dim
            ),
            nam_depth=(
                nam_depth
            ),
            nam_dropout=(
                nam_dropout
            ),
            categorical_embedding_dim=(
                categorical_embedding_dim
            ),
            proposer_d_model=(
                proposer_d_model
            ),
            proposer_n_heads=(
                proposer_n_heads
            ),
            proposer_n_layers=(
                proposer_n_layers
            ),
            proposer_dropout=(
                proposer_dropout
            ),
            nam_config=(
                nam_config
            ),
            proposer_config=(
                proposer_config
            ),
            device=device,
        )
    )

    return override_discovery_candidate_k(
        base_result,
        candidate_k=candidate_k,
        selection_threshold=(
            selection_threshold
        ),
    )