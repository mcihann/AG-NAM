from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from agnam.benchmarking.synthetic_protocol import (
    PairSetRecovery,
    evaluate_pair_set_recovery,
)
from agnam.data.preprocessing import TabularPreprocessor
from agnam.interpretation.interaction_metrics import canonical_pair
from agnam.interpretation.surface_reproducibility import (
    compute_isr,
    diagonal_purified_vector,
    evaluate_pairwise_raw_grid,
)
from agnam.models.pairwise_interaction import PairwiseResidualNetwork
from agnam.training.discovery import ReproducibleDiscoveryResult
from agnam.training.isr import (
    ISRConfig,
    _pair_specs,
    _pairwise_seed,
    _selected_runs_for_pair,
    sample_reference_indices,
)
from agnam.training.pairwise_trainer import (
    PairwiseTrainingConfig,
    train_pairwise_residual_network,
)
from agnam.utils.reproducibility import seed_everything

Pair = tuple[str, str]


@dataclass(frozen=True)
class PairAgreementResult:
    feature_j: str
    feature_k: str
    selection_count: int
    selection_probability: float
    mean_rank: float
    raw_agreement: float
    purified_agreement: float
    n_surfaces: int

    @property
    def pair(self) -> Pair:
        return canonical_pair(self.feature_j, self.feature_k)


@dataclass(frozen=True)
class ISRNoveltyResult:
    reference_indices: np.ndarray
    pair_agreements: tuple[PairAgreementResult, ...]


@dataclass(frozen=True)
class VariantRecoveryResult:
    variant: str
    pairs: tuple[Pair, ...]
    recovery: PairSetRecovery
    false_negative_rate: float


def _raw_diagonal_vector(raw_grid: np.ndarray) -> np.ndarray:
    raw_grid = np.asarray(raw_grid, dtype=np.float64)
    if raw_grid.ndim != 2 or raw_grid.shape[0] != raw_grid.shape[1]:
        raise ValueError("raw_grid must be a square two-dimensional array.")
    if not np.isfinite(raw_grid).all():
        raise ValueError("raw_grid must contain finite values.")
    return np.diag(raw_grid).copy()


def evaluate_isr_novelty_for_discovery(
    X: pd.DataFrame,
    discovery: ReproducibleDiscoveryResult,
    *,
    numeric_features: Sequence[str] | None = None,
    categorical_features: Sequence[str] | None = None,
    config: ISRConfig | None = None,
    device: torch.device | None = None,
    verbose: bool = True,
) -> ISRNoveltyResult:
    """Fit each ISR pairwise model once and compute both raw and purified
    cross-run surface agreement on the identical reference support.

    This is a reviewer-requested post-submission analysis. It intentionally
    mirrors the locked ISR fitting/seeding path while retaining the raw
    diagonal surface vector in addition to the purified vector.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")
    if config is None:
        config = ISRConfig()

    reference_indices = sample_reference_indices(
        len(X), reference_size=config.reference_size, seed=config.reference_seed
    )
    X_reference = X.iloc[reference_indices].reset_index(drop=True)

    results: list[PairAgreementResult] = []
    accepted_candidates = discovery.selection.accepted_interactions

    if verbose:
        print("\n=== REVIEWER ISR NOVELTY EVALUATION ===")
        print("Selection-stable candidates:", len(accepted_candidates))
        print("Reference support:", len(X_reference))

    for candidate_index, candidate in enumerate(accepted_candidates, start=1):
        pair = canonical_pair(candidate.feature_j, candidate.feature_k)
        selected_runs = _selected_runs_for_pair(pair, discovery.runs)
        if len(selected_runs) != candidate.selection_count:
            raise RuntimeError(
                "Selected-run count disagrees with selection reproducibility summary."
            )
        if len(selected_runs) < 2:
            raise RuntimeError("Surface agreement requires at least two selected runs.")

        if verbose:
            print(
                f"[{candidate_index}/{len(accepted_candidates)}] "
                f"{pair[0]} x {pair[1]} "
                f"(pi={candidate.selection_probability:.2f}, runs={len(selected_runs)})"
            )

        raw_vectors: list[np.ndarray] = []
        purified_vectors: list[np.ndarray] = []

        for run in selected_runs:
            X_train = X.iloc[run.train_indices].reset_index(drop=True)
            X_validation = X.iloc[run.validation_indices].reset_index(drop=True)

            preprocessor = TabularPreprocessor(
                numeric_features=numeric_features,
                categorical_features=categorical_features,
            )
            train_data = preprocessor.fit_transform(X_train)
            validation_data = preprocessor.transform(X_validation)
            reference_data = preprocessor.transform(X_reference)

            pair_specs = _pair_specs(train_data, pair)
            pair_seed = _pairwise_seed(run.seed, pair_specs)
            seed_everything(pair_seed, deterministic=True)

            model = PairwiseResidualNetwork(
                feature_specs=pair_specs,
                feature_embedding_dim=config.feature_embedding_dim,
                hidden_dim=config.hidden_dim,
                depth=config.depth,
                dropout=config.dropout,
            )
            training_config = PairwiseTrainingConfig(
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
                batch_size=config.batch_size,
                max_epochs=config.max_epochs,
                patience=config.patience,
                min_delta=config.min_delta,
                seed=pair_seed,
                deterministic=True,
            )
            train_pairwise_residual_network(
                model=model,
                train_data=train_data,
                train_residuals=run.train_residuals,
                val_data=validation_data,
                val_residuals=run.validation_residuals,
                config=training_config,
                device=device,
            )

            raw_grid = evaluate_pairwise_raw_grid(
                model=model,
                reference_data=reference_data,
                batch_size=config.surface_batch_size,
                device=device,
            )
            raw_vectors.append(_raw_diagonal_vector(raw_grid))
            purified_vectors.append(diagonal_purified_vector(raw_grid))

        raw = compute_isr(tuple(raw_vectors))
        purified = compute_isr(tuple(purified_vectors))

        results.append(
            PairAgreementResult(
                feature_j=pair[0],
                feature_k=pair[1],
                selection_count=int(candidate.selection_count),
                selection_probability=float(candidate.selection_probability),
                mean_rank=float(candidate.mean_rank),
                raw_agreement=float(raw.isr),
                purified_agreement=float(purified.isr),
                n_surfaces=len(selected_runs),
            )
        )

        if verbose:
            print(
                f"  raw={raw.isr:.4f}; purified={purified.isr:.4f}"
            )

    results.sort(
        key=lambda item: (
            -item.selection_probability,
            item.mean_rank,
            item.feature_j,
            item.feature_k,
        )
    )
    return ISRNoveltyResult(
        reference_indices=reference_indices,
        pair_agreements=tuple(results),
    )


def build_isr_variant_sets(
    discovery: ReproducibleDiscoveryResult,
    novelty: ISRNoveltyResult,
    *,
    selection_threshold: float = 0.60,
    strict_selection_threshold: float = 0.80,
    agreement_threshold: float = 0.60,
) -> Mapping[str, tuple[Pair, ...]]:
    """Construct fixed reviewer-requested ISR novelty variants.

    count_matched_selection is restricted to the original selection-stable
    set and keeps the top N pairs by selection probability / mean rank,
    where N equals the number retained by purified ISR.
    """
    if not 0.0 < selection_threshold <= 1.0:
        raise ValueError("selection_threshold must lie in (0, 1].")
    if not 0.0 < strict_selection_threshold <= 1.0:
        raise ValueError("strict_selection_threshold must lie in (0, 1].")
    if strict_selection_threshold < selection_threshold:
        raise ValueError("strict_selection_threshold must be >= selection_threshold.")
    if not 0.0 < agreement_threshold <= 1.0:
        raise ValueError("agreement_threshold must lie in (0, 1].")

    agreement_by_pair = {item.pair: item for item in novelty.pair_agreements}

    stable_summaries = tuple(
        item
        for item in discovery.selection.interactions
        if item.selection_probability >= selection_threshold
    )
    stable_pairs = tuple(
        canonical_pair(item.feature_j, item.feature_k)
        for item in stable_summaries
    )

    missing = [pair for pair in stable_pairs if pair not in agreement_by_pair]
    if missing:
        raise ValueError(f"Missing surface-agreement results for stable pairs: {missing}")

    strict_pairs = tuple(
        canonical_pair(item.feature_j, item.feature_k)
        for item in discovery.selection.interactions
        if item.selection_probability >= strict_selection_threshold
    )

    raw_pairs = tuple(
        pair
        for pair in stable_pairs
        if agreement_by_pair[pair].raw_agreement >= agreement_threshold
    )
    purified_pairs = tuple(
        pair
        for pair in stable_pairs
        if agreement_by_pair[pair].purified_agreement >= agreement_threshold
    )

    n_target = len(purified_pairs)
    count_matched_pairs = stable_pairs[:n_target]

    return {
        "selection_stable_060": stable_pairs,
        "selection_stable_080": strict_pairs,
        "count_matched_selection": count_matched_pairs,
        "raw_surface_agreement": raw_pairs,
        "purified_isr": purified_pairs,
    }


def evaluate_isr_variant_recovery(
    variants: Mapping[str, tuple[Pair, ...]],
    *,
    true_pairs: Sequence[Pair],
) -> tuple[VariantRecoveryResult, ...]:
    true_pairs = tuple(canonical_pair(*pair) for pair in true_pairs)
    n_true = len(set(true_pairs))
    results: list[VariantRecoveryResult] = []
    for name, pairs in variants.items():
        recovery = evaluate_pair_set_recovery(
            predicted_pairs=pairs,
            true_pairs=true_pairs,
        )
        fnr = (
            float(recovery.false_negative / n_true)
            if n_true > 0
            else float("nan")
        )
        results.append(
            VariantRecoveryResult(
                variant=name,
                pairs=tuple(pairs),
                recovery=recovery,
                false_negative_rate=fnr,
            )
        )
    return tuple(results)
