"""Reviewer-only adapter: external interaction proposals to AG-NAM ISR.

This module does NOT train external models or manufacture AG-NAM proposer
artifacts. It packages independently computed full pair rankings together
with genuine, split-aligned MainEffectNAM residual targets.

The returned view provides the attributes consumed by
``evaluate_isr_for_discovery`` (``runs``, ``selection``, run split arrays,
residual arrays, seeds, and ``top_k_pairs``). It is intentionally separate
from the original AG-NAM ``InteractionDiscoveryRun`` data class.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Sequence

import numpy as np

from agnam.interpretation.interaction_metrics import canonical_pair
from agnam.interpretation.interaction_scoring import RankedInteraction
from agnam.interpretation.selection_reproducibility import (
    SelectionReproducibilityResult,
    summarize_selection_reproducibility,
)

Pair = tuple[str, str]


@dataclass(frozen=True)
class ExternalRunInput:
    """One *actual* external scoring run and its aligned residual targets.

    All indices address rows of the same outer-development dataframe.
    ``pair_scores`` must cover every unordered feature pair. The residuals
    must be generated using the supplied train/validation split; this
    provenance cannot be established from arrays alone and must be logged
    by the experiment runner.
    """

    run_index: int
    seed: int
    train_indices: np.ndarray
    validation_indices: np.ndarray
    scoring_indices: np.ndarray
    train_residuals: np.ndarray
    validation_residuals: np.ndarray
    pair_scores: Mapping[Pair, float]


@dataclass(frozen=True)
class ExternalProposalRun:
    """ISR-facing run without fictitious AG-NAM proposer model results."""

    run_index: int
    seed: int
    train_indices: np.ndarray
    validation_indices: np.ndarray
    scoring_indices: np.ndarray
    train_residuals: np.ndarray
    validation_residuals: np.ndarray
    k: int
    ranking: tuple[RankedInteraction, ...]

    @property
    def top_k_pairs(self) -> tuple[Pair, ...]:
        return tuple(
            canonical_pair(item.feature_j, item.feature_k)
            for item in self.ranking[: self.k]
        )


@dataclass(frozen=True)
class ExternalISRDiscoveryView:
    """The actual, narrowly defined interface consumed by AG-NAM ISR."""

    runs: tuple[ExternalProposalRun, ...]
    selection: SelectionReproducibilityResult


def _indices(values: np.ndarray, n_rows: int, label: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0 or array.dtype.kind not in "iu":
        raise ValueError(f"{label} must be a nonempty 1-D integer array")
    array = np.array(array, dtype=np.int64, copy=True)
    if (array < 0).any() or (array >= n_rows).any():
        raise ValueError(f"{label} contains an out-of-bounds index")
    if np.unique(array).size != array.size:
        raise ValueError(f"{label} contains duplicate indices")
    return array


def _residuals(values: np.ndarray, n_rows: int, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size != n_rows or not np.isfinite(array).all():
        raise ValueError(f"{label} must contain {n_rows} finite residuals")
    return np.array(array, dtype=np.float64, copy=True)


def _ranking(
    scores: Mapping[Pair, float],
    feature_names: tuple[str, ...],
) -> tuple[RankedInteraction, ...]:
    lookup = {name: i for i, name in enumerate(feature_names)}
    expected = {
        canonical_pair(a, b) for a, b in combinations(feature_names, 2)
    }
    normalized: dict[Pair, float] = {}
    for pair, value in scores.items():
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError("Each pair-score key must be a two-name tuple")
        a, b = pair
        if a not in lookup or b not in lookup or a == b:
            raise ValueError(f"Invalid pair: {pair!r}")
        key = canonical_pair(a, b)
        if key in normalized:
            raise ValueError(f"Duplicate unordered pair: {key}")
        score = float(value)
        if not np.isfinite(score):
            raise ValueError(f"Nonfinite score for pair: {key}")
        normalized[key] = score
    if set(normalized) != expected:
        raise ValueError(
            f"Incomplete pair universe: expected {len(expected)}, "
            f"received {len(normalized)}"
        )
    ordered = sorted(normalized.items(), key=lambda item: (-item[1], item[0]))
    return tuple(
        RankedInteraction(
            feature_j=pair[0],
            feature_k=pair[1],
            index_j=lookup[pair[0]],
            index_k=lookup[pair[1]],
            score=score,
            rank=rank,
        )
        for rank, (pair, score) in enumerate(ordered, start=1)
    )


def build_external_isr_view(
    *,
    runs: Sequence[ExternalRunInput],
    feature_names: Sequence[str],
    n_development_rows: int,
    k: int,
    selection_threshold: float = 0.60,
) -> ExternalISRDiscoveryView:
    """Validate and assemble five-run external proposals for common ISR.

    No ground-truth interaction labels are accepted by this function.
    The caller must create residual targets from each run's *actual*
    training/validation data and record their provenance separately.
    """
    features = tuple(feature_names)
    if (
        len(features) < 2
        or any(not isinstance(name, str) or not name for name in features)
        or len(set(features)) != len(features)
    ):
        raise ValueError("feature_names must be unique, nonempty strings")
    if not isinstance(n_development_rows, int) or n_development_rows < 3:
        raise ValueError("n_development_rows must be at least 3")
    n_pairs = len(features) * (len(features) - 1) // 2
    if not isinstance(k, int) or not 1 <= k <= n_pairs:
        raise ValueError("k must be between 1 and the total pair count")
    if not 0.0 < selection_threshold <= 1.0:
        raise ValueError("selection_threshold must lie in (0, 1]")
    if len(runs) < 2:
        raise ValueError("ISR requires at least two independent runs")

    prepared: list[ExternalProposalRun] = []
    seen_run_indices: set[int] = set()
    seen_seeds: set[int] = set()
    for run in runs:
        if run.run_index in seen_run_indices or run.seed in seen_seeds:
            raise ValueError("Run indices and seeds must be unique")
        seen_run_indices.add(run.run_index)
        seen_seeds.add(run.seed)
        train = _indices(run.train_indices, n_development_rows, "train_indices")
        validation = _indices(
            run.validation_indices, n_development_rows, "validation_indices"
        )
        scoring = _indices(
            run.scoring_indices, n_development_rows, "scoring_indices"
        )
        if (
            np.intersect1d(train, validation).size
            or np.intersect1d(train, scoring).size
            or np.intersect1d(validation, scoring).size
        ):
            raise ValueError("Train, validation and scoring splits must be disjoint")
        if np.unique(np.concatenate([train, validation, scoring])).size != n_development_rows:
            raise ValueError("The three splits must cover all development rows")
        prepared.append(
            ExternalProposalRun(
                run_index=int(run.run_index),
                seed=int(run.seed),
                train_indices=train,
                validation_indices=validation,
                scoring_indices=scoring,
                train_residuals=_residuals(
                    run.train_residuals, len(train), "train_residuals"
                ),
                validation_residuals=_residuals(
                    run.validation_residuals,
                    len(validation),
                    "validation_residuals",
                ),
                k=k,
                ranking=_ranking(run.pair_scores, features),
            )
        )
    prepared.sort(key=lambda run: run.run_index)
    selection = summarize_selection_reproducibility(
        tuple(run.ranking for run in prepared),
        k=k,
        threshold=selection_threshold,
    )
    if any(item.selection_count < 2 for item in selection.accepted_interactions):
        raise ValueError("A selected interaction has fewer than two supporting runs")
    return ExternalISRDiscoveryView(runs=tuple(prepared), selection=selection)
