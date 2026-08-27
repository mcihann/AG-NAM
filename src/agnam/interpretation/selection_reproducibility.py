from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from agnam.interpretation.interaction_metrics import canonical_pair
from agnam.interpretation.interaction_scoring import RankedInteraction


Pair = tuple[str, str]


@dataclass(frozen=True)
class StableInteractionSummary:
    feature_j: str
    feature_k: str

    selection_count: int
    selection_probability: float

    mean_rank: float
    median_rank: float

    ranks: tuple[int, ...]
    selected_runs: tuple[bool, ...]

    accepted: bool


@dataclass(frozen=True)
class SelectionReproducibilityResult:
    n_runs: int
    k: int
    threshold: float

    interactions: tuple[StableInteractionSummary, ...]

    mean_pairwise_jaccard: float

    @property
    def accepted_interactions(
        self,
    ) -> tuple[StableInteractionSummary, ...]:
        return tuple(
            item
            for item in self.interactions
            if item.accepted
        )


def _ranking_to_pairs(
    ranking: tuple[RankedInteraction, ...],
) -> list[Pair]:
    pairs: list[Pair] = []

    for item in ranking:
        pair = canonical_pair(
            item.feature_j,
            item.feature_k,
        )

        if pair in pairs:
            raise ValueError(
                f"Duplicate interaction in ranking: {pair}"
            )

        pairs.append(pair)

    return pairs


def _mean_pairwise_jaccard(
    selected_sets: list[set[Pair]],
) -> float:
    if len(selected_sets) == 1:
        return 1.0

    values: list[float] = []

    for first, second in combinations(
        selected_sets,
        2,
    ):
        union = first | second

        if len(union) == 0:
            values.append(1.0)
            continue

        intersection = first & second

        values.append(
            len(intersection)
            / len(union)
        )

    return float(
        np.mean(values)
    )


def summarize_selection_reproducibility(
    rankings: tuple[
        tuple[RankedInteraction, ...],
        ...
    ],
    *,
    k: int,
    threshold: float = 0.60,
) -> SelectionReproducibilityResult:
    """
    Summarize interaction-selection reproducibility.

    Selection probability:

        pi_jk =
            number of runs where pair is in Top-K
            --------------------------------------
                         number of runs

    Full-ranking positions are retained even when an interaction
    falls outside Top-K. No artificial K+1 rank is assigned.
    """
    if len(rankings) == 0:
        raise ValueError(
            "At least one interaction ranking is required."
        )

    if not 0.0 < threshold <= 1.0:
        raise ValueError(
            "threshold must lie in (0, 1]."
        )

    reference_pairs = _ranking_to_pairs(
        rankings[0]
    )

    if len(reference_pairs) == 0:
        raise ValueError(
            "Interaction rankings cannot be empty."
        )

    if not 1 <= k <= len(reference_pairs):
        raise ValueError(
            "k must lie between 1 and the number of ranked pairs."
        )

    reference_set = set(
        reference_pairs
    )

    ranking_pairs: list[list[Pair]] = []
    selected_sets: list[set[Pair]] = []

    for ranking in rankings:
        pairs = _ranking_to_pairs(
            ranking
        )

        if set(pairs) != reference_set:
            raise ValueError(
                "All runs must rank the same interaction universe."
            )

        ranking_pairs.append(
            pairs
        )

        selected_sets.append(
            set(
                pairs[:k]
            )
        )

    summaries: list[
        StableInteractionSummary
    ] = []

    n_runs = len(
        rankings
    )

    for pair in sorted(
        reference_set
    ):
        ranks = tuple(
            pairs.index(pair) + 1
            for pairs in ranking_pairs
        )

        selected_runs = tuple(
            pair in selected
            for selected in selected_sets
        )

        selection_count = int(
            sum(selected_runs)
        )

        selection_probability = (
            selection_count / n_runs
        )

        summaries.append(
            StableInteractionSummary(
                feature_j=pair[0],
                feature_k=pair[1],
                selection_count=selection_count,
                selection_probability=float(
                    selection_probability
                ),
                mean_rank=float(
                    np.mean(ranks)
                ),
                median_rank=float(
                    np.median(ranks)
                ),
                ranks=ranks,
                selected_runs=selected_runs,
                accepted=bool(
                    selection_probability
                    >= threshold
                ),
            )
        )

    summaries.sort(
        key=lambda item: (
            -item.selection_probability,
            item.mean_rank,
            item.feature_j,
            item.feature_k,
        )
    )

    mean_jaccard = (
        _mean_pairwise_jaccard(
            selected_sets
        )
    )

    return SelectionReproducibilityResult(
        n_runs=n_runs,
        k=k,
        threshold=threshold,
        interactions=tuple(
            summaries
        ),
        mean_pairwise_jaccard=(
            mean_jaccard
        ),
    )