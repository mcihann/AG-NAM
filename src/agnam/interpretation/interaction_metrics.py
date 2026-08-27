from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score

from agnam.interpretation.interaction_scoring import RankedInteraction


Pair = tuple[str, str]


@dataclass(frozen=True)
class InteractionRecoveryMetrics:
    k: int
    precision_at_k: float
    recall_at_k: float
    ndcg_at_k: float
    interaction_auprc: float
    exact_recovery_at_k: float
    false_discovery_rate_at_k: float
    true_ranks: dict[Pair, int]


def canonical_pair(
    feature_a: str,
    feature_b: str,
) -> Pair:
    if feature_a == feature_b:
        raise ValueError(
            "An interaction requires two distinct features."
        )

    return tuple(
        sorted(
            (feature_a, feature_b)
        )
    )


def evaluate_interaction_ranking(
    ranked_interactions: tuple[RankedInteraction, ...],
    true_interactions: tuple[Pair, ...],
    *,
    k: int,
) -> InteractionRecoveryMetrics:
    if k <= 0:
        raise ValueError(
            "k must be positive."
        )

    if len(ranked_interactions) == 0:
        raise ValueError(
            "ranked_interactions cannot be empty."
        )

    true_pairs = {
        canonical_pair(a, b)
        for a, b in true_interactions
    }

    if len(true_pairs) == 0:
        raise ValueError(
            "At least one true interaction is required."
        )

    ranked_pairs: list[Pair] = []
    scores: list[float] = []

    for item in ranked_interactions:
        pair = canonical_pair(
            item.feature_j,
            item.feature_k,
        )

        if pair in ranked_pairs:
            raise ValueError(
                f"Duplicate ranked interaction: {pair}"
            )

        ranked_pairs.append(pair)
        scores.append(float(item.score))

    ranked_pair_set = set(ranked_pairs)

    missing_true_pairs = (
        true_pairs - ranked_pair_set
    )

    if missing_true_pairs:
        raise ValueError(
            "True interactions are missing from the "
            f"ranking universe: {sorted(missing_true_pairs)}"
        )

    k_effective = min(
        k,
        len(ranked_pairs),
    )

    top_k = ranked_pairs[
        :k_effective
    ]

    hits = sum(
        pair in true_pairs
        for pair in top_k
    )

    precision_at_k = (
        hits / k_effective
    )

    recall_at_k = (
        hits / len(true_pairs)
    )

    false_discovery_rate = (
        1.0 - precision_at_k
    )

    exact_recovery = float(
        hits == len(true_pairs)
    )

    relevance = np.asarray(
        [
            1.0 if pair in true_pairs else 0.0
            for pair in ranked_pairs
        ],
        dtype=np.float64,
    )

    score_array = np.asarray(
        scores,
        dtype=np.float64,
    )

    interaction_auprc = float(
        average_precision_score(
            relevance,
            score_array,
        )
    )

    top_relevance = relevance[
        :k_effective
    ]

    discounts = np.log2(
        np.arange(
            2,
            k_effective + 2,
            dtype=np.float64,
        )
    )

    dcg = float(
        np.sum(
            top_relevance
            / discounts
        )
    )

    ideal_hits = min(
        len(true_pairs),
        k_effective,
    )

    ideal_relevance = np.zeros(
        k_effective,
        dtype=np.float64,
    )

    ideal_relevance[
        :ideal_hits
    ] = 1.0

    idcg = float(
        np.sum(
            ideal_relevance
            / discounts
        )
    )

    ndcg_at_k = (
        dcg / idcg
        if idcg > 0.0
        else 0.0
    )

    true_ranks = {
        pair: (
            ranked_pairs.index(pair)
            + 1
        )
        for pair in sorted(true_pairs)
    }

    return InteractionRecoveryMetrics(
        k=k_effective,
        precision_at_k=float(
            precision_at_k
        ),
        recall_at_k=float(
            recall_at_k
        ),
        ndcg_at_k=float(
            ndcg_at_k
        ),
        interaction_auprc=(
            interaction_auprc
        ),
        exact_recovery_at_k=(
            exact_recovery
        ),
        false_discovery_rate_at_k=float(
            false_discovery_rate
        ),
        true_ranks=true_ranks,
    )