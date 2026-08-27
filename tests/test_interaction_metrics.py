import numpy as np
import pytest

from agnam.interpretation.interaction_metrics import (
    canonical_pair,
    evaluate_interaction_ranking,
)
from agnam.interpretation.interaction_scoring import (
    RankedInteraction,
)


def make_ranked(
    pairs_and_scores,
):
    return tuple(
        RankedInteraction(
            feature_j=a,
            feature_k=b,
            index_j=0,
            index_k=1,
            score=score,
            rank=rank,
        )
        for rank, (
            a,
            b,
            score,
        ) in enumerate(
            pairs_and_scores,
            start=1,
        )
    )


def test_canonical_pair_is_order_invariant():
    assert canonical_pair(
        "x4",
        "x3",
    ) == canonical_pair(
        "x3",
        "x4",
    )


def test_perfect_top_k_recovery():
    ranked = make_ranked(
        [
            ("x1", "x2", 0.9),
            ("x3", "x4", 0.8),
            ("x1", "x3", 0.2),
            ("x2", "x4", 0.1),
        ]
    )

    metrics = (
        evaluate_interaction_ranking(
            ranked_interactions=ranked,
            true_interactions=(
                ("x1", "x2"),
                ("x3", "x4"),
            ),
            k=2,
        )
    )

    assert np.isclose(
        metrics.precision_at_k,
        1.0,
    )

    assert np.isclose(
        metrics.recall_at_k,
        1.0,
    )

    assert np.isclose(
        metrics.ndcg_at_k,
        1.0,
    )

    assert np.isclose(
        metrics.interaction_auprc,
        1.0,
    )

    assert metrics.exact_recovery_at_k == 1.0


def test_imperfect_top_k_recovery():
    ranked = make_ranked(
        [
            ("x1", "x3", 0.9),
            ("x1", "x2", 0.8),
            ("x2", "x4", 0.4),
            ("x3", "x4", 0.3),
        ]
    )

    metrics = (
        evaluate_interaction_ranking(
            ranked_interactions=ranked,
            true_interactions=(
                ("x1", "x2"),
                ("x3", "x4"),
            ),
            k=2,
        )
    )

    assert np.isclose(
        metrics.precision_at_k,
        0.5,
    )

    assert np.isclose(
        metrics.recall_at_k,
        0.5,
    )

    assert metrics.exact_recovery_at_k == 0.0


def test_true_ranks_are_reported():
    ranked = make_ranked(
        [
            ("a", "b", 0.9),
            ("a", "c", 0.7),
            ("b", "c", 0.5),
        ]
    )

    metrics = (
        evaluate_interaction_ranking(
            ranked_interactions=ranked,
            true_interactions=(
                ("a", "c"),
                ("b", "c"),
            ),
            k=2,
        )
    )

    assert metrics.true_ranks[
        canonical_pair("a", "c")
    ] == 2

    assert metrics.true_ranks[
        canonical_pair("b", "c")
    ] == 3


def test_duplicate_ranking_raises():
    ranked = make_ranked(
        [
            ("a", "b", 0.9),
            ("b", "a", 0.8),
        ]
    )

    with pytest.raises(ValueError):
        evaluate_interaction_ranking(
            ranked_interactions=ranked,
            true_interactions=(
                ("a", "b"),
            ),
            k=1,
        )


def test_invalid_k_raises():
    ranked = make_ranked(
        [
            ("a", "b", 0.9),
        ]
    )

    with pytest.raises(ValueError):
        evaluate_interaction_ranking(
            ranked_interactions=ranked,
            true_interactions=(
                ("a", "b"),
            ),
            k=0,
        )