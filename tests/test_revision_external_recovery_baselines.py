import numpy as np

from agnam.revision.external_recovery_baselines import (
    canonical_pair,
    deterministic_ranking,
    evaluate_pair_ranking,
    full_pair_universe,
)


def test_full_pair_universe_and_canonicalization():
    names = ("x1", "x2", "x3")
    assert full_pair_universe(names) == (
        ("x1", "x2"),
        ("x1", "x3"),
        ("x2", "x3"),
    )
    assert canonical_pair("x3", "x1") == ("x1", "x3")


def test_deterministic_ranking_ties():
    scores = {
        ("x2", "x3"): 0.5,
        ("x1", "x3"): 0.5,
        ("x1", "x2"): 0.9,
    }
    assert deterministic_ranking(scores) == (
        ("x1", "x2", 0.9),
        ("x1", "x3", 0.5),
        ("x2", "x3", 0.5),
    )


def test_ranking_metrics_perfect():
    names = ("x1", "x2", "x3", "x4")
    scores = {
        ("x1", "x2"): 4.0,
        ("x1", "x3"): 3.0,
        ("x1", "x4"): 0.2,
        ("x2", "x3"): 0.1,
        ("x2", "x4"): 0.0,
        ("x3", "x4"): -0.1,
    }
    metrics = evaluate_pair_ranking(
        feature_names=names,
        pair_scores=scores,
        true_pairs=(("x1", "x2"), ("x1", "x3")),
        candidate_k=2,
    )
    assert metrics.interaction_auprc == 1.0
    assert metrics.interaction_auroc == 1.0
    assert metrics.precision_at_k == 1.0
    assert metrics.recall_at_k == 1.0
    assert metrics.f1_at_k == 1.0


def test_pair_universe_mismatch_rejected():
    names = ("x1", "x2", "x3")
    scores = {
        ("x1", "x2"): 1.0,
    }
    try:
        evaluate_pair_ranking(
            feature_names=names,
            pair_scores=scores,
            true_pairs=(("x1", "x2"),),
            candidate_k=1,
        )
    except ValueError as exc:
        assert "Pair-score universe mismatch" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
