from __future__ import annotations

from agnam.interpretation.interaction_scoring import RankedInteraction
from agnam.revision.attention_ablation import ATTENTION_SCORE_MODES
from agnam.revision.attention_ablation_runner import (
    summarize_attention_ablation_rankings,
)


def _ranking(order, scores):
    return tuple(
        RankedInteraction(
            feature_j=pair[0],
            feature_k=pair[1],
            index_j=0,
            index_k=1,
            score=float(score),
            rank=index + 1,
        )
        for index, (pair, score) in enumerate(zip(order, scores))
    )


def test_attention_ablation_summary_uses_matched_runs_and_stability():
    pairs = [("x1", "x2"), ("x1", "x3"), ("x2", "x3")]

    run_a = _ranking(pairs, [0.9, 0.5, 0.1])
    run_b = _ranking(
        [("x1", "x2"), ("x2", "x3"), ("x1", "x3")],
        [0.8, 0.4, 0.2],
    )

    rankings = {
        mode: (run_a, run_b)
        for mode in ATTENTION_SCORE_MODES
    }

    run_metrics, summary, full_rankings = summarize_attention_ablation_rankings(
        rankings,
        true_pairs=(("x1", "x2"),),
        k=1,
        selection_threshold=0.60,
    )

    assert len(run_metrics) == 6
    assert len(summary) == 3
    assert len(full_rankings) == 18

    for _, row in summary.iterrows():
        assert row["interaction_auprc_mean"] == 1.0
        assert row["precision_at_k_mean"] == 1.0
        assert row["recall_at_k_mean"] == 1.0
        assert row["f1_at_k_mean"] == 1.0
        assert row["mean_pairwise_jaccard"] == 1.0
        assert row["selection_stable_count"] == 1
        assert row["selection_stable_precision"] == 1.0
        assert row["selection_stable_recall"] == 1.0
        assert row["selection_stable_f1"] == 1.0


def test_attention_ablation_summary_preserves_all_score_modes():
    ranking = _ranking(
        [("x1", "x2"), ("x1", "x3"), ("x2", "x3")],
        [0.9, 0.5, 0.1],
    )

    rankings = {
        mode: (ranking,)
        for mode in ATTENTION_SCORE_MODES
    }

    _, summary, _ = summarize_attention_ablation_rankings(
        rankings,
        true_pairs=(("x1", "x2"),),
        k=1,
        selection_threshold=0.60,
    )

    assert set(summary["mode"]) == set(ATTENTION_SCORE_MODES)
