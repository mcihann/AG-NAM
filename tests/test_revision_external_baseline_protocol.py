import pytest

from agnam.revision.external_baseline_protocol import (
    BASELINE_SPECS,
    REQUIRED_BASELINES,
    deterministic_rank_pair_scores,
    project_scored_interaction_sets_to_pairs,
    unordered_pairs,
    validate_full_pair_scores,
)


def test_required_reviewer_baselines_are_present():
    reviewer_names = {
        spec.name
        for spec in BASELINE_SPECS
        if spec.requested_by_reviewer
    }
    assert reviewer_names == set(REQUIRED_BASELINES)


def test_higher_order_projection_uses_max_score():
    names = ("x1", "x2", "x3", "x4")
    scores = project_scored_interaction_sets_to_pairs(
        names,
        [
            (("x1", "x2", "x3"), 0.4),
            (("x1", "x2"), 0.8),
        ],
    )

    assert scores[("x1", "x2")] == 0.8
    assert scores[("x1", "x3")] == 0.4
    assert scores[("x2", "x3")] == 0.4
    assert scores[("x1", "x4")] == 0.0

    validate_full_pair_scores(names, scores)


def test_deterministic_tie_breaking():
    scores = {
        ("x2", "x3"): 0.5,
        ("x1", "x3"): 0.5,
        ("x1", "x2"): 0.9,
    }
    ranked = deterministic_rank_pair_scores(scores)

    assert ranked[0] == ("x1", "x2", 0.9)
    assert ranked[1] == ("x1", "x3", 0.5)
    assert ranked[2] == ("x2", "x3", 0.5)


def test_pair_universe_rejects_duplicates():
    with pytest.raises(ValueError):
        unordered_pairs(("x1", "x1", "x2"))
