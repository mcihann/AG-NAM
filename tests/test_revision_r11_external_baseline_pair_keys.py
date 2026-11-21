"""Regression tests for nonalphabetical feature names in baseline pair projection."""

import pytest

from agnam.revision.external_baseline_protocol import (
    deterministic_rank_pair_scores,
    project_scored_interaction_sets_to_pairs,
    unordered_pairs,
    validate_full_pair_scores,
)


def test_r11_nonalphabetical_features_project_without_keyerror():
    names = ("z", "a", "m")
    scores = project_scored_interaction_sets_to_pairs(names, [(("z", "a"), 0.9)])
    assert set(scores) == set(unordered_pairs(names))
    assert len(scores) == 3
    assert scores[("z", "a")] == pytest.approx(0.9)
    assert scores[("z", "m")] == pytest.approx(0.0)
    assert scores[("a", "m")] == pytest.approx(0.0)
    validate_full_pair_scores(names, scores)


def test_r11_scored_set_member_order_is_immaterial():
    names = ("z", "a", "m")
    left = project_scored_interaction_sets_to_pairs(names, [(("a", "z"), 0.75)])
    right = project_scored_interaction_sets_to_pairs(names, [(("z", "a"), 0.75)])
    assert left == right
    assert left[("z", "a")] == pytest.approx(0.75)


def test_r11_higher_order_set_projects_to_all_unordered_pairs():
    names = ("z", "a", "m")
    scores = project_scored_interaction_sets_to_pairs(
        names, [(("m", "z", "a"), 0.8), (("z", "a"), 0.4)]
    )
    assert set(scores) == set(unordered_pairs(names))
    assert all(value == pytest.approx(0.8) for value in scores.values())
    validate_full_pair_scores(names, scores)


def test_r11_validator_accepts_equivalent_pair_orientation():
    names = ("z", "a", "m")
    scores = {("a", "z"): 0.9, ("m", "z"): 0.2, ("a", "m"): 0.0}
    validate_full_pair_scores(names, scores)


def test_r11_validator_rejects_missing_pair():
    names = ("z", "a", "m")
    with pytest.raises(ValueError, match="Pair universe mismatch"):
        validate_full_pair_scores(names, {("z", "a"): 0.9})


def test_r11_unknown_feature_is_rejected():
    with pytest.raises(ValueError, match="Unknown features"):
        project_scored_interaction_sets_to_pairs(
            ("z", "a", "m"), [(("z", "does_not_exist"), 0.9)]
        )


def test_r11_ranking_is_deterministic_after_projection():
    scores = project_scored_interaction_sets_to_pairs(
        ("z", "a", "m"), [(("z", "a"), 0.9), (("m", "a"), 0.4)]
    )
    assert deterministic_rank_pair_scores(scores) == (
        ("a", "z", 0.9),
        ("a", "m", 0.4),
        ("m", "z", 0.0),
    )


def test_r11_alphabetical_feature_order_remains_compatible():
    names = ("a", "m", "z")
    scores = project_scored_interaction_sets_to_pairs(names, [(("z", "a"), 0.9)])
    assert scores[("a", "z")] == pytest.approx(0.9)
    validate_full_pair_scores(names, scores)
