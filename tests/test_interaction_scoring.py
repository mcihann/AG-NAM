import numpy as np
import pandas as pd
import torch

from agnam.data.preprocessing import TabularPreprocessor
from agnam.interpretation.interaction_scoring import (
    candidate_count,
    compute_interaction_scores,
    rank_symmetric_interactions,
)
from agnam.models.attention_proposer import ResidualAttentionProposer


def make_data():
    return pd.DataFrame(
        {
            "x1": [0.1, 0.2, 0.3, 0.4, 0.5],
            "x2": [1.0, 2.0, 1.5, 2.5, 3.0],
            "x3": [-1.0, 0.0, 1.0, 0.5, -0.5],
            "group": ["A", "B", "A", "B", "C"],
        }
    )


def build_model_and_data():
    X = make_data()

    preprocessor = TabularPreprocessor()
    transformed = preprocessor.fit_transform(X)

    torch.manual_seed(42)

    model = ResidualAttentionProposer(
        feature_specs=transformed.feature_specs,
        d_model=16,
        n_heads=4,
        n_layers=2,
        dropout=0.0,
    )

    return model, transformed


def test_candidate_count_protocol():
    assert candidate_count(5) == 5
    assert candidate_count(10) == 5
    assert candidate_count(20) == 19
    assert candidate_count(50) == 20


def test_candidate_count_handles_small_toy_models():
    assert candidate_count(1) == 0
    assert candidate_count(2) == 1
    assert candidate_count(3) == 3


def test_score_matrix_shapes():
    model, transformed = build_model_and_data()

    result = compute_interaction_scores(
        model=model,
        transformed=transformed,
        batch_size=2,
        device=torch.device("cpu"),
    )

    assert result.directed_score_matrix.shape == (
        4,
        4,
    )

    assert result.symmetric_score_matrix.shape == (
        4,
        4,
    )

    assert len(
        result.layer_score_matrices
    ) == 2


def test_symmetric_matrix_is_symmetric():
    model, transformed = build_model_and_data()

    result = compute_interaction_scores(
        model=model,
        transformed=transformed,
        batch_size=5,
        device=torch.device("cpu"),
    )

    np.testing.assert_allclose(
        result.symmetric_score_matrix,
        result.symmetric_score_matrix.T,
        atol=1e-12,
        rtol=1e-12,
    )


def test_symmetric_diagonal_is_zero():
    model, transformed = build_model_and_data()

    result = compute_interaction_scores(
        model=model,
        transformed=transformed,
        batch_size=5,
        device=torch.device("cpu"),
    )

    np.testing.assert_allclose(
        np.diag(
            result.symmetric_score_matrix
        ),
        0.0,
        atol=0.0,
    )


def test_scores_are_nonnegative_and_finite():
    model, transformed = build_model_and_data()

    result = compute_interaction_scores(
        model=model,
        transformed=transformed,
        batch_size=5,
        device=torch.device("cpu"),
    )

    assert np.isfinite(
        result.symmetric_score_matrix
    ).all()

    assert np.all(
        result.symmetric_score_matrix >= 0.0
    )


def test_ranking_contains_each_pair_once():
    model, transformed = build_model_and_data()

    result = compute_interaction_scores(
        model=model,
        transformed=transformed,
        batch_size=5,
        device=torch.device("cpu"),
    )

    assert len(
        result.ranked_interactions
    ) == 6

    seen = set()

    for item in result.ranked_interactions:
        pair = tuple(
            sorted(
                [
                    item.feature_j,
                    item.feature_k,
                ]
            )
        )

        assert pair not in seen
        seen.add(pair)


def test_manual_ranking_orders_scores_correctly():
    matrix = np.array(
        [
            [0.0, 0.2, 0.9],
            [0.2, 0.0, 0.4],
            [0.9, 0.4, 0.0],
        ]
    )

    ranked = rank_symmetric_interactions(
        symmetric_score_matrix=matrix,
        feature_names=("a", "b", "c"),
    )

    assert (
        ranked[0].feature_j,
        ranked[0].feature_k,
    ) == (
        "a",
        "c",
    )

    assert ranked[0].rank == 1

    assert (
        ranked[-1].feature_j,
        ranked[-1].feature_k,
    ) == (
        "a",
        "b",
    )


def test_scoring_is_batch_size_invariant():
    model, transformed = build_model_and_data()

    first = compute_interaction_scores(
        model=model,
        transformed=transformed,
        batch_size=1,
        device=torch.device("cpu"),
    )

    second = compute_interaction_scores(
        model=model,
        transformed=transformed,
        batch_size=5,
        device=torch.device("cpu"),
    )

    np.testing.assert_allclose(
        first.symmetric_score_matrix,
        second.symmetric_score_matrix,
        atol=1e-7,
        rtol=1e-6,
    )