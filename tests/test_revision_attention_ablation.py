import numpy as np
import pandas as pd
import torch

from agnam.data.preprocessing import TabularPreprocessor
from agnam.interpretation.interaction_scoring import (
    compute_interaction_scores,
)
from agnam.models.attention_proposer import ResidualAttentionProposer
from agnam.revision.attention_ablation import (
    ATTENTION_SCORE_MODES,
    compute_attention_score_ablation,
)


def _build_model_and_data():
    X = pd.DataFrame(
        {
            "x1": [0.1, 0.2, 0.3, 0.4, 0.5],
            "x2": [1.0, 2.0, 1.5, 2.5, 3.0],
            "x3": [-1.0, 0.0, 1.0, 0.5, -0.5],
            "group": ["A", "B", "A", "B", "C"],
        }
    )

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


def test_attention_ablation_returns_all_modes():
    model, transformed = _build_model_and_data()

    result = compute_attention_score_ablation(
        model=model,
        transformed=transformed,
        batch_size=2,
        device=torch.device("cpu"),
    )

    for mode in ATTENTION_SCORE_MODES:
        score_result = result.by_mode(mode)
        assert score_result.symmetric_score_matrix.shape == (4, 4)
        assert len(score_result.ranked_interactions) == 6
        assert np.isfinite(score_result.symmetric_score_matrix).all()
        assert np.all(score_result.symmetric_score_matrix >= 0.0)
        np.testing.assert_allclose(
            score_result.symmetric_score_matrix,
            score_result.symmetric_score_matrix.T,
            atol=1e-12,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            np.diag(score_result.symmetric_score_matrix),
            0.0,
            atol=0.0,
        )


def test_grad_x_attention_matches_locked_primary_scorer():
    model, transformed = _build_model_and_data()

    primary = compute_interaction_scores(
        model=model,
        transformed=transformed,
        batch_size=2,
        device=torch.device("cpu"),
    )

    ablation = compute_attention_score_ablation(
        model=model,
        transformed=transformed,
        batch_size=2,
        device=torch.device("cpu"),
    )

    np.testing.assert_allclose(
        primary.symmetric_score_matrix,
        ablation.grad_x_attention.symmetric_score_matrix,
        atol=1e-12,
        rtol=1e-12,
    )

    primary_pairs = [
        (item.feature_j, item.feature_k)
        for item in primary.ranked_interactions
    ]
    ablation_pairs = [
        (item.feature_j, item.feature_k)
        for item in ablation.grad_x_attention.ranked_interactions
    ]

    assert primary_pairs == ablation_pairs


def test_attention_ablation_is_batch_size_invariant():
    model, transformed = _build_model_and_data()

    first = compute_attention_score_ablation(
        model=model,
        transformed=transformed,
        batch_size=1,
        device=torch.device("cpu"),
    )

    second = compute_attention_score_ablation(
        model=model,
        transformed=transformed,
        batch_size=5,
        device=torch.device("cpu"),
    )

    for mode in ATTENTION_SCORE_MODES:
        np.testing.assert_allclose(
            first.by_mode(mode).symmetric_score_matrix,
            second.by_mode(mode).symmetric_score_matrix,
            atol=1e-7,
            rtol=1e-6,
        )
