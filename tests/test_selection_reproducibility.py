import numpy as np
import pytest
import torch

from agnam.data.synthetic import generate_s1
from agnam.interpretation.interaction_scoring import (
    RankedInteraction,
)
from agnam.interpretation.selection_reproducibility import (
    summarize_selection_reproducibility,
)
from agnam.training.discovery import (
    run_single_interaction_discovery,
)
from agnam.training.nam_trainer import (
    NAMTrainingConfig,
)
from agnam.training.proposer_trainer import (
    ProposerTrainingConfig,
)


def make_ranking(
    ordered_pairs,
):
    return tuple(
        RankedInteraction(
            feature_j=a,
            feature_k=b,
            index_j=0,
            index_k=1,
            score=float(
                len(ordered_pairs)
                - index
            ),
            rank=index + 1,
        )
        for index, (
            a,
            b,
        ) in enumerate(
            ordered_pairs
        )
    )


def small_nam_config(
    seed: int,
) -> NAMTrainingConfig:
    return NAMTrainingConfig(
        learning_rate=1e-3,
        weight_decay=1e-5,
        batch_size=128,
        max_epochs=2,
        patience=2,
        seed=seed,
        deterministic=True,
    )


def small_proposer_config(
    seed: int,
) -> ProposerTrainingConfig:
    return ProposerTrainingConfig(
        learning_rate=1e-3,
        weight_decay=1e-5,
        batch_size=128,
        max_epochs=2,
        patience=2,
        seed=seed,
        deterministic=True,
    )


def test_selection_probabilities_are_correct():
    universe = [
        ("a", "b"),
        ("a", "c"),
        ("a", "d"),
        ("b", "c"),
        ("b", "d"),
        ("c", "d"),
    ]

    run1 = make_ranking(
        universe
    )

    run2 = make_ranking(
        [
            ("a", "b"),
            ("a", "d"),
            ("a", "c"),
            ("b", "c"),
            ("b", "d"),
            ("c", "d"),
        ]
    )

    run3 = make_ranking(
        [
            ("a", "b"),
            ("a", "c"),
            ("b", "d"),
            ("a", "d"),
            ("b", "c"),
            ("c", "d"),
        ]
    )

    result = (
        summarize_selection_reproducibility(
            rankings=(
                run1,
                run2,
                run3,
            ),
            k=2,
            threshold=0.60,
        )
    )

    lookup = {
        (
            item.feature_j,
            item.feature_k,
        ): item
        for item in result.interactions
    }

    assert np.isclose(
        lookup[
            ("a", "b")
        ].selection_probability,
        1.0,
    )

    assert np.isclose(
        lookup[
            ("a", "c")
        ].selection_probability,
        2.0 / 3.0,
    )


def test_threshold_controls_acceptance():
    rankings = (
        make_ranking(
            [
                ("a", "b"),
                ("a", "c"),
                ("b", "c"),
            ]
        ),
        make_ranking(
            [
                ("a", "c"),
                ("a", "b"),
                ("b", "c"),
            ]
        ),
        make_ranking(
            [
                ("a", "b"),
                ("b", "c"),
                ("a", "c"),
            ]
        ),
    )

    result = (
        summarize_selection_reproducibility(
            rankings=rankings,
            k=1,
            threshold=0.60,
        )
    )

    lookup = {
        (
            item.feature_j,
            item.feature_k,
        ): item
        for item in result.interactions
    }

    assert lookup[
        ("a", "b")
    ].accepted

    assert not lookup[
        ("a", "c")
    ].accepted


def test_full_ranks_are_preserved():
    rankings = (
        make_ranking(
            [
                ("a", "b"),
                ("a", "c"),
                ("b", "c"),
            ]
        ),
        make_ranking(
            [
                ("b", "c"),
                ("a", "c"),
                ("a", "b"),
            ]
        ),
    )

    result = (
        summarize_selection_reproducibility(
            rankings=rankings,
            k=1,
            threshold=0.50,
        )
    )

    lookup = {
        (
            item.feature_j,
            item.feature_k,
        ): item
        for item in result.interactions
    }

    assert lookup[
        ("a", "b")
    ].ranks == (
        1,
        3,
    )

    assert np.isclose(
        lookup[
            ("a", "b")
        ].mean_rank,
        2.0,
    )


def test_jaccard_is_valid():
    rankings = (
        make_ranking(
            [
                ("a", "b"),
                ("a", "c"),
                ("b", "c"),
            ]
        ),
        make_ranking(
            [
                ("a", "b"),
                ("b", "c"),
                ("a", "c"),
            ]
        ),
    )

    result = (
        summarize_selection_reproducibility(
            rankings=rankings,
            k=2,
            threshold=0.60,
        )
    )

    # {ab, ac} vs {ab, bc}
    # intersection = 1
    # union = 3
    assert np.isclose(
        result.mean_pairwise_jaccard,
        1.0 / 3.0,
    )


def test_mismatched_ranking_universe_raises():
    first = make_ranking(
        [
            ("a", "b"),
            ("a", "c"),
            ("b", "c"),
        ]
    )

    second = make_ranking(
        [
            ("a", "b"),
            ("a", "d"),
            ("b", "c"),
        ]
    )

    with pytest.raises(
        ValueError
    ):
        summarize_selection_reproducibility(
            rankings=(
                first,
                second,
            ),
            k=2,
        )


def test_single_discovery_run_partitions_are_disjoint():
    dataset = generate_s1(
        seed=7,
        n=300,
    )

    result = (
        run_single_interaction_discovery(
            X=dataset.X,
            y=dataset.y.to_numpy(),
            run_index=0,
            seed=7,
            crossfit_n_splits=2,
            main_early_stop_fraction=0.20,
            nam_hidden_dim=8,
            nam_depth=1,
            nam_dropout=0.0,
            proposer_d_model=16,
            proposer_n_heads=4,
            proposer_n_layers=1,
            proposer_dropout=0.0,
            nam_config=small_nam_config(
                seed=7
            ),
            proposer_config=small_proposer_config(
                seed=7
            ),
            device=torch.device(
                "cpu"
            ),
        )
    )

    train = set(
        result.train_indices.tolist()
    )

    validation = set(
        result.validation_indices.tolist()
    )

    scoring = set(
        result.scoring_indices.tolist()
    )

    assert train.isdisjoint(
        validation
    )

    assert train.isdisjoint(
        scoring
    )

    assert validation.isdisjoint(
        scoring
    )

    combined = (
        train
        | validation
        | scoring
    )

    assert combined == set(
        range(300)
    )

    assert len(
        result.top_k_pairs
    ) == 19


def test_discovery_run_retains_residual_targets():
    """
    Regression test for ISR integration.

    A discovery run must retain the exact leakage-safe residual
    targets that were used during proposer training and validation.
    """
    dataset = generate_s1(
        seed=17,
        n=300,
    )

    result = (
        run_single_interaction_discovery(
            X=dataset.X,
            y=dataset.y.to_numpy(),
            run_index=0,
            seed=17,
            crossfit_n_splits=2,
            main_early_stop_fraction=0.20,
            nam_hidden_dim=8,
            nam_depth=1,
            nam_dropout=0.0,
            proposer_d_model=16,
            proposer_n_heads=4,
            proposer_n_layers=1,
            proposer_dropout=0.0,
            nam_config=small_nam_config(
                seed=17
            ),
            proposer_config=small_proposer_config(
                seed=17
            ),
            device=torch.device(
                "cpu"
            ),
        )
    )

    assert result.train_residuals.shape == (
        len(
            result.train_indices
        ),
    )

    assert (
        result.validation_residuals.shape
        == (
            len(
                result.validation_indices
            ),
        )
    )

    assert np.isfinite(
        result.train_residuals
    ).all()

    assert np.isfinite(
        result.validation_residuals
    ).all()

    assert np.all(
        result.train_residuals >= -1.0
    )

    assert np.all(
        result.train_residuals <= 1.0
    )

    assert np.all(
        result.validation_residuals >= -1.0
    )

    assert np.all(
        result.validation_residuals <= 1.0
    )