from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agnam.interpretation.selection_reproducibility import (
    SelectionReproducibilityResult,
    StableInteractionSummary,
)
from agnam.revision.isr_novelty import (
    ISRNoveltyResult,
    PairAgreementResult,
    _raw_diagonal_vector,
    build_isr_variant_sets,
    evaluate_isr_variant_recovery,
)


@dataclass
class DummyDiscovery:
    selection: SelectionReproducibilityResult


def _summary(pair, count, prob, mean_rank):
    return StableInteractionSummary(
        feature_j=pair[0],
        feature_k=pair[1],
        selection_count=count,
        selection_probability=prob,
        mean_rank=mean_rank,
        median_rank=mean_rank,
        ranks=(1, 1, 1, 1, 1),
        selected_runs=(True, True, True, True, True),
        accepted=prob >= 0.60,
    )


def test_raw_diagonal_vector():
    grid = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(_raw_diagonal_vector(grid), [1.0, 4.0])


def test_variant_sets_are_count_matched_and_thresholded():
    summaries = (
        _summary(("x1", "x2"), 5, 1.0, 2.0),
        _summary(("x1", "x3"), 4, 0.8, 3.0),
        _summary(("x2", "x3"), 3, 0.6, 4.0),
        _summary(("x2", "x4"), 2, 0.4, 5.0),
    )
    selection = SelectionReproducibilityResult(
        n_runs=5,
        k=3,
        threshold=0.60,
        interactions=summaries,
        mean_pairwise_jaccard=0.5,
    )
    novelty = ISRNoveltyResult(
        reference_indices=np.arange(4),
        pair_agreements=(
            PairAgreementResult("x1", "x2", 5, 1.0, 2.0, 0.7, 0.8, 5),
            PairAgreementResult("x1", "x3", 4, 0.8, 3.0, 0.4, 0.7, 4),
            PairAgreementResult("x2", "x3", 3, 0.6, 4.0, 0.9, 0.2, 3),
        ),
    )
    variants = build_isr_variant_sets(DummyDiscovery(selection), novelty)
    assert variants["selection_stable_060"] == (
        ("x1", "x2"), ("x1", "x3"), ("x2", "x3")
    )
    assert variants["selection_stable_080"] == (("x1", "x2"), ("x1", "x3"))
    assert variants["purified_isr"] == (("x1", "x2"), ("x1", "x3"))
    assert variants["raw_surface_agreement"] == (("x1", "x2"), ("x2", "x3"))
    assert variants["count_matched_selection"] == (("x1", "x2"), ("x1", "x3"))


def test_variant_recovery_reports_false_negative_rate():
    variants = {
        "a": (("x1", "x2"),),
        "b": (("x1", "x2"), ("x3", "x4")),
    }
    results = evaluate_isr_variant_recovery(
        variants,
        true_pairs=(("x1", "x2"), ("x3", "x4")),
    )
    lookup = {item.variant: item for item in results}
    assert lookup["a"].recovery.true_positive == 1
    assert lookup["a"].false_negative_rate == 0.5
    assert lookup["b"].recovery.false_negative == 0
    assert lookup["b"].false_negative_rate == 0.0
