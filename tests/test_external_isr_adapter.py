from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from agnam.revision.external_isr_adapter import (
    ExternalRunInput,
    build_external_isr_view,
)
from agnam.training.isr import ISRConfig, evaluate_isr_for_discovery


FEATURES = ("x1", "x2", "x3")


def _run(index: int, top_pair=("x1", "x2")) -> ExternalRunInput:
    scores = {
        ("x1", "x2"): 0.9 if top_pair == ("x1", "x2") else 0.1,
        ("x1", "x3"): 0.9 if top_pair == ("x1", "x3") else 0.2,
        ("x2", "x3"): 0.3,
    }
    return ExternalRunInput(
        run_index=index,
        seed=40000 + index,
        train_indices=np.array([0, 1]),
        validation_indices=np.array([2, 3]),
        scoring_indices=np.array([4, 5]),
        train_residuals=np.array([0.1, -0.2]),
        validation_residuals=np.array([0.3, -0.4]),
        pair_scores=scores,
    )


def _build(runs):
    return build_external_isr_view(
        runs=runs,
        feature_names=FEATURES,
        n_development_rows=6,
        k=1,
        selection_threshold=0.60,
    )


def test_repeated_selection_and_top_k():
    view = _build([_run(0), _run(1), _run(2, ("x1", "x3"))])
    accepted = view.selection.accepted_interactions
    assert len(accepted) == 1
    assert (accepted[0].feature_j, accepted[0].feature_k) == ("x1", "x2")
    assert accepted[0].selection_count == 2
    assert accepted[0].selection_probability == pytest.approx(2 / 3)
    assert view.runs[0].top_k_pairs == (("x1", "x2"),)
    assert view.runs[2].top_k_pairs == (("x1", "x3"),)


def test_missing_pair_rejected():
    bad_scores = {("x1", "x2"): 1.0, ("x1", "x3"): 0.5}
    with pytest.raises(ValueError, match="Incomplete pair universe"):
        _build([replace(_run(0), pair_scores=bad_scores), _run(1)])


def test_duplicate_undirected_pair_rejected():
    bad_scores = dict(_run(0).pair_scores)
    bad_scores[("x2", "x1")] = 0.9
    with pytest.raises(ValueError, match="Duplicate unordered pair"):
        _build([replace(_run(0), pair_scores=bad_scores), _run(1)])


def test_nonfinite_scores_rejected():
    bad_scores = dict(_run(0).pair_scores)
    bad_scores[("x1", "x2")] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        _build([replace(_run(0), pair_scores=bad_scores), _run(1)])


def test_overlapping_splits_rejected():
    with pytest.raises(ValueError, match="disjoint"):
        _build([replace(_run(0), scoring_indices=np.array([3, 5])), _run(1)])


def test_residual_length_rejected():
    with pytest.raises(ValueError, match="train_residuals"):
        _build([replace(_run(0), train_residuals=np.array([0.1])), _run(1)])


def test_real_isr_entry_point_accepts_view_without_candidates():
    view = build_external_isr_view(
        runs=[_run(0), _run(1, ("x1", "x3"))],
        feature_names=FEATURES,
        n_development_rows=6,
        k=1,
        selection_threshold=1.0,
    )
    assert view.selection.accepted_interactions == ()
    X = pd.DataFrame({"x1": range(6), "x2": range(10, 16), "x3": range(20, 26)})
    result = evaluate_isr_for_discovery(
        X=X,
        discovery=view,
        config=ISRConfig(reference_size=2),
        verbose=False,
    )
    assert result.retained_interactions == ()


@pytest.mark.parametrize(
    ("computed_isr", "expected_retained"),
    [(0.80, True), (0.40, False)],
)
def test_real_isr_entry_point_with_selected_pair(
    monkeypatch: pytest.MonkeyPatch,
    computed_isr: float,
    expected_retained: bool,
):
    """Exercise the nonempty ISR path, without performing neural training.

    Selection, split handling, preprocessing, model construction and ISR
    result assembly use production code. The expensive residual-model fit,
    surface evaluation, and ISR score are controlled test doubles.
    """
    from types import SimpleNamespace

    import agnam.training.isr as isr_module

    view = _build([_run(0), _run(1), _run(2)])
    assert len(view.selection.accepted_interactions) == 1

    calls = {"trained": 0, "grids": 0, "agreements": 0}

    def fake_train(**kwargs):
        calls["trained"] += 1
        assert len(kwargs["train_residuals"]) == 2
        assert len(kwargs["val_residuals"]) == 2
        return SimpleNamespace(best_epoch=0, best_validation_mse=0.01)

    def fake_grid(**kwargs):
        calls["grids"] += 1
        return object()

    def fake_agreement(vectors):
        calls["agreements"] += 1
        assert len(vectors) == 3
        return SimpleNamespace(
            isr=computed_isr,
            pairwise_correlations=(computed_isr,) * 3,
        )

    monkeypatch.setattr(isr_module, "train_pairwise_residual_network", fake_train)
    monkeypatch.setattr(isr_module, "evaluate_pairwise_raw_grid", fake_grid)
    monkeypatch.setattr(
        isr_module,
        "diagonal_purified_vector",
        lambda _grid: np.array([0.1, 0.2, 0.3], dtype=float),
    )
    monkeypatch.setattr(isr_module, "compute_isr", fake_agreement)

    X = pd.DataFrame(
        {"x1": [0, 1, 2, 3, 4, 5],
         "x2": [10, 11, 12, 13, 14, 15],
         "x3": [20, 21, 22, 23, 24, 25]}
    )
    result = evaluate_isr_for_discovery(
        X=X,
        discovery=view,
        config=ISRConfig(reference_size=2, isr_threshold=0.60),
        verbose=False,
    )
    assert calls == {"trained": 3, "grids": 3, "agreements": 1}
    assert len(result.interactions) == 1
    assert result.interactions[0].final_accepted is expected_retained
    assert len(result.retained_interactions) == int(expected_retained)
