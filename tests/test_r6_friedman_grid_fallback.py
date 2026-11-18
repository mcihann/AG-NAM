"""Regression tests for R6's percentile-grid failure on skewed features."""
import numpy as np
import pytest
from agnam.revision import external_recovery_baselines as b


def sample():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(400, 4))
    X[:, 1] = 0.
    X[:12, 1] = np.arange(1, 13)
    y = (X[:, 0] + X[:, 2] * X[:, 3] > 0).astype(int)
    return X, y


def test_skewed_support_fallback_and_pair_axes(monkeypatch):
    X, y = sample()
    args = dict(X_fit=X, y_fit=y, X_score=X, feature_names=tuple('abcd'), seed=42)
    with pytest.raises(ValueError, match='percentiles are too close'):
        b.friedman_h_pair_scores(**args)
    original = b.partial_dependence
    calls = []
    def spy(model, X, features, **kwargs):
        res = original(model, X, features, **kwargs)
        calls.append((features, res['grid_values']))
        return res
    monkeypatch.setattr(b, 'partial_dependence', spy)
    affected = []
    scores, *_ = b.friedman_h_pair_scores(**args, allow_degenerate_grid_fallback=True,
                                         grid_fallback_features=affected)
    assert affected == ['b']
    assert set(scores) == set(b.full_pair_universe(tuple('abcd')))
    assert np.isfinite(list(scores.values())).all()
    singles = {fs[0]: grids[0] for fs, grids in calls if len(fs) == 1 and isinstance(fs[0], int)}
    assert len(singles[1]) == 10
    assert singles[1][0] == 0 and singles[1][-1] == 12
    assert set(singles[1]).issubset(set(X[:, 1]))
    for fs, grids in calls:
        if len(grids) == 2:
            indices = fs[0] if isinstance(fs[0], tuple) else fs
            for j, grid in zip(indices, grids):
                np.testing.assert_array_equal(grid, singles[j])


def test_regular_support_scores_unchanged():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(200, 3)); y = (X[:, 0] + X[:, 1]*X[:, 2] > 0).astype(int)
    args = dict(X_fit=X, y_fit=y, X_score=X, feature_names=tuple('abc'), seed=42)
    original, *_ = b.friedman_h_pair_scores(**args)
    affected = []
    fixed, *_ = b.friedman_h_pair_scores(**args, allow_degenerate_grid_fallback=True,
                                         grid_fallback_features=affected)
    assert affected == []
    assert fixed == original


def test_unrelated_errors_not_swallowed(monkeypatch):
    X, y = sample()
    def fail(*args, **kwargs):
        raise ValueError('unrelated error')
    monkeypatch.setattr(b, 'partial_dependence', fail)
    with pytest.raises(ValueError, match='unrelated error'):
        b.friedman_h_pair_scores(X_fit=X, y_fit=y, X_score=X, feature_names=tuple('abcd'),
                               seed=42, allow_degenerate_grid_fallback=True)
