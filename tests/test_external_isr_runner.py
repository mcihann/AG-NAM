"""Fast, deterministic tests of the reviewer-only hybrid runner orchestration.

All expensive training/scoring is mocked. These tests prove integration and
cache/seed/split wiring, not the empirical quality of any baseline method.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from agnam.revision import external_isr_runner as runner


def _dataset():
    rng = np.random.default_rng(123)
    X = pd.DataFrame(rng.normal(size=(120, 4)), columns=[f"x{i}" for i in range(1, 5)])
    y = pd.Series(np.tile([0, 1], 60))
    return SimpleNamespace(X=X, y=y, true_interactions=(("x1", "x2"),))


def _stub_expensive_components(monkeypatch):
    counts = {"residuals": 0, "scores": 0, "isr": 0}
    monkeypatch.setattr(runner, "get_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(
        runner, "generate_synthetic_scenario", lambda scenario, seed: _dataset()
    )

    def fake_residual_targets(**kwargs):
        counts["residuals"] += 1
        assert len(kwargs["X_train"]) == len(kwargs["y_train"])
        assert len(kwargs["X_validation"]) == len(kwargs["y_validation"])
        assert kwargs["crossfit_n_splits"] == 5
        return SimpleNamespace(
            train_residuals=np.linspace(0.0, 0.5, len(kwargs["X_train"])),
            validation_residuals=np.linspace(0.0, 0.5, len(kwargs["X_validation"])),
        )

    monkeypatch.setattr(
        runner, "generate_train_validation_residual_targets", fake_residual_targets
    )

    def fake_friedman(**kwargs):
        counts["scores"] += 1
        assert kwargs["X_fit"].shape[0] > 0
        assert kwargs["X_score"].shape[0] > 0
        names = kwargs["feature_names"]
        scores = {
            (names[i], names[j]): float(100 - 10 * i - j)
            for i in range(len(names)) for j in range(i + 1, len(names))
        }
        return scores, 0.25, 0.05, 0.0, 0.0, 0.0

    monkeypatch.setattr(runner, "friedman_h_pair_scores", fake_friedman)

    def fake_isr(*, X, discovery, config, device, verbose):
        counts["isr"] += 1
        assert len(discovery.runs) == 5
        assert config.isr_threshold == 0.6
        candidates = discovery.selection.accepted_interactions
        assert all(candidate.selection_count >= 3 for candidate in candidates)
        objects = tuple(SimpleNamespace(
            feature_j=candidate.feature_j,
            feature_k=candidate.feature_k,
            selection_count=candidate.selection_count,
            selection_probability=candidate.selection_probability,
            isr=0.99 if candidate.feature_j == "x1" else 0.1,
            final_accepted=(candidate.feature_j == "x1"),
            surface_runs=(object(), object(), object()),
        ) for candidate in candidates)
        return SimpleNamespace(
            interactions=objects,
            retained_interactions=tuple(x for x in objects if x.final_accepted),
        )

    monkeypatch.setattr(runner, "evaluate_isr_for_discovery", fake_isr)
    return counts


def test_locked_split_is_disjoint_and_covers_all_rows():
    y = np.tile([0, 1], 50)
    a, b, c = runner._discovery_split(y, 40000)
    assert tuple(map(len, (a, b, c))) == (60, 20, 20)
    assert sorted(np.concatenate((a, b, c)).tolist()) == list(range(100))
    assert not set(a) & set(b) and not set(a) & set(c) and not set(b) & set(c)
    assert all(set(np.unique(y[idx])) == {0, 1} for idx in (a, b, c))


def test_realization_proposal_cache_then_isr_then_resume(monkeypatch, tmp_path):
    counts = _stub_expensive_components(monkeypatch)
    root = tmp_path / "results"
    output = runner.run_external_isr_hybrid_realization(
        scenario="S1", realization_number=1,
        methods=("friedman_h_gbdt",), stage="proposals", output_root=root,
    )
    assert counts == {"residuals": 5, "scores": 5, "isr": 0}
    for i in range(1, 6):
        frame = pd.read_csv(
            output / "proposal_runs" / "friedman_h_gbdt" / f"run_{i:02d}_scores.csv"
        )
        assert list(frame.columns) == ["feature_a", "feature_b", "score", "rank"]
        assert "is_true" not in frame
        assert len(frame) == 6
        assert (output / "residual_runs" / f"run_{i:02d}.npz").exists()
    output2 = runner.run_external_isr_hybrid_realization(
        scenario="S1", realization_number=1,
        methods=("friedman_h_gbdt",), stage="isr", output_root=root,
    )
    assert output2 == output
    assert counts["isr"] == 1
    result = pd.read_csv(output / "friedman_h_gbdt_summary.csv").iloc[0]
    assert result["n_runs"] == 5
    assert result["n_isr"] <= result["n_selection"]
    assert result["selection_tp"] >= result["isr_tp"]
    assert result["method"] == "friedman_h_gbdt"
    runner.run_external_isr_hybrid_realization(
        scenario="S1", realization_number=1,
        methods=("friedman_h_gbdt",), stage="both", output_root=root,
    )
    assert counts == {"residuals": 5, "scores": 5, "isr": 1}


def test_corrupt_proposal_metadata_is_rejected(monkeypatch, tmp_path):
    _stub_expensive_components(monkeypatch)
    root = tmp_path / "results"
    folder = runner.run_external_isr_hybrid_realization(
        scenario="S1", realization_number=1,
        methods=("friedman_h_gbdt",), stage="proposals", output_root=root,
    )
    path = folder / "proposal_runs" / "friedman_h_gbdt" / "run_01_manifest.json"
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace('"dataset_seed": 10000', '"dataset_seed": 99999'),
                    encoding="utf-8")
    with pytest.raises(ValueError, match="metadata mismatch"):
        runner.run_external_isr_hybrid_realization(
            scenario="S1", realization_number=1,
            methods=("friedman_h_gbdt",), stage="isr", output_root=root,
        )


def test_invalid_method_and_stage_rejected(tmp_path):
    with pytest.raises(ValueError, match="Invalid or duplicate"):
        runner.run_external_isr_hybrid_realization(
            scenario="S1", realization_number=1,
            methods=("friedman_h_gbdt", "friedman_h_gbdt"),
            output_root=tmp_path,
        )
    with pytest.raises(ValueError, match="stage must"):
        runner.run_external_isr_hybrid_realization(
            scenario="S1", realization_number=1,
            stage="unknown", output_root=tmp_path,
        )


def test_corrupt_residual_metadata_is_rejected(monkeypatch, tmp_path):
    _stub_expensive_components(monkeypatch)
    root = tmp_path / "results"
    folder = runner.run_external_isr_hybrid_realization(
        scenario="S1", realization_number=1,
        methods=("friedman_h_gbdt",), stage="proposals", output_root=root,
    )
    path = folder / "residual_runs" / "run_01.json"
    data = path.read_text(encoding="utf-8")
    path.write_text(data.replace('"run_seed": 40000', '"run_seed": 99999'),
                    encoding="utf-8")
    with pytest.raises(ValueError, match="Residual cache metadata mismatch"):
        runner.run_external_isr_hybrid_realization(
            scenario="S1", realization_number=1,
            methods=("friedman_h_gbdt",), stage="isr", output_root=root,
        )


def test_proposals_never_access_ground_truth(monkeypatch, tmp_path):
    _stub_expensive_components(monkeypatch)
    real_prepare = runner._prepare_development

    class GuardedDataset:
        @property
        def X(self):
            return _dataset().X
        @property
        def y(self):
            return _dataset().y
        @property
        def true_interactions(self):
            raise AssertionError("Truth accessed during proposals")

    monkeypatch.setattr(
        runner, "generate_synthetic_scenario",
        lambda scenario, seed: GuardedDataset(),
    )
    runner.run_external_isr_hybrid_realization(
        scenario="S1", realization_number=1,
        methods=("friedman_h_gbdt",), stage="proposals", output_root=tmp_path,
    )
