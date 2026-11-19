from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from agnam.revision.strength_fnr_analysis import (
    audit_realx_frame,
    summarize_strength_fnr,
    run_strength_fnr_analysis,
)
from agnam.revision.predictive_threshold_runner import (
    verify_r4_threshold_parity,
    _pairs_hash,
)


def sample_realx() -> pd.DataFrame:
    rows = []
    for task in range(9):
        for strength, tp in (("weak", 0), ("moderate", 1), ("strong", 2)):
            for realization in range(5):
                rows.append({
                    "task_id": task,
                    "strength_name": strength,
                    "realization_index": realization,
                    "n_true_interactions": 3,
                    "selection_recall": tp / 3,
                    "isr_recall": max(tp - (1 if task == 0 and realization == 0 else 0), 0) / 3,
                    "delta_auroc": .01,
                })
        for realization in range(5):
            rows.append({
                "task_id": task,
                "strength_name": np.nan,
                "realization_index": realization,
                "n_true_interactions": 0,
                "selection_recall": np.nan,
                "isr_recall": np.nan,
                "delta_auroc": 0,
            })
    return pd.DataFrame(rows)


def test_realx_audit_135_positive_and_45_null():
    pos, null = audit_realx_frame(sample_realx())
    assert len(pos) == 135
    assert len(null) == 45
    assert pos.selection_tp.sum() == 135
    assert pos.isr_tp.sum() == 133
    assert pos.tp_lost_by_isr.sum() == 2


def test_fnr_bootstrap_is_task_clustered_deterministic():
    pos, _ = audit_realx_frame(sample_realx())
    a, da = summarize_strength_fnr(pos, n_boot=200, seed=123)
    b, db = summarize_strength_fnr(pos, n_boot=200, seed=123)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(da, db)
    assert a.n_true.tolist() == [135, 135, 135]
    assert np.isclose(a.loc[a.strength_name == "weak", "isr_fnr"].iloc[0], 1.)
    assert np.all(a.isr_fnr_ci_low <= a.isr_fnr)
    assert np.all(a.isr_fnr <= a.isr_fnr_ci_high)


def test_null_runs_never_count_as_zero_fnr():
    pos, null = audit_realx_frame(sample_realx())
    assert (null.n_true_interactions == 0).all()
    assert "selection_fnr" not in null.columns
    assert "isr_fnr" not in null.columns
    assert pos.n_true_interactions.min() == 3


def test_bad_realx_recovery_is_rejected():
    frame = sample_realx()
    frame.loc[0, "isr_recall"] = 1
    with pytest.raises(ValueError, match="ISR TP exceeds"):
        audit_realx_frame(frame)


def test_noninteger_tp_is_rejected():
    frame = sample_realx()
    frame.loc[0, "selection_recall"] = 0.2
    with pytest.raises(ValueError, match="Non-integer implied TP"):
        audit_realx_frame(frame)


def test_null_mislabeled_rejected():
    frame = sample_realx()
    frame.loc[frame.n_true_interactions == 0, "strength_name"] = "weak"
    with pytest.raises(ValueError, match="Null runs"):
        audit_realx_frame(frame)


def test_r9a_outputs_and_manifest(tmp_path):
    source = tmp_path / "realx.csv"
    sample_realx().to_csv(source, index=False)
    out = run_strength_fnr_analysis(source, tmp_path / "out", n_boot=200)
    assert all(p.exists() for p in out.values())
    m = json.loads(out["manifest"].read_text(encoding="utf8"))
    assert m["n_positive_runs"] == 135
    assert m["n_null_runs"] == 45
    assert m["n_task_clusters"] == 9
    assert len(pd.read_csv(out["summary"])) == 3
    assert len(pd.read_csv(out["per_task"])) == 27


def _grid_data():
    data = []
    for s in (40, 60, 80):
        for i in (40, 60, 80):
            data.append({
                "family": "threshold_grid",
                "config_id": f"threshold_s{s:02d}_i{i:02d}",
                "selection_threshold": s / 100,
                "isr_threshold": i / 100,
                "n_discovery_runs": 5,
                "candidate_k": 19,
                "n_selection_retained": 3,
                "selection_tp": 1,
                "selection_fp": 2,
                "selection_fn": 2,
                "selection_precision": 1/3,
                "selection_recall": 1/3,
                "selection_f1": 1/3,
                "n_isr_retained": 2,
                "isr_tp": 1,
                "isr_fp": 1,
                "isr_fn": 2,
                "isr_precision": 0.5,
                "isr_recall": 1/3,
                "isr_f1": 0.4,
                "fp_reduction_vs_selection": 0.5,
            })
    return pd.DataFrame(data)


def test_r4_parity_full_9_row_grid():
    df = _grid_data()
    verify_r4_threshold_parity(df, df)
    bad = df.copy()
    bad.loc[3, "isr_tp"] = 0
    with pytest.raises(ValueError, match="parity failure"):
        verify_r4_threshold_parity(bad, df)


def test_r4_parity_refuses_incomplete():
    df = _grid_data()
    with pytest.raises(ValueError, match="9"):
        verify_r4_threshold_parity(df.iloc[:8], df)


def test_pair_signature_order_invariant():
    a = (("a", "b"), ("b", "c"))
    b = (("c", "b"), ("b", "a"))
    assert _pairs_hash(a) == _pairs_hash(b)
    assert _pairs_hash(()) != _pairs_hash(a)
