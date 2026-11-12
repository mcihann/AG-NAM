from pathlib import Path

import pandas as pd
import pytest

from agnam.revision.attention_ablation_analysis import (
    analyze_synthetic_attention_ablation,
    load_synthetic_attention_summaries,
)


def _row(realization: int, mode: str, value: float) -> dict:
    return {
        "scenario": "S1",
        "realization_number": realization,
        "mode": mode,
        "interaction_auprc_mean": value,
        "precision_at_k_mean": value,
        "recall_at_k_mean": value,
        "f1_at_k_mean": value,
        "ndcg_at_k_mean": value,
        "mean_pairwise_jaccard": value,
        "selection_stable_precision": value,
        "selection_stable_recall": value,
        "selection_stable_f1": value,
        "selection_stable_count": value,
        "selection_stable_tp": value,
        "selection_stable_fp": value,
        "selection_stable_fn": value,
    }


def _write_realization(root: Path, realization: int) -> None:
    directory = root / "S1"
    directory.mkdir(parents=True, exist_ok=True)
    rows = [
        _row(realization, "raw_attention", 0.1),
        _row(realization, "gradient_only", 0.4),
        _row(realization, "grad_x_attention", 0.3),
    ]
    pd.DataFrame(rows).to_csv(
        directory / f"realization_{realization:02d}_summary.csv", index=False
    )


def test_incomplete_scenario_is_rejected_by_default(tmp_path: Path):
    _write_realization(tmp_path, 1)
    with pytest.raises(ValueError, match="Incomplete S1"):
        load_synthetic_attention_summaries(
            tmp_path, scenarios=("S1",), expected_realizations=2
        )


def test_paired_difference_direction_is_left_minus_right(tmp_path: Path):
    _write_realization(tmp_path, 1)
    _write_realization(tmp_path, 2)
    result = analyze_synthetic_attention_ablation(
        tmp_path,
        scenarios=("S1",),
        expected_realizations=2,
        bootstrap_resamples=100,
    )
    row = result.paired_comparisons[
        (result.paired_comparisons["metric"] == "interaction_auprc_mean")
        & (result.paired_comparisons["left_mode"] == "gradient_only")
        & (result.paired_comparisons["right_mode"] == "grad_x_attention")
    ].iloc[0]
    assert row["mean_difference"] == pytest.approx(0.1)
    assert row["wins_left"] == 2
    assert row["wins_right"] == 0
