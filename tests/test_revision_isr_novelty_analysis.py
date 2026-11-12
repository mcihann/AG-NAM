from pathlib import Path

import pandas as pd

from agnam.revision.isr_novelty_analysis import (
    RevisionISRAnalysisConfig,
    aggregate_variant_summary,
    paired_purified_comparisons,
    load_raw_vs_purified_structure,
)


def _summary_rows(realization: int) -> pd.DataFrame:
    variants = [
        "selection_stable_060",
        "selection_stable_080",
        "count_matched_selection",
        "raw_surface_agreement",
        "purified_isr",
    ]
    rows = []
    for idx, variant in enumerate(variants):
        rows.append(
            {
                "scenario": "S1",
                "realization_number": realization,
                "variant": variant,
                "n_retained": 10 - idx,
                "true_positive": 1 + (variant == "purified_isr"),
                "false_positive": 8 - idx,
                "false_negative": 2 - (variant == "purified_isr"),
                "precision": 0.1 + 0.1 * idx,
                "recall": 0.3 + 0.1 * idx,
                "f1": 0.15 + 0.1 * idx,
                "false_negative_rate": 0.7 - 0.1 * idx,
                "fp_reduction_vs_selection": 0.1 * idx,
            }
        )
    return pd.DataFrame(rows)


def test_variant_and_paired_aggregation():
    data = pd.concat(
        [_summary_rows(1), _summary_rows(2)],
        ignore_index=True,
    )
    summary = aggregate_variant_summary(data)
    assert len(summary) == 5

    paired = paired_purified_comparisons(
        data,
        bootstrap_seed=123,
        bootstrap_resamples=100,
    )
    row = paired[
        (paired["comparison"] == "purified_isr - count_matched_selection")
        & (paired["metric"] == "f1")
    ].iloc[0]
    assert row["mean_difference"] > 0
    assert row["wins_for_purified"] == 2


def test_raw_vs_purified_structure(tmp_path: Path):
    scenario_dir = tmp_path / "S1"
    scenario_dir.mkdir(parents=True)

    frame = pd.DataFrame(
        {
            "feature_j": ["x1", "x2", "x3"],
            "feature_k": ["x4", "x5", "x6"],
            "raw_pass_060": [True, True, False],
            "purified_pass_060": [True, False, True],
        }
    )
    frame.to_csv(
        scenario_dir / "realization_01_pair_agreements.csv",
        index=False,
    )

    config = RevisionISRAnalysisConfig(
        input_root=tmp_path,
        output_root=tmp_path / "out",
        scenarios=("S1",),
        expected_realizations=1,
    )
    out = load_raw_vs_purified_structure(config)
    assert out.loc[0, "n_raw_retained"] == 2
    assert out.loc[0, "n_purified_retained"] == 2
    assert out.loc[0, "n_overlap"] == 1
    assert out.loc[0, "raw_purified_jaccard"] == 1 / 3
