import pandas as pd

from agnam.revision.sensitivity_analysis import (
    configuration_summary,
    paired_vs_reference,
)


def _data():
    rows = []
    for realization in (1, 2):
        for config_id, value, is_reference in (
            ("runs_03", 0.4, False),
            ("runs_05", 0.5, True),
            ("runs_10", 0.6, False),
        ):
            row = {
                "scenario": "S1",
                "realization_number": realization,
                "family": "run_count",
                "config_id": config_id,
                "selection_threshold": 0.6,
                "isr_threshold": 0.6,
                "n_discovery_runs": int(config_id.split("_")[1]),
                "k_mode": "locked",
                "candidate_k": 19,
                "reference_size": 512,
                "is_reference": is_reference,
            }
            for metric in (
                "n_selection_retained",
                "selection_tp",
                "selection_fp",
                "selection_fn",
                "selection_precision",
                "selection_recall",
                "selection_f1",
                "n_isr_retained",
                "isr_tp",
                "isr_fp",
                "isr_fn",
                "isr_precision",
                "isr_recall",
                "isr_f1",
                "fp_reduction_vs_selection",
            ):
                row[metric] = value
            rows.append(row)
    return pd.DataFrame(rows)


def test_configuration_summary():
    summary = configuration_summary(_data())
    assert len(summary) == 3
    ref = summary[summary["config_id"] == "runs_05"].iloc[0]
    assert ref["isr_f1_mean"] == 0.5


def test_paired_vs_reference_direction():
    paired = paired_vs_reference(
        _data(),
        bootstrap_seed=1,
        bootstrap_resamples=100,
    )
    row = paired[
        (paired["config_id"] == "runs_10")
        & (paired["metric"] == "isr_f1")
    ].iloc[0]

    assert row["mean_difference_config_minus_reference"] > 0
    assert row["wins_for_config"] == 2
    assert row["losses_for_config"] == 0
