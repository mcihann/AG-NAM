from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

SCORE_MODES = ("raw_attention", "gradient_only", "grad_x_attention")
METRICS = (
    "interaction_auprc_mean",
    "precision_at_k_mean",
    "recall_at_k_mean",
    "f1_at_k_mean",
    "ndcg_at_k_mean",
    "mean_pairwise_jaccard",
    "selection_stable_precision",
    "selection_stable_recall",
    "selection_stable_f1",
    "selection_stable_count",
    "selection_stable_tp",
    "selection_stable_fp",
    "selection_stable_fn",
)

PAIRWISE_COMPARISONS = (
    ("gradient_only", "grad_x_attention"),
    ("grad_x_attention", "raw_attention"),
)


@dataclass(frozen=True)
class AttentionAblationAnalysis:
    realization_summary: pd.DataFrame
    mode_summary: pd.DataFrame
    paired_comparisons: pd.DataFrame


def load_synthetic_attention_summaries(
    input_root: Path,
    *,
    scenarios: Sequence[str] | None = None,
    expected_realizations: int = 20,
    allow_incomplete: bool = False,
) -> pd.DataFrame:
    input_root = Path(input_root)
    if scenarios is None:
        scenarios = tuple(
            sorted(path.name for path in input_root.iterdir() if path.is_dir())
        )
    else:
        scenarios = tuple(scenarios)

    frames: list[pd.DataFrame] = []
    for scenario in scenarios:
        scenario_dir = input_root / scenario
        files = sorted(scenario_dir.glob("realization_*_summary.csv"))
        if not files:
            raise FileNotFoundError(f"No summary CSVs found for {scenario}: {scenario_dir}")

        scenario_frames = [pd.read_csv(path) for path in files]
        frame = pd.concat(scenario_frames, ignore_index=True)

        required = {
            "scenario",
            "realization_number",
            "mode",
            *METRICS,
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(
                f"{scenario} summary files are missing columns: {sorted(missing)}"
            )

        observed_scenarios = set(frame["scenario"].astype(str))
        if observed_scenarios != {scenario}:
            raise ValueError(
                f"Scenario directory {scenario} contains mismatched labels: "
                f"{sorted(observed_scenarios)}"
            )

        counts = frame.groupby("realization_number")["mode"].nunique()
        if (counts != len(SCORE_MODES)).any():
            bad = counts[counts != len(SCORE_MODES)].to_dict()
            raise ValueError(
                f"Each realization must contain all {len(SCORE_MODES)} score modes; "
                f"violations: {bad}"
            )

        duplicates = frame.duplicated(["scenario", "realization_number", "mode"])
        if duplicates.any():
            raise ValueError(f"Duplicate realization/mode rows detected for {scenario}.")

        realized = sorted(frame["realization_number"].unique().tolist())
        if not allow_incomplete:
            expected = list(range(1, expected_realizations + 1))
            if realized != expected:
                missing_realizations = sorted(set(expected).difference(realized))
                extra_realizations = sorted(set(realized).difference(expected))
                raise ValueError(
                    f"Incomplete {scenario}: expected realizations 1-{expected_realizations}; "
                    f"missing={missing_realizations}, extra={extra_realizations}."
                )

        frames.append(frame)

    result = pd.concat(frames, ignore_index=True)
    return result.sort_values(["scenario", "realization_number", "mode"]).reset_index(
        drop=True
    )


def summarize_modes(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (scenario, mode), group in frame.groupby(["scenario", "mode"], sort=True):
        row = {
            "scenario": scenario,
            "mode": mode,
            "n_realizations": int(group["realization_number"].nunique()),
        }
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="raise").to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _bootstrap_mean_ci(
    differences: np.ndarray,
    *,
    rng: np.random.Generator,
    n_resamples: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    if differences.ndim != 1 or len(differences) == 0:
        raise ValueError("differences must be a non-empty one-dimensional array")
    indices = rng.integers(0, len(differences), size=(n_resamples, len(differences)))
    means = differences[indices].mean(axis=1)
    return (
        float(np.quantile(means, alpha / 2.0)),
        float(np.quantile(means, 1.0 - alpha / 2.0)),
    )


def paired_mode_comparisons(
    frame: pd.DataFrame,
    *,
    comparisons: Iterable[tuple[str, str]] = PAIRWISE_COMPARISONS,
    bootstrap_seed: int = 26090806,
    bootstrap_resamples: int = 10_000,
    tolerance: float = 1e-12,
) -> pd.DataFrame:
    rows: list[dict] = []
    rng = np.random.default_rng(bootstrap_seed)

    for scenario, scenario_frame in frame.groupby("scenario", sort=True):
        for left_mode, right_mode in comparisons:
            for metric in METRICS:
                pivot = scenario_frame.pivot(
                    index="realization_number",
                    columns="mode",
                    values=metric,
                )
                if left_mode not in pivot.columns or right_mode not in pivot.columns:
                    raise ValueError(
                        f"Missing mode for paired comparison: {left_mode}, {right_mode}"
                    )

                paired = pivot[[left_mode, right_mode]].dropna()
                differences = (
                    paired[left_mode].to_numpy(dtype=float)
                    - paired[right_mode].to_numpy(dtype=float)
                )
                ci_low, ci_high = _bootstrap_mean_ci(
                    differences,
                    rng=rng,
                    n_resamples=bootstrap_resamples,
                )

                rows.append(
                    {
                        "scenario": scenario,
                        "metric": metric,
                        "left_mode": left_mode,
                        "right_mode": right_mode,
                        "difference_definition": "left_minus_right",
                        "n_pairs": int(len(differences)),
                        "mean_difference": float(np.mean(differences)),
                        "bootstrap_ci_low": ci_low,
                        "bootstrap_ci_high": ci_high,
                        "wins_left": int(np.sum(differences > tolerance)),
                        "ties": int(np.sum(np.abs(differences) <= tolerance)),
                        "wins_right": int(np.sum(differences < -tolerance)),
                    }
                )

    return pd.DataFrame(rows)


def analyze_synthetic_attention_ablation(
    input_root: Path,
    *,
    scenarios: Sequence[str] | None = None,
    expected_realizations: int = 20,
    allow_incomplete: bool = False,
    bootstrap_seed: int = 26090806,
    bootstrap_resamples: int = 10_000,
) -> AttentionAblationAnalysis:
    realization_summary = load_synthetic_attention_summaries(
        input_root,
        scenarios=scenarios,
        expected_realizations=expected_realizations,
        allow_incomplete=allow_incomplete,
    )
    mode_summary = summarize_modes(realization_summary)
    paired = paired_mode_comparisons(
        realization_summary,
        bootstrap_seed=bootstrap_seed,
        bootstrap_resamples=bootstrap_resamples,
    )
    return AttentionAblationAnalysis(
        realization_summary=realization_summary,
        mode_summary=mode_summary,
        paired_comparisons=paired,
    )
