from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd


METRICS = (
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
)

HIGHER_IS_BETTER = {
    "selection_tp": True,
    "selection_fp": False,
    "selection_fn": False,
    "selection_precision": True,
    "selection_recall": True,
    "selection_f1": True,
    "isr_tp": True,
    "isr_fp": False,
    "isr_fn": False,
    "isr_precision": True,
    "isr_recall": True,
    "isr_f1": True,
    "fp_reduction_vs_selection": True,
}

REFERENCE_BY_FAMILY = {
    "threshold_grid": "threshold_s60_i60",
    "run_count": "runs_05",
    "candidate_k": "k_locked",
    "reference_support": "support_512",
}

EXPECTED_CONFIGS = {
    "threshold_grid": {
        "threshold_s40_i40",
        "threshold_s40_i60",
        "threshold_s40_i80",
        "threshold_s60_i40",
        "threshold_s60_i60",
        "threshold_s60_i80",
        "threshold_s80_i40",
        "threshold_s80_i60",
        "threshold_s80_i80",
    },
    "run_count": {"runs_03", "runs_05", "runs_10"},
    "candidate_k": {"k_half", "k_locked", "k_double"},
    "reference_support": {"support_128", "support_256", "support_512"},
}


@dataclass(frozen=True)
class SensitivityAnalysisConfig:
    input_root: Path
    output_root: Path
    scenarios: tuple[str, ...]
    expected_realizations: int = 20
    bootstrap_seed: int = 26090806
    bootstrap_resamples: int = 10_000
    allow_incomplete: bool = False


def _bootstrap_mean_ci(values, *, seed, n_resamples):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    if values.size == 1:
        return float(values[0]), float(values[0])

    rng = np.random.default_rng(seed)
    draws = rng.choice(
        values,
        size=(n_resamples, values.size),
        replace=True,
    ).mean(axis=1)

    return (
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def load_sensitivity_results(config: SensitivityAnalysisConfig) -> pd.DataFrame:
    frames = []

    for scenario in config.scenarios:
        paths = sorted(
            (config.input_root / scenario).glob("realization_*_summary.csv")
        )

        if (
            not config.allow_incomplete
            and len(paths) != config.expected_realizations
        ):
            raise ValueError(
                f"{scenario}: expected {config.expected_realizations} "
                f"summary files, found {len(paths)}."
            )

        for path in paths:
            frame = pd.read_csv(path)

            if len(frame) != 18:
                raise ValueError(
                    f"{path}: expected 18 sensitivity rows, found {len(frame)}."
                )

            for family, expected in EXPECTED_CONFIGS.items():
                observed = set(
                    frame.loc[frame["family"] == family, "config_id"]
                )
                if observed != expected:
                    raise ValueError(
                        f"{path}: {family} configuration set mismatch. "
                        f"Expected {sorted(expected)}, observed {sorted(observed)}."
                    )

            frames.append(frame)

    if not frames:
        raise ValueError("No sensitivity summary files found.")

    data = pd.concat(frames, ignore_index=True)

    key = ["scenario", "realization_number", "family", "config_id"]
    if data.duplicated(key).any():
        raise ValueError("Duplicate sensitivity rows detected.")

    return data.sort_values(key).reset_index(drop=True)


def configuration_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (scenario, family, config_id), group in data.groupby(
        ["scenario", "family", "config_id"],
        sort=True,
    ):
        first = group.iloc[0]
        row = {
            "scenario": scenario,
            "family": family,
            "config_id": config_id,
            "n_realizations": int(group["realization_number"].nunique()),
            "selection_threshold": float(first["selection_threshold"]),
            "isr_threshold": float(first["isr_threshold"]),
            "n_discovery_runs": int(first["n_discovery_runs"]),
            "k_mode": str(first["k_mode"]),
            "candidate_k": int(first["candidate_k"]),
            "reference_size": int(first["reference_size"]),
            "is_reference": bool(first["is_reference"]),
        }

        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sd"] = float(values.std(ddof=1))
            row[f"{metric}_median"] = float(values.median())

        rows.append(row)

    return pd.DataFrame(rows)


def paired_vs_reference(
    data: pd.DataFrame,
    *,
    bootstrap_seed: int,
    bootstrap_resamples: int,
) -> pd.DataFrame:
    rows = []

    for scenario_index, (scenario, scenario_frame) in enumerate(
        data.groupby("scenario", sort=True)
    ):
        for family_index, (family, family_frame) in enumerate(
            scenario_frame.groupby("family", sort=True)
        ):
            reference_id = REFERENCE_BY_FAMILY[family]
            config_ids = sorted(set(family_frame["config_id"]))

            reference = family_frame[
                family_frame["config_id"] == reference_id
            ].set_index("realization_number")

            for config_index, config_id in enumerate(config_ids):
                if config_id == reference_id:
                    continue

                candidate = family_frame[
                    family_frame["config_id"] == config_id
                ].set_index("realization_number")

                common = reference.index.intersection(candidate.index)
                if len(common) == 0:
                    continue

                for metric_index, metric in enumerate(METRICS):
                    a = pd.to_numeric(
                        candidate.loc[common, metric],
                        errors="coerce",
                    ).to_numpy(dtype=float)
                    b = pd.to_numeric(
                        reference.loc[common, metric],
                        errors="coerce",
                    ).to_numpy(dtype=float)

                    diff = a - b
                    diff = diff[np.isfinite(diff)]

                    seed = (
                        bootstrap_seed
                        + scenario_index * 100_000
                        + family_index * 10_000
                        + config_index * 100
                        + metric_index
                    )
                    ci_low, ci_high = _bootstrap_mean_ci(
                        diff,
                        seed=seed,
                        n_resamples=bootstrap_resamples,
                    )

                    orientation = HIGHER_IS_BETTER.get(metric)
                    if orientation is None:
                        wins = ties = losses = np.nan
                    else:
                        oriented = diff if orientation else -diff
                        wins = int(np.sum(oriented > 0))
                        ties = int(np.sum(oriented == 0))
                        losses = int(np.sum(oriented < 0))

                    rows.append(
                        {
                            "scenario": scenario,
                            "family": family,
                            "config_id": config_id,
                            "reference_config_id": reference_id,
                            "metric": metric,
                            "n": int(diff.size),
                            "mean_difference_config_minus_reference": (
                                float(np.mean(diff)) if diff.size else np.nan
                            ),
                            "median_difference_config_minus_reference": (
                                float(np.median(diff)) if diff.size else np.nan
                            ),
                            "ci95_low": ci_low,
                            "ci95_high": ci_high,
                            "wins_for_config": wins,
                            "ties": ties,
                            "losses_for_config": losses,
                        }
                    )

    return pd.DataFrame(rows)


def family_range_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []

    focus_metrics = (
        "n_isr_retained_mean",
        "isr_precision_mean",
        "isr_recall_mean",
        "isr_f1_mean",
        "fp_reduction_vs_selection_mean",
    )

    for (scenario, family), group in summary.groupby(
        ["scenario", "family"],
        sort=True,
    ):
        row = {
            "scenario": scenario,
            "family": family,
            "n_configurations": int(len(group)),
            "reference_config_id": REFERENCE_BY_FAMILY[family],
        }

        for metric in focus_metrics:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
            row[f"{metric}_range"] = float(values.max() - values.min())

        rows.append(row)

    return pd.DataFrame(rows)


def run_analysis(config: SensitivityAnalysisConfig) -> dict[str, Path]:
    config.output_root.mkdir(parents=True, exist_ok=True)

    data = load_sensitivity_results(config)
    summary = configuration_summary(data)
    paired = paired_vs_reference(
        data,
        bootstrap_seed=config.bootstrap_seed,
        bootstrap_resamples=config.bootstrap_resamples,
    )
    ranges = family_range_summary(summary)

    outputs = {
        "realization_summary": config.output_root / "realization_summary_long.csv",
        "configuration_summary": config.output_root / "configuration_summary.csv",
        "paired_vs_reference": config.output_root / "paired_vs_reference.csv",
        "family_range_summary": config.output_root / "family_range_summary.csv",
        "manifest": config.output_root / "analysis_manifest.json",
    }

    data.to_csv(outputs["realization_summary"], index=False)
    summary.to_csv(outputs["configuration_summary"], index=False)
    paired.to_csv(outputs["paired_vs_reference"], index=False)
    ranges.to_csv(outputs["family_range_summary"], index=False)

    manifest = {
        "analysis_status": "reviewer_requested_post_submission",
        "analysis_name": "parameter_sensitivity_aggregate",
        "scenarios": list(config.scenarios),
        "expected_realizations_per_scenario": config.expected_realizations,
        "allow_incomplete": config.allow_incomplete,
        "bootstrap_seed": config.bootstrap_seed,
        "bootstrap_resamples": config.bootstrap_resamples,
        "formal_hypothesis_tests": False,
        "locked_primary_results_modified": False,
        "reference_by_family": REFERENCE_BY_FAMILY,
        "difference_definition": "configuration minus submitted-family reference",
        "outputs": {
            key: str(path)
            for key, path in outputs.items()
            if key != "manifest"
        },
    }

    outputs["manifest"].write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    return outputs
