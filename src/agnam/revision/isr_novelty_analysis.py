from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd


EXPECTED_VARIANTS = (
    "selection_stable_060",
    "selection_stable_080",
    "count_matched_selection",
    "raw_surface_agreement",
    "purified_isr",
)

SUMMARY_METRICS = (
    "n_retained",
    "true_positive",
    "false_positive",
    "false_negative",
    "precision",
    "recall",
    "f1",
    "false_negative_rate",
    "fp_reduction_vs_selection",
)

HIGHER_IS_BETTER = {
    "true_positive": True,
    "false_positive": False,
    "false_negative": False,
    "precision": True,
    "recall": True,
    "f1": True,
    "false_negative_rate": False,
    "fp_reduction_vs_selection": True,
}


@dataclass(frozen=True)
class RevisionISRAnalysisConfig:
    input_root: Path
    output_root: Path
    scenarios: tuple[str, ...]
    expected_realizations: int = 20
    bootstrap_seed: int = 26090806
    bootstrap_resamples: int = 10_000
    allow_incomplete: bool = False


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    seed: int,
    n_resamples: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (np.nan, np.nan)
    if values.size == 1:
        return (float(values[0]), float(values[0]))

    rng = np.random.default_rng(seed)
    n = values.size
    samples = rng.choice(values, size=(n_resamples, n), replace=True)
    means = samples.mean(axis=1)
    alpha = 1.0 - confidence
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return lo, hi


def _summary_paths(input_root: Path, scenario: str) -> list[Path]:
    return sorted((input_root / scenario).glob("realization_*_summary.csv"))


def _pair_paths(input_root: Path, scenario: str) -> list[Path]:
    return sorted((input_root / scenario).glob("realization_*_pair_agreements.csv"))


def load_revision_summaries(config: RevisionISRAnalysisConfig) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for scenario in config.scenarios:
        paths = _summary_paths(config.input_root, scenario)
        if not config.allow_incomplete and len(paths) != config.expected_realizations:
            raise ValueError(
                f"{scenario}: expected {config.expected_realizations} summary files, "
                f"found {len(paths)}."
            )

        for path in paths:
            frame = pd.read_csv(path)
            if set(frame["variant"]) != set(EXPECTED_VARIANTS):
                raise ValueError(
                    f"{path}: variant set does not match the locked revision protocol."
                )
            if len(frame) != len(EXPECTED_VARIANTS):
                raise ValueError(f"{path}: expected exactly 5 variant rows.")
            frames.append(frame)

    if not frames:
        raise ValueError("No ISR novelty summary files were found.")

    data = pd.concat(frames, ignore_index=True)

    key_cols = ["scenario", "realization_number", "variant"]
    if data.duplicated(key_cols).any():
        raise ValueError("Duplicate scenario/realization/variant rows detected.")

    for scenario in config.scenarios:
        part = data[data["scenario"] == scenario]
        n_realizations = part["realization_number"].nunique()
        if (
            not config.allow_incomplete
            and n_realizations != config.expected_realizations
        ):
            raise ValueError(
                f"{scenario}: expected {config.expected_realizations} unique "
                f"realizations, found {n_realizations}."
            )

    return data.sort_values(
        ["scenario", "realization_number", "variant"]
    ).reset_index(drop=True)


def aggregate_variant_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    for (scenario, variant), group in data.groupby(
        ["scenario", "variant"], sort=True
    ):
        row = {
            "scenario": scenario,
            "variant": variant,
            "n_realizations": int(group["realization_number"].nunique()),
        }
        for metric in SUMMARY_METRICS:
            vals = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(vals.mean())
            row[f"{metric}_sd"] = float(vals.std(ddof=1))
            row[f"{metric}_median"] = float(vals.median())
        rows.append(row)

    return pd.DataFrame(rows)


def paired_purified_comparisons(
    data: pd.DataFrame,
    *,
    bootstrap_seed: int,
    bootstrap_resamples: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    comparators = tuple(v for v in EXPECTED_VARIANTS if v != "purified_isr")

    for scenario, scenario_frame in data.groupby("scenario", sort=True):
        wide = scenario_frame.pivot(
            index="realization_number",
            columns="variant",
            values=list(SUMMARY_METRICS),
        )

        for comparator_index, comparator in enumerate(comparators):
            for metric_index, metric in enumerate(SUMMARY_METRICS):
                a = pd.to_numeric(
                    wide[(metric, "purified_isr")], errors="coerce"
                ).to_numpy(dtype=float)
                b = pd.to_numeric(
                    wide[(metric, comparator)], errors="coerce"
                ).to_numpy(dtype=float)
                diff = a - b
                finite = np.isfinite(diff)
                diff = diff[finite]

                seed = (
                    bootstrap_seed
                    + 10_000 * comparator_index
                    + 100 * metric_index
                    + sum(ord(ch) for ch in str(scenario))
                )
                ci_lo, ci_hi = _bootstrap_mean_ci(
                    diff,
                    seed=seed,
                    n_resamples=bootstrap_resamples,
                )

                higher = HIGHER_IS_BETTER.get(metric)
                if higher is None:
                    wins = ties = losses = np.nan
                else:
                    oriented = diff if higher else -diff
                    wins = int(np.sum(oriented > 0))
                    ties = int(np.sum(oriented == 0))
                    losses = int(np.sum(oriented < 0))

                rows.append(
                    {
                        "scenario": scenario,
                        "comparison": f"purified_isr - {comparator}",
                        "metric": metric,
                        "n": int(diff.size),
                        "mean_difference": float(np.mean(diff))
                        if diff.size else np.nan,
                        "median_difference": float(np.median(diff))
                        if diff.size else np.nan,
                        "ci95_low": ci_lo,
                        "ci95_high": ci_hi,
                        "wins_for_purified": wins,
                        "ties": ties,
                        "losses_for_purified": losses,
                    }
                )

    return pd.DataFrame(rows)


def load_raw_vs_purified_structure(
    config: RevisionISRAnalysisConfig,
) -> pd.DataFrame:
    rows: list[dict] = []

    for scenario in config.scenarios:
        paths = _pair_paths(config.input_root, scenario)
        if not config.allow_incomplete and len(paths) != config.expected_realizations:
            raise ValueError(
                f"{scenario}: expected {config.expected_realizations} pair-agreement "
                f"files, found {len(paths)}."
            )

        for path in paths:
            frame = pd.read_csv(path)
            required = {
                "feature_j",
                "feature_k",
                "raw_pass_060",
                "purified_pass_060",
            }
            missing = required.difference(frame.columns)
            if missing:
                raise ValueError(f"{path}: missing columns {sorted(missing)}")

            raw_set = {
                (str(r.feature_j), str(r.feature_k))
                for r in frame.itertuples()
                if bool(r.raw_pass_060)
            }
            purified_set = {
                (str(r.feature_j), str(r.feature_k))
                for r in frame.itertuples()
                if bool(r.purified_pass_060)
            }
            union = raw_set | purified_set
            overlap = raw_set & purified_set

            realization_number = int(
                path.name.split("_")[1]
            )

            rows.append(
                {
                    "scenario": scenario,
                    "realization_number": realization_number,
                    "n_raw_retained": len(raw_set),
                    "n_purified_retained": len(purified_set),
                    "n_overlap": len(overlap),
                    "n_raw_only": len(raw_set - purified_set),
                    "n_purified_only": len(purified_set - raw_set),
                    "raw_purified_jaccard": (
                        len(overlap) / len(union) if union else 1.0
                    ),
                    "sets_identical": raw_set == purified_set,
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["scenario", "realization_number"]
    ).reset_index(drop=True)


def aggregate_structure_summary(structure: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for scenario, group in structure.groupby("scenario", sort=True):
        rows.append(
            {
                "scenario": scenario,
                "n_realizations": int(len(group)),
                "mean_raw_retained": float(group["n_raw_retained"].mean()),
                "mean_purified_retained": float(
                    group["n_purified_retained"].mean()
                ),
                "mean_overlap": float(group["n_overlap"].mean()),
                "mean_raw_only": float(group["n_raw_only"].mean()),
                "mean_purified_only": float(group["n_purified_only"].mean()),
                "mean_raw_purified_jaccard": float(
                    group["raw_purified_jaccard"].mean()
                ),
                "identical_set_fraction": float(
                    group["sets_identical"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def run_analysis(config: RevisionISRAnalysisConfig) -> dict[str, Path]:
    config.output_root.mkdir(parents=True, exist_ok=True)

    data = load_revision_summaries(config)
    variant_summary = aggregate_variant_summary(data)
    paired = paired_purified_comparisons(
        data,
        bootstrap_seed=config.bootstrap_seed,
        bootstrap_resamples=config.bootstrap_resamples,
    )
    structure = load_raw_vs_purified_structure(config)
    structure_summary = aggregate_structure_summary(structure)

    outputs = {
        "realization_summary": config.output_root / "realization_summary_long.csv",
        "variant_summary": config.output_root / "variant_summary.csv",
        "paired_comparisons": config.output_root / "paired_comparisons.csv",
        "raw_vs_purified_structure": (
            config.output_root / "raw_vs_purified_structure.csv"
        ),
        "structure_summary": config.output_root / "structure_summary.csv",
        "manifest": config.output_root / "analysis_manifest.json",
    }

    data.to_csv(outputs["realization_summary"], index=False)
    variant_summary.to_csv(outputs["variant_summary"], index=False)
    paired.to_csv(outputs["paired_comparisons"], index=False)
    structure.to_csv(outputs["raw_vs_purified_structure"], index=False)
    structure_summary.to_csv(outputs["structure_summary"], index=False)

    manifest = {
        "analysis_status": "reviewer_requested_post_submission",
        "analysis_name": "isr_novelty_ablation_aggregate",
        "scenarios": list(config.scenarios),
        "expected_realizations_per_scenario": config.expected_realizations,
        "allow_incomplete": config.allow_incomplete,
        "bootstrap_seed": config.bootstrap_seed,
        "bootstrap_resamples": config.bootstrap_resamples,
        "formal_hypothesis_tests": False,
        "locked_primary_results_modified": False,
        "primary_revision_contrasts": [
            "purified_isr vs selection_stable_080",
            "purified_isr vs count_matched_selection",
            "purified_isr vs raw_surface_agreement",
            "purified_isr vs selection_stable_060",
        ],
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
