"""R9-B: strict, read-only summary of the 20-realization threshold follow-up.

Exploratory, reviewer-requested analyses. The selected subset and nine threshold
configurations were fixed before viewing predictive outcomes; analyses do not
optimize, select, or refit any predictive models.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_ROOT = Path("results/revision/predictive_threshold_r9/v1")
DEFAULT_R4_ROOT = Path("results/revision/sensitivity/synthetic")
DEFAULT_ORIGINAL = Path("results/synthetic/master_results.csv")
SCENARIOS = ("S1", "S2", "S3", "S4")
LEVELS = (40, 60, 80)
CONFIG_IDS = tuple(f"threshold_s{s}_i{i}" for s in LEVELS for i in LEVELS)
REFERENCE = "threshold_s60_i60"
NECESSARY_FILES = (
    "predictive_summary.csv", "pair_sets.json", "final_model_cache.json", "manifest.json",
)
TO_CHECK = (
    ("predictive_summary.csv", "summary_sha256"),
    ("pair_sets.json", "pair_sets_sha256"),
    ("final_model_cache.json", "model_cache_sha256"),
)
FLOAT_METRICS = ("auroc", "auprc", "balanced_accuracy", "f1")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _close(a, b, *, tol: float = 1e-10) -> bool:
    return bool(np.isclose(float(a), float(b), rtol=0, atol=tol))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_one(
    root: Path,
    scenario: str,
    realization: int,
    *,
    verify_sources: bool,
    r4_root: Path,
    original_path: Path,
) -> tuple[pd.DataFrame, dict]:
    folder = root / scenario / f"realization_{realization:02d}"
    _require(all((folder / name).is_file() for name in NECESSARY_FILES), f"Missing R9-B files: {folder}")
    manifest = _read_json(folder / "manifest.json")
    pair_sets = _read_json(folder / "pair_sets.json")
    cache = _read_json(folder / "final_model_cache.json")
    df = pd.read_csv(folder / "predictive_summary.csv")

    _require(manifest.get("scenario") == scenario, f"Scenario mismatch: {folder}")
    _require(int(manifest.get("realization_number", -1)) == realization, f"Realization mismatch: {folder}")
    _require(manifest.get("r4_recovery_parity") == "9/9", f"R4 parity unverified: {folder}")
    _require(manifest.get("locked_primary_results_modified") is False, f"Original modified flag: {folder}")
    _require(manifest.get("n_threshold_configs") == 9, f"Threshold count mismatch: {folder}")
    _require(manifest.get("subset_definition") == "S1-S4 realizations 1-5 (20 total), fixed before R9-B outcomes", f"Subset definition mismatch: {folder}")
    _require(len(df) == 9 and set(df.config_id) == set(CONFIG_IDS), f"Invalid 3x3 grid: {folder}")
    _require(df.config_id.is_unique, f"Duplicate configuration IDs: {folder}")
    _require(set(df.scenario) == {scenario} and set(df.realization_number) == {realization}, f"CSV IDs inconsistent: {folder}")
    _require(int(df.pair_hash.nunique()) == int(manifest.get("n_unique_pair_sets", -1)), f"Unique-set mismatch: {folder}")
    for name, key in TO_CHECK:
        _require(manifest.get(key) == _sha256(folder / name), f"SHA256 mismatch {folder / name}")

    _require(isinstance(pair_sets, list) and len(pair_sets) == 9, f"Invalid pair sets: {folder}")
    pairs_by_config = {rec["config_id"]: rec for rec in pair_sets}
    _require(len(pairs_by_config) == 9 and set(pairs_by_config) == set(CONFIG_IDS), f"Pair-set config IDs mismatch: {folder}")
    models = cache.get("models", {})
    _require(set(models) == set(df.pair_hash), f"Final model cache coverage mismatch: {folder}")
    _require(cache.get("source_hashes") == manifest.get("source_hashes"), f"Cache source mismatch: {folder}")
    _require(cache.get("pair_sets_sha256") == manifest.get("pair_sets_sha256"), f"Cache/pair source mismatch: {folder}")

    if verify_sources:
        expected = {
            "r4": _sha256(r4_root / scenario / f"realization_{realization:02d}_summary.csv"),
            "original": _sha256(original_path),
        }
        _require(manifest.get("source_hashes") == expected, f"Original/R4 source hash mismatch: {folder}")

    for row in df.itertuples(index=False):
        item = pairs_by_config[row.config_id]
        _require(item["pair_hash"] == row.pair_hash, f"Pair hash mismatch: {folder} {row.config_id}")
        _require(len(item["isr_pairs"]) == int(row.n_isr_retained), f"Pair count mismatch: {folder} {row.config_id}")
        _require(row.pair_hash == row.model_fit_cache_key, f"Cache-key mismatch: {folder} {row.config_id}")
        model = models[row.pair_hash]
        _require(item["isr_pairs"] == model["pairs"], f"Cached model pair identity mismatch: {folder} {row.config_id}")
        for metric in FLOAT_METRICS:
            value = "agnam_auroc" if metric == "auroc" else metric
            _require(_close(getattr(row, value), model[metric]), f"Cached {metric} mismatch: {folder} {row.config_id}")
        _require(_close(row.delta_auroc_vs_main, row.agnam_auroc - row.main_auroc_locked), f"AUROC delta mismatch: {folder} {row.config_id}")
        _require(np.isfinite(row.agnam_auroc) and 0 <= row.agnam_auroc <= 1, f"Invalid AUROC: {folder}")
        if int(row.n_isr_retained) == 0:
            _require(_close(row.agnam_auroc, row.main_auroc_locked), f"Zero-pair final AUROC != main AUROC: {folder}")
    reference = df.set_index("config_id").loc[REFERENCE]
    _require(_close(reference.agnam_auroc, manifest["original_full_auroc"], tol=1e-6), f"Original full AUROC parity failure: {folder}")
    _require(_close(reference.main_auroc_locked, manifest["main_auroc"], tol=1e-10), f"Original main AUROC parity failure: {folder}")
    _require(all(_close(v, manifest["main_auroc"]) for v in df.main_auroc_locked), f"Main NAM differs by threshold: {folder}")
    df = df.copy()
    df["selection_threshold"] = df.config_id.str.extract(r"threshold_s(\d+)_i")[0].astype(int) / 100
    df["isr_threshold"] = df.config_id.str.extract(r"_i(\d+)$")[0].astype(int) / 100
    df["reference_auroc"] = float(reference.agnam_auroc)
    df["paired_delta_vs_reference"] = df.agnam_auroc - df.reference_auroc
    df["zero_pairs"] = df.n_isr_retained.eq(0)
    audit = {
        "scenario": scenario,
        "realization_number": realization,
        "n_configs": len(df),
        "n_distinct_pair_sets": df.pair_hash.nunique(),
        "n_zero_pair_configs": int(df.zero_pairs.sum()),
        "r4_recovery_parity": manifest["r4_recovery_parity"],
        "reference_auroc_parity_abs_delta": abs(float(reference.agnam_auroc) - float(manifest["original_full_auroc"])),
        "summary_sha256": manifest["summary_sha256"],
        "cache_sha256": manifest["model_cache_sha256"],
        "pair_sets_sha256": manifest["pair_sets_sha256"],
    }
    return df, audit


def _group_summary(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return frame.groupby(group_cols, sort=True).agg(
        n=("agnam_auroc", "size"),
        mean_auroc=("agnam_auroc", "mean"),
        sd_auroc=("agnam_auroc", "std"),
        mean_pairs=("n_isr_retained", "mean"),
        mean_delta_vs_main=("delta_auroc_vs_main", "mean"),
        mean_paired_delta_vs_reference=("paired_delta_vs_reference", "mean"),
        zero_pair_configs=("zero_pairs", "sum"),
    ).reset_index()


def _bootstrap_deltas(frame: pd.DataFrame, *, n_bootstrap: int, seed: int) -> dict[str, tuple[float, float]]:
    """Descriptive percentile CIs; paired resampling of realizations within scenario."""
    matrix = frame.pivot(index=["scenario", "realization_number"], columns="config_id", values="paired_delta_vs_reference")
    matrix = matrix.reindex(columns=CONFIG_IDS)
    groups = [matrix.loc[scenario].to_numpy(dtype=float) for scenario in SCENARIOS]
    rng = np.random.default_rng(seed)
    draws = np.zeros((n_bootstrap, len(CONFIG_IDS)), dtype=float)
    for b in range(n_bootstrap):
        draws[b] = np.concatenate([a[rng.integers(0, len(a), size=len(a))] for a in groups], axis=0).mean(axis=0)
    limits = np.percentile(draws, [2.5, 97.5], axis=0)
    return {key: (float(limits[0, i]), float(limits[1, i])) for i, key in enumerate(CONFIG_IDS)}


def analyze_predictive_threshold(
    *,
    input_root: str | Path = DEFAULT_ROOT,
    output_root: str | Path | None = None,
    r4_root: str | Path = DEFAULT_R4_ROOT,
    original_path: str | Path = DEFAULT_ORIGINAL,
    verify_sources: bool = True,
    bootstrap_repeats: int = 10000,
    bootstrap_seed: int = 2026,
) -> dict[str, Path]:
    if bootstrap_repeats < 100:
        raise ValueError("bootstrap_repeats must be >= 100")
    root = Path(input_root)
    out = Path(output_root) if output_root is not None else root / "analysis"
    _require(out.resolve() != root.resolve(), "Output directory must differ from input root")
    frames, audits = [], []
    for scenario in SCENARIOS:
        for n in range(1, 6):
            data, audit = _load_one(root, scenario, n, verify_sources=verify_sources, r4_root=Path(r4_root), original_path=Path(original_path))
            frames.append(data)
            audits.append(audit)
    data = pd.concat(frames, ignore_index=True)
    _require(len(data) == 180, "Expected 180 predictive evaluations")
    config_summary = _group_summary(data, ["config_id", "selection_threshold", "isr_threshold"])
    scenario_summary = _group_summary(data, ["scenario", "config_id", "selection_threshold", "isr_threshold"])
    paired = data.groupby("config_id", sort=True).agg(
        n=("paired_delta_vs_reference", "size"),
        mean_paired_delta=("paired_delta_vs_reference", "mean"),
        median_paired_delta=("paired_delta_vs_reference", "median"),
        min_paired_delta=("paired_delta_vs_reference", "min"),
        max_paired_delta=("paired_delta_vs_reference", "max"),
        wins=("paired_delta_vs_reference", lambda x: int((x > 1e-12).sum())),
        ties=("paired_delta_vs_reference", lambda x: int((abs(x) <= 1e-12).sum())),
        losses=("paired_delta_vs_reference", lambda x: int((x < -1e-12).sum())),
    ).reset_index()
    ci = _bootstrap_deltas(data, n_bootstrap=bootstrap_repeats, seed=bootstrap_seed)
    paired["delta_ci95_low"] = paired.config_id.map(lambda c: ci[c][0])
    paired["delta_ci95_high"] = paired.config_id.map(lambda c: ci[c][1])

    # Factor-level descriptives, averaging all three thresholds of the other
    # factor within each realization. Not significance tests.
    sel_factor = data.groupby(["scenario", "realization_number", "selection_threshold"], as_index=False).agg(auroc=("agnam_auroc", "mean"), pairs=("n_isr_retained", "mean"))
    isr_factor = data.groupby(["scenario", "realization_number", "isr_threshold"], as_index=False).agg(auroc=("agnam_auroc", "mean"), pairs=("n_isr_retained", "mean"))
    factor_summary = pd.concat([
        sel_factor.groupby("selection_threshold", as_index=False).agg(n=("auroc", "size"), mean_auroc=("auroc", "mean"), mean_pairs=("pairs", "mean")).rename(columns={"selection_threshold": "threshold"}).assign(factor="selection"),
        isr_factor.groupby("isr_threshold", as_index=False).agg(n=("auroc", "size"), mean_auroc=("auroc", "mean"), mean_pairs=("pairs", "mean")).rename(columns={"isr_threshold": "threshold"}).assign(factor="isr"),
    ], ignore_index=True)[["factor", "threshold", "n", "mean_auroc", "mean_pairs"]]

    _require(not out.exists() or out.is_dir(), f"Not a directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    output_data = {
        "realization_summary_long.csv": data,
        "configuration_summary.csv": config_summary,
        "scenario_configuration_summary.csv": scenario_summary,
        "paired_vs_reference.csv": paired,
        "threshold_factor_summary.csv": factor_summary,
        "realization_audit.csv": pd.DataFrame(audits),
    }
    for name, values in output_data.items():
        values.to_csv(out / name, index=False, float_format="%.12g")
    manifest = {
        "analysis": "R9-B predictive threshold sensitivity posthoc descriptive aggregation",
        "status": "reviewer_requested_post_submission_exploratory",
        "subset": "S1-S4, first five realizations per scenario (fixed prior to predictive outcomes)",
        "selection_thresholds": [0.40, 0.60, 0.80],
        "isr_thresholds": [0.40, 0.60, 0.80],
        "reference_configuration": REFERENCE,
        "total_realizations": len(audits),
        "evaluations": len(data),
        "total_final_model_fits": int(sum(a["n_distinct_pair_sets"] for a in audits)),
        "zero_pair_evaluations": int(data.zero_pairs.sum()),
        "all_source_hashes_verified": bool(verify_sources),
        "all_20_r4_recovery_parity": True,
        "all_20_original_final_auroc_parity": True,
        "inference": "None (descriptive comparisons); percentile CIs are exploratory and multiplicity-unadjusted",
        "bootstrap_design": "paired realization-level resampling within each of four scenarios; equal weighting of scenarios",
        "bootstrap_repeats": bootstrap_repeats,
        "bootstrap_seed": bootstrap_seed,
        "original_benchmark_modified": False,
        "file_hashes": {name: _sha256(out / name) for name in output_data},
        "inputs": {f"{a['scenario']}/realization_{a['realization_number']:02d}": {
            "summary_sha256": a["summary_sha256"],
            "cache_sha256": a["cache_sha256"],
            "pair_sets_sha256": a["pair_sets_sha256"],
        } for a in audits},
    }
    (out / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    return {name: out / name for name in output_data} | {"manifest": out / "analysis_manifest.json"}
