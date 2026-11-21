"""Evaluate the independent TreeExtra Additive Groves interaction baseline.

R11 reviewer revision, synthetic S1-S4 experiments. This script ONLY reads an
existing, completed TreeExtra run and writes three new result artifacts.
It never re-trains the model, modifies existing baseline files, or uses
hold-out test labels for model selection.

Example (from AG-NAM repository root):
    python scripts/run_revision_additive_groves_analysis.py --scenario S1 --realization 1

Prerequisites:
    - editable/importable `agnam` Python package (existing conda environment)
    - completed TreeExtra logs in
      results/revision/external_interaction_baselines/additive_groves/S1/realization_01
    - existing reference manifest in
      results/revision/external_interaction_baselines/core_synthetic/S1/

Scoring policy:
    Native TreeExtra produces signed standardized performance loss for each
    tested pair. Higher scores indicate stronger interaction evidence.
    Pairs omitted by backward feature selection get an identical score strictly
    below all tested pairs. This retains the original full pair universe.
    "Small interaction is present" counts as a native positive. The native
    binary F1 is distinct from rank-based F1@K.

Caution:
    Wall-clock time and peak memory cannot be recovered retroactively from
    earlier console runs. These fields are NA rather than fabricated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from agnam.benchmarking.synthetic_protocol import SyntheticBenchmarkProtocol
from agnam.revision.external_recovery_baselines import (
    canonical_pair,
    deterministic_ranking,
    evaluate_pair_ranking,
    full_pair_universe,
)
from agnam.revision.external_recovery_runner import _prepare_data


_PAIR_RE = re.compile(
    r"Restricting interaction between\s+(x\d+)\s+and\s+(x\d+)"
    r"[\s\S]*?"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?)"
    r"\s+standard deviations from the mean\.\s*"
    r"((?:Small\s+)?Interaction is (?:present|absent))\.",
    re.IGNORECASE,
)
_ATTRIBUTE_RE = re.compile(r"^([^:\s]+)\s*:\s*[^\n]+$", re.MULTILINE)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _attribute_names(path: Path) -> tuple[str, ...]:
    content = path.read_text(encoding="utf-8", errors="replace")
    # TreeExtra attribute metadata places the response column first.
    head = content.split("contexts:", 1)[0]
    entries = _ATTRIBUTE_RE.findall(head)
    if not entries or entries[0].lower() != "class":
        raise ValueError(f"Expected 'class' as first attribute in {path}")
    return tuple(entries[1:])


def _excluded_features(path: Path) -> set[str]:
    content = path.read_text(encoding="utf-8", errors="replace")
    if "contexts:" not in content:
        raise ValueError(f"Missing contexts: section: {path}")
    lines = content.split("contexts:", 1)[1].splitlines()
    excluded: set[str] = set()
    for line in lines:
        match = re.fullmatch(r"\s*([^\s]+)\s+never\.\s*", line)
        if match:
            excluded.add(match.group(1))
        elif line.strip():
            raise ValueError(f"Unrecognized feature-selection context: {line!r}")
    return excluded


def _parse_log(text: str) -> dict[tuple[str, str], tuple[float, str]]:
    parsed: dict[tuple[str, str], tuple[float, str]] = {}
    raw_matches = list(_PAIR_RE.finditer(text))
    if not raw_matches:
        raise ValueError("No completed TreeExtra pairwise interaction tests in log")
    for match in raw_matches:
        a, b, score, decision = match.groups()
        pair = canonical_pair(a, b)
        if pair in parsed:
            raise ValueError(f"Duplicate interaction test: {pair}")
        score_float = float(score)
        if not math.isfinite(score_float):
            raise ValueError(f"Non-finite score for {pair}")
        parsed[pair] = (score_float, decision.lower().strip())
    return parsed


def _native_counts(
    parsed: dict[tuple[str, str], tuple[float, str]],
    truth: set[tuple[str, str]],
) -> dict[str, float | int]:
    positive = {pair for pair, (_, decision) in parsed.items() if decision.endswith("present")}
    tp = len(positive & truth)
    fp = len(positive - truth)
    fn = len(truth - positive)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "native_tp": tp,
        "native_fp": fp,
        "native_fn": fn,
        "native_precision": float(precision),
        "native_recall": float(recall),
        "native_f1": float(f1),
    }


def analyze(repo: Path, scenario: str, realization: int, overwrite: bool = False) -> None:
    scenario = scenario.upper()
    source_manifest = (
        repo / "results/revision/external_interaction_baselines/core_synthetic"
        / scenario / f"realization_{realization:02d}_manifest.json"
    )
    workdir = (
        repo / "results/revision/external_interaction_baselines/additive_groves"
        / scenario / f"realization_{realization:02d}"
    )
    if not source_manifest.is_file():
        raise FileNotFoundError(f"Locked existing-baseline manifest not found: {source_manifest}")
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    if source.get("scenario") != scenario or source.get("realization_number") != realization:
        raise ValueError("Source scenario/realization mismatch")

    log_path = workdir / "ag_interactions.log"
    fs_log_path = workdir / "ag_fs.log"
    source_attr = workdir / "data.attr"
    selected_attr = workdir / "data.fs.attr"
    train_file = workdir / "data.train"
    validation_file = workdir / "data.valid"
    for required in (log_path, fs_log_path, source_attr, selected_attr, train_file, validation_file):
        if not required.is_file():
            raise FileNotFoundError(f"Required completed-run artifact missing: {required}")

    log = log_path.read_text(encoding="utf-8", errors="replace")
    if "Interaction is" not in log and "interaction is" not in log.lower():
        raise ValueError("TreeExtra log does not contain completed interaction decisions")
    parsed = _parse_log(log)
    feature_names = _attribute_names(source_attr)
    selected_feature_names = _attribute_names(selected_attr)
    if selected_feature_names != feature_names:
        raise ValueError("TreeExtra .fs.attr changed attribute column order")
    excluded = _excluded_features(selected_attr)
    if not excluded.issubset(set(feature_names)):
        raise ValueError("Unknown feature listed as excluded")
    selected = set(feature_names) - excluded
    if len(parsed) != math.comb(len(selected), 2):
        raise ValueError(
            f"Expected all {math.comb(len(selected), 2)} pairs of selected features; "
            f"got {len(parsed)}"
        )
    expected_tested = {
        canonical_pair(a, b)
        for i, a in enumerate(feature_names) if a in selected
        for b in feature_names[i + 1:] if b in selected
    }
    if set(parsed) != expected_tested:
        raise ValueError("TreeExtra tested pairs do not exactly match selected features")

    # This is used exclusively to retrieve the existing *ground-truth* labels.
    # It uses exactly the dataset/split seeds from the reference baseline's manifest.
    spec = SimpleNamespace(
        scenario=scenario,
        dataset_seed=int(source["dataset_seed"]),
        outer_split_seed=int(source["outer_split_seed"]),
        final_split_seed=int(source["final_split_seed"]),
    )
    _, _, _, _, generated_names, generated_truth = _prepare_data(
        spec, SyntheticBenchmarkProtocol()
    )
    if tuple(generated_names) != feature_names:
        raise ValueError("Regenerated features disagree with TreeExtra attribute file")
    truth = {canonical_pair(*pair) for pair in generated_truth}
    if len(truth) != int(source["n_true_interactions"]):
        raise ValueError("Ground-truth pair count disagrees with existing manifest")

    universe = full_pair_universe(feature_names)
    raw_scores = {pair: score for pair, (score, _) in parsed.items()}
    # One shared floor: eliminated pairs have equal, below-tested evidence.
    floor_score = min(raw_scores.values()) - 1.0 if raw_scores else -1.0
    full_scores = {pair: raw_scores.get(pair, floor_score) for pair in universe}
    candidate_k = int(source["candidate_k"])
    metrics = evaluate_pair_ranking(
        feature_names=feature_names,
        pair_scores=full_scores,
        true_pairs=truth,
        candidate_k=candidate_k,
    )
    native = _native_counts(parsed, truth)

    rows = [
        {
            "scenario": scenario,
            "realization_number": realization,
            "method": "additive_groves",
            "feature_a": a,
            "feature_b": b,
            "score": score,
            "rank": rank,
            "is_true": canonical_pair(a, b) in truth,
        }
        for rank, (a, b, score) in enumerate(deterministic_ranking(full_scores), start=1)
    ]
    pair_df = pd.DataFrame(rows)
    if len(pair_df) != math.comb(len(feature_names), 2):
        raise AssertionError("Incomplete full pair output")

    # Same column names and order as the existing external recovery summary.
    summary = {
        "scenario": scenario,
        "realization_number": realization,
        "dataset_seed": spec.dataset_seed,
        "method": "additive_groves",
        "candidate_k": candidate_k,
        **asdict(metrics),
        "fit_seconds": np.nan,
        "scoring_seconds": np.nan,
        "standalone_seconds": np.nan,
        "rss_start_mib": np.nan,
        "rss_end_mib": np.nan,
        "rss_peak_mib": np.nan,
        "cuda_peak_allocated_mib": np.nan,
        "shared_mlp_best_epoch": np.nan,
        "shared_mlp_best_val_auroc": np.nan,
    }
    ref_csv = source_manifest.with_name(
        f"realization_{realization:02d}_summary.csv"
    )
    if ref_csv.is_file():
        existing_columns = pd.read_csv(ref_csv, nrows=0).columns.tolist()
        if set(existing_columns) != set(summary):
            raise ValueError(
                "Summary schema mismatch; missing="
                f"{set(existing_columns)-set(summary)}, "
                f"extra={set(summary)-set(existing_columns)}"
            )
        summary_df = pd.DataFrame([summary], columns=existing_columns)
    else:
        summary_df = pd.DataFrame([summary])

    fs_text = fs_log_path.read_text(encoding="utf-8", errors="replace")
    stats = list(
        re.finditer(
            r"Average performance:\s*([0-9.]+),\s*std:\s*([0-9.]+)",
            fs_text,
        )
    )
    performance_info = (
        {"mean": float(stats[-1].group(1)), "std": float(stats[-1].group(2))}
        if stats else None
    )

    manifest = {
        "analysis_status": "reviewer_requested_post_submission",
        "method": "additive_groves",
        "implementation": "TreeExtra C++ 2.7.0; layered Additive Groves",
        "scenario": scenario,
        "realization_number": realization,
        "reference_manifest": str(source_manifest.relative_to(repo)),
        "dataset_seed": spec.dataset_seed,
        "outer_split_seed": spec.outer_split_seed,
        "final_split_seed": spec.final_split_seed,
        "baseline_seed": source.get("baseline_seed"),
        "candidate_k": candidate_k,
        "n_features": len(feature_names),
        "n_all_pairs": len(universe),
        "n_selected_features": len(selected),
        "selected_features": sorted(selected, key=feature_names.index),
        "n_tested_pairs": len(parsed),
        "n_untested_pairs": len(universe) - len(parsed),
        "n_true_pairs": len(truth),
        "truth_pairs": [list(p) for p in sorted(truth)],
        "score_type": "signed_standard_deviations_from_reference_mean",
        "untested_score_policy": "same_value_strictly_below_minimum_tested_score",
        "untested_floor_score": floor_score,
        "tie_policy": "sklearn_average_precision_roc_auc_ndcg_and_deterministic_lexical_rank",
        "native_decision_policy": "interaction_present_or_small_interaction_present",
        **native,
        "rank_metrics": asdict(metrics),
        "fs_reference_performance": performance_info,
        "original_interaction_command": next(
            (line.strip() for line in log.splitlines() if line.strip().startswith("ag_interactions ")),
            None,
        ),
        "wall_time_and_peak_memory": "unavailable_from_past_runs; NA, not zero",
        "input_sha256": {
            p.name: _sha256(p)
            for p in (source_manifest, source_attr, selected_attr, train_file,
                      validation_file, fs_log_path, log_path)
        },
        "locked_primary_results_modified": False,
    }

    pairs_path = workdir / "additive_groves_pair_scores.csv"
    summary_path = workdir / "additive_groves_summary.csv"
    manifest_path = workdir / "additive_groves_manifest.json"
    outputs = (pairs_path, summary_path, manifest_path)
    existing = [p for p in outputs if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing Additive Groves files without "
            f"--overwrite: {[str(p) for p in existing]}"
        )
    pair_df.to_csv(pairs_path, index=False, float_format="%.12g")
    summary_df.to_csv(summary_path, index=False, float_format="%.12g")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print(f"=== VERIFIED ADDITIVE GROVES {scenario} R{realization:02d} ===")
    print(f"PAIR UNIVERSE: {len(universe)} | TESTED: {len(parsed)} | UNTESTED: {len(universe)-len(parsed)}")
    print(f"NATIVE TP: {native['native_tp']} | FP: {native['native_fp']} | FN: {native['native_fn']} | F1: {native['native_f1']:.6f}")
    for name, value in asdict(metrics).items():
        print(f"{name}: {value:.8f}")
    print("OUTPUTS:")
    for output in outputs:
        print(f"  {output.relative_to(repo)}")
    print("Runtime/memory: NA (not retrospectively measured)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=["S1", "S2", "S3", "S4"], default="S1")
    parser.add_argument("--realization", type=int, default=1)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true", help="Replace this script's own output files")
    args = parser.parse_args()
    if args.realization < 1:
        parser.error("--realization must be >=1")
    analyze(args.repo_root.resolve(), args.scenario, args.realization, args.overwrite)


if __name__ == "__main__":
    main()
