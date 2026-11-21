"""R12: Read-only, realization-matched AG-NAM versus native pairwise SIAN audit.

Runs NO training. Reads frozen AG-NAM phase-4 checkpoints and R12 SIAN outputs.
Writes a NEW ZIP with per-realization comparisons, summary, source checksums,
and the original matching AG-NAM checkpoints (no model weights).

Usage (from AG-NAM repository root):
  python scripts/collect_r12_paired_agnam_sian.py --scenario S1
  python scripts/collect_r12_paired_agnam_sian.py --scenario S2

Do not claim paired test-prediction equality if the AG-NAM checkpoint lacks
raw test labels; matching is established via shared generator/split seed
identities and the original locked 60/20/20 sampling procedure.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import statistics
from zipfile import ZipFile, ZIP_DEFLATED

from sklearn.metrics import roc_auc_score, average_precision_score

S1_TO_S4 = ("S1", "S2", "S3", "S4")
EXPECTED_TRUE_PAIRS = {"S1": 3, "S2": 4, "S3": 2, "S4": 4}
EXPECTED_TEST_SIZE = {"S1": 1000, "S2": 1600, "S3": 1600, "S4": 2000}
METRICS = ["agnam_auroc", "sian_auroc", "delta_sian_minus_agnam_auroc",
           "agnam_auprc", "sian_auprc", "delta_sian_minus_agnam_auprc",
           "agnam_isr_f1", "sian_selected_f1", "agnam_n_isr_retained",
           "sian_n_selected", "agnam_recovery_precision", "agnam_recovery_recall",
           "sian_recovery_precision", "sian_recovery_recall", "sian_wall_seconds"]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        return list(csv.DictReader(fp))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def positive_int(x) -> int:
    value = int(x)
    if value < 0:
        raise ValueError(f"Negative value {value}")
    return value


def finite_float(x) -> float:
    v = float(x)
    if not math.isfinite(v):
        raise ValueError(f"Non-finite metric {x}")
    return v


def truth(x: str) -> bool:
    a = str(x).strip().lower()
    if a not in {"true", "false"}:
        raise ValueError(f"Expected literal True or False; got {x}")
    return a == "true"


def csv_render(rows: list[dict]) -> bytes:
    if not rows:
        raise ValueError("No rows")
    cols = list(rows[0])
    s = io.StringIO(newline="")
    w = csv.DictWriter(s, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)
    return s.getvalue().encode("utf-8")


def recovered_f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def check_one(repo: Path, scenario: str, number: int) -> tuple[dict, list[Path]]:
    """Never trust a reported SIAN metric without checking its saved predictions."""
    old = repo / "results" / "synthetic" / scenario / f"realization_{number:02d}.csv"
    old_pairs = repo / "results" / "synthetic" / scenario / f"realization_{number:02d}_pairs.csv"
    new_dir = repo / "results" / "revision" / "r12_native_sian_pairwise" / scenario / f"realization_{number:02d}"
    new_summary = new_dir / "summary.json"
    new_pairs = new_dir / "selected_pairs.csv"
    new_preds = new_dir / "test_predictions.csv"
    new_cost = repo / "results" / "revision" / "r12_cost_audit" / f"r12_{scenario.lower()}_{number:02d}_sian" / "measurement.json"
    sources = [old, old_pairs, new_summary, new_pairs, new_preds, new_cost]
    for f in sources:
        if not f.is_file():
            raise FileNotFoundError(f"{scenario} realization {number:02d} missing: {f}")
    older = read_rows(old)
    if len(older) != 1:
        raise ValueError(f"Expected exactly one AG-NAM checkpoint record: {old}")
    a = older[0]
    if a["scenario"] != scenario or int(a["realization_index"]) != number - 1:
        raise ValueError(f"AG-NAM realization mismatch: {old}")
    b = read_json(new_summary)
    cost = read_json(new_cost)
    if (b.get("scenario") != scenario or b.get("realization") != number
            or b.get("status") != "complete" or b.get("sian_version") != "0.9.5"):
        raise ValueError(f"SIAN realization not complete/compatible: {new_summary}")
    if (cost.get("experiment_run_status") != "complete" or cost.get("returncode") != 0
            or cost.get("label") != f"r12_{scenario.lower()}_{number:02d}_sian"):
        raise ValueError(f"SIAN external cost invalid: {new_cost}")
    fields = {
        "dataset_seed": int(b["data_seed"]),
        "outer_split_seed": int(b["split_seeds"]["outer"]),
        "final_split_seed": int(b["split_seeds"]["final"]),
        "discovery_base_seed": int(b["training_seeds"]["reference"]),
        "final_model_seed": int(b["training_seeds"]["final"]),
    }
    for key, expected in fields.items():
        if int(a[key]) != expected:
            raise ValueError(f"Seed mismatch {scenario} #{number} {key}: AGNAM={a[key]} SIAN={expected}")
    n_true = EXPECTED_TRUE_PAIRS[scenario]
    if int(a["n_true_interactions"]) != n_true:
        raise ValueError("Incorrect AG-NAM true-pair count")
    old_p = read_rows(old_pairs)
    old_pairs_set = set()
    true_set = set()
    for pair in old_p:
        p = tuple(sorted((pair["feature_j"], pair["feature_k"])))
        if truth(pair["is_true"]):
            true_set.add(p)
        if truth(pair["isr_retained"]):
            old_pairs_set.add(p)
    if len(true_set) != n_true or len(old_pairs_set) != int(a["n_isr_retained"]):
        raise ValueError("AG-NAM pair audit does not match checkpoint")
    old_tp = len(true_set & old_pairs_set)
    old_fp = len(old_pairs_set - true_set)
    old_fn = n_true - old_tp
    old_prec = old_tp / len(old_pairs_set) if old_pairs_set else 0.0
    old_rec = old_tp / n_true
    old_f1 = recovered_f1(old_tp, old_fp, old_fn)
    for key, recomputed in [("isr_precision", old_prec), ("isr_recall", old_rec), ("isr_f1", old_f1)]:
        if abs(finite_float(a[key]) - recomputed) > 2e-7:
            raise ValueError(f"AG-NAM {key} disagreement at {scenario} #{number}")
    selected = read_rows(new_pairs)
    si_pairs = set()
    for p in selected:
        pair = tuple(sorted((p["feature_a"], p["feature_b"])))
        if pair in si_pairs:
            raise ValueError("Duplicate native SIAN selected pair")
        si_pairs.add(pair)
        if truth(p["is_true"]) != (pair in true_set):
            raise ValueError(f"SIAN selected-pair truth mismatch: {pair}")
    si_tp = len(si_pairs & true_set)
    si_fp = len(si_pairs - true_set)
    si_fn = n_true - si_tp
    si_prec = si_tp / len(si_pairs) if si_pairs else 0.0
    si_rec = si_tp / n_true
    si_f1 = recovered_f1(si_tp, si_fp, si_fn)
    r = b["interaction_recovery"]
    for k, v in [("true_positive", si_tp), ("false_positive", si_fp), ("false_negative", si_fn),
                 ("precision", si_prec), ("recall", si_rec), ("f1", si_f1)]:
        if abs(float(r[k]) - v) > 2e-7:
            raise ValueError(f"SIAN {k} mismatch {scenario} #{number}")
    pred = read_rows(new_preds)
    if len(pred) != EXPECTED_TEST_SIZE[scenario]:
        raise ValueError("Wrong SIAN test set size")
    labels = [int(x["y_test"]) for x in pred]
    scores = [float(x["sian_probability"]) for x in pred]
    if not set(labels) == {0, 1} or not all(0 <= x <= 1 and math.isfinite(x) for x in scores):
        raise ValueError("Invalid SIAN test predictions")
    aoc = float(roc_auc_score(labels, scores))
    ap = float(average_precision_score(labels, scores))
    if abs(aoc - float(b["test_predictive_metrics"]["test_auroc"])) > 2e-7 or abs(ap-float(b["test_predictive_metrics"]["test_auprc"])) > 2e-7:
        raise ValueError("Native SIAN predictive recomputation does not match summary")
    ag_auc = finite_float(a["agnam_auroc"])
    ag_ap = finite_float(a["agnam_auprc"])
    if not (0 <= ag_auc <= 1 and 0 <= ag_ap <= 1):
        raise ValueError("Invalid AG-NAM checkpoint prediction metrics")
    rec = {
        "scenario": scenario, "realization": number,
        "dataset_seed": fields["dataset_seed"],
        "outer_split_seed": fields["outer_split_seed"],
        "final_split_seed": fields["final_split_seed"],
        "agnam_auroc": ag_auc, "sian_auroc": aoc,
        "delta_sian_minus_agnam_auroc": aoc - ag_auc,
        "agnam_auprc": ag_ap, "sian_auprc": ap,
        "delta_sian_minus_agnam_auprc": ap - ag_ap,
        "agnam_isr_f1": old_f1, "sian_selected_f1": si_f1,
        "agnam_n_isr_retained": len(old_pairs_set), "sian_n_selected": len(si_pairs),
        "agnam_recovery_precision": old_prec, "agnam_recovery_recall": old_rec,
        "sian_recovery_precision": si_prec, "sian_recovery_recall": si_rec,
        "agnam_tp": old_tp, "agnam_fp": old_fp, "agnam_fn": old_fn,
        "sian_tp": si_tp, "sian_fp": si_fp, "sian_fn": si_fn,
        "sian_wall_seconds": finite_float(cost["elapsed_wall_seconds"]),
        "sian_git": cost["git"]["head"],
        "sian_test_split_hash": b["split_index_sha256"]["test"],
    }
    return rec, sources


def summarize(rows: list[dict]) -> dict:
    out = {"scenario": rows[0]["scenario"], "n": len(rows)}
    for k in METRICS:
        v = [float(r[k]) for r in rows]
        out[k + "_mean"] = statistics.mean(v)
        out[k + "_sd"] = statistics.stdev(v) if len(v) > 1 else 0.0
    out["sian_wins_auroc"] = sum(r["delta_sian_minus_agnam_auroc"] > 1e-12 for r in rows)
    out["agnam_wins_auroc"] = sum(r["delta_sian_minus_agnam_auroc"] < -1e-12 for r in rows)
    out["ties_auroc"] = len(rows) - out["sian_wins_auroc"] - out["agnam_wins_auroc"]
    return out


def collect(repo: Path, scenario: str, out: Path) -> dict:
    if scenario not in S1_TO_S4:
        raise ValueError("Scenario must be S1, S2, S3, S4")
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing archive: {out}")
    runs, sources = [], []
    for number in range(1, 21):
        r, files = check_one(repo, scenario, number)
        runs.append(r)
        sources.extend(files)
    if len({r["dataset_seed"] for r in runs}) != 20:
        raise ValueError("Data seeds repeated")
    if len({r["sian_git"] for r in runs}) != 1:
        raise ValueError("Different SIAN code revisions within scenario")
    if len({r["sian_test_split_hash"] for r in runs}) != 20:
        raise ValueError("SIAN test indices repeated")
    summary = summarize(runs)
    run_bytes = csv_render(runs)
    summary_bytes = csv_render([summary])
    source_hashes = {p.relative_to(repo).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    manifest = {
        "analysis": "r12_exploratory_paired_agnam_vs_pairwise_native_sian",
        "scope": "read-only matched synthetic generator/split seed audit; no model training",
        "scenario": scenario, "n": len(runs),
        "siancode_git_heads": sorted({r["sian_git"] for r in runs}),
        "source_sha256": source_hashes,
        "paired_rows_sha256": hashlib.sha256(run_bytes).hexdigest(),
        "paired_summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
        "restrictions": [
            "R12 comparison is exploratory and post-review: no new confirmatory claim",
            "SIAN native package constrained to MAX_K=2 with R12-set FIS thresholds",
            "Recorded AG-NAM checkpoint metrics are used; no raw AG-NAM test predictions exported",
            "Shared data/split protocol and seeds verified; raw y_test equality not provable from checkpoint alone",
            "No apples-to-apples AG-NAM runtime until isolated Full AG-NAM measured",
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(out, mode="x", compression=ZIP_DEFLATED) as z:
        z.writestr("R12_PAIRED_PER_REALIZATION.csv", run_bytes)
        z.writestr("R12_PAIRED_DESCRIPTIVE_SUMMARY.csv", summary_bytes)
        z.writestr("R12_PAIRED_MANIFEST.json", json.dumps(manifest, indent=2, ensure_ascii=False))
        # Raw source files can be used by a third party to verify the pairing.
        for src in sources:
            z.write(src, arcname=src.relative_to(repo).as_posix())
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", choices=S1_TO_S4, default="S1")
    p.add_argument("--output", default=None)
    args = p.parse_args()
    root = Path.cwd().resolve()
    target = Path(args.output or f"AG_NAM_R12_{args.scenario}_Matched_AGNAM_SIAN_Evidence.zip").resolve()
    summary = collect(root, args.scenario, target)
    print("=== R12 READ-ONLY MATCHED COMPARISON ===")
    print("SCENARIO:", args.scenario, "N:", summary["n"])
    print("SIAN minus AG-NAM mean AUROC:", summary["delta_sian_minus_agnam_auroc_mean"])
    print("SIAN minus AG-NAM mean AUPRC:", summary["delta_sian_minus_agnam_auprc_mean"])
    print("SIAN / AG-NAM AUROC wins:", summary["sian_wins_auroc"], "/", summary["agnam_wins_auroc"])
    print("AG-NAM ISR F1 vs SIAN selected F1:", summary["agnam_isr_f1_mean"], "/", summary["sian_selected_f1_mean"])
    print("ARCHIVE:", target)
    print("TRAINING: NO | EXISTING OUTPUTS MODIFIED: NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
