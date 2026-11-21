"""Read-only R12 SIAN native-pairwise evidence audit and ZIP export.

Recomputes predictive and selected-pair recovery metrics from per-run CSVs,
refuses incomplete runs, writes only the requested NEW ZIP artifact. Model
state_dict files remain on the user's machine and are not duplicated in ZIP.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import zipfile

from sklearn.metrics import average_precision_score, roc_auc_score

RESULT_ROOT = Path("results/revision/r12_native_sian_pairwise")
COST_ROOT = Path("results/revision/r12_cost_audit")
EXPECTED_TRUE_COUNTS = {"S1": 3, "S2": 4, "S3": 2, "S4": 4}
EXPECTED_SPLIT_SIZES = {"S1": (3000, 1000, 1000), "S2": (4800, 1600, 1600),
                        "S3": (4800, 1600, 1600), "S4": (6000, 2000, 2000)}


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def close(a: float, b: float, atol: float = 2e-7) -> bool:
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= atol


def check_run(repo: Path, scenario: str, number: int) -> tuple[dict, list[Path]]:
    root = repo / RESULT_ROOT / scenario / f"realization_{number:02d}"
    cost = repo / COST_ROOT / f"r12_{scenario.lower()}_{number:02d}_sian"
    files = [root / "summary.json", root / "selected_pairs.csv",
             root / "test_predictions.csv", cost / "measurement.json"]
    for f in files:
        if not f.is_file():
            raise FileNotFoundError(f"Missing evidence for {scenario}-{number:02d}: {f}")
    summary = json.loads(files[0].read_text(encoding="utf-8"))
    measure = json.loads(files[3].read_text(encoding="utf-8"))
    if not (summary.get("status") == "complete" and summary.get("scenario") == scenario
            and summary.get("realization") == number and summary.get("sian_version") == "0.9.5"):
        raise ValueError(f"Wrong or incomplete SIAN summary for {scenario}-{number:02d}")
    if not (measure.get("returncode") == 0 and measure.get("experiment_run_status") == "complete"):
        raise ValueError("Incomplete external wall-clock measurement")
    if (not isinstance(summary.get("split_index_sha256"), dict)
            or set(summary["split_index_sha256"]) != {"train", "validation", "test"}):
        raise ValueError("Missing split hashes")
    sizes = tuple(summary["data_counts"][key] for key in ("train", "validation", "test"))
    if sizes != EXPECTED_SPLIT_SIZES[scenario]:
        raise ValueError(f"Unexpected split sizes: {sizes}")
    preds = read_csv(files[2])
    if len(preds) != sizes[2]:
        raise ValueError("Test prediction count does not match test split")
    labels = [int(row["y_test"]) for row in preds]
    scores = [float(row["sian_probability"]) for row in preds]
    if not all(0.0 <= s <= 1.0 and math.isfinite(s) for s in scores):
        raise ValueError("Invalid test probability")
    if set(labels) != {0, 1}:
        raise ValueError("Test set must contain both binary labels")
    auc = float(roc_auc_score(labels, scores))
    ap = float(average_precision_score(labels, scores))
    if not close(auc, summary["test_predictive_metrics"]["test_auroc"]):
        raise ValueError("AUROC discrepancy")
    if not close(ap, summary["test_predictive_metrics"]["test_auprc"]):
        raise ValueError("AUPRC discrepancy")
    pairs = read_csv(files[1])
    names = {(r["feature_a"], r["feature_b"]) for r in pairs}
    if len(names) != len(pairs):
        raise ValueError("Duplicate selected pairs")
    tp = sum(str(r["is_true"]).strip().lower() == "true" for r in pairs)
    fp = len(pairs) - tp
    fn = EXPECTED_TRUE_COUNTS[scenario] - tp
    if fn < 0:
        raise ValueError("TP exceeds known true-pair count")
    r = summary["interaction_recovery"]
    prec = tp / len(pairs) if pairs else 0.0
    recall = tp / EXPECTED_TRUE_COUNTS[scenario]
    f1 = 2 * prec * recall / (prec + recall) if prec + recall else 0.0
    if (not all(int(r[k]) == v for k, v in (("true_positive", tp), ("false_positive", fp),
                                            ("false_negative", fn)))
        or not all(close(float(r[k]), v) for k, v in (("precision", prec),
                                                   ("recall", recall), ("f1", f1)))):
        raise ValueError("Recovery metric discrepancy")
    if summary["n_selected_pairs"] != len(pairs):
        raise ValueError("Selected pair count mismatch")
    if summary["native_fis_settings"].get("max_interaction_degree") != 2:
        raise ValueError("Not the locked pairwise-constrained SIAN protocol")
    expected_label = f"r12_{scenario.lower()}_{number:02d}_sian"
    if measure["label"] != expected_label:
        raise ValueError("Cost label mismatch")
    row = {"scenario": scenario, "realization": number,
           "n_test": len(preds), "n_selected_pairs": len(pairs),
           "tp": tp, "fp": fp, "fn": fn,
           "precision": prec, "recall": recall, "f1": f1,
           "test_auroc": auc, "test_auprc": ap,
           "wall_seconds_external": measure["elapsed_wall_seconds"],
           "wall_seconds_internal": summary["stage_elapsed_seconds"]["whole_pipeline_seconds"],
           "peak_rss_mib_sampled": measure.get("peak_process_tree_rss_mib_sampled"),
           "peak_gpu_allocated_mib": summary.get("peak_torch_cuda_allocated_mib"),
           "best_epoch": summary["sian_train"]["best_epoch"],
           "epochs_ran": summary["sian_train"]["epochs_ran"],
           "git_head": measure.get("git", {}).get("head", ""),
           "selected_pairs": sorted(names)}
    return row, files


def summarize(rows: list[dict]) -> list[dict]:
    result = []
    scenarios = sorted(set(r["scenario"] for r in rows))
    for scenario in scenarios:
        items = [r for r in rows if r["scenario"] == scenario]
        d = {"scenario": scenario, "n": len(items)}
        for metric in ("test_auroc", "test_auprc", "precision", "recall", "f1",
                       "n_selected_pairs", "tp", "fp", "fn", "wall_seconds_external",
                       "peak_rss_mib_sampled", "peak_gpu_allocated_mib"):
            values = [float(r[metric]) for r in items if r[metric] is not None]
            d[metric + "_mean"] = statistics.mean(values) if values else None
            d[metric + "_sd"] = statistics.stdev(values) if len(values) > 1 else None
        pair_sets = [set(tuple(p) for p in r["selected_pairs"]) for r in items]
        similarities = []
        for a in range(len(pair_sets)):
            for b in range(a + 1, len(pair_sets)):
                union = pair_sets[a] | pair_sets[b]
                similarities.append(len(pair_sets[a] & pair_sets[b]) / len(union) if union else 1.0)
        d["mean_pairwise_jaccard_descriptive"] = (
            statistics.mean(similarities) if similarities else None)
        result.append(d)
    return result


def csv_bytes(rows: list[dict]) -> bytes:
    if not rows:
        return b""
    buffer = io.StringIO()
    columns = [k for k in rows[0] if k != "selected_pairs"]
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    writer.writerows({k: row.get(k) for k in columns} for row in rows)
    return buffer.getvalue().encode("utf-8")


def collect(repo: Path, scenario: str, start: int, end: int, output: Path) -> Path:
    if scenario not in EXPECTED_TRUE_COUNTS or not 1 <= start <= end <= 20:
        raise ValueError("Scenario or range out of bounds")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite prior evidence: {output}")
    rows, data_files = [], []
    for number in range(start, end + 1):
        row, paths = check_run(repo, scenario, number)
        rows.append(row)
        data_files.extend(paths)
    # Mixing different code commits can happen if the repository is modified;
    # flag this visibly instead of silently presenting results as one frozen run.
    heads = sorted({r["git_head"] for r in rows})
    row_bytes = csv_bytes(rows)
    summary = summarize(rows)
    summary_bytes = csv_bytes(summary)
    report = {"status": "independently_audited_r12_pairwise_sian",
              "scenario": scenario, "start": start, "end": end,
              "run_count": len(rows), "git_heads": heads,
              "mixed_git_revisions": len(heads) != 1,
              "scope": "SIAN native FIS plus native sparse neural model, MAX_K=2",
              "limitations": ["No unrestricted higher-order SIAN benchmark",
                              "No matched AG-NAM cost without isolated full-path measurement",
                              "One realization ranking AUPRC is not available from a selected set",
                              "No superiority inference performed", "Experimental thresholds are R12 adaptations"],
              "file_sha256": {str(p.relative_to(repo)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in data_files},
              "summary_csv_sha256": hashlib.sha256(summary_bytes).hexdigest(),
              "by_realization_csv_sha256": hashlib.sha256(row_bytes).hexdigest()}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for p in data_files:
            z.write(p, arcname=str(p.relative_to(repo)).replace("\\", "/"))
        z.writestr("R12_BATCH_PER_REALIZATION.csv", row_bytes)
        z.writestr("R12_BATCH_DESCRIPTIVE_SUMMARY.csv", summary_bytes)
        z.writestr("R12_BATCH_EVIDENCE_MANIFEST.json", json.dumps(report, indent=2))
    print("=== R12 NATIVE SIAN BATCH EVIDENCE ===")
    print("AUDITED RUNS:", len(rows), "SCENARIO:", scenario)
    print("MIXED GIT REVISIONS:", len(heads) != 1)
    print("DESCRIPTIVE SUMMARY:", summary)
    print("ARCHIVE:", output)
    print("NO TRAINING | EXISTING RESULTS MODIFIED: NO")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=tuple(EXPECTED_TRUE_COUNTS), default="S1")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=5)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    filename = (f"AG_NAM_R12_SIAN_{args.scenario}_R{args.start:02d}_R{args.end:02d}_Evidence.zip")
    output = Path(args.output or filename).resolve()
    collect(Path.cwd().resolve(), args.scenario, args.start, args.end, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
