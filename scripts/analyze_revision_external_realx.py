"""Descriptive post-submission analysis. No training or new hypothesis tests."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from agnam.revision.external_realx_runner import (
    ROOT, select_specs, validate_outputs, paths_for, hash_file, read_frozen_anchor,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strength", choices=["moderate", "null", "both"], default="both")
    args = parser.parse_args()
    specs = select_specs(strength=args.strength)
    frames, provenance = [], {}
    evidence_hashes = set()
    for s in specs:
        anchor, anchor_hash, evidence_hash = read_frozen_anchor(s)
        evidence_hashes.add(evidence_hash)
        a, _, _ = validate_outputs(s, anchor_hash=anchor_hash, evidence_hash=evidence_hash)
        frames.append(a)
        for p in paths_for(s):
            provenance[p.as_posix()] = hash_file(p)
    if len(evidence_hashes) != 1:
        raise ValueError("Frozen evidence differs across selected runs")
    data = pd.concat(frames, ignore_index=True)
    if len(data) != len(specs)*3 or data.duplicated(["run_id", "method"]).any():
        raise ValueError("Incomplete or duplicate experiment records")
    # The original AG-NAM AP averages five runs, unlike the single-run external detector AP.
    data["ap_minus_agnam_mean_discovery_ap"] = data.interaction_auprc - data.agnam_mean_discovery_ap
    data["matched_f1_minus_agnam_isr"] = data.f1_at_matched - data.agnam_isr_f1
    metrics = ["interaction_auprc", "interaction_auroc", "ndcg", "random_prevalence",
               "precision_at_k", "recall_at_k", "f1_at_k", "fp_at_k", "fn_at_k",
               "f1_at_true_count", "matched_count", "precision_at_matched", "recall_at_matched",
               "f1_at_matched", "fp_at_matched", "fn_at_matched", "agnam_mean_discovery_ap",
               "agnam_isr_f1", "ap_minus_agnam_mean_discovery_ap", "matched_f1_minus_agnam_isr",
               "fit_seconds", "scoring_seconds", "standalone_seconds", "rss_peak_mib", "cuda_peak_allocated_mib",
               "score_mean", "score_max", "positive_score_fraction"]
    dataset = data.groupby(["strength", "task_id", "dataset_name", "feature_type", "method"], sort=False)[metrics].agg(["mean", "std"])
    dataset.columns = [f"{a}_{b}" for a,b in dataset.columns]
    dataset = dataset.reset_index()
    # Equal weight to each dataset (not a new confirmatory test).
    mean_cols = [f"{m}_mean" for m in metrics]
    macro = dataset.groupby(["strength", "method"], sort=False)[mean_cols].agg(["mean", "std"])
    macro.columns = [f"{a}_across_datasets_{b}" for a,b in macro.columns]
    macro = macro.reset_index()
    out = ROOT / f"analysis_{args.strength}"
    out.mkdir(parents=True, exist_ok=True)
    files = {"experiment_summary.csv": data, "dataset_method_summary.csv": dataset,
             "macro_method_summary.csv": macro}
    for name, frame in files.items():
        tmp = out / (name+".tmp")
        frame.to_csv(tmp, index=False)
        tmp.replace(out/name)
    manifest = {"strength_scope": args.strength, "n_runs": len(specs), "n_method_evaluations": len(data),
                "frozen_evidence_sha256": next(iter(evidence_hashes)), "inputs": provenance,
                "outputs": {name: hash_file(out/name) for name in files}, "formal_hypothesis_tests": False,
                "comparison_note": "External AP uses one model/run; AG-NAM AP averages five discovery runs. "
                                   "Null positive-class ranking metrics undefined; fixed-K false pairs are forced."}
    path = out/"analysis_manifest.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp.replace(path)
    print("VERIFIED RUNS:", len(specs), "METHOD EVALUATIONS:", len(data))
    print(macro[["strength", "method", "interaction_auprc_mean_across_datasets_mean", "f1_at_k_mean_across_datasets_mean",
                 "f1_at_matched_mean_across_datasets_mean", "standalone_seconds_mean_across_datasets_mean"]].to_string(index=False))
    print("SAVED", out)


if __name__ == "__main__":
    main()
