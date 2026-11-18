"""Post-submission Real-X detector comparisons; never writes frozen evidence."""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from dataclasses import asdict
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from agnam.benchmarking.realx_protocol import RealXProtocol, build_realx_schedule
from agnam.data.preprocessing import TabularPreprocessor
from agnam.revision.external_recovery_baselines import (
    canonical_pair, deterministic_ranking, evaluate_pair_ranking,
    friedman_h_pair_scores, full_pair_universe, nid_pair_scores_from_model,
    sian_fid_pair_scores, train_shared_sian_mlp,
)
from agnam.revision.external_recovery_runner import METHODS
from agnam.utils.reproducibility import get_device, seed_everything

ROOT = Path("results/revision/external_interaction_baselines/core_realx_v1")
FROZEN_ROOT = Path("results/realx/benchmark")
STRENGTHS = ("moderate", "null")
SCHEMA = "r6_realx_external_v1"


def select_specs(*, strength="moderate", task_id=None, realization=None):
    if strength not in (*STRENGTHS, "both"):
        raise ValueError("Invalid strength")
    specs = tuple(s for s in build_realx_schedule() if
                  s.strength_name in STRENGTHS and
                  (strength == "both" or s.strength_name == strength) and
                  (task_id is None or s.task_id == task_id) and
                  (realization is None or s.realization == realization))
    if not specs:
        raise ValueError("No matching locked Real-X specs")
    return specs


def hash_file(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def hash_int(values):
    return sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def read_csv(path, **kwargs):
    # Pandas' default NA vocabulary would silently erase the null condition.
    return pd.read_csv(path, keep_default_na=False, na_values=[""], **kwargs)


def parse_pairs(value):
    if pd.isna(value) or value == "":
        return set()
    return {canonical_pair(*p.split("::")) for p in str(value).split("|")}


def encode_original_features(X_fit, X_score):
    """One scalar per original feature; training-only fit, original column order."""
    pre = TabularPreprocessor()
    train = pre.fit_transform(X_fit)
    score = pre.transform(X_score)
    names = tuple(str(c) for c in X_fit.columns)
    def matrix(data):
        out = np.empty((len(data), len(names)), dtype=np.float64)
        for f in data.feature_specs:
            source = data.numeric if f.kind == "numeric" else data.categorical
            out[:, f.original_index] = source[:, f.transformed_index]
        if not np.isfinite(out).all():
            raise ValueError("Nonfinite encoded predictor")
        return out
    return matrix(train), matrix(score), names


def read_frozen_anchor(spec, frozen_root=FROZEN_ROOT):
    from agnam.benchmarking.realx_batch import checkpoint_path, validate_checkpoint
    from agnam.benchmarking.realx_freeze import verify_manifest_files
    root = Path(frozen_root)
    manifest_path = root / "freeze/realx_evidence_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest["freeze_status"] != "FROZEN" or manifest["benchmark_id"] != RealXProtocol().benchmark_id
            or manifest["run_checkpoint_count"] != 180 or len(manifest["files"]) != 182):
        raise ValueError("Original evidence is not frozen")
    # Verify all original locked files, not just the selected checkpoint.
    evidence_hash = verify_manifest_files(manifest, output_root=root)
    path = checkpoint_path(spec, output_root=root)
    rel = path.relative_to(root).as_posix()
    if not any(Path(f["relative_path"]).as_posix() == rel and f["role"] == "checkpoint"
               for f in manifest["files"]):
        raise ValueError("Selected checkpoint absent from freeze manifest")
    valid, reason = validate_checkpoint(path, spec)
    if not valid:
        raise ValueError(f"Invalid frozen checkpoint {spec.run_id}: {reason}")
    return read_csv(path).iloc[0], hash_file(path), evidence_hash


def prepare_realx(spec, anchor):
    from agnam.benchmarking.realx_generator import generate_realx_run
    from agnam.benchmarking.realx_openml import load_realx_openml_X, validate_realx_feature_type
    from agnam.benchmarking.realx_runner import (
        get_dataset_specification, derive_realx_model_seeds, realx_candidate_k,
        validate_locked_split,
    )
    loaded = load_realx_openml_X(get_dataset_specification(spec.task_id))
    validate_realx_feature_type(loaded)
    data = generate_realx_run(loaded.X, spec, protocol=RealXProtocol())
    train, val, test = data.train_indices, data.validation_indices, data.test_indices
    validate_locked_split(n_samples=len(data.X), train_indices=train,
                          validation_indices=val, test_indices=test)
    seed = derive_realx_model_seeds(spec).discovery_base_seed
    k = realx_candidate_k(data.X.shape[1])
    checks = {"n_samples": len(data.X), "n_features": data.X.shape[1],
              "train_size": len(train), "validation_size": len(val), "test_size": len(test),
              "discovery_base_seed": seed, "expected_candidate_k": k,
              "observed_candidate_k": k, "n_true_interactions": len(data.active_true_pairs)}
    for key, value in checks.items():
        if int(anchor[key]) != value:
            raise ValueError(f"Frozen anchor mismatch: {key}")
    if not np.isclose(float(anchor["positive_fraction"]), np.mean(data.y), atol=1e-12, rtol=0):
        raise ValueError("Generated prevalence differs from frozen anchor")
    truth = {canonical_pair(*p) for p in data.active_true_pairs}
    template = {canonical_pair(*p) for p in data.template_true_pairs}
    if truth != parse_pairs(anchor["true_pairs"]) or template != parse_pairs(anchor["template_true_pairs"]):
        raise ValueError("Generated truth differs from frozen anchor")
    X_fit, X_score, names = encode_original_features(data.X.iloc[train], data.X.iloc[val])
    y_fit, y_score = np.asarray(data.y[train]), np.asarray(data.y[val])
    if set(np.unique(y_fit)) != {0, 1} or set(np.unique(y_score)) != {0, 1}:
        raise ValueError("Both classes are required in fitting and scoring")
    provenance = {"source_row_hash": hash_int(data.source_row_positions),
                  "train_index_hash": hash_int(train), "validation_index_hash": hash_int(val),
                  "test_index_hash": hash_int(test), "generated_label_hash": hash_int(data.y),
                  "fit_matrix_sha256": sha256(np.asarray(X_fit, dtype="<f8").tobytes()).hexdigest(),
                  "score_matrix_sha256": sha256(np.asarray(X_score, dtype="<f8").tobytes()).hexdigest(),
                  "openml_dataset_id": int(loaded.openml_dataset_id),
                  "n_categorical": len(loaded.categorical_columns),
                  "n_numeric": len(loaded.numeric_columns)}
    audit_path = Path("results/realx/audit/realx_generator_audit.csv")
    if audit_path.is_file():
        audit = read_csv(audit_path)
        row = audit.loc[(audit.task_id == spec.task_id) & (audit.realization == spec.realization)]
        if len(row) != 1 or str(row.iloc[0]["source_row_hash"]) != provenance["source_row_hash"]:
            raise ValueError("Source rows differ from the original generator audit")
        provenance["original_generator_source_rows_verified"] = True
    else:
        provenance["original_generator_source_rows_verified"] = False
    return X_fit, y_fit, X_score, y_score, names, truth, seed, k, provenance


def ranking_metrics(names, scores, truth, k, matched_count):
    universe = set(full_pair_universe(names))
    truth = {canonical_pair(*pair) for pair in truth}
    if set(scores) != universe or not truth <= universe or not all(np.isfinite(list(scores.values()))):
        raise ValueError("Invalid pair universe, truth, or nonfinite score")
    if not 0 < k <= len(universe) or not 0 <= matched_count <= len(universe):
        raise ValueError("Invalid candidate budget")
    rank = deterministic_ranking(scores)
    def at_count(n):
        selected = {(a, b) for a, b, _ in rank[:n]}
        tp = len(selected & truth)
        fp = len(selected - truth)
        fn = len(truth - selected)
        # Null: ranking discrimination and positive-class recall/F1 are undefined.
        return dict(tp=tp, fp=fp, fn=fn, precision=tp/n if n else (0.0 if truth else np.nan),
                    recall=tp/len(truth) if truth else np.nan,
                    f1=2*tp/(2*tp+fp+fn) if truth else np.nan)
    if truth:
        metrics = asdict(evaluate_pair_ranking(feature_names=names, pair_scores=scores,
                                               true_pairs=truth, candidate_k=k))
    else:
        metrics = {"interaction_auprc": np.nan, "interaction_auroc": np.nan, "ndcg": np.nan,
                   "precision_at_true_count": np.nan, "recall_at_true_count": np.nan,
                   "f1_at_true_count": np.nan}
    for label, n in (("k", k), ("matched", matched_count)):
        for key, value in at_count(n).items():
            metrics[f"{key}_at_{label}"] = value
    metrics.update(n_pairs=len(universe), n_true=len(truth), random_prevalence=len(truth)/len(universe),
                   score_mean=float(np.mean(list(scores.values()))),
                   score_max=float(np.max(list(scores.values()))),
                   positive_score_fraction=float(np.mean(np.array(list(scores.values())) > 0)))
    return metrics, rank


def paths_for(spec, root=ROOT):
    directory = Path(root) / f"task_{spec.task_id}" / f"r{spec.realization:02d}_{spec.strength_name}"
    return directory / "summary.csv", directory / "pair_scores.csv", directory / "manifest.json"


def validate_outputs(spec, root=ROOT, *, anchor_hash=None, evidence_hash=None):
    summary, pairs, manifest_path = paths_for(spec, root)
    if not all(p.is_file() for p in (summary, pairs, manifest_path)):
        raise ValueError("Incomplete output triplet")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["schema"] != SCHEMA or manifest["spec"] != asdict(spec) or tuple(manifest["methods"]) != METHODS:
        raise ValueError("Protocol mismatch")
    if anchor_hash is not None and manifest["frozen_checkpoint_sha256"] != anchor_hash:
        raise ValueError("Frozen checkpoint hash mismatch")
    if evidence_hash is not None and manifest["frozen_evidence_sha256"] != evidence_hash:
        raise ValueError("Frozen evidence hash mismatch")
    for path in (summary, pairs):
        if hash_file(path) != manifest["outputs"][path.name]:
            raise ValueError(f"Output hash mismatch: {path}")
    a, b = read_csv(summary), read_csv(pairs, dtype={"feature_a": str, "feature_b": str})
    if len(a) != len(METHODS) or set(a.method) != set(METHODS) or set(b.method) != set(METHODS):
        raise ValueError("Method coverage mismatch")
    if set(a.run_id) != {spec.run_id} or set(b.run_id) != {spec.run_id} or set(a.strength) != {spec.strength_name}:
        raise ValueError("Run ID mismatch")
    names = manifest["feature_names"]
    universe = set(full_pair_universe(names))
    for method in METHODS:
        r = a.loc[a.method == method].iloc[0]
        q = b.loc[b.method == method]
        if (len(q) != int(r.n_pairs) or q[["feature_a", "feature_b"]].duplicated().any()
                or sorted(q["rank"].tolist()) != list(range(1, len(q)+1))
                or set(zip(q.feature_a, q.feature_b)) != universe
                or not np.isfinite(q.score).all() or int(q.is_true.sum()) != spec.active_true_interaction_count
                or int(r.n_true) != spec.active_true_interaction_count):
            raise ValueError(f"Invalid complete ranking: {method}")
    return a, b, manifest


def run_external_realx(spec, *, root=ROOT, frozen_root=FROZEN_ROOT, force=False):
    if spec not in select_specs(strength="both"):
        raise ValueError("Only locked moderate/null specs are allowed")
    root, frozen_root = Path(root), Path(frozen_root)
    if root.resolve() == frozen_root.resolve() or frozen_root.resolve() in root.resolve().parents:
        raise ValueError("Refusing to write under frozen benchmark")
    anchor, anchor_hash, evidence_hash = read_frozen_anchor(spec, frozen_root)
    summary_path, pairs_path, manifest_path = paths_for(spec, root)
    if not force and any(p.exists() for p in (summary_path, pairs_path, manifest_path)):
        validate_outputs(spec, root, anchor_hash=anchor_hash, evidence_hash=evidence_hash)
        print("SKIP valid:", spec.run_id)
        return summary_path
    X_fit, y_fit, X_score, y_score, names, truth, seed, k, provenance = prepare_realx(spec, anchor)
    matched_count = int(anchor["n_isr_retained"])
    if matched_count != len(parse_pairs(anchor["isr_pairs"])):
        raise ValueError("Frozen ISR cardinality mismatch")
    if version("sian") != "0.9.5":
        raise ValueError("R5a/R6 requires sian==0.9.5")
    device = get_device()
    seed_everything(seed)
    common = {"run_id": spec.run_id, "task_id": spec.task_id, "dataset_name": spec.dataset_name,
              "feature_type": spec.feature_type, "realization": spec.realization,
              "strength": spec.strength_name, "candidate_k": k, "matched_count": matched_count,
              "baseline_seed": seed, "n_features": len(names), "n_fit": len(y_fit), "n_score": len(y_score),
              "agnam_mean_discovery_ap": anchor["mean_interaction_auprc"] if truth else np.nan,
              "agnam_selection_f1": anchor["selection_f1"] if truth else np.nan,
              "agnam_isr_f1": anchor["isr_f1"] if truth else np.nan}
    rows, pair_rows = [], []
    def record(method, scores, fit, scoring, start, end, peak, cuda, epoch=None, val_auc=None):
        metrics, rank = ranking_metrics(names, scores, truth, k, matched_count)
        rows.append({**common, "method": method, **metrics, "fit_seconds": fit,
                     "scoring_seconds": scoring, "standalone_seconds": fit+scoring,
                     "rss_start_mib": start, "rss_end_mib": end, "rss_peak_mib": peak,
                     "cuda_peak_allocated_mib": cuda, "shared_mlp_best_epoch": epoch,
                     "shared_mlp_best_val_auroc": val_auc})
        pair_rows.extend({"run_id": spec.run_id, "method": method, "feature_a": a, "feature_b": b,
                          "score": sc, "rank": i, "is_true": (a,b) in truth}
                         for i, (a,b,sc) in enumerate(rank, 1))
    print("START", spec.run_id, "device", device, "p", len(names), "K", k)
    h_grid_fallback_features = []
    h, fit, scoring, start, end, peak = friedman_h_pair_scores(
        X_fit=X_fit, y_fit=y_fit, X_score=X_score, feature_names=names, seed=seed, grid_resolution=10,
        allow_degenerate_grid_fallback=True, grid_fallback_features=h_grid_fallback_features)
    if h_grid_fallback_features:
        print("  Friedman-H degenerate percentile grid fallback:", h_grid_fallback_features, flush=True)
    record(METHODS[0], h, fit, scoring, start, end, peak, 0.0)
    print("  Friedman-H complete", flush=True)
    shared = train_shared_sian_mlp(X_fit, y_fit, X_score, y_score, seed=seed, device=device)
    t = time.perf_counter()
    nid = nid_pair_scores_from_model(shared.model, names)
    scoring = time.perf_counter()-t
    record(METHODS[1], nid, shared.training_seconds, scoring, shared.rss_start_mib,
           shared.rss_end_mib, shared.rss_peak_mib, shared.cuda_peak_allocated_mib,
           shared.best_epoch, shared.best_val_auroc)
    print("  NID complete", flush=True)
    sian, scoring, start, end, peak, cuda = sian_fid_pair_scores(
        model=shared.model, scaler=shared.scaler, X_score=X_score, feature_names=names,
        seed=seed, device=device, agg_k=32, explainer_batch_size=512)
    record(METHODS[2], sian, shared.training_seconds, scoring, shared.rss_start_mib, end,
           max(shared.rss_peak_mib, peak), max(shared.cuda_peak_allocated_mib, cuda),
           shared.best_epoch, shared.best_val_auroc)
    # Write manifest last. Incomplete/invalid triplets are never silently accepted.
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.unlink(missing_ok=True)
    for frame, path in ((pd.DataFrame(rows), summary_path), (pd.DataFrame(pair_rows), pairs_path)):
        tmp = path.with_suffix(".tmp")
        frame.to_csv(tmp, index=False)
        tmp.replace(path)
    manifest = {"schema": SCHEMA, "spec": asdict(spec), "methods": list(METHODS),
                "frozen_checkpoint_sha256": anchor_hash, "frozen_evidence_sha256": evidence_hash,
                "provenance": provenance, "feature_names": list(names), "seed": seed,
                "encoding": "train-only TabularPreprocessor; scalar original columns; numeric missing flags omitted",
                "device": str(device), "versions": {p: version(p) for p in
                    ("numpy", "pandas", "torch", "scikit-learn", "openml", "sian", "psutil")},
                "settings": {"friedman_h": {"trees": 200, "learning_rate": 0.05, "depth": 3, "grid": 10,
                                             "grid_policy": "original_05_95_with_observed_support_fallback_v1",
                                             "grid_fallback_features": h_grid_fallback_features},
                             "shared_mlp": {"widths": [128,64,32], "optimizer": "Adagrad", "lr": 0.005,
                                            "batch_size": 32, "epochs": 300, "l1": 5e-5, "patience": 20},
                             "sian_fid": {"agg_k":32, "batch_size":512, "score":"old_arch_inter_score",
                                          "masking":"triangle_marginal"}},
                "outputs": {p.name: hash_file(p) for p in (summary_path, pairs_path)},
                "post_submission": True, "formal_hypothesis_tests": False}
    tmp = manifest_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    tmp.replace(manifest_path)
    validate_outputs(spec, root, anchor_hash=anchor_hash, evidence_hash=evidence_hash)
    print(pd.DataFrame(rows)[["method", "interaction_auprc", "f1_at_k", "f1_at_matched", "standalone_seconds"]].to_string(index=False))
    print("SAVED", summary_path)
    return summary_path
