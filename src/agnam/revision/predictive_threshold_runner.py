"""R9-B: R4-compatible, reviewer-requested predictive threshold follow-up.

This is a separate post-submission exploratory analysis. It reuses the locked
synthetic generator/split/model trainer. A hard R4 recovery-parity audit runs
before any final predictive training. Identical pair sets share final-fit cache.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from agnam.benchmarking.synthetic_protocol import (
    SyntheticBenchmarkProtocol,
    SyntheticRealizationSpec,
    candidate_count_from_feature_count,
)
from agnam.benchmarking.synthetic_runner import (
    SyntheticModelConfig,
    _build_discovery_configs,
    _build_isr_config,
    _prepare_final_prediction_data,
    _train_interaction_model,
    generate_synthetic_scenario,
    normalize_pairs,
)
from agnam.revision.sensitivity import (
    build_sensitivity_configurations,
    isr_pairs_at_threshold,
    make_discovery_view,
    recovery_metrics,
    selection_pairs_from_discovery,
)
from agnam.training.discovery import run_reproducible_interaction_discovery
from agnam.training.isr import evaluate_isr_for_discovery
from agnam.utils.reproducibility import get_device


SENSITIVITY_ROOT = Path("results/revision/sensitivity/synthetic")
ORIGINAL_ROOT = Path("results/synthetic/master_results.csv")
DEFAULT_ROOT = Path("results/revision/predictive_threshold_r9/v1")
AUDIT_COLUMNS = (
    "n_selection_retained", "selection_tp", "selection_fp", "selection_fn",
    "selection_precision", "selection_recall", "selection_f1",
    "n_isr_retained", "isr_tp", "isr_fp", "isr_fn",
    "isr_precision", "isr_recall", "isr_f1",
    "fp_reduction_vs_selection",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pairs_hash(pairs) -> str:
    payload = json.dumps([list(p) for p in normalize_pairs(pairs)], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _save_json_atomic(path: Path, payload: dict | list) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def verify_r4_threshold_parity(
    created: pd.DataFrame,
    archived: pd.DataFrame,
) -> None:
    """Fail closed if any R4 pair-recovery summary differs."""
    archived = archived[archived.family == "threshold_grid"]
    if len(created) != 9 or len(archived) != 9:
        raise ValueError("Expected exactly 9 R4/R9 threshold-grid rows")
    if set(created.config_id) != set(archived.config_id):
        raise ValueError("Threshold configuration IDs disagree with R4")
    base = archived.set_index("config_id")
    proposed = created.set_index("config_id")
    for config_id in proposed.index:
        for col in AUDIT_COLUMNS + ("candidate_k", "n_discovery_runs", "selection_threshold", "isr_threshold"):
            left = float(proposed.loc[config_id, col])
            right = float(base.loc[config_id, col])
            if not np.isclose(left, right, atol=1e-7, rtol=1e-7, equal_nan=True):
                raise ValueError(
                    f"R4 parity failure {config_id} {col}: newly={left} archived={right}; "
                    "no final model training performed"
                )


def _original_reference(spec: SyntheticRealizationSpec, path: Path) -> dict[str, float]:
    df = pd.read_csv(path)
    subset = df[(df.scenario == spec.scenario) & (df.realization_index == spec.realization_index)]
    if len(subset) != 1:
        raise ValueError("Original synthetic reference realization missing or duplicated")
    row = subset.iloc[0]
    if not (np.isfinite(float(row.main_auroc)) and np.isfinite(float(row.agnam_auroc))):
        raise ValueError("Original benchmark lacks finite final AUROC")
    return {
        "main_auroc": float(row.main_auroc),
        "full_auroc": float(row.agnam_auroc),
    }


def run_predictive_threshold_realization(
    spec: SyntheticRealizationSpec,
    *,
    protocol: SyntheticBenchmarkProtocol | None = None,
    model_config: SyntheticModelConfig | None = None,
    output_root: str | Path = DEFAULT_ROOT,
    r4_root: str | Path = SENSITIVITY_ROOT,
    original_path: str | Path = ORIGINAL_ROOT,
    device=None,
    force: bool = False,
    verbose: bool = True,
) -> dict[str, Path]:
    protocol = protocol or SyntheticBenchmarkProtocol()
    model_config = model_config or SyntheticModelConfig()
    device = device if device is not None else get_device()
    if spec.scenario not in ("S1", "S2", "S3", "S4") or not 1 <= spec.realization_number <= 5:
        raise ValueError("R9-B prespecified exploratory subset is realization 1-5 per scenario")
    root = Path(output_root) / spec.scenario / f"realization_{spec.realization_number:02d}"
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": root / "predictive_summary.csv",
        "pair_sets": root / "pair_sets.json",
        "cache": root / "final_model_cache.json",
        "manifest": root / "manifest.json",
    }
    r4_source = Path(r4_root) / spec.scenario / f"realization_{spec.realization_number:02d}_summary.csv"
    original_path = Path(original_path)
    source_hashes = {"r4": _sha256(r4_source), "original": _sha256(original_path)}
    if paths["manifest"].exists() and paths["summary"].exists() and not force:
        saved = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        if saved.get("source_hashes") != source_hashes:
            raise RuntimeError("Existing R9-B output used different source files; audit before proceeding")
        if saved.get("n_threshold_configs") != 9 or saved.get("summary_sha256") != _sha256(paths["summary"]):
            raise RuntimeError("Existing R9-B output is incomplete/corrupt; do not silently skip")
        if verbose:
            print("R9-B SKIP verified completed:", root)
        return paths

    # Repeat exactly the outer split used in the locked synthetic benchmark.
    dataset = generate_synthetic_scenario(spec.scenario, seed=spec.dataset_seed)
    X, y = dataset.X, dataset.y.to_numpy()
    dev_idx, test_idx = train_test_split(
        np.arange(len(X)),
        test_size=protocol.outer_test_fraction,
        random_state=spec.outer_split_seed,
        stratify=y,
    )
    X_dev = X.iloc[dev_idx].reset_index(drop=True)
    y_dev = y[dev_idx]
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_test = y[test_idx]
    true_pairs = normalize_pairs(dataset.true_interactions)

    nam_cfg, proposer_cfg = _build_discovery_configs(spec=spec, config=model_config)
    if verbose:
        print("R9-B discovery (5 locked seeds):", spec.scenario, spec.realization_number)
    discovery = run_reproducible_interaction_discovery(
        X=X_dev,
        y=y_dev,
        n_runs=5,
        base_seed=spec.discovery_base_seed,
        selection_threshold=0.60,
        crossfit_n_splits=protocol.residual_crossfit_folds,
        main_early_stop_fraction=0.20,
        nam_hidden_dim=model_config.nam_hidden_dim,
        nam_depth=model_config.nam_depth,
        nam_dropout=model_config.nam_dropout,
        categorical_embedding_dim=model_config.categorical_embedding_dim,
        proposer_d_model=model_config.proposer_d_model,
        proposer_n_heads=model_config.proposer_n_heads,
        proposer_n_layers=model_config.proposer_n_layers,
        proposer_dropout=model_config.proposer_dropout,
        nam_config=nam_cfg,
        proposer_config=proposer_cfg,
        device=device,
    )
    locked_k = candidate_count_from_feature_count(X_dev.shape[1], protocol)
    configs = tuple(c for c in build_sensitivity_configurations() if c.family == "threshold_grid")
    views: dict[float, object] = {}
    isr_cache: dict[float, object] = {}
    generated = []
    pair_rows = []
    for cfg in configs:
        s = float(cfg.selection_threshold)
        if s not in views:
            views[s] = make_discovery_view(discovery.runs, n_runs=5, k=locked_k, selection_threshold=s)
        view = views[s]
        selection_pairs = selection_pairs_from_discovery(view)
        if s not in isr_cache:
            base_config = _build_isr_config(protocol=protocol, config=model_config)
            eval_config = replace(base_config, reference_size=512, isr_threshold=0.40)
            isr_cache[s] = evaluate_isr_for_discovery(
                X=X_dev, discovery=view, config=eval_config, device=device, verbose=False
            )
        isr_pairs = isr_pairs_at_threshold(isr_cache[s], threshold=cfg.isr_threshold)
        measures = recovery_metrics(
            selection_pairs=selection_pairs, isr_pairs=isr_pairs, true_pairs=true_pairs
        )
        generated.append({
            "config_id": cfg.config_id, "family": cfg.family,
            "selection_threshold": s,
            "isr_threshold": float(cfg.isr_threshold),
            "n_discovery_runs": 5, "candidate_k": locked_k,
            **measures,
        })
        pair_rows.append({
            "config_id": cfg.config_id,
            "selection_pairs": [list(v) for v in selection_pairs],
            "isr_pairs": [list(v) for v in isr_pairs],
            "pair_hash": _pairs_hash(isr_pairs),
        })

    verify_r4_threshold_parity(pd.DataFrame(generated), pd.read_csv(r4_source))
    if verbose:
        print("R9-B VERIFIED R4 RECOVERY PARITY: 9/9")
    _save_json_atomic(paths["pair_sets"], pair_rows)

    # Only model fits vary. The locked Main NAM benchmark is the shared baseline.
    original = _original_reference(spec, original_path)
    final_data = _prepare_final_prediction_data(
        X_outer_dev=X_dev, y_outer_dev=y_dev,
        X_test=X_test, y_test=y_test,
        final_split_seed=spec.final_split_seed,
    )
    cached = {}
    if paths["cache"].exists() and not force:
        prior = json.loads(paths["cache"].read_text(encoding="utf-8"))
        if prior.get("source_hashes") != source_hashes or prior.get("pair_sets_sha256") != _sha256(paths["pair_sets"]):
            raise RuntimeError("Stale final-model cache; cannot resume safely")
        cached = prior["models"]

    rows = []
    # Always evaluate the reference threshold first to catch AUROC mismatches
    # before performing other expensive final fits.
    ordered = sorted(pair_rows, key=lambda r: (r["config_id"] != "threshold_s60_i60", r["config_id"]))
    models_fit_in_run = 0
    for item in ordered:
        key = item["pair_hash"]
        if key not in cached:
            start = perf_counter()
            pred, decomposition_error = _train_interaction_model(
                data=final_data,
                interaction_pairs=normalize_pairs(item["isr_pairs"]),
                seed=spec.final_model_seed,
                config=model_config,
                device=device,
            )
            cached[key] = {
                "pairs": item["isr_pairs"],
                "auroc": float(pred.auroc),
                "auprc": float(pred.auprc),
                "balanced_accuracy": float(pred.balanced_accuracy),
                "f1": float(pred.f1),
                "best_epoch": int(pred.best_epoch),
                "max_decomposition_error": float(decomposition_error),
                "final_fit_seconds": float(perf_counter() - start),
            }
            models_fit_in_run += 1
            _save_json_atomic(paths["cache"], {
                "source_hashes": source_hashes,
                "pair_sets_sha256": _sha256(paths["pair_sets"]),
                "models": cached,
            })
        if item["config_id"] == "threshold_s60_i60":
            delta = abs(cached[key]["auroc"] - original["full_auroc"])
            if delta > 1e-6:
                raise RuntimeError(
                    f"R9-B original AUROC parity failed: {delta:.8g}. "
                    "Stop: no nonreference final models will be trained"
                )
        rows.append({
            "scenario": spec.scenario,
            "realization_index": spec.realization_index,
            "realization_number": spec.realization_number,
            "config_id": item["config_id"],
            "pair_hash": key,
            "n_isr_retained": len(item["isr_pairs"]),
            "main_auroc_locked": original["main_auroc"],
            "agnam_auroc": cached[key]["auroc"],
            "delta_auroc_vs_main": cached[key]["auroc"] - original["main_auroc"],
            "auprc": cached[key]["auprc"],
            "balanced_accuracy": cached[key]["balanced_accuracy"],
            "f1": cached[key]["f1"],
            "best_epoch": cached[key]["best_epoch"],
            "max_decomposition_error": cached[key]["max_decomposition_error"],
            "model_fit_cache_key": key,
        })
        if verbose:
            print("R9-B", item["config_id"], "pairs", len(item["isr_pairs"]), "AUROC", f"{cached[key]['auroc']:.6f}")

    frame = pd.DataFrame(rows).sort_values("config_id").reset_index(drop=True)
    if len(frame) != 9:
        raise RuntimeError("R9-B output must contain 9 threshold configurations")
    frame.to_csv(paths["summary"], index=False)
    manifest = {
        "analysis": "R9-B predictive threshold sensitivity",
        "status": "reviewer_requested_post_submission_exploratory",
        "subset_definition": "S1-S4 realizations 1-5 (20 total), fixed before R9-B outcomes",
        "scenario": spec.scenario,
        "realization_number": spec.realization_number,
        "source_hashes": source_hashes,
        "n_threshold_configs": len(frame),
        "n_unique_pair_sets": int(frame.pair_hash.nunique()),
        "new_final_models_fit_this_invocation": models_fit_in_run,
        "r4_recovery_parity": "9/9",
        "original_auroc_parity_absolute_tolerance": 1e-6,
        "main_auroc": original["main_auroc"],
        "original_full_auroc": original["full_auroc"],
        "summary_sha256": _sha256(paths["summary"]),
        "pair_sets_sha256": _sha256(paths["pair_sets"]),
        "model_cache_sha256": _sha256(paths["cache"]),
        "locked_primary_results_modified": False,
    }
    _save_json_atomic(paths["manifest"], manifest)
    if verbose:
        print("R9-B COMPLETE:", len(frame), "threshold configs;", frame.pair_hash.nunique(), "distinct pair sets")
        print("R9-B SAVED:", root)
    return paths
