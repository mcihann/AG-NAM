"""R12: Native SIAN (v0.9.5) pairwise-constrained synthetic pilot.

A full *native* SIAN workflow: reference MLP fit -> SIAN's own
layerwise feature interaction selection (FIS) -> native SIAN sparse
additive-model fit. MAX_K=2 is intentionally fixed for a fair pairwise
recovery task. This is NOT the unconstrained higher-order SIAN model.

No experiment is executed without --execute. No existing R1-R11 outputs
are read to choose the model and no existing output is overwritten.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
from itertools import combinations
from pathlib import Path
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from agnam.benchmarking.synthetic_protocol import (
    SyntheticBenchmarkProtocol, build_synthetic_schedule, evaluate_pair_set_recovery,
)
from agnam.benchmarking.synthetic_runner import generate_synthetic_scenario, normalize_pairs
from agnam.revision.external_recovery_baselines import (
    PeakRSSMonitor, train_shared_sian_mlp,
)
from agnam.utils.reproducibility import get_device


# Locked, *not* tuned on the test set. These are an R12-specific adaptation
# of the installed native layerwise FIS, not claims about the paper defaults.
R12_FIS = {
    "max_interaction_degree": 2,
    "theta_percentile_mode": True,
    "theta_thresholds": {1: 0.50, 2: 0.10},
    "tau_thresholds": {2: 0.50},
    "interaction_support_n": 32,
    "explainer_batch_size": 512,
    "max_frontier": 1000,
}
R12_TRAIN = {
    "hidden_sizes": [128, 64, 32],
    "small_sizes": [0, 16, 12, 8, 1],
    "batch_size": 32,
    "max_epochs": 300,
    "learning_rate": 0.005,
    "l1_coefficient": 5e-5,
    "early_stop_patience": 20,
    "gradient_clip_norm": 1.0,
}


def get_spec(scenario: str, realization: int, protocol=None):
    protocol = protocol or SyntheticBenchmarkProtocol()
    found = [s for s in build_synthetic_schedule(protocol)
             if s.scenario == scenario.upper() and s.realization_number == int(realization)]
    if len(found) != 1:
        raise ValueError("Unknown synthetic scenario/realization")
    return found[0]


def native_api_guard():
    version = importlib.metadata.version("sian")
    if version != "0.9.5":
        raise RuntimeError(f"R12 requires sian==0.9.5, found {version}")
    from sian.models.models import SIAN
    from sian.interpret.explainer2 import JamArchipelago, unmasked_FID_Hyperparameters
    from sian.interpret.basic_wrappers import MixedModelWrapperTorch
    from sian.fis.feature_interaction_selection import (
        layerwise_FIS_Hyperparameters,
        layerwise_feature_interaction_selection_algorithm,
    )
    required = {"sizes", "indices", "dnn_on_or_off", "small_sizes"}
    if not required.issubset(inspect.signature(SIAN).parameters):
        raise RuntimeError("Unexpected native SIAN constructor API")
    return (SIAN, JamArchipelago, unmasked_FID_Hyperparameters,
            MixedModelWrapperTorch, layerwise_FIS_Hyperparameters,
            layerwise_feature_interaction_selection_algorithm)


def create_splits(spec, protocol):
    ds = generate_synthetic_scenario(spec.scenario, seed=spec.dataset_seed)
    X = ds.X.reset_index(drop=True)
    y = ds.y.to_numpy()
    all_ids = np.arange(len(y), dtype=np.int64)
    dev_ids, test_ids = train_test_split(
        all_ids, test_size=protocol.outer_test_fraction,
        random_state=spec.outer_split_seed, stratify=y,
    )
    train_local, valid_local = train_test_split(
        np.arange(len(dev_ids)),
        test_size=protocol.final_validation_fraction_within_development,
        random_state=spec.final_split_seed, stratify=y[dev_ids],
    )
    train_ids, valid_ids = dev_ids[train_local], dev_ids[valid_local]
    assert not (set(train_ids) & set(valid_ids) or set(train_ids) & set(test_ids)
                or set(valid_ids) & set(test_ids))
    assert len(train_ids) + len(valid_ids) + len(test_ids) == len(y)
    return {
        "X_train": X.iloc[train_ids].to_numpy(dtype=np.float32),
        "y_train": y[train_ids].astype(np.float32),
        "X_valid": X.iloc[valid_ids].to_numpy(dtype=np.float32),
        "y_valid": y[valid_ids].astype(np.float32),
        "X_test": X.iloc[test_ids].to_numpy(dtype=np.float32),
        "y_test": y[test_ids].astype(np.float32),
        "feature_names": tuple(str(f) for f in X.columns),
        "true_pairs": normalize_pairs(ds.true_interactions),
        "indices": {"train": train_ids, "validation": valid_ids, "test": test_ids},
    }


def hash_ids(indices):
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def pair_structure(indices, feature_names):
    """Explicitly discard higher-order terms; require unique in-range indices."""
    p = len(feature_names)
    selected_singletons, selected_pairs = set(), set()
    for group in indices:
        if not isinstance(group, (list, tuple)):
            raise ValueError(f"Unexpected FIS interaction type: {type(group)}")
        ids = tuple(sorted(int(q) for q in group))
        if len(ids) != len(set(ids)) or any(q < 0 or q >= p for q in ids):
            raise ValueError(f"Invalid FIS indices: {group}")
        if len(ids) == 1:
            selected_singletons.add(ids)
        elif len(ids) == 2:
            selected_pairs.add(ids)
        elif len(ids) != 0:
            raise ValueError(f"FIS emitted order >2: {group}")
    native_terms = tuple(sorted(selected_singletons | selected_pairs))
    names = tuple(sorted(tuple(sorted((feature_names[j], feature_names[k])))
                         for j, k in selected_pairs))
    return native_terms, names


def build_explainer(model, device, n_features, api):
    _, JamArchipelago, hypers_class, wrapper, _, _ = api
    groups = {"D": n_features, "D0": n_features, "C": 2}
    for i in range(n_features):
        groups[i] = [i]
    hypers = hypers_class(
        "triangle_marginal", "binary_classification", "old_arch_inter_score",
        ["inc_inter_score", "rem_inter_score", "old_arch_inter_score"],
        device, groups,
    )
    e = JamArchipelago(wrapper(model, device), hypers)
    e.batch_size = R12_FIS["explainer_batch_size"]
    e.verbose = False
    return e


def native_pairwise_fis(mlp, scorer_support, names, device, api):
    _, _, _, _, hyper_class, native_select = api
    explainer = build_explainer(mlp, device, len(names), api)
    config = hyper_class(
        MAX_K=2,
        tau_thresholds=dict(R12_FIS["tau_thresholds"]),
        theta_thresholds=dict(R12_FIS["theta_thresholds"]),
        explainer=explainer,
        AGG_K=R12_FIS["interaction_support_n"],
        theta_percentile_mode=True,
        FRONTIER_MAX_SIZE=R12_FIS["max_frontier"],
    )
    raw_interactions, details = native_select(
        config, scorer_support, R12_FIS["interaction_support_n"]
    )
    terms, pairs = pair_structure(raw_interactions, names)
    if not terms:
        raise RuntimeError("Native SIAN FIS returned no selectable terms; no fallback was invented")
    return terms, pairs, details


def _logits(model, batch):
    dnn, additive, _ = model(batch)
    return (dnn + additive).reshape(-1)


def fit_final_sian(terms, X_train, y_train, X_valid, y_valid,
                   *, seed, device, native_sian_class):
    random.seed(int(seed)); np.random.seed(int(seed)); torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    p = X_train.shape[1]
    model = native_sian_class(
        sizes=[p, *R12_TRAIN["hidden_sizes"], 1],
        indices=list(terms),
        dnn_on_or_off=False,
        small_sizes=[p, 16, 12, 8, 1],
    ).to(device)
    optim = torch.optim.Adagrad(model.parameters(), lr=R12_TRAIN["learning_rate"])
    loss_func = torch.nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(int(seed))
    loader = DataLoader(TensorDataset(torch.from_numpy(X_train),
                                      torch.from_numpy(y_train)),
                        batch_size=R12_TRAIN["batch_size"], shuffle=True,
                        generator=generator, num_workers=0)
    vX = torch.from_numpy(X_valid).to(device)
    vy = torch.from_numpy(y_valid).to(device)
    best_state = copy.deepcopy(model.state_dict())
    best_auc, best_epoch, n_bad = -float("inf"), 0, 0
    epochs_ran = 0
    for epoch in range(1, R12_TRAIN["max_epochs"] + 1):
        epochs_ran = epoch
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad(set_to_none=True)
            loss = loss_func(_logits(model, xb), yb)
            if R12_TRAIN["l1_coefficient"]:
                penalty = sum(w.abs().sum() for w in model.parameters())
                loss = loss + R12_TRAIN["l1_coefficient"] * penalty
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), R12_TRAIN["gradient_clip_norm"])
            optim.step()
        model.eval()
        with torch.no_grad():
            val_logits = _logits(model, vX)
            val_proba = torch.sigmoid(val_logits).cpu().numpy()
        score = float(roc_auc_score(y_valid, val_proba))
        if score > best_auc + 1e-8:
            best_auc, best_epoch, n_bad = score, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            n_bad += 1
            if n_bad >= R12_TRAIN["early_stop_patience"]:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, {"best_epoch": best_epoch, "epochs_ran": epochs_ran,
                   "best_validation_auroc": best_auc}


def predict_test(model, X_test, y_test, device):
    # *Only* called after all FIS and early stopping has finished.
    with torch.no_grad():
        proba = torch.sigmoid(_logits(model, torch.from_numpy(X_test).to(device)))
        proba = proba.detach().cpu().numpy().reshape(-1)
    if not np.isfinite(proba).all() or len(proba) != len(y_test):
        raise RuntimeError("Non-finite or mis-sized test predictions")
    return proba, {"test_auroc": float(roc_auc_score(y_test, proba)),
                   "test_auprc": float(average_precision_score(y_test, proba))}


def run_pilot(scenario, realization, output_root):
    api = native_api_guard()
    protocol = SyntheticBenchmarkProtocol()
    spec = get_spec(scenario, realization, protocol)
    final_dir = Path(output_root) / spec.scenario / f"realization_{realization:02d}"
    staging = Path(str(final_dir) + ".inprogress")
    if final_dir.exists() or staging.exists():
        raise FileExistsError(f"Refusing overwrite of completed/in-progress run: {final_dir}")
    staging.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    device = get_device()
    created = datetime.now(timezone.utc).isoformat()
    whole_start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    try:
        with PeakRSSMonitor() as memory:
            data = create_splits(spec, protocol)
            stages = {}
            start = time.perf_counter()
            trained = train_shared_sian_mlp(
                data["X_train"], data["y_train"],
                data["X_valid"], data["y_valid"],
                seed=spec.discovery_base_seed, device=device,
                hidden_sizes=tuple(R12_TRAIN["hidden_sizes"]),
                batch_size=R12_TRAIN["batch_size"],
                max_epochs=R12_TRAIN["max_epochs"],
                learning_rate=R12_TRAIN["learning_rate"],
                l1_coefficient=R12_TRAIN["l1_coefficient"],
                patience=R12_TRAIN["early_stop_patience"],
            )
            stages["reference_mlp_seconds"] = time.perf_counter() - start
            start = time.perf_counter()
            support_size = min(R12_FIS["interaction_support_n"], len(data["X_valid"]))
            if support_size != R12_FIS["interaction_support_n"]:
                raise RuntimeError("Not enough validation samples for locked support size")
            rnd = np.random.default_rng(spec.discovery_base_seed)
            support_indices = rnd.choice(len(data["X_valid"]), support_size, replace=False)
            support = trained.scaler.transform(data["X_valid"])[support_indices].astype(np.float32)
            terms, selected_pairs, details = native_pairwise_fis(
                trained.model, support, data["feature_names"], device, api
            )
            stages["native_fis_seconds"] = time.perf_counter() - start
            # Ground truth has not participated in training/threshold decisions.
            recovery = asdict(evaluate_pair_set_recovery(selected_pairs, data["true_pairs"]))
            start = time.perf_counter()
            trX = trained.scaler.transform(data["X_train"]).astype(np.float32)
            vaX = trained.scaler.transform(data["X_valid"]).astype(np.float32)
            teX = trained.scaler.transform(data["X_test"]).astype(np.float32)
            final_model, training_stats = fit_final_sian(
                terms, trX, data["y_train"], vaX, data["y_valid"],
                seed=spec.final_model_seed, device=device,
                native_sian_class=api[0],
            )
            stages["native_sian_fit_seconds"] = time.perf_counter() - start
            probs, predictive = predict_test(final_model, teX, data["y_test"], device)
            pd.DataFrame({"y_test": data["y_test"].astype(int),
                          "sian_probability": probs}).to_csv(staging / "test_predictions.csv", index=False)
            pd.DataFrame([{"feature_a": a, "feature_b": b,
                           "is_true": (a, b) in set(data["true_pairs"])}
                          for a, b in selected_pairs],
                         columns=["feature_a", "feature_b", "is_true"]).to_csv(
                             staging / "selected_pairs.csv", index=False)
            torch.save({"state_dict": final_model.state_dict(),
                        "terms": terms, "training_settings": R12_TRAIN}, staging / "sian_state_dict.pt")
            if device.type == "cuda":
                torch.cuda.synchronize()
                cuda_peak_mib = float(torch.cuda.max_memory_allocated(device) / 1048576)
            else:
                cuda_peak_mib = None
        stages["whole_pipeline_seconds"] = time.perf_counter() - whole_start
        result = {
            "analysis": "r12_native_pairwise_constrained_sian_end_to_end",
            "interpretation": "Descriptive pilot only, NOT full unrestricted-order SIAN and NOT AG-NAM head-to-head proof",
            "status": "complete", "created_utc": created,
            "scenario": spec.scenario, "realization": spec.realization_number,
            "sian_version": importlib.metadata.version("sian"),
            "torch_version": torch.__version__, "device": str(device),
            "data_seed": spec.dataset_seed,
            "split_seeds": {"outer": spec.outer_split_seed, "final": spec.final_split_seed},
            "training_seeds": {"reference": spec.discovery_base_seed,
                               "final": spec.final_model_seed},
            "data_counts": {k: len(data["indices"][k]) for k in data["indices"]},
            "split_index_sha256": {k: hash_ids(v) for k,v in data["indices"].items()},
            "feature_names": data["feature_names"],
            "native_fis_settings": R12_FIS,
            "training_settings": R12_TRAIN,
            "selected_native_terms_indexed": terms,
            "selected_pair_names": selected_pairs,
            "n_selected_pairs": len(selected_pairs),
            "interaction_recovery": recovery,
            "test_predictive_metrics": predictive,
            "reference_mlp_best_epoch": trained.best_epoch,
            "sian_train": training_stats,
            "stage_elapsed_seconds": stages,
            "peak_process_rss_mib_sampled": memory.peak_mib,
            "peak_torch_cuda_allocated_mib": cuda_peak_mib,
            "test_data_policy": "never used for FIS, model selection, scaling or early stopping",
            "cautions": [
                "SIAN FIS thresholds are fixed R12 adaptation; not paper-original parameters.",
                "Pairwise restriction MAX_K=2 changes original higher-order SIAN scope.",
                "A selected-set F1 is not a full-universe ranking AUPRC.",
                "One realization provides no evidence of statistical superiority.",
                "GPU allocated peak and sampled RSS have different definitions.",
            ],
        }
        (staging / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False,
                                                            default=lambda x: float(x)), encoding="utf-8")
        staging.rename(final_dir)
        print("=== R12 NATIVE SIAN PAIRWISE PILOT COMPLETED ===")
        print("OUTPUT:", final_dir)
        print("SELECTED PAIRS:", len(selected_pairs), "| Recovery F1:", recovery["f1"])
        print("TEST AUROC:", predictive["test_auroc"])
        print("WALL CLOCK (s):", stages["whole_pipeline_seconds"])
        print("NOTE: descriptive pilot; no established full-SIAN superiority claim")
    except BaseException as e:
        (staging / "failure.json").write_text(json.dumps({"status": "failed", "error": repr(e)}, indent=2),
                                               encoding="utf-8")
        print("PILOT FAILED; data left in:", staging)
        raise


def parse_args(argv=None):
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--scenario", choices=("S1", "S2", "S3", "S4"), default="S1")
    a.add_argument("--realization", type=int, default=1)
    a.add_argument("--output-root", default="results/revision/r12_native_sian_pairwise")
    a.add_argument("--execute", action="store_true", help="Required to permit model training")
    return a.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    spec = get_spec(args.scenario, args.realization)
    print("=== R12 NATIVE SIAN (PAIRWISE-CONSTRAINED) ===")
    print("SPEC:", spec.scenario, spec.realization_number,
          "| train/valid/test = 60/20/20 | seed =", spec.dataset_seed)
    api = native_api_guard()
    print("SIAN 0.9.5 native API: OK")
    print("FIS:", R12_FIS)
    if not args.execute:
        print("CHECK ONLY: No training, outputs or modifications. Add --execute for S1 pilot.")
        return 0
    run_pilot(args.scenario, args.realization, args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
