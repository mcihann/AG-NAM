"""Post-submission external-proposal x common-ISR synthetic pilot.

This file is deliberately separate from the locked primary benchmark and
from the locked, one-run external recovery baseline results.

Stage 'proposals' generates 5 split-matched external rankings and authentic,
shared MainEffectNAM residuals.  Stage 'isr' applies the existing AG-NAM ISR
to the resulting selection-stable pairs; ground truth is used ONLY to score
the results, never to construct rankings/residual targets or accept pairs.

NID and SIAN-FID share one fitted MLP when requested in the same invocation.
Residual targets are common across proposal methods by design: the resulting
pipelines must be described as 'external proposal + shared AG-NAM ISR hybrid',
not as independent implementations of the external models.
"""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import time
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from agnam.benchmarking.synthetic_protocol import (
    SyntheticBenchmarkProtocol,
    build_synthetic_schedule,
    candidate_count_from_feature_count,
    compute_false_positive_reduction,
    evaluate_pair_set_recovery,
)
from agnam.benchmarking.synthetic_runner import (
    SyntheticModelConfig,
    _build_discovery_configs,
    _build_isr_config,
    generate_synthetic_scenario,
    normalize_pairs,
)
from agnam.revision.external_isr_adapter import (
    ExternalRunInput,
    build_external_isr_view,
)
from agnam.revision.external_recovery_baselines import (
    deterministic_ranking,
    evaluate_pair_ranking,
    friedman_h_pair_scores,
    nid_pair_scores_from_model,
    sian_fid_pair_scores,
    train_shared_sian_mlp,
)
from agnam.training.isr import evaluate_isr_for_discovery
from agnam.training.residual_targets import (
    generate_train_validation_residual_targets,
)
from agnam.utils.reproducibility import get_device

METHODS = (
    "friedman_h_gbdt",
    "nid_shared_mlp",
    "sian_fid_shared_mlp",
)
DEFAULT_OUTPUT_ROOT = Path("results/revision/external_isr_hybrid/core_synthetic")


def _spec_for(scenario: str, realization: int, protocol):
    matches = [
        item for item in build_synthetic_schedule(protocol)
        if item.scenario == scenario.upper()
        and item.realization_number == realization
    ]
    if len(matches) != 1:
        raise ValueError(f"Invalid synthetic realization: {scenario} R{realization}")
    return matches[0]


def _prepare_development(spec, protocol):
    dataset = generate_synthetic_scenario(spec.scenario, seed=spec.dataset_seed)
    X = dataset.X.reset_index(drop=True)
    y = dataset.y.to_numpy(dtype=np.int64)
    all_indices = np.arange(len(X))
    outer_development_indices, outer_test_indices = train_test_split(
        all_indices,
        test_size=protocol.outer_test_fraction,
        random_state=spec.outer_split_seed,
        stratify=y,
    )
    X_dev = X.iloc[outer_development_indices].reset_index(drop=True)
    y_dev = y[outer_development_indices]
    # Outer-test indices and ground-truth interactions are NEVER returned to
    # the proposal / residual / ISR training functions.
    return X_dev, y_dev, dataset, outer_test_indices


def _discovery_split(y: np.ndarray, seed: int):
    indices = np.arange(len(y))
    dev_indices, scoring_indices = train_test_split(
        indices, test_size=0.20, random_state=seed, stratify=y,
    )
    train_indices, validation_indices = train_test_split(
        dev_indices, test_size=0.25, random_state=seed + 1,
        stratify=y[dev_indices],
    )
    return (
        np.asarray(train_indices, dtype=np.int64),
        np.asarray(validation_indices, dtype=np.int64),
        np.asarray(scoring_indices, dtype=np.int64),
    )


def _fingerprint(*arrays: np.ndarray) -> str:
    digest = sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _save_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _save_residual_cache(path: Path, *, arrays: dict, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)
    _save_json(path.with_suffix(".json"), metadata)


def _load_residual_cache(path: Path, *, spec, run_index: int,
                         split: tuple, n_development_rows: int,
                         expected_feature_names: tuple[str, ...]):
    meta_path = path.with_suffix(".json")
    if not path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"Incomplete residual cache: {path}")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    seed = spec.discovery_base_seed + run_index
    expected = {
        "scenario": spec.scenario,
        "realization_number": spec.realization_number,
        "dataset_seed": spec.dataset_seed,
        "outer_split_seed": spec.outer_split_seed,
        "run_index": run_index,
        "run_seed": seed,
        "n_development_rows": n_development_rows,
        "feature_names": list(expected_feature_names),
        "split_sha256": _fingerprint(*split),
        "residual_source": "AG-NAM MainEffectNAM crossfit + train-only validation fit",
    }
    for name, expected_value in expected.items():
        if metadata.get(name) != expected_value:
            raise ValueError(f"Residual cache metadata mismatch: {path}: {name}")
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "train_indices", "validation_indices", "scoring_indices",
            "train_residuals", "validation_residuals",
        }
        if set(payload.files) != required:
            raise ValueError(f"Unexpected residual cache fields: {path}")
        arrays = {key: np.array(payload[key], copy=True) for key in required}
    for name, expected_array in zip(
        ("train_indices", "validation_indices", "scoring_indices"), split
    ):
        if not np.array_equal(arrays[name], expected_array):
            raise ValueError(f"Residual cache split mismatch: {path}: {name}")
    if len(arrays["train_residuals"]) != len(split[0]) or len(
        arrays["validation_residuals"]
    ) != len(split[1]):
        raise ValueError(f"Residual length mismatch in {path}")
    if any(not np.isfinite(arrays[name]).all() for name in (
        "train_residuals", "validation_residuals"
    )):
        raise ValueError(f"Nonfinite cached residuals: {path}")
    return arrays, metadata


def _get_residual_cache(
    path: Path, *, spec, protocol, model_config, X_dev, y_dev,
    split: tuple, run_index: int, device, force: bool,
):
    if not force and path.exists() and path.with_suffix(".json").exists():
        return _load_residual_cache(
            path, spec=spec, run_index=run_index, split=split,
            n_development_rows=len(X_dev),
            expected_feature_names=tuple(str(c) for c in X_dev.columns),
        )
    if not force and (path.exists() or path.with_suffix(".json").exists()):
        raise RuntimeError(f"Partial residual cache; repair explicitly: {path}")
    train_idx, val_idx, score_idx = split
    run_seed = spec.discovery_base_seed + run_index
    nam_config, _ = _build_discovery_configs(spec, model_config)
    nam_config = replace(nam_config, seed=run_seed)
    start = time.perf_counter()
    targets = generate_train_validation_residual_targets(
        X_train=X_dev.iloc[train_idx].reset_index(drop=True),
        y_train=y_dev[train_idx],
        X_validation=X_dev.iloc[val_idx].reset_index(drop=True),
        y_validation=y_dev[val_idx],
        crossfit_n_splits=protocol.residual_crossfit_folds,
        main_early_stop_fraction=0.20,
        hidden_dim=model_config.nam_hidden_dim,
        depth=model_config.nam_depth,
        dropout=model_config.nam_dropout,
        categorical_embedding_dim=model_config.categorical_embedding_dim,
        training_config=nam_config,
        seed=run_seed,
        device=device,
    )
    elapsed = time.perf_counter() - start
    arrays = {
        "train_indices": train_idx.copy(),
        "validation_indices": val_idx.copy(),
        "scoring_indices": score_idx.copy(),
        "train_residuals": np.asarray(targets.train_residuals, dtype=np.float64),
        "validation_residuals": np.asarray(
            targets.validation_residuals, dtype=np.float64
        ),
    }
    metadata = {
        "scenario": spec.scenario,
        "realization_number": spec.realization_number,
        "dataset_seed": spec.dataset_seed,
        "outer_split_seed": spec.outer_split_seed,
        "run_index": run_index,
        "run_seed": run_seed,
        "n_development_rows": len(X_dev),
        "feature_names": [str(c) for c in X_dev.columns],
        "split_sha256": _fingerprint(*split),
        "residual_source": "AG-NAM MainEffectNAM crossfit + train-only validation fit",
        "residual_wall_seconds": float(elapsed),
    }
    _save_residual_cache(path, arrays=arrays, metadata=metadata)
    return arrays, metadata


def _proposal_paths(root: Path, method: str, run_index: int):
    folder = root / "proposal_runs" / method
    stem = f"run_{run_index + 1:02d}"
    return folder / f"{stem}_scores.csv", folder / f"{stem}_manifest.json"


def _load_proposal_scores(path: Path, manifest_path: Path, *, method: str,
                          spec, seed: int, features: tuple, split_sha256: str,
                          candidate_k: int):
    if not path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Incomplete proposal cache: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "run_index": seed - spec.discovery_base_seed,
        "method": method,
        "scenario": spec.scenario,
        "realization_number": spec.realization_number,
        "dataset_seed": spec.dataset_seed,
        "run_seed": seed,
        "candidate_k": candidate_k,
        "split_sha256": split_sha256,
        "feature_names": list(features),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Proposal cache metadata mismatch: {path}: {key}")
    frame = pd.read_csv(path)
    if list(frame.columns) != ["feature_a", "feature_b", "score", "rank"]:
        raise ValueError(f"Unexpected proposal columns in {path}")
    if frame.isna().any().any():
        raise ValueError(f"Missing proposal values in {path}")
    if frame["rank"].tolist() != list(range(1, len(frame) + 1)):
        raise ValueError(f"Non-contiguous ranks in {path}")
    pairs = [tuple(sorted((str(a), str(b)))) for a, b in zip(
        frame.feature_a, frame.feature_b
    )]
    if len(set(pairs)) != len(pairs):
        raise ValueError(f"Duplicate proposal pairs in {path}")
    scores = {pair: float(score) for pair, score in zip(pairs, frame.score)}
    if deterministic_ranking(scores) != tuple(
        (pair[0], pair[1], float(score)) for pair, score in zip(pairs, frame.score)
    ):
        raise ValueError(f"Non-deterministic saved ranking order: {path}")
    if not np.isfinite(np.asarray(frame.score, dtype=np.float64)).all():
        raise ValueError(f"Nonfinite scores in {path}")
    return scores, manifest


def _save_proposal_scores(root, method, run_index, scores, *, spec, seed,
                          features, split_sha256, candidate_k, timings):
    score_path, manifest_path = _proposal_paths(root, method, run_index)
    ranking = deterministic_ranking(scores)
    frame = pd.DataFrame(
        [(a, b, float(value), rank) for rank, (a, b, value) in enumerate(
            ranking, start=1
        )], columns=["feature_a", "feature_b", "score", "rank"],
    )
    manifest = {
        "method": method,
        "scenario": spec.scenario,
        "realization_number": spec.realization_number,
        "dataset_seed": spec.dataset_seed,
        "run_seed": seed,
        "run_index": run_index,
        "candidate_k": candidate_k,
        "split_sha256": split_sha256,
        "feature_names": list(features),
        **timings,
    }
    _save_frame(score_path, frame)
    _save_json(manifest_path, manifest)


def _method_scores(method: str, *, X_train, y_train, X_val, y_val,
                   X_scoring, features, seed, device, shared_training):
    if method == "friedman_h_gbdt":
        scores, fit_sec, score_sec, *_ = friedman_h_pair_scores(
            X_fit=X_train.to_numpy(),
            y_fit=y_train,
            X_score=X_scoring.to_numpy(),
            feature_names=features,
            seed=seed,
            grid_resolution=10,
        )
        return scores, {
            "fit_seconds": float(fit_sec),
            "scoring_seconds": float(score_sec),
            "standalone_seconds": float(fit_sec + score_sec),
        }
    if shared_training is None:
        raise RuntimeError("Missing shared MLP training result")
    if method == "nid_shared_mlp":
        started = time.perf_counter()
        scores = nid_pair_scores_from_model(shared_training.model, features)
        score_sec = time.perf_counter() - started
        return scores, {
            "fit_seconds": float(shared_training.training_seconds),
            "scoring_seconds": float(score_sec),
            "standalone_seconds": float(shared_training.training_seconds + score_sec),
        }
    if method == "sian_fid_shared_mlp":
        scores, score_sec, *_ = sian_fid_pair_scores(
            model=shared_training.model,
            scaler=shared_training.scaler,
            X_score=X_scoring.to_numpy(),
            feature_names=features,
            seed=seed,
            device=device,
            agg_k=32,
            explainer_batch_size=512,
        )
        return scores, {
            "fit_seconds": float(shared_training.training_seconds),
            "scoring_seconds": float(score_sec),
            "standalone_seconds": float(shared_training.training_seconds + score_sec),
        }
    raise ValueError(f"Unknown method: {method}")


def _run_proposals(*, root, methods, spec, protocol, model_config,
                   X_dev, y_dev, device, candidate_k, force):
    features = tuple(str(column) for column in X_dev.columns)
    for run_index in range(protocol.n_discovery_runs):
        seed = spec.discovery_base_seed + run_index
        split = _discovery_split(y_dev, seed)
        split_sha = _fingerprint(*split)
        residual_path = root / "residual_runs" / f"run_{run_index + 1:02d}.npz"
        arrays, residual_meta = _get_residual_cache(
            residual_path, spec=spec, protocol=protocol,
            model_config=model_config, X_dev=X_dev, y_dev=y_dev,
            split=split, run_index=run_index, device=device, force=force,
        )
        pending = []
        for method in methods:
            path, manifest_path = _proposal_paths(root, method, run_index)
            if not force and path.exists() and manifest_path.exists():
                _load_proposal_scores(
                    path, manifest_path, method=method, spec=spec, seed=seed,
                    features=features, split_sha256=split_sha,
                    candidate_k=candidate_k,
                )
                print(f"CACHED: {method} run {run_index + 1}/5")
            elif not force and (path.exists() or manifest_path.exists()):
                raise RuntimeError(f"Partial proposal cache: {path}")
            else:
                pending.append(method)
        if not pending:
            continue
        train_idx, val_idx, scoring_idx = split
        X_train = X_dev.iloc[train_idx].reset_index(drop=True)
        X_val = X_dev.iloc[val_idx].reset_index(drop=True)
        X_scoring = X_dev.iloc[scoring_idx].reset_index(drop=True)
        shared = None
        if any(method in pending for method in METHODS[1:]):
            shared = train_shared_sian_mlp(
                X_train.to_numpy(), y_dev[train_idx],
                X_val.to_numpy(), y_dev[val_idx],
                seed=seed, device=device,
            )
        for method in pending:
            scores, timings = _method_scores(
                method, X_train=X_train, y_train=y_dev[train_idx],
                X_val=X_val, y_val=y_dev[val_idx],
                X_scoring=X_scoring, features=features,
                seed=seed, device=device, shared_training=shared,
            )
            _save_proposal_scores(
                root, method, run_index, scores, spec=spec, seed=seed,
                features=features, split_sha256=split_sha,
                candidate_k=candidate_k, timings=timings,
            )
            # This read validates the full persisted pair-score universe later
            # through build_external_isr_view, but catch early cache damage too.
            path, manifest_path = _proposal_paths(root, method, run_index)
            _load_proposal_scores(
                path, manifest_path, method=method, spec=spec, seed=seed,
                features=features, split_sha256=split_sha,
                candidate_k=candidate_k,
            )
            print(f"SAVED: {method} run {run_index + 1}/5")
        del arrays, residual_meta, shared


def _load_view(root, method, spec, protocol, X_dev, y_dev, candidate_k):
    features = tuple(str(column) for column in X_dev.columns)
    inputs = []
    timings = []
    residual_times = []
    for run_index in range(protocol.n_discovery_runs):
        seed = spec.discovery_base_seed + run_index
        split = _discovery_split(y_dev, seed)
        residual_path = root / "residual_runs" / f"run_{run_index + 1:02d}.npz"
        arrays, residual_meta = _load_residual_cache(
            residual_path, spec=spec, run_index=run_index, split=split,
            n_development_rows=len(X_dev), expected_feature_names=features,
        )
        score_path, score_manifest_path = _proposal_paths(root, method, run_index)
        scores, score_meta = _load_proposal_scores(
            score_path, score_manifest_path, method=method, spec=spec,
            seed=seed, features=features,
            split_sha256=_fingerprint(*split), candidate_k=candidate_k,
        )
        inputs.append(ExternalRunInput(
            run_index=run_index,
            seed=seed,
            train_indices=arrays["train_indices"],
            validation_indices=arrays["validation_indices"],
            scoring_indices=arrays["scoring_indices"],
            train_residuals=arrays["train_residuals"],
            validation_residuals=arrays["validation_residuals"],
            pair_scores=scores,
        ))
        timings.append(float(score_meta["standalone_seconds"]))
        residual_times.append(float(residual_meta["residual_wall_seconds"]))
    view = build_external_isr_view(
        runs=inputs,
        feature_names=features,
        n_development_rows=len(X_dev),
        k=candidate_k,
        selection_threshold=protocol.selection_threshold,
    )
    return view, np.asarray(timings), np.asarray(residual_times)


def _run_isr(*, root, methods, spec, protocol, model_config,
             X_dev, y_dev, dataset, device, candidate_k, force):
    for method in methods:
        summary_path = root / f"{method}_summary.csv"
        isr_pairs_path = root / f"{method}_isr_pairs.csv"
        manifest_path = root / f"{method}_manifest.json"
        outputs = (summary_path, isr_pairs_path, manifest_path)
        if not force and any(path.exists() for path in outputs):
            if all(path.exists() for path in outputs):
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
                if existing.get("scenario") != spec.scenario or (
                    existing.get("realization_number") != spec.realization_number
                ) or existing.get("method") != method:
                    raise ValueError(f"Existing ISR manifest mismatch: {method}")
                print(f"CACHED ISR: {method}")
                continue
            raise RuntimeError(f"Partial ISR output: {method}; use --force")
        view, durations, residual_durations = _load_view(
            root, method, spec, protocol, X_dev, y_dev, candidate_k,
        )
        print(f"ISR: {method} | selection candidates="
              f"{len(view.selection.accepted_interactions)}", flush=True)
        isr_started = time.perf_counter()
        result = evaluate_isr_for_discovery(
            X=X_dev,
            discovery=view,
            config=_build_isr_config(protocol, model_config),
            device=device,
            verbose=True,
        )
        isr_seconds = time.perf_counter() - isr_started
        selected = normalize_pairs(
            (item.feature_j, item.feature_k)
            for item in view.selection.accepted_interactions
        )
        retained = normalize_pairs(
            (item.feature_j, item.feature_k)
            for item in result.retained_interactions
        )
        if not set(retained).issubset(set(selected)):
            raise RuntimeError("ISR added a previously unselected pair")
        # Ground truth is introduced only AFTER both selection and ISR finish.
        truth = normalize_pairs(dataset.true_interactions)
        before = evaluate_pair_set_recovery(selected, truth)
        after = evaluate_pair_set_recovery(retained, truth)
        fp_reduction = compute_false_positive_reduction(before, after)
        per_run_ap = [evaluate_pair_ranking(
            feature_names=tuple(str(c) for c in X_dev.columns),
            pair_scores={
                (r.feature_j, r.feature_k): r.score for r in run.ranking
            },
            true_pairs=truth,
            candidate_k=candidate_k,
        ).interaction_auprc for run in view.runs]
        table = pd.DataFrame([{
            "feature_a": item.feature_j,
            "feature_b": item.feature_k,
            "selection_count": item.selection_count,
            "selection_probability": item.selection_probability,
            "isr": item.isr,
            "isr_retained": item.final_accepted,
            "is_true": (item.feature_j, item.feature_k) in set(truth),
            "n_surface_runs": len(item.surface_runs),
        } for item in result.interactions], columns=[
            "feature_a", "feature_b", "selection_count",
            "selection_probability", "isr", "isr_retained",
            "is_true", "n_surface_runs",
        ])
        metrics = {
            "scenario": spec.scenario,
            "realization_number": spec.realization_number,
            "dataset_seed": spec.dataset_seed,
            "method": method,
            "n_runs": len(view.runs),
            "candidate_k": candidate_k,
            "selection_threshold": protocol.selection_threshold,
            "isr_threshold": protocol.isr_threshold,
            "n_true": len(truth),
            "interaction_ap_mean": float(np.mean(per_run_ap)),
            "interaction_ap_std": float(np.std(per_run_ap, ddof=1)),
            "selection_jaccard": view.selection.mean_pairwise_jaccard,
            "n_selection": len(selected),
            "n_isr": len(retained),
            "selection_tp": before.true_positive,
            "selection_fp": before.false_positive,
            "selection_fn": before.false_negative,
            "selection_precision": before.precision,
            "selection_recall": before.recall,
            "selection_f1": before.f1,
            "isr_tp": after.true_positive,
            "isr_fp": after.false_positive,
            "isr_fn": after.false_negative,
            "isr_precision": after.precision,
            "isr_recall": after.recall,
            "isr_f1": after.f1,
            "fp_reduction": fp_reduction,
            "proposal_standalone_seconds_sum": float(np.sum(durations)),
            "residual_seconds_sum_shared": float(np.sum(residual_durations)),
            "isr_wall_seconds": float(isr_seconds),
            "interpretation": "external proposal + shared AG-NAM ISR hybrid",
        }
        _save_frame(isr_pairs_path, table)
        _save_frame(summary_path, pd.DataFrame([metrics]))
        _save_json(root / f"{method}_manifest.json", {
            "analysis_status": "reviewer_requested_post_submission",
            "original_primary_results_modified": False,
            "method": method,
            "scenario": spec.scenario,
            "realization_number": spec.realization_number,
            "dataset_seed": spec.dataset_seed,
            "discovery_seeds": [
                spec.discovery_base_seed + i for i in range(protocol.n_discovery_runs)
            ],
            "n_runs": protocol.n_discovery_runs,
            "selection_threshold": protocol.selection_threshold,
            "isr_threshold": protocol.isr_threshold,
            "residual_source": "shared AG-NAM MainEffectNAM",
            "validation_name": "AG-NAM ISR",
            "protocol_not_equal_budget_to_original_single_run_baselines": True,
            "no_formal_hypothesis_tests": True,
        })
        print(f"SAVED ISR: {method}: {summary_path}")


def run_external_isr_hybrid_realization(*, scenario: str,
                                        realization_number: int,
                                        methods: Sequence[str] = METHODS,
                                        stage: str = "proposals",
                                        output_root: str | Path = DEFAULT_OUTPUT_ROOT,
                                        force: bool = False):
    """Run/resume the reviewer-only pilot without modifying primary outputs."""
    if stage not in {"proposals", "isr", "both"}:
        raise ValueError("stage must be proposals, isr, or both")
    methods = tuple(methods)
    if not methods or len(methods) != len(set(methods)) or set(methods) - set(METHODS):
        raise ValueError(f"Invalid or duplicate method list: {methods}")
    protocol = SyntheticBenchmarkProtocol()
    model_config = SyntheticModelConfig()
    spec = _spec_for(scenario, int(realization_number), protocol)
    root = Path(output_root) / spec.scenario / f"realization_{spec.realization_number:02d}"
    root.mkdir(parents=True, exist_ok=True)
    X_dev, y_dev, dataset, _ = _prepare_development(spec, protocol)
    features = tuple(str(c) for c in X_dev.columns)
    candidate_k = candidate_count_from_feature_count(len(features), protocol)
    device = get_device()
    print(f"Reviewer hybrid: {spec.scenario} R{spec.realization_number:02d}"
          f" | methods={methods} | stage={stage} | device={device}", flush=True)
    if stage in {"proposals", "both"}:
        _run_proposals(
            root=root, methods=methods, spec=spec, protocol=protocol,
            model_config=model_config, X_dev=X_dev, y_dev=y_dev,
            device=device, candidate_k=candidate_k, force=force,
        )
        for method in methods:
            view, _, _ = _load_view(
                root, method, spec, protocol, X_dev, y_dev, candidate_k,
            )
            print(f"VERIFIED: {method}, {len(view.runs)} runs, "
                  f"{len(view.selection.accepted_interactions)} stable pairs",
                  flush=True)
    if stage in {"isr", "both"}:
        _run_isr(
            root=root, methods=methods, spec=spec, protocol=protocol,
            model_config=model_config, X_dev=X_dev, y_dev=y_dev,
            dataset=dataset, device=device, candidate_k=candidate_k,
            force=force,
        )
    return root
