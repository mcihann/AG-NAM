from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from agnam.benchmarking.synthetic_protocol import (
    SyntheticBenchmarkProtocol,
    build_synthetic_schedule,
    candidate_count_from_feature_count,
)
from agnam.benchmarking.synthetic_runner import (
    generate_synthetic_scenario,
    normalize_pairs,
)
from agnam.revision.external_recovery_baselines import (
    deterministic_ranking,
    evaluate_pair_ranking,
    friedman_h_pair_scores,
    nid_pair_scores_from_model,
    sian_fid_pair_scores,
    train_shared_sian_mlp,
)
from agnam.utils.reproducibility import get_device


METHODS = (
    "friedman_h_gbdt",
    "nid_shared_mlp",
    "sian_fid_shared_mlp",
)


def _find_spec(
    scenario: str,
    realization_number: int,
    protocol: SyntheticBenchmarkProtocol,
):
    scenario = scenario.upper()
    for spec in build_synthetic_schedule(protocol):
        if (
            spec.scenario == scenario
            and spec.realization_number == int(realization_number)
        ):
            return spec
    raise ValueError(
        f"Unknown scenario-realization: "
        f"{scenario}-{realization_number}"
    )


def _prepare_data(spec, protocol):
    dataset = generate_synthetic_scenario(
        spec.scenario,
        seed=spec.dataset_seed,
    )

    X = dataset.X.reset_index(drop=True)
    y = dataset.y.to_numpy()
    true_pairs = normalize_pairs(dataset.true_interactions)

    all_idx = np.arange(len(X))
    outer_dev_idx, _ = train_test_split(
        all_idx,
        test_size=protocol.outer_test_fraction,
        random_state=spec.outer_split_seed,
        stratify=y,
    )

    X_outer = X.iloc[outer_dev_idx].reset_index(drop=True)
    y_outer = y[outer_dev_idx]

    local_idx = np.arange(len(X_outer))
    fit_idx, score_idx = train_test_split(
        local_idx,
        test_size=(
            protocol
            .final_validation_fraction_within_development
        ),
        random_state=spec.final_split_seed,
        stratify=y_outer,
    )

    X_fit = X_outer.iloc[fit_idx].reset_index(drop=True)
    y_fit = y_outer[fit_idx]

    X_score = X_outer.iloc[score_idx].reset_index(drop=True)
    y_score = y_outer[score_idx]

    return (
        X_fit,
        y_fit,
        X_score,
        y_score,
        tuple(str(c) for c in X.columns),
        true_pairs,
    )


def _metric_row(
    *,
    method,
    scenario,
    realization_number,
    dataset_seed,
    candidate_k,
    metrics,
    fit_seconds,
    scoring_seconds,
    standalone_seconds,
    rss_start_mib,
    rss_end_mib,
    rss_peak_mib,
    cuda_peak_allocated_mib,
    shared_mlp_best_epoch=None,
    shared_mlp_best_val_auroc=None,
):
    return {
        "scenario": scenario,
        "realization_number": realization_number,
        "dataset_seed": dataset_seed,
        "method": method,
        "candidate_k": candidate_k,
        **asdict(metrics),
        "fit_seconds": float(fit_seconds),
        "scoring_seconds": float(scoring_seconds),
        "standalone_seconds": float(standalone_seconds),
        "rss_start_mib": float(rss_start_mib),
        "rss_end_mib": float(rss_end_mib),
        "rss_peak_mib": float(rss_peak_mib),
        "cuda_peak_allocated_mib": float(
            cuda_peak_allocated_mib
        ),
        "shared_mlp_best_epoch": (
            None
            if shared_mlp_best_epoch is None
            else int(shared_mlp_best_epoch)
        ),
        "shared_mlp_best_val_auroc": (
            None
            if shared_mlp_best_val_auroc is None
            else float(shared_mlp_best_val_auroc)
        ),
    }


def run_external_recovery_realization(
    *,
    scenario: str,
    realization_number: int,
    output_root: str | Path = (
        "results/revision/external_interaction_baselines/core_synthetic"
    ),
    methods=METHODS,
    force: bool = False,
):
    protocol = SyntheticBenchmarkProtocol()
    spec = _find_spec(
        scenario,
        realization_number,
        protocol,
    )

    output_dir = (
        Path(output_root)
        / spec.scenario
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = f"realization_{spec.realization_number:02d}"
    summary_path = output_dir / f"{stem}_summary.csv"
    pairs_path = output_dir / f"{stem}_pair_scores.csv"
    manifest_path = output_dir / f"{stem}_manifest.json"

    if (
        not force
        and summary_path.exists()
        and pairs_path.exists()
        and manifest_path.exists()
    ):
        print(
            f"Skipping existing external recovery result: "
            f"{spec.scenario} realization "
            f"{spec.realization_number}"
        )
        return summary_path, pairs_path, manifest_path

    requested = tuple(methods)
    unknown = sorted(set(requested).difference(METHODS))
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")

    (
        X_fit,
        y_fit,
        X_score,
        y_score,
        feature_names,
        true_pairs,
    ) = _prepare_data(spec, protocol)

    candidate_k = candidate_count_from_feature_count(
        len(feature_names),
        protocol,
    )

    device = get_device()
    rows = []
    pair_rows = []

    shared_training = None

    # ----------------------------------------------------------
    # Friedman H
    # ----------------------------------------------------------
    if "friedman_h_gbdt" in requested:
        (
            scores,
            elapsed,
            rss_start,
            rss_end,
            rss_peak,
        ) = friedman_h_pair_scores(
            X_fit=X_fit.to_numpy(),
            y_fit=y_fit,
            X_score=X_score.to_numpy(),
            feature_names=feature_names,
            seed=spec.discovery_base_seed,
            grid_resolution=10,
        )

        metrics = evaluate_pair_ranking(
            feature_names=feature_names,
            pair_scores=scores,
            true_pairs=true_pairs,
            candidate_k=candidate_k,
        )

        rows.append(
            _metric_row(
                method="friedman_h_gbdt",
                scenario=spec.scenario,
                realization_number=spec.realization_number,
                dataset_seed=spec.dataset_seed,
                candidate_k=candidate_k,
                metrics=metrics,
                fit_seconds=elapsed,
                scoring_seconds=0.0,
                standalone_seconds=elapsed,
                rss_start_mib=rss_start,
                rss_end_mib=rss_end,
                rss_peak_mib=rss_peak,
                cuda_peak_allocated_mib=0.0,
            )
        )

        truth = set(true_pairs)
        for rank, (a, b, score) in enumerate(
            deterministic_ranking(scores),
            start=1,
        ):
            pair_rows.append(
                {
                    "scenario": spec.scenario,
                    "realization_number": spec.realization_number,
                    "method": "friedman_h_gbdt",
                    "feature_a": a,
                    "feature_b": b,
                    "score": score,
                    "rank": rank,
                    "is_true": (a, b) in truth,
                }
            )

    # ----------------------------------------------------------
    # Shared neural backbone
    # ----------------------------------------------------------
    if (
        "nid_shared_mlp" in requested
        or "sian_fid_shared_mlp" in requested
    ):
        shared_training = train_shared_sian_mlp(
            X_fit.to_numpy(),
            y_fit,
            X_score.to_numpy(),
            y_score,
            seed=spec.discovery_base_seed,
            device=device,
        )

    # ----------------------------------------------------------
    # NID
    # ----------------------------------------------------------
    if "nid_shared_mlp" in requested:
        start = time.perf_counter()
        scores = nid_pair_scores_from_model(
            shared_training.model,
            feature_names,
        )
        scoring_seconds = time.perf_counter() - start

        metrics = evaluate_pair_ranking(
            feature_names=feature_names,
            pair_scores=scores,
            true_pairs=true_pairs,
            candidate_k=candidate_k,
        )

        rows.append(
            _metric_row(
                method="nid_shared_mlp",
                scenario=spec.scenario,
                realization_number=spec.realization_number,
                dataset_seed=spec.dataset_seed,
                candidate_k=candidate_k,
                metrics=metrics,
                fit_seconds=shared_training.training_seconds,
                scoring_seconds=scoring_seconds,
                standalone_seconds=(
                    shared_training.training_seconds
                    + scoring_seconds
                ),
                rss_start_mib=shared_training.rss_start_mib,
                rss_end_mib=shared_training.rss_end_mib,
                rss_peak_mib=shared_training.rss_peak_mib,
                cuda_peak_allocated_mib=(
                    shared_training.cuda_peak_allocated_mib
                ),
                shared_mlp_best_epoch=shared_training.best_epoch,
                shared_mlp_best_val_auroc=(
                    shared_training.best_val_auroc
                ),
            )
        )

        truth = set(true_pairs)
        for rank, (a, b, score) in enumerate(
            deterministic_ranking(scores),
            start=1,
        ):
            pair_rows.append(
                {
                    "scenario": spec.scenario,
                    "realization_number": spec.realization_number,
                    "method": "nid_shared_mlp",
                    "feature_a": a,
                    "feature_b": b,
                    "score": score,
                    "rank": rank,
                    "is_true": (a, b) in truth,
                }
            )

    # ----------------------------------------------------------
    # SIAN FID
    # ----------------------------------------------------------
    if "sian_fid_shared_mlp" in requested:
        (
            scores,
            scoring_seconds,
            rss_start,
            rss_end,
            rss_peak,
            cuda_peak,
        ) = sian_fid_pair_scores(
            model=shared_training.model,
            scaler=shared_training.scaler,
            X_score=X_score.to_numpy(),
            feature_names=feature_names,
            seed=spec.discovery_base_seed,
            device=device,
            agg_k=32,
            explainer_batch_size=512,
        )

        metrics = evaluate_pair_ranking(
            feature_names=feature_names,
            pair_scores=scores,
            true_pairs=true_pairs,
            candidate_k=candidate_k,
        )

        rows.append(
            _metric_row(
                method="sian_fid_shared_mlp",
                scenario=spec.scenario,
                realization_number=spec.realization_number,
                dataset_seed=spec.dataset_seed,
                candidate_k=candidate_k,
                metrics=metrics,
                fit_seconds=shared_training.training_seconds,
                scoring_seconds=scoring_seconds,
                standalone_seconds=(
                    shared_training.training_seconds
                    + scoring_seconds
                ),
                rss_start_mib=min(
                    shared_training.rss_start_mib,
                    rss_start,
                ),
                rss_end_mib=rss_end,
                rss_peak_mib=max(
                    shared_training.rss_peak_mib,
                    rss_peak,
                ),
                cuda_peak_allocated_mib=max(
                    shared_training.cuda_peak_allocated_mib,
                    cuda_peak,
                ),
                shared_mlp_best_epoch=shared_training.best_epoch,
                shared_mlp_best_val_auroc=(
                    shared_training.best_val_auroc
                ),
            )
        )

        truth = set(true_pairs)
        for rank, (a, b, score) in enumerate(
            deterministic_ranking(scores),
            start=1,
        ):
            pair_rows.append(
                {
                    "scenario": spec.scenario,
                    "realization_number": spec.realization_number,
                    "method": "sian_fid_shared_mlp",
                    "feature_a": a,
                    "feature_b": b,
                    "score": score,
                    "rank": rank,
                    "is_true": (a, b) in truth,
                }
            )

    summary = pd.DataFrame(rows)
    pair_frame = pd.DataFrame(pair_rows)

    summary.to_csv(summary_path, index=False)
    pair_frame.to_csv(pairs_path, index=False)

    manifest = {
        "analysis_status": "reviewer_requested_post_submission",
        "analysis_name": "core_external_interaction_recovery",
        "scenario": spec.scenario,
        "realization_number": spec.realization_number,
        "dataset_seed": spec.dataset_seed,
        "outer_split_seed": spec.outer_split_seed,
        "final_split_seed": spec.final_split_seed,
        "baseline_seed": spec.discovery_base_seed,
        "methods": list(requested),
        "candidate_k": int(candidate_k),
        "n_features": len(feature_names),
        "n_true_interactions": len(true_pairs),
        "shared_neural_backbone": {
            "implementation": "sian.models.models.MLP",
            "hidden_sizes": [128, 64, 32],
            "optimizer": "Adagrad",
            "learning_rate": 0.005,
            "batch_size": 32,
            "max_epochs": 300,
            "l1_coefficient": 5e-5,
            "gradient_clip_norm": 1.0,
            "early_stopping_patience": 20,
        },
        "sian_fid": {
            "package_version": "0.9.5",
            "masking_style": "triangle_marginal",
            "score_type": "old_arch_inter_score",
            "agg_k": 32,
            "explainer_batch_size": 512,
        },
        "friedman_h": {
            "estimator": "GradientBoostingClassifier",
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 3,
            "grid_resolution": 10,
            "percentiles": [0.05, 0.95],
            "response_method": "decision_function",
            "partial_dependence_method": "recursion",
        },
        "formal_hypothesis_tests": False,
        "locked_primary_results_modified": False,
        "outputs": {
            "summary": str(summary_path),
            "pair_scores": str(pairs_path),
        },
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("Saved external recovery outputs:")
    print(summary_path)
    print(pairs_path)
    print(manifest_path)

    return summary_path, pairs_path, manifest_path
