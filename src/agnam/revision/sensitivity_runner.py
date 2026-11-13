from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
import json

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
    generate_synthetic_scenario,
    normalize_pairs,
)
from agnam.training.discovery import run_reproducible_interaction_discovery
from agnam.training.isr import evaluate_isr_for_discovery
from agnam.utils.reproducibility import get_device

from agnam.revision.sensitivity import (
    build_sensitivity_configurations,
    candidate_k_levels,
    isr_pairs_at_threshold,
    make_discovery_view,
    recovery_metrics,
    selection_pairs_from_discovery,
)


def _outer_development_data(
    spec: SyntheticRealizationSpec,
    protocol: SyntheticBenchmarkProtocol,
):
    dataset = generate_synthetic_scenario(
        spec.scenario,
        seed=spec.dataset_seed,
    )
    X = dataset.X
    y = dataset.y.to_numpy()

    indices = np.arange(len(X))
    dev_idx, _ = train_test_split(
        indices,
        test_size=protocol.outer_test_fraction,
        random_state=spec.outer_split_seed,
        stratify=y,
    )

    X_dev = X.iloc[dev_idx].reset_index(drop=True)
    y_dev = y[dev_idx]
    true_pairs = normalize_pairs(dataset.true_interactions)

    return X_dev, y_dev, true_pairs


def run_synthetic_sensitivity_realization(
    spec: SyntheticRealizationSpec,
    *,
    output_root: str | Path = "results/revision/sensitivity/synthetic",
    protocol: SyntheticBenchmarkProtocol | None = None,
    model_config: SyntheticModelConfig | None = None,
    device=None,
    force: bool = False,
    verbose: bool = True,
) -> dict[str, Path]:
    if protocol is None:
        protocol = SyntheticBenchmarkProtocol()
    if model_config is None:
        model_config = SyntheticModelConfig()
    if device is None:
        device = get_device()

    scenario_dir = Path(output_root) / spec.scenario
    scenario_dir.mkdir(parents=True, exist_ok=True)

    stem = f"realization_{spec.realization_number:02d}"
    summary_path = scenario_dir / f"{stem}_summary.csv"
    manifest_path = scenario_dir / f"{stem}_manifest.json"

    if manifest_path.exists() and summary_path.exists() and not force:
        if verbose:
            print(
                f"SKIP {spec.scenario} realization "
                f"{spec.realization_number}: manifest already exists"
            )
        return {
            "summary": summary_path,
            "manifest": manifest_path,
        }

    X_dev, y_dev, true_pairs = _outer_development_data(
        spec,
        protocol,
    )

    nam_config, proposer_config = _build_discovery_configs(
        spec=spec,
        config=model_config,
    )

    # Train the maximal prespecified run stream once. The first 3/5 runs
    # are deterministic prefixes of this same stream.
    base_discovery = run_reproducible_interaction_discovery(
        X=X_dev,
        y=y_dev,
        n_runs=10,
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
        nam_config=nam_config,
        proposer_config=proposer_config,
        device=device,
    )

    locked_k = candidate_count_from_feature_count(
        X_dev.shape[1],
        protocol,
    )
    k_levels = candidate_k_levels(
        n_features=X_dev.shape[1],
        locked_k=locked_k,
    )

    configs = build_sensitivity_configurations()

    discovery_cache = {}
    isr_cache = {}

    def get_view(config):
        k = k_levels[config.k_mode]
        key = (
            config.n_discovery_runs,
            k,
            float(config.selection_threshold),
        )
        if key not in discovery_cache:
            discovery_cache[key] = make_discovery_view(
                base_discovery.runs,
                n_runs=config.n_discovery_runs,
                k=k,
                selection_threshold=config.selection_threshold,
            )
        return discovery_cache[key], k

    def get_isr(config, discovery, k):
        # The actual numeric ISR score is independent of the acceptance
        # threshold. Use 0.40 internally so one deterministic surface fit
        # can support all threshold rows; final acceptance is re-thresholded
        # below from item.isr.
        key = (
            config.n_discovery_runs,
            k,
            float(config.selection_threshold),
            int(config.reference_size),
        )
        if key not in isr_cache:
            base_isr_config = _build_isr_config(
                protocol=protocol,
                config=model_config,
            )
            eval_config = replace(
                base_isr_config,
                reference_size=int(config.reference_size),
                isr_threshold=0.40,
            )
            isr_cache[key] = evaluate_isr_for_discovery(
                X=X_dev,
                discovery=discovery,
                config=eval_config,
                device=device,
                verbose=False,
            )
        return isr_cache[key]

    rows = []
    for index, config in enumerate(configs, start=1):
        if verbose:
            print(
                f"[{index:02d}/{len(configs)}] "
                f"{spec.scenario} R{spec.realization_number:02d} "
                f"{config.config_id}"
            )

        discovery, k = get_view(config)
        selection_pairs = selection_pairs_from_discovery(discovery)
        isr_result = get_isr(config, discovery, k)
        isr_pairs = isr_pairs_at_threshold(
            isr_result,
            threshold=config.isr_threshold,
        )

        metrics = recovery_metrics(
            selection_pairs=selection_pairs,
            isr_pairs=isr_pairs,
            true_pairs=true_pairs,
        )

        rows.append(
            {
                "scenario": spec.scenario,
                "realization_index": spec.realization_index,
                "realization_number": spec.realization_number,
                "dataset_seed": spec.dataset_seed,
                "outer_split_seed": spec.outer_split_seed,
                "discovery_base_seed": spec.discovery_base_seed,
                "config_id": config.config_id,
                "family": config.family,
                "is_reference": config.is_reference,
                "selection_threshold": config.selection_threshold,
                "isr_threshold": config.isr_threshold,
                "n_discovery_runs": config.n_discovery_runs,
                "k_mode": config.k_mode,
                "candidate_k": k,
                "reference_size": config.reference_size,
                "n_features": X_dev.shape[1],
                "n_true_interactions": len(true_pairs),
                **metrics,
            }
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(summary_path, index=False)

    manifest = {
        "analysis_status": "reviewer_requested_post_submission",
        "analysis_name": "parameter_sensitivity",
        "scenario": spec.scenario,
        "realization_number": spec.realization_number,
        "dataset_seed": spec.dataset_seed,
        "outer_split_seed": spec.outer_split_seed,
        "discovery_base_seed": spec.discovery_base_seed,
        "discovery_seeds_trained": [
            spec.discovery_base_seed + i
            for i in range(10)
        ],
        "locked_reference": {
            "selection_threshold": 0.60,
            "isr_threshold": 0.60,
            "n_discovery_runs": 5,
            "candidate_k": locked_k,
            "reference_size": 512,
            "reference_seed": protocol.reference_seed,
        },
        "k_levels": k_levels,
        "families": {
            "threshold_grid": {
                "selection": [0.40, 0.60, 0.80],
                "isr": [0.40, 0.60, 0.80],
            },
            "run_count": [3, 5, 10],
            "candidate_k": ["half", "locked", "double"],
            "reference_support": [128, 256, 512],
        },
        "n_output_configurations": int(len(frame)),
        "formal_hypothesis_tests": False,
        "locked_primary_results_modified": False,
        "computation_reuse": {
            "max_discovery_runs_trained_once": 10,
            "run_prefixes": [3, 5, 10],
            "k_reuses_full_rankings": True,
            "isr_threshold_applied_post_surface_agreement": True,
            "pairwise_evaluation_cache_within_realization": True,
        },
        "outputs": {
            "summary": str(summary_path),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    if verbose:
        print("\nSaved revision outputs:")
        print(summary_path)
        print(manifest_path)

    return {
        "summary": summary_path,
        "manifest": manifest_path,
    }
