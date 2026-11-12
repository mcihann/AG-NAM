from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from agnam.benchmarking.synthetic_protocol import (
    SyntheticBenchmarkProtocol,
    build_synthetic_schedule,
)
from agnam.benchmarking.synthetic_runner import (
    SyntheticModelConfig,
    _build_discovery_configs,
    _build_isr_config,
    generate_synthetic_scenario,
    normalize_pairs,
)
from agnam.revision.isr_novelty import (
    build_isr_variant_sets,
    evaluate_isr_novelty_for_discovery,
    evaluate_isr_variant_recovery,
)
from agnam.training.discovery import run_reproducible_interaction_discovery
from agnam.utils.reproducibility import get_device


def _find_spec(scenario: str, realization_number: int):
    protocol = SyntheticBenchmarkProtocol()
    scenario = scenario.upper()
    matches = [
        spec
        for spec in build_synthetic_schedule(protocol)
        if spec.scenario == scenario and spec.realization_number == realization_number
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one locked spec for {scenario} realization {realization_number}."
        )
    return protocol, matches[0]


def run_synthetic_isr_novelty(
    *,
    scenario: str,
    realization_number: int,
    results_root: str | Path = "results/revision/isr_novelty/synthetic",
    device: torch.device | None = None,
    verbose: bool = True,
) -> dict[str, Path]:
    protocol, spec = _find_spec(scenario, realization_number)
    model_config = SyntheticModelConfig()
    if device is None:
        device = get_device()

    dataset = generate_synthetic_scenario(spec.scenario, seed=spec.dataset_seed)
    X = dataset.X
    y = dataset.y.to_numpy()
    true_pairs = normalize_pairs(dataset.true_interactions)

    all_indices = np.arange(len(X))
    outer_dev_idx, _ = train_test_split(
        all_indices,
        test_size=protocol.outer_test_fraction,
        random_state=spec.outer_split_seed,
        stratify=y,
    )
    X_outer_dev = X.iloc[outer_dev_idx].reset_index(drop=True)
    y_outer_dev = y[outer_dev_idx]

    nam_config, proposer_config = _build_discovery_configs(
        spec=spec,
        config=model_config,
    )
    discovery = run_reproducible_interaction_discovery(
        X=X_outer_dev,
        y=y_outer_dev,
        n_runs=protocol.n_discovery_runs,
        base_seed=spec.discovery_base_seed,
        selection_threshold=protocol.selection_threshold,
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

    isr_config = _build_isr_config(protocol=protocol, config=model_config)
    novelty = evaluate_isr_novelty_for_discovery(
        X=X_outer_dev,
        discovery=discovery,
        config=isr_config,
        device=device,
        verbose=verbose,
    )
    variants = build_isr_variant_sets(
        discovery,
        novelty,
        selection_threshold=protocol.selection_threshold,
        strict_selection_threshold=0.80,
        agreement_threshold=protocol.isr_threshold,
    )
    recoveries = evaluate_isr_variant_recovery(
        variants,
        true_pairs=true_pairs,
    )

    base = Path(results_root) / spec.scenario
    base.mkdir(parents=True, exist_ok=True)
    stem = f"realization_{spec.realization_number:02d}"
    pair_path = base / f"{stem}_pair_agreements.csv"
    summary_path = base / f"{stem}_summary.csv"
    manifest_path = base / f"{stem}_manifest.json"

    pair_frame = pd.DataFrame(
        [
            {
                **asdict(item),
                "raw_pass_060": item.raw_agreement >= protocol.isr_threshold,
                "purified_pass_060": item.purified_agreement >= protocol.isr_threshold,
            }
            for item in novelty.pair_agreements
        ]
    )
    pair_frame.to_csv(pair_path, index=False)

    selection_fp = next(
        item.recovery.false_positive
        for item in recoveries
        if item.variant == "selection_stable_060"
    )
    summary_rows = []
    for item in recoveries:
        rec = item.recovery
        fp_reduction = (
            float((selection_fp - rec.false_positive) / selection_fp)
            if selection_fp > 0
            else float("nan")
        )
        summary_rows.append(
            {
                "scenario": spec.scenario,
                "realization_index": spec.realization_index,
                "realization_number": spec.realization_number,
                "dataset_seed": spec.dataset_seed,
                "outer_split_seed": spec.outer_split_seed,
                "discovery_base_seed": spec.discovery_base_seed,
                "variant": item.variant,
                "n_retained": len(item.pairs),
                "true_positive": rec.true_positive,
                "false_positive": rec.false_positive,
                "false_negative": rec.false_negative,
                "precision": rec.precision,
                "recall": rec.recall,
                "f1": rec.f1,
                "false_negative_rate": item.false_negative_rate,
                "fp_reduction_vs_selection": fp_reduction,
            }
        )
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    manifest = {
        "analysis_status": "reviewer_requested_post_submission",
        "analysis_name": "isr_novelty_ablation",
        "scenario": spec.scenario,
        "realization_number": spec.realization_number,
        "dataset_seed": spec.dataset_seed,
        "outer_split_seed": spec.outer_split_seed,
        "discovery_base_seed": spec.discovery_base_seed,
        "discovery_seeds": list(spec.discovery_seeds),
        "selection_threshold": protocol.selection_threshold,
        "strict_selection_threshold": 0.80,
        "agreement_threshold": protocol.isr_threshold,
        "reference_size": protocol.reference_size,
        "reference_seed": protocol.reference_seed,
        "variants": list(variants.keys()),
        "count_matched_rule": (
            "Within the original selection-stable set, retain the first N pairs "
            "under the locked selection-probability/mean-rank ordering, where N "
            "equals the purified-ISR retained count."
        ),
        "raw_surface_definition": (
            "Pearson agreement across raw diagonal h_jk(x_j_i, x_k_i) vectors "
            "on the common locked reference support."
        ),
        "purified_surface_definition": (
            "Pearson agreement across functional-ANOVA-purified diagonal vectors "
            "on the same common reference support."
        ),
        "formal_hypothesis_tests": False,
        "locked_primary_results_modified": False,
        "outputs": {
            "pair_agreements": str(pair_path),
            "summary": str(summary_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "pair_agreements": pair_path,
        "summary": summary_path,
        "manifest": manifest_path,
    }
