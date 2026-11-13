from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil
from typing import Iterable

import numpy as np

from agnam.benchmarking.synthetic_protocol import (
    evaluate_pair_set_recovery,
    compute_false_positive_reduction,
)
from agnam.benchmarking.synthetic_runner import normalize_pairs
from agnam.interpretation.selection_reproducibility import (
    summarize_selection_reproducibility,
)
from agnam.training.discovery import (
    InteractionDiscoveryRun,
    ReproducibleDiscoveryResult,
)


THRESHOLD_VALUES = (0.40, 0.60, 0.80)
RUN_VALUES = (3, 5, 10)
SUPPORT_VALUES = (128, 256, 512)


@dataclass(frozen=True)
class SensitivityConfiguration:
    config_id: str
    family: str
    selection_threshold: float
    isr_threshold: float
    n_discovery_runs: int
    k_mode: str
    reference_size: int
    is_reference: bool = False


def candidate_k_levels(
    *,
    n_features: int,
    locked_k: int,
) -> dict[str, int]:
    if n_features < 2:
        raise ValueError("n_features must be at least 2.")
    p = n_features * (n_features - 1) // 2
    if not 1 <= locked_k <= p:
        raise ValueError("locked_k must lie in [1, P].")

    return {
        "half": max(1, int(ceil(locked_k / 2.0))),
        "locked": int(locked_k),
        "double": min(int(p), int(2 * locked_k)),
    }


def build_sensitivity_configurations() -> tuple[SensitivityConfiguration, ...]:
    rows: list[SensitivityConfiguration] = []

    for selection_threshold in THRESHOLD_VALUES:
        for isr_threshold in THRESHOLD_VALUES:
            rows.append(
                SensitivityConfiguration(
                    config_id=(
                        f"threshold_s{int(selection_threshold * 100):02d}"
                        f"_i{int(isr_threshold * 100):02d}"
                    ),
                    family="threshold_grid",
                    selection_threshold=selection_threshold,
                    isr_threshold=isr_threshold,
                    n_discovery_runs=5,
                    k_mode="locked",
                    reference_size=512,
                    is_reference=(
                        np.isclose(selection_threshold, 0.60)
                        and np.isclose(isr_threshold, 0.60)
                    ),
                )
            )

    for n_runs in RUN_VALUES:
        rows.append(
            SensitivityConfiguration(
                config_id=f"runs_{n_runs:02d}",
                family="run_count",
                selection_threshold=0.60,
                isr_threshold=0.60,
                n_discovery_runs=n_runs,
                k_mode="locked",
                reference_size=512,
                is_reference=(n_runs == 5),
            )
        )

    for k_mode in ("half", "locked", "double"):
        rows.append(
            SensitivityConfiguration(
                config_id=f"k_{k_mode}",
                family="candidate_k",
                selection_threshold=0.60,
                isr_threshold=0.60,
                n_discovery_runs=5,
                k_mode=k_mode,
                reference_size=512,
                is_reference=(k_mode == "locked"),
            )
        )

    for reference_size in SUPPORT_VALUES:
        rows.append(
            SensitivityConfiguration(
                config_id=f"support_{reference_size}",
                family="reference_support",
                selection_threshold=0.60,
                isr_threshold=0.60,
                n_discovery_runs=5,
                k_mode="locked",
                reference_size=reference_size,
                is_reference=(reference_size == 512),
            )
        )

    if len(rows) != 18:
        raise RuntimeError("Sensitivity matrix must contain exactly 18 rows.")

    return tuple(rows)


def make_discovery_view(
    base_runs: tuple[InteractionDiscoveryRun, ...],
    *,
    n_runs: int,
    k: int,
    selection_threshold: float,
) -> ReproducibleDiscoveryResult:
    if n_runs < 1 or n_runs > len(base_runs):
        raise ValueError("n_runs is outside the available discovery-run range.")

    selected_runs = tuple(
        replace(run, k=int(k))
        for run in base_runs[:n_runs]
    )

    rankings = tuple(
        run.score_result.ranked_interactions
        for run in selected_runs
    )

    selection = summarize_selection_reproducibility(
        rankings=rankings,
        k=int(k),
        threshold=float(selection_threshold),
    )

    return ReproducibleDiscoveryResult(
        runs=selected_runs,
        selection=selection,
    )


def selection_pairs_from_discovery(
    discovery: ReproducibleDiscoveryResult,
) -> tuple[tuple[str, str], ...]:
    return normalize_pairs(
        (
            (item.feature_j, item.feature_k)
            for item in discovery.selection.accepted_interactions
        )
    )


def isr_pairs_at_threshold(
    isr_result,
    *,
    threshold: float,
) -> tuple[tuple[str, str], ...]:
    return normalize_pairs(
        (
            (item.feature_j, item.feature_k)
            for item in isr_result.interactions
            if float(item.isr) >= float(threshold)
        )
    )


def recovery_metrics(
    *,
    selection_pairs,
    isr_pairs,
    true_pairs,
) -> dict[str, float | int]:
    selection = evaluate_pair_set_recovery(
        predicted_pairs=selection_pairs,
        true_pairs=true_pairs,
    )
    isr = evaluate_pair_set_recovery(
        predicted_pairs=isr_pairs,
        true_pairs=true_pairs,
    )

    fp_reduction = compute_false_positive_reduction(
        selection_recovery=selection,
        isr_recovery=isr,
    )

    return {
        "n_selection_retained": len(selection_pairs),
        "selection_tp": selection.true_positive,
        "selection_fp": selection.false_positive,
        "selection_fn": selection.false_negative,
        "selection_precision": selection.precision,
        "selection_recall": selection.recall,
        "selection_f1": selection.f1,
        "n_isr_retained": len(isr_pairs),
        "isr_tp": isr.true_positive,
        "isr_fp": isr.false_positive,
        "isr_fn": isr.false_negative,
        "isr_precision": isr.precision,
        "isr_recall": isr.recall,
        "isr_f1": isr.f1,
        "fp_reduction_vs_selection": fp_reduction,
    }
