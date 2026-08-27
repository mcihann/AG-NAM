from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from agnam.interpretation.interaction_metrics import (
    canonical_pair,
)


Pair = tuple[str, str]

SYNTHETIC_SCENARIOS = (
    "S1",
    "S2",
    "S3",
    "S4",
)


@dataclass(frozen=True)
class SyntheticBenchmarkProtocol:
    """
    Locked primary protocol for the synthetic AG-NAM benchmark.

    The benchmark contains:

        4 scenarios
        x 20 independent realizations
        = 80 primary synthetic experiments

    This object contains only protocol-level constants.
    """

    scenarios: tuple[str, ...] = SYNTHETIC_SCENARIOS

    n_realizations: int = 20
    n_discovery_runs: int = 5

    outer_test_fraction: float = 0.20
    final_validation_fraction_within_development: float = 0.25

    residual_crossfit_folds: int = 5

    selection_threshold: float = 0.60
    isr_threshold: float = 0.60

    reference_size: int = 512
    reference_seed: int = 2026

    candidate_fraction: float = 0.10
    candidate_minimum: int = 5
    candidate_maximum: int = 20

    dataset_seed_base: int = 10_000
    outer_split_seed_base: int = 20_000
    final_split_seed_base: int = 30_000
    discovery_seed_base: int = 40_000
    random_pair_seed_base: int = 80_000
    final_model_seed_base: int = 90_000

    def __post_init__(
        self,
    ) -> None:
        if self.n_realizations < 1:
            raise ValueError(
                "n_realizations must be at least 1."
            )

        if self.n_discovery_runs < 1:
            raise ValueError(
                "n_discovery_runs must be at least 1."
            )

        if not 0.0 < self.outer_test_fraction < 1.0:
            raise ValueError(
                "outer_test_fraction must lie in (0, 1)."
            )

        if not (
            0.0
            < self.final_validation_fraction_within_development
            < 1.0
        ):
            raise ValueError(
                "final validation fraction must lie in (0, 1)."
            )

        if self.residual_crossfit_folds < 2:
            raise ValueError(
                "residual_crossfit_folds must be at least 2."
            )

        if not 0.0 < self.selection_threshold <= 1.0:
            raise ValueError(
                "selection_threshold must lie in (0, 1]."
            )

        if not 0.0 < self.isr_threshold <= 1.0:
            raise ValueError(
                "isr_threshold must lie in (0, 1]."
            )

        if self.reference_size < 1:
            raise ValueError(
                "reference_size must be positive."
            )

        if not 0.0 < self.candidate_fraction <= 1.0:
            raise ValueError(
                "candidate_fraction must lie in (0, 1]."
            )

        if self.candidate_minimum < 1:
            raise ValueError(
                "candidate_minimum must be positive."
            )

        if (
            self.candidate_maximum
            < self.candidate_minimum
        ):
            raise ValueError(
                "candidate_maximum must be >= candidate_minimum."
            )


@dataclass(frozen=True)
class SyntheticRealizationSpec:
    """
    Complete deterministic seed specification for one
    scenario-realization combination.
    """

    scenario: str
    realization_index: int

    dataset_seed: int
    outer_split_seed: int
    final_split_seed: int

    discovery_base_seed: int

    random_pair_seed: int
    final_model_seed: int

    n_discovery_runs: int

    @property
    def realization_number(
        self,
    ) -> int:
        """
        Human-facing realization number: 1..20.
        """
        return (
            self.realization_index
            + 1
        )

    @property
    def discovery_seeds(
        self,
    ) -> tuple[int, ...]:
        return tuple(
            self.discovery_base_seed
            + run_index
            for run_index in range(
                self.n_discovery_runs
            )
        )


@dataclass(frozen=True)
class PairSetRecovery:
    """
    Recovery statistics for a predicted interaction set.
    """

    true_positive: int
    false_positive: int
    false_negative: int

    precision: float
    recall: float
    f1: float


@dataclass
class SyntheticBenchmarkRecord:
    """
    One row of the final synthetic benchmark table.

    Fields belonging to comparison models that have not yet been
    executed may remain NaN until later benchmark phases.
    """

    scenario: str
    realization_index: int

    dataset_seed: int
    outer_split_seed: int
    final_split_seed: int
    discovery_base_seed: int
    random_pair_seed: int
    final_model_seed: int

    n_samples: int
    n_features: int
    positive_fraction: float
    n_true_interactions: int

    candidate_k: int

    mean_pairwise_jaccard: float = np.nan

    mean_interaction_auprc: float = np.nan
    std_interaction_auprc: float = np.nan

    n_selection_stable: int = 0

    selection_precision: float = np.nan
    selection_recall: float = np.nan
    selection_f1: float = np.nan

    n_isr_retained: int = 0

    isr_precision: float = np.nan
    isr_recall: float = np.nan
    isr_f1: float = np.nan

    false_positive_reduction: float = np.nan

    main_auroc: float = np.nan
    main_auprc: float = np.nan
    main_balanced_accuracy: float = np.nan
    main_f1: float = np.nan

    agnam_auroc: float = np.nan
    agnam_auprc: float = np.nan
    agnam_balanced_accuracy: float = np.nan
    agnam_f1: float = np.nan

    delta_auroc: float = np.nan
    delta_auprc: float = np.nan
    delta_balanced_accuracy: float = np.nan
    delta_f1: float = np.nan

    oracle_auroc: float = np.nan
    oracle_auprc: float = np.nan

    random_pair_auroc: float = np.nan
    random_pair_auprc: float = np.nan

    no_isr_auroc: float = np.nan
    no_isr_auprc: float = np.nan

    single_run_auroc: float = np.nan
    single_run_auprc: float = np.nan

    max_decomposition_error: float = np.nan

    runtime_seconds: float = np.nan


def candidate_count_from_feature_count(
    n_features: int,
    protocol: SyntheticBenchmarkProtocol | None = None,
) -> int:
    """
    Locked candidate-count rule:

        P = p(p - 1) / 2

        K =
            clip(
                ceil(candidate_fraction * P),
                candidate_minimum,
                candidate_maximum
            )

    K is additionally capped by the number of available pairs.
    """
    if protocol is None:
        protocol = (
            SyntheticBenchmarkProtocol()
        )

    if n_features < 2:
        return 0

    n_pairs = (
        n_features
        * (
            n_features
            - 1
        )
        // 2
    )

    proposed = int(
        np.ceil(
            protocol.candidate_fraction
            * n_pairs
        )
    )

    proposed = max(
        protocol.candidate_minimum,
        proposed,
    )

    proposed = min(
        protocol.candidate_maximum,
        proposed,
    )

    return min(
        n_pairs,
        proposed,
    )


def build_synthetic_schedule(
    protocol: SyntheticBenchmarkProtocol | None = None,
) -> tuple[
    SyntheticRealizationSpec,
    ...
]:
    """
    Build the complete deterministic synthetic benchmark schedule.

    Seed blocks are intentionally separated by purpose.

    Scenario offsets prevent S1-S4 from sharing the same seed stream.

    Discovery seed blocks reserve 10 integer seeds for each
    realization. The primary protocol uses only the first five.
    """
    if protocol is None:
        protocol = (
            SyntheticBenchmarkProtocol()
        )

    specifications: list[
        SyntheticRealizationSpec
    ] = []

    for scenario_index, scenario in enumerate(
        protocol.scenarios
    ):
        for realization_index in range(
            protocol.n_realizations
        ):
            scenario_offset = (
                scenario_index
                * 1_000
            )

            dataset_seed = (
                protocol.dataset_seed_base
                + scenario_offset
                + realization_index
            )

            outer_split_seed = (
                protocol.outer_split_seed_base
                + scenario_offset
                + realization_index
            )

            final_split_seed = (
                protocol.final_split_seed_base
                + scenario_offset
                + realization_index
            )

            # Each realization receives a block of 10 seeds.
            # With B=5, seeds base ... base+4 are used.
            discovery_base_seed = (
                protocol.discovery_seed_base
                + scenario_index
                * 10_000
                + realization_index
                * 10
            )

            random_pair_seed = (
                protocol.random_pair_seed_base
                + scenario_offset
                + realization_index
            )

            final_model_seed = (
                protocol.final_model_seed_base
                + scenario_offset
                + realization_index
            )

            specifications.append(
                SyntheticRealizationSpec(
                    scenario=scenario,
                    realization_index=(
                        realization_index
                    ),
                    dataset_seed=(
                        dataset_seed
                    ),
                    outer_split_seed=(
                        outer_split_seed
                    ),
                    final_split_seed=(
                        final_split_seed
                    ),
                    discovery_base_seed=(
                        discovery_base_seed
                    ),
                    random_pair_seed=(
                        random_pair_seed
                    ),
                    final_model_seed=(
                        final_model_seed
                    ),
                    n_discovery_runs=(
                        protocol
                        .n_discovery_runs
                    ),
                )
            )

    return tuple(
        specifications
    )


def canonical_pair_set(
    pairs: Iterable[
        Pair
    ],
) -> set[Pair]:
    """
    Convert an iterable of feature pairs into a canonical set.
    """
    return {
        canonical_pair(
            feature_a,
            feature_b,
        )
        for (
            feature_a,
            feature_b
        ) in pairs
    }


def evaluate_pair_set_recovery(
    predicted_pairs: Iterable[
        Pair
    ],
    true_pairs: Iterable[
        Pair
    ],
) -> PairSetRecovery:
    """
    Evaluate exact interaction-set recovery.
    """
    predicted = canonical_pair_set(
        predicted_pairs
    )

    truth = canonical_pair_set(
        true_pairs
    )

    if len(truth) == 0:
        raise ValueError(
            "At least one true interaction is required."
        )

    true_positive = len(
        predicted
        & truth
    )

    false_positive = len(
        predicted
        - truth
    )

    false_negative = len(
        truth
        - predicted
    )

    if len(predicted) == 0:
        precision = 0.0
    else:
        precision = (
            true_positive
            / len(predicted)
        )

    recall = (
        true_positive
        / len(truth)
    )

    if (
        precision
        + recall
        == 0.0
    ):
        f1 = 0.0
    else:
        f1 = (
            2.0
            * precision
            * recall
            / (
                precision
                + recall
            )
        )

    return PairSetRecovery(
        true_positive=(
            true_positive
        ),
        false_positive=(
            false_positive
        ),
        false_negative=(
            false_negative
        ),
        precision=float(
            precision
        ),
        recall=float(
            recall
        ),
        f1=float(
            f1
        ),
    )


def compute_false_positive_reduction(
    selection_recovery: PairSetRecovery,
    isr_recovery: PairSetRecovery,
) -> float:
    """
    Fraction of selection-stage false positives removed by ISR.

        (FP_selection - FP_ISR)
        ----------------------
              FP_selection

    If the selection stage contains no false positives, the quantity
    is undefined and NaN is returned.
    """
    if (
        selection_recovery
        .false_positive
        == 0
    ):
        return float(
            "nan"
        )

    if (
        isr_recovery.false_positive
        > selection_recovery.false_positive
    ):
        raise ValueError(
            "ISR false positives cannot exceed selection-stage "
            "false positives when ISR is a filtering operation."
        )

    return float(
        (
            selection_recovery
            .false_positive
            - isr_recovery
            .false_positive
        )
        / selection_recovery
        .false_positive
    )


def records_to_frame(
    records: Iterable[
        SyntheticBenchmarkRecord
    ],
) -> pd.DataFrame:
    """
    Convert benchmark records into one deterministic tabular schema.
    """
    rows = [
        asdict(
            record
        )
        for record in records
    ]

    return pd.DataFrame(
        rows
    )


def save_records_csv(
    records: Iterable[
        SyntheticBenchmarkRecord
    ],
    path: str | Path,
) -> Path:
    """
    Save benchmark records as CSV.
    """
    output_path = Path(
        path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame = records_to_frame(
        records
    )

    frame.to_csv(
        output_path,
        index=False,
    )

    return output_path