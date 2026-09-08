from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


# =============================================================================
# DATASETS
# =============================================================================


@dataclass(frozen=True)
class RealXDatasetSpec:
    task_id: int
    dataset_name: str
    feature_type: str

    overall_registry_position: int

    feature_type_rank: int
    feature_type_count: int

    selection_anchor: str


LOCKED_REALX_DATASETS = (
    RealXDatasetSpec(
        task_id=49,
        dataset_name="tic-tac-toe",
        feature_type="categorical",
        overall_registry_position=1,
        feature_type_rank=1,
        feature_type_count=3,
        selection_anchor="first",
    ),
    RealXDatasetSpec(
        task_id=3,
        dataset_name="kr-vs-kp",
        feature_type="categorical",
        overall_registry_position=2,
        feature_type_rank=2,
        feature_type_count=3,
        selection_anchor="lower_median",
    ),
    RealXDatasetSpec(
        task_id=14952,
        dataset_name="PhishingWebsites",
        feature_type="categorical",
        overall_registry_position=3,
        feature_type_rank=3,
        feature_type_count=3,
        selection_anchor="last",
    ),
    RealXDatasetSpec(
        task_id=125920,
        dataset_name="dresses-sales",
        feature_type="mixed",
        overall_registry_position=4,
        feature_type_rank=1,
        feature_type_count=10,
        selection_anchor="first",
    ),
    RealXDatasetSpec(
        task_id=31,
        dataset_name="credit-g",
        feature_type="mixed",
        overall_registry_position=8,
        feature_type_rank=5,
        feature_type_count=10,
        selection_anchor="lower_median",
    ),
    RealXDatasetSpec(
        task_id=7592,
        dataset_name="adult",
        feature_type="mixed",
        overall_registry_position=13,
        feature_type_rank=10,
        feature_type_count=10,
        selection_anchor="last",
    ),
    RealXDatasetSpec(
        task_id=3913,
        dataset_name="kc2",
        feature_type="numeric",
        overall_registry_position=14,
        feature_type_rank=1,
        feature_type_count=15,
        selection_anchor="first",
    ),
    RealXDatasetSpec(
        task_id=3902,
        dataset_name="pc4",
        feature_type="numeric",
        overall_registry_position=21,
        feature_type_rank=8,
        feature_type_count=15,
        selection_anchor="lower_median",
    ),
    RealXDatasetSpec(
        task_id=3904,
        dataset_name="jm1",
        feature_type="numeric",
        overall_registry_position=28,
        feature_type_rank=15,
        feature_type_count=15,
        selection_anchor="last",
    ),
)


# =============================================================================
# SIGNAL-STRENGTH CONDITIONS
# =============================================================================


@dataclass(frozen=True)
class RealXStrengthSpec:
    name: str
    interaction_coefficient: float
    active_ground_truth: bool


LOCKED_REALX_STRENGTHS = (
    RealXStrengthSpec(
        name="null",
        interaction_coefficient=0.0,
        active_ground_truth=False,
    ),
    RealXStrengthSpec(
        name="weak",
        interaction_coefficient=0.5,
        active_ground_truth=True,
    ),
    RealXStrengthSpec(
        name="moderate",
        interaction_coefficient=1.0,
        active_ground_truth=True,
    ),
    RealXStrengthSpec(
        name="strong",
        interaction_coefficient=1.5,
        active_ground_truth=True,
    ),
)


# =============================================================================
# PROTOCOL
# =============================================================================


@dataclass(frozen=True)
class RealXProtocol:
    benchmark_id: str = "agnam_realx_semisynthetic_v1"

    n_realizations: int = 5
    max_rows: int = 5000

    train_fraction: float = 0.60
    validation_fraction: float = 0.20
    test_fraction: float = 0.20

    n_true_interactions: int = 3
    n_main_effects: int = 3
    min_eligible_features: int = 9

    max_states_per_feature: int = 8

    main_effect_coefficient: float = 1.0
    target_expected_prevalence: float = 0.50

    discovery_runs: int = 5
    residual_crossfit_folds: int = 5

    selection_threshold: float = 0.60
    isr_threshold: float = 0.60

    base_seed: int = 26_090_800

    interaction_purification_solver: str = (
        "direct_weighted_least_squares_projection"
    )

    minimum_interaction_df: int = 1
    surface_max_attempts: int = 32

    purification_tolerance: float = 1e-10
    purification_max_iterations: int = 1000

    primary_strength: str = "moderate"
    alpha: float = 0.05

    bootstrap_resamples: int = 10_000
    bootstrap_seed: int = 26_090_806

    def __post_init__(
        self,
    ) -> None:
        if self.n_realizations != 5:
            raise ValueError(
                "Real-X v1 requires exactly 5 realizations."
            )

        if self.max_rows != 5000:
            raise ValueError(
                "Real-X v1 max_rows is locked to 5000."
            )

        split_total = (
            self.train_fraction
            + self.validation_fraction
            + self.test_fraction
        )

        if not np.isclose(
            split_total,
            1.0,
            atol=1e-12,
            rtol=0.0,
        ):
            raise ValueError(
                "Train/validation/test fractions must sum to 1."
            )

        if self.n_true_interactions != 3:
            raise ValueError(
                "Real-X v1 requires exactly "
                "3 ground-truth interactions."
            )

        if self.n_main_effects != 3:
            raise ValueError(
                "Real-X v1 requires exactly 3 main-effect features."
            )

        required_features = (
            2 * self.n_true_interactions
            + self.n_main_effects
        )

        if (
            self.min_eligible_features
            < required_features
        ):
            raise ValueError(
                "min_eligible_features is too small "
                "for disjoint main and interaction features."
            )

        if self.max_states_per_feature < 2:
            raise ValueError(
                "max_states_per_feature must be at least 2."
            )

        if not (
            0.0
            < self.target_expected_prevalence
            < 1.0
        ):
            raise ValueError(
                "Target prevalence must lie strictly in (0, 1)."
            )

        if self.discovery_runs != 5:
            raise ValueError(
                "Real-X v1 discovery_runs is locked to 5."
            )

        if self.residual_crossfit_folds != 5:
            raise ValueError(
                "Real-X v1 residual cross-fitting is locked to 5."
            )

        if not np.isclose(
            self.selection_threshold,
            0.60,
            atol=1e-12,
            rtol=0.0,
        ):
            raise ValueError(
                "Real-X v1 selection threshold is locked to 0.60."
            )

        if not np.isclose(
            self.isr_threshold,
            0.60,
            atol=1e-12,
            rtol=0.0,
        ):
            raise ValueError(
                "Real-X v1 ISR threshold is locked to 0.60."
            )

        if (
            self.interaction_purification_solver
            != "direct_weighted_least_squares_projection"
        ):
            raise ValueError(
                "Real-X v1 purification solver is locked to "
                "direct_weighted_least_squares_projection."
            )

        if self.minimum_interaction_df != 1:
            raise ValueError(
                "Real-X v1 minimum interaction df is locked to 1."
            )

        if self.surface_max_attempts != 32:
            raise ValueError(
                "Real-X v1 surface_max_attempts is locked to 32."
            )

        if self.primary_strength != "moderate":
            raise ValueError(
                "Real-X v1 primary condition is locked to moderate."
            )

        if not (
            0.0
            < self.alpha
            < 1.0
        ):
            raise ValueError(
                "alpha must lie in (0, 1)."
            )

        if self.bootstrap_resamples < 1000:
            raise ValueError(
                "At least 1000 bootstrap resamples are required."
            )

        if self.purification_tolerance <= 0.0:
            raise ValueError(
                "Purification tolerance must be positive."
            )

        if self.purification_max_iterations < 1:
            raise ValueError(
                "Purification iteration limit must be positive."
            )


# =============================================================================
# RUN SCHEDULE
# =============================================================================


@dataclass(frozen=True)
class RealXRunSpec:
    run_id: str

    dataset_index: int

    task_id: int
    dataset_name: str
    feature_type: str

    realization: int

    strength_name: str
    interaction_coefficient: float
    active_ground_truth: bool

    active_true_interaction_count: int

    row_sample_seed: int
    split_seed: int
    feature_seed: int
    surface_seed: int
    label_seed: int


def _seed_family(
    *,
    protocol: RealXProtocol,
    dataset_index: int,
    realization: int,
) -> dict[str, int]:
    group_seed = (
        protocol.base_seed
        + dataset_index * 10_000
        + realization * 100
    )

    return {
        "row_sample_seed": group_seed + 11,
        "split_seed": group_seed + 21,
        "feature_seed": group_seed + 31,
        "surface_seed": group_seed + 41,
        "label_seed": group_seed + 51,
    }


def build_realx_schedule(
    protocol: RealXProtocol | None = None,
) -> tuple[
    RealXRunSpec,
    ...,
]:
    if protocol is None:
        protocol = RealXProtocol()

    rows = []

    for (
        dataset_index,
        dataset,
    ) in enumerate(
        LOCKED_REALX_DATASETS
    ):
        for realization in range(
            1,
            protocol.n_realizations + 1,
        ):
            seeds = _seed_family(
                protocol=protocol,
                dataset_index=dataset_index,
                realization=realization,
            )

            for strength in LOCKED_REALX_STRENGTHS:
                active_count = (
                    protocol.n_true_interactions
                    if strength.active_ground_truth
                    else 0
                )

                run_id = (
                    "realx_"
                    f"{dataset.task_id}_"
                    f"r{realization:02d}_"
                    f"{strength.name}"
                )

                rows.append(
                    RealXRunSpec(
                        run_id=run_id,
                        dataset_index=dataset_index,
                        task_id=dataset.task_id,
                        dataset_name=dataset.dataset_name,
                        feature_type=dataset.feature_type,
                        realization=realization,
                        strength_name=strength.name,
                        interaction_coefficient=(
                            strength.interaction_coefficient
                        ),
                        active_ground_truth=(
                            strength.active_ground_truth
                        ),
                        active_true_interaction_count=active_count,
                        row_sample_seed=seeds["row_sample_seed"],
                        split_seed=seeds["split_seed"],
                        feature_seed=seeds["feature_seed"],
                        surface_seed=seeds["surface_seed"],
                        label_seed=seeds["label_seed"],
                    )
                )

    return tuple(
        rows
    )


def expected_realx_run_count(
    protocol: RealXProtocol | None = None,
) -> int:
    if protocol is None:
        protocol = RealXProtocol()

    return (
        len(
            LOCKED_REALX_DATASETS
        )
        * len(
            LOCKED_REALX_STRENGTHS
        )
        * protocol.n_realizations
    )


# =============================================================================
# CONFIG LOCK
# =============================================================================


def canonical_realx_config_dict() -> dict[str, Any]:
    protocol = RealXProtocol()

    return {
        "benchmark_id": protocol.benchmark_id,

        "selection_rule": {
            "rule": (
                "first_lower_median_last_"
                "within_feature_type"
            ),
            "performance_used": False,
            "source_registry": (
                "locked_28_task_openml_registry"
            ),
        },

        "datasets": [
            {
                "task_id": dataset.task_id,
                "dataset_name": dataset.dataset_name,
                "feature_type": dataset.feature_type,
                "overall_registry_position": (
                    dataset.overall_registry_position
                ),
                "feature_type_rank": (
                    dataset.feature_type_rank
                ),
                "feature_type_count": (
                    dataset.feature_type_count
                ),
                "selection_anchor": (
                    dataset.selection_anchor
                ),
            }
            for dataset in LOCKED_REALX_DATASETS
        ],

        "strengths": [
            {
                "name": strength.name,
                "interaction_coefficient": (
                    strength.interaction_coefficient
                ),
                "active_ground_truth": (
                    strength.active_ground_truth
                ),
            }
            for strength in LOCKED_REALX_STRENGTHS
        ],

        "protocol": {
            "n_realizations": 5,
            "max_rows": 5000,

            "train_fraction": 0.60,
            "validation_fraction": 0.20,
            "test_fraction": 0.20,

            "n_true_interactions": 3,
            "n_main_effects": 3,
            "min_eligible_features": 9,

            "max_states_per_feature": 8,

            "main_effect_coefficient": 1.0,
            "target_expected_prevalence": 0.50,
            "link": "logistic",

            "discovery_runs": 5,
            "residual_crossfit_folds": 5,
            "selection_threshold": 0.60,
            "isr_threshold": 0.60,

            "candidate_k_rule": (
                "min_p_minus_1_20"
            ),

            "base_seed": 26_090_800,
        },

        "generator": {
            "original_labels_used": False,
            "preserve_real_X": True,

            "missing_values": (
                "explicit_missing_state"
            ),

            "numeric_state_encoding": (
                "empirical_quantile_states"
            ),

            "categorical_state_encoding": (
                "top_levels_plus_other"
            ),

            "main_effect_surface": (
                "centered_random_state_effect"
            ),

            "interaction_surface": (
                "gaussian_cell_surface"
            ),

            "interaction_purification": (
                "empirical_weighted_two_way_"
                "functional_anova"
            ),

            "interaction_purification_solver": (
                "direct_weighted_least_squares_projection"
            ),

            "interaction_pair_eligibility": (
                "positive_empirical_interaction_df"
            ),

            "minimum_interaction_df": 1,

            "interaction_pair_matching": (
                "seeded_disjoint_backtracking"
            ),

            "surface_max_attempts": 32,

            "component_standardization": (
                "unit_standard_deviation"
            ),

            "common_random_numbers_across_strengths": True,

            "intercept_calibration": (
                "target_expected_prevalence"
            ),

            "purification_tolerance": 1e-10,
            "purification_max_iterations": 1000,
        },

        "comparators": [
            "Main NAM",
            "Full AG-NAM",
            "Oracle AG-NAM",
            "Random-Pair NAM",
            "No-ISR AG-NAM",
            "Single-Run AG-NAM",
        ],

        "statistics": {
            "experimental_unit": "dataset",

            "realization_aggregation": (
                "mean_over_5_realizations"
            ),

            "primary_condition": "moderate",

            "primary_metric": (
                "interaction_auprc_minus_"
                "random_prevalence"
            ),

            "primary_test": (
                "one_sided_wilcoxon_greater"
            ),

            "alpha": 0.05,

            "multiplicity_adjustment": (
                "none_single_primary"
            ),

            "bootstrap_resamples": 10_000,
            "bootstrap_seed": 26_090_806,

            "secondary_inference": (
                "descriptive_only"
            ),
        },
    }


def load_realx_config(
    path: str | Path = (
        "configs/realx_semisynthetic_v1.yaml"
    ),
) -> dict[str, Any]:
    path = Path(
        path
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Real-X protocol config not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        loaded = yaml.safe_load(
            handle
        )

    if not isinstance(
        loaded,
        dict,
    ):
        raise ValueError(
            "Real-X config must contain a YAML mapping."
        )

    return loaded


def validate_realx_config_against_lock(
    config: dict[str, Any],
) -> None:
    expected = (
        canonical_realx_config_dict()
    )

    if config != expected:
        raise RuntimeError(
            "Real-X semi-synthetic config "
            "does not exactly match the "
            "locked v1 protocol."
        )


def load_and_validate_realx_config(
    path: str | Path = (
        "configs/realx_semisynthetic_v1.yaml"
    ),
) -> dict[str, Any]:
    config = load_realx_config(
        path
    )

    validate_realx_config_against_lock(
        config
    )

    return config