from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agnam.benchmarking.openml_protocol import (
    OPENML_TASKS,
)


@dataclass(frozen=True)
class ExternalBaselineProtocol:
    """
    Locked external-baseline protocol for the real-world OpenML
    benchmark.

    External baseline choices and principal hyperparameters are fixed
    before AG-NAM OpenML predictive results are inspected.
    """

    # ---------------------------------------------------------
    # PACKAGE LOCK
    # ---------------------------------------------------------

    ebm_package: str = "interpret-core"
    ebm_version: str = "0.7.8"

    catboost_package: str = "catboost"
    catboost_version: str = "1.2.10"

    # ---------------------------------------------------------
    # EBM
    # ---------------------------------------------------------

    ebm_validation_size: float = 0.15

    ebm_outer_bags: int = 14
    ebm_inner_bags: int = 0

    ebm_learning_rate: float = 0.015

    ebm_max_rounds: int = 50_000

    ebm_early_stopping_rounds: int = 100
    ebm_early_stopping_tolerance: float = 1e-5

    ebm_max_bins: int = 1024
    ebm_max_interaction_bins: int = 64

    ebm_min_samples_leaf: int = 4
    ebm_max_leaves: int = 2

    ebm_n_jobs: int = -2

    # EBM receives the same locked candidate interaction budget rule
    # used by AG-NAM:
    #
    # P = p(p - 1) / 2
    #
    # K = clip(
    #       ceil(0.10 * P),
    #       5,
    #       20
    #     )
    #
    # This deliberately gives EBM a relatively strong interaction
    # capacity. AG-NAM is not advantaged by constraining EBM to the
    # smaller post-ISR interaction-set size.

    interaction_fraction: float = 0.10
    interaction_minimum: int = 5
    interaction_maximum: int = 20

    # ---------------------------------------------------------
    # CATBOOST
    # ---------------------------------------------------------

    # CPU is explicitly locked because CatBoost GPU training is
    # nondeterministic according to the official implementation
    # documentation.

    catboost_task_type: str = "CPU"

    catboost_iterations: int = 2000

    catboost_learning_rate: float = 0.03
    catboost_depth: int = 6

    catboost_l2_leaf_reg: float = 3.0

    catboost_loss_function: str = "Logloss"
    catboost_eval_metric: str = "AUC"

    catboost_early_stopping_rounds: int = 100

    catboost_bootstrap_type: str = "Bayesian"
    catboost_bagging_temperature: float = 1.0

    catboost_random_strength: float = 1.0

    catboost_thread_count: int = -1

    catboost_allow_writing_files: bool = False

    catboost_verbose: bool = False

    # ---------------------------------------------------------
    # BASELINE SEEDS
    # ---------------------------------------------------------

    ebm_seed_base: int = 120_000
    catboost_seed_base: int = 130_000

    # ---------------------------------------------------------
    # REAL-WORLD INFERENCE
    # ---------------------------------------------------------

    alpha: float = 0.05

    bootstrap_resamples: int = 10_000
    bootstrap_seed: int = 20_260_907

    primary_metric: str = "AUROC"

    primary_comparison: str = (
        "agnam_vs_main_auroc"
    )

    secondary_comparisons: tuple[
        str,
        ...
    ] = (
        "agnam_vs_main_auprc",
        "agnam_vs_ebm_auroc",
        "agnam_vs_ebm_auprc",
        "agnam_vs_catboost_auroc",
        "agnam_vs_catboost_auprc",
        "agnam_vs_random_auroc",
        "agnam_vs_random_auprc",
        "agnam_vs_single_run_auroc",
        "agnam_vs_single_run_auprc",
    )

    def __post_init__(
        self,
    ) -> None:
        if (
            self.ebm_version
            != "0.7.8"
        ):
            raise ValueError(
                "Primary benchmark requires "
                "interpret-core 0.7.8."
            )

        if (
            self.catboost_version
            != "1.2.10"
        ):
            raise ValueError(
                "Primary benchmark requires "
                "CatBoost 1.2.10."
            )

        if (
            self.catboost_task_type
            != "CPU"
        ):
            raise ValueError(
                "Primary benchmark locks "
                "CatBoost to CPU training."
            )

        if (
            self.interaction_fraction
            != 0.10
        ):
            raise ValueError(
                "Interaction fraction "
                "must remain 0.10."
            )

        if (
            self.interaction_minimum
            != 5
        ):
            raise ValueError(
                "Minimum interaction budget "
                "must remain 5."
            )

        if (
            self.interaction_maximum
            != 20
        ):
            raise ValueError(
                "Maximum interaction budget "
                "must remain 20."
            )

        if (
            self.alpha
            != 0.05
        ):
            raise ValueError(
                "Alpha must remain 0.05."
            )


@dataclass(frozen=True)
class ExternalBaselineExecutionSpec:
    task_id: int
    dataset_name: str
    task_index: int

    ebm_seed: int
    catboost_seed: int


def external_interaction_budget(
    n_features: int,
    *,
    protocol: (
        ExternalBaselineProtocol
        | None
    ) = None,
) -> int:
    """
    Interaction budget shared conceptually with the locked AG-NAM
    candidate-set rule.

    P = p(p - 1) / 2

    K = clip(
        ceil(0.10 * P),
        5,
        20
    )

    K is capped by the actual number of possible feature pairs.
    """
    if protocol is None:
        protocol = (
            ExternalBaselineProtocol()
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
            protocol
            .interaction_fraction
            * n_pairs
        )
    )

    proposed = max(
        protocol
        .interaction_minimum,
        proposed,
    )

    proposed = min(
        protocol
        .interaction_maximum,
        proposed,
    )

    return min(
        n_pairs,
        proposed,
    )


def build_external_baseline_schedule(
    protocol: (
        ExternalBaselineProtocol
        | None
    ) = None,
) -> tuple[
    ExternalBaselineExecutionSpec,
    ...
]:
    """
    Build one deterministic EBM/CatBoost seed specification for every
    locked OpenML task.
    """
    if protocol is None:
        protocol = (
            ExternalBaselineProtocol()
        )

    rows = []

    for task_index, task in enumerate(
        OPENML_TASKS
    ):
        rows.append(
            ExternalBaselineExecutionSpec(
                task_id=(
                    task.task_id
                ),
                dataset_name=(
                    task.dataset_name
                ),
                task_index=(
                    task_index
                ),
                ebm_seed=(
                    protocol
                    .ebm_seed_base
                    + task_index
                ),
                catboost_seed=(
                    protocol
                    .catboost_seed_base
                    + task_index
                ),
            )
        )

    return tuple(
        rows
    )


def ebm_locked_parameters(
    *,
    n_features: int,
    seed: int,
    protocol: (
        ExternalBaselineProtocol
        | None
    ) = None,
) -> dict:
    """
    Return the explicit locked EBM parameter dictionary.

    No InterpretML import occurs here.
    """
    if protocol is None:
        protocol = (
            ExternalBaselineProtocol()
        )

    interactions = (
        external_interaction_budget(
            n_features,
            protocol=(
                protocol
            ),
        )
    )

    return {
        "interactions": (
            interactions
        ),
        "validation_size": (
            protocol
            .ebm_validation_size
        ),
        "outer_bags": (
            protocol
            .ebm_outer_bags
        ),
        "inner_bags": (
            protocol
            .ebm_inner_bags
        ),
        "learning_rate": (
            protocol
            .ebm_learning_rate
        ),
        "max_rounds": (
            protocol
            .ebm_max_rounds
        ),
        "early_stopping_rounds": (
            protocol
            .ebm_early_stopping_rounds
        ),
        "early_stopping_tolerance": (
            protocol
            .ebm_early_stopping_tolerance
        ),
        "max_bins": (
            protocol
            .ebm_max_bins
        ),
        "max_interaction_bins": (
            protocol
            .ebm_max_interaction_bins
        ),
        "min_samples_leaf": (
            protocol
            .ebm_min_samples_leaf
        ),
        "max_leaves": (
            protocol
            .ebm_max_leaves
        ),
        "n_jobs": (
            protocol
            .ebm_n_jobs
        ),
        "random_state": (
            seed
        ),
    }


def catboost_locked_parameters(
    *,
    seed: int,
    protocol: (
        ExternalBaselineProtocol
        | None
    ) = None,
) -> dict:
    """
    Return the explicit locked CatBoost parameter dictionary.

    No CatBoost import occurs here.
    """
    if protocol is None:
        protocol = (
            ExternalBaselineProtocol()
        )

    return {
        "iterations": (
            protocol
            .catboost_iterations
        ),
        "learning_rate": (
            protocol
            .catboost_learning_rate
        ),
        "depth": (
            protocol
            .catboost_depth
        ),
        "l2_leaf_reg": (
            protocol
            .catboost_l2_leaf_reg
        ),
        "loss_function": (
            protocol
            .catboost_loss_function
        ),
        "eval_metric": (
            protocol
            .catboost_eval_metric
        ),
        "random_seed": (
            seed
        ),
        "task_type": (
            protocol
            .catboost_task_type
        ),
        "bootstrap_type": (
            protocol
            .catboost_bootstrap_type
        ),
        "bagging_temperature": (
            protocol
            .catboost_bagging_temperature
        ),
        "random_strength": (
            protocol
            .catboost_random_strength
        ),
        "thread_count": (
            protocol
            .catboost_thread_count
        ),
        "allow_writing_files": (
            protocol
            .catboost_allow_writing_files
        ),
        "verbose": (
            protocol
            .catboost_verbose
        ),
    }