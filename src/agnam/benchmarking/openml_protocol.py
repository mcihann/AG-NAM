from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class OpenMLTaskSpec:
    task_id: int
    dataset_name: str
    feature_type: str

    expected_n_samples: int
    expected_n_predictors: int

    expected_n_numeric: int
    expected_n_categorical: int


OPENML_TASKS = (
    OpenMLTaskSpec(
        49,
        "tic-tac-toe",
        "categorical",
        958,
        9,
        0,
        9,
    ),
    OpenMLTaskSpec(
        3,
        "kr-vs-kp",
        "categorical",
        3196,
        36,
        0,
        36,
    ),
    OpenMLTaskSpec(
        14952,
        "PhishingWebsites",
        "categorical",
        11055,
        30,
        0,
        30,
    ),
    OpenMLTaskSpec(
        125920,
        "dresses-sales",
        "mixed",
        500,
        12,
        1,
        11,
    ),
    OpenMLTaskSpec(
        14954,
        "cylinder-bands",
        "mixed",
        540,
        37,
        18,
        19,
    ),
    OpenMLTaskSpec(
        9971,
        "ilpd",
        "mixed",
        583,
        10,
        9,
        1,
    ),
    OpenMLTaskSpec(
        29,
        "credit-approval",
        "mixed",
        690,
        15,
        6,
        9,
    ),
    OpenMLTaskSpec(
        31,
        "credit-g",
        "mixed",
        1000,
        20,
        7,
        13,
    ),
    OpenMLTaskSpec(
        3021,
        "sick",
        "mixed",
        3772,
        29,
        7,
        22,
    ),
    OpenMLTaskSpec(
        167141,
        "churn",
        "mixed",
        5000,
        20,
        16,
        4,
    ),
    OpenMLTaskSpec(
        14965,
        "bank-marketing",
        "mixed",
        45211,
        16,
        7,
        9,
    ),
    OpenMLTaskSpec(
        219,
        "electricity",
        "mixed",
        45312,
        8,
        7,
        1,
    ),
    OpenMLTaskSpec(
        7592,
        "adult",
        "mixed",
        48842,
        14,
        6,
        8,
    ),
    OpenMLTaskSpec(
        3913,
        "kc2",
        "numeric",
        522,
        21,
        21,
        0,
    ),
    OpenMLTaskSpec(
        146819,
        "climate-model-simulation-crashes",
        "numeric",
        540,
        18,
        18,
        0,
    ),
    OpenMLTaskSpec(
        9946,
        "wdbc",
        "numeric",
        569,
        30,
        30,
        0,
    ),
    OpenMLTaskSpec(
        15,
        "breast-w",
        "numeric",
        699,
        9,
        9,
        0,
    ),
    OpenMLTaskSpec(
        37,
        "diabetes",
        "numeric",
        768,
        8,
        8,
        0,
    ),
    OpenMLTaskSpec(
        9957,
        "qsar-biodeg",
        "numeric",
        1055,
        41,
        41,
        0,
    ),
    OpenMLTaskSpec(
        3918,
        "pc1",
        "numeric",
        1109,
        21,
        21,
        0,
    ),
    OpenMLTaskSpec(
        3902,
        "pc4",
        "numeric",
        1458,
        37,
        37,
        0,
    ),
    OpenMLTaskSpec(
        3903,
        "pc3",
        "numeric",
        1563,
        37,
        37,
        0,
    ),
    OpenMLTaskSpec(
        3917,
        "kc1",
        "numeric",
        2109,
        21,
        21,
        0,
    ),
    OpenMLTaskSpec(
        9978,
        "ozone-level-8hr",
        "numeric",
        2534,
        72,
        72,
        0,
    ),
    OpenMLTaskSpec(
        43,
        "spambase",
        "numeric",
        4601,
        57,
        57,
        0,
    ),
    OpenMLTaskSpec(
        146820,
        "wilt",
        "numeric",
        4839,
        5,
        5,
        0,
    ),
    OpenMLTaskSpec(
        9952,
        "phoneme",
        "numeric",
        5404,
        5,
        5,
        0,
    ),
    OpenMLTaskSpec(
        3904,
        "jm1",
        "numeric",
        10885,
        21,
        21,
        0,
    ),
)


@dataclass(frozen=True)
class OpenMLBenchmarkProtocol:
    """
    Locked primary real-world benchmark protocol.
    """

    repeat: int = 0
    fold: int = 0
    sample: int = 0

    final_validation_fraction: float = 0.20
    final_split_seed: int = 50_000

    n_discovery_runs: int = 5
    residual_crossfit_folds: int = 5

    selection_threshold: float = 0.60
    isr_threshold: float = 0.60

    reference_size: int = 512
    reference_seed: int = 2026

    candidate_fraction: float = 0.10
    candidate_minimum: int = 5
    candidate_maximum: int = 20

    discovery_seed_base: int = 60_000
    random_pair_seed_base: int = 70_000
    final_model_seed_base: int = 80_000

    positive_class_rule: str = (
        "minority_in_outer_development"
    )

    primary_metric: str = "AUROC"

    def __post_init__(
        self,
    ) -> None:
        if self.repeat < 0:
            raise ValueError(
                "repeat cannot be negative."
            )

        if self.fold < 0:
            raise ValueError(
                "fold cannot be negative."
            )

        if self.sample < 0:
            raise ValueError(
                "sample cannot be negative."
            )

        if not (
            0.0
            < self.final_validation_fraction
            < 1.0
        ):
            raise ValueError(
                "final_validation_fraction "
                "must lie in (0, 1)."
            )

        if self.n_discovery_runs != 5:
            raise ValueError(
                "Primary benchmark requires "
                "five discovery runs."
            )

        if self.residual_crossfit_folds != 5:
            raise ValueError(
                "Primary benchmark requires "
                "five residual cross-fitting folds."
            )

        if self.selection_threshold != 0.60:
            raise ValueError(
                "Primary selection threshold "
                "must remain 0.60."
            )

        if self.isr_threshold != 0.60:
            raise ValueError(
                "Primary ISR threshold "
                "must remain 0.60."
            )


@dataclass(frozen=True)
class OpenMLExecutionSpec:
    task_id: int
    dataset_name: str
    task_index: int

    discovery_base_seed: int
    random_pair_seed: int
    final_split_seed: int
    final_model_seed: int

    @property
    def discovery_seeds(
        self,
    ) -> tuple[int, ...]:
        return tuple(
            self.discovery_base_seed
            + index
            for index in range(5)
        )


def build_openml_schedule(
    protocol: OpenMLBenchmarkProtocol | None = None,
) -> tuple[
    OpenMLExecutionSpec,
    ...
]:
    if protocol is None:
        protocol = (
            OpenMLBenchmarkProtocol()
        )

    rows = []

    for task_index, task in enumerate(
        OPENML_TASKS
    ):
        rows.append(
            OpenMLExecutionSpec(
                task_id=(
                    task.task_id
                ),
                dataset_name=(
                    task.dataset_name
                ),
                task_index=(
                    task_index
                ),
                discovery_base_seed=(
                    protocol.discovery_seed_base
                    + task_index * 10
                ),
                random_pair_seed=(
                    protocol.random_pair_seed_base
                    + task_index
                ),
                final_split_seed=(
                    protocol.final_split_seed
                    + task_index
                ),
                final_model_seed=(
                    protocol.final_model_seed_base
                    + task_index
                ),
            )
        )

    return tuple(
        rows
    )


def openml_registry_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "task_id": task.task_id,
                "dataset_name": (
                    task.dataset_name
                ),
                "feature_type": (
                    task.feature_type
                ),
                "expected_n_samples": (
                    task.expected_n_samples
                ),
                "expected_n_predictors": (
                    task.expected_n_predictors
                ),
                "expected_n_numeric": (
                    task.expected_n_numeric
                ),
                "expected_n_categorical": (
                    task.expected_n_categorical
                ),
            }
            for task in OPENML_TASKS
        ]
    )