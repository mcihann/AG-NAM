from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from agnam.benchmarking.openml_protocol import (
    OPENML_TASKS,
)
from agnam.benchmarking.openml_statistics import (
    RealWorldStatisticalProtocol,
    analyze_openml_comparisons,
    build_openml_diagnostic_summary,
    openml_completion_table,
    validate_openml_master,
)


@dataclass(frozen=True)
class OpenMLPublicationExportPaths:
    predictive_by_task: Path
    model_summary: Path
    inference_table: Path
    structure_by_task: Path
    feature_type_summary: Path
    sparsity_performance: Path
    interaction_count_distribution: Path
    case_study_selection: Path
    diagnostic_summary: Path
    completion_table: Path
    freeze_manifest_json: Path


MODEL_METRIC_COLUMNS = {
    "Main NAM": {
        "AUROC": "main_auroc",
        "AUPRC": "main_auprc",
        "Balanced Accuracy": "main_balanced_accuracy",
        "F1": "main_f1",
    },
    "AG-NAM": {
        "AUROC": "agnam_auroc",
        "AUPRC": "agnam_auprc",
        "Balanced Accuracy": "agnam_balanced_accuracy",
        "F1": "agnam_f1",
    },
    "Random-Pair NAM": {
        "AUROC": "random_pair_auroc",
        "AUPRC": "random_pair_auprc",
    },
    "No-ISR AG-NAM": {
        "AUROC": "no_isr_auroc",
        "AUPRC": "no_isr_auprc",
    },
    "Single-Run AG-NAM": {
        "AUROC": "single_run_auroc",
        "AUPRC": "single_run_auprc",
    },
    "EBM": {
        "AUROC": "ebm_auroc",
        "AUPRC": "ebm_auprc",
        "Balanced Accuracy": "ebm_balanced_accuracy",
        "F1": "ebm_f1",
    },
    "CatBoost": {
        "AUROC": "catboost_auroc",
        "AUPRC": "catboost_auprc",
        "Balanced Accuracy": "catboost_balanced_accuracy",
        "F1": "catboost_f1",
    },
}


def _finite_summary(values) -> dict:
    array = pd.to_numeric(
        pd.Series(values),
        errors="coerce",
    ).to_numpy(dtype=np.float64)

    array = array[
        np.isfinite(array)
    ]

    if len(array) == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "iqr": np.nan,
            "minimum": np.nan,
            "maximum": np.nan,
        }

    q25 = float(
        np.quantile(
            array,
            0.25,
        )
    )

    q75 = float(
        np.quantile(
            array,
            0.75,
        )
    )

    return {
        "n": int(len(array)),
        "mean": float(
            np.mean(array)
        ),
        "std": (
            float(
                np.std(
                    array,
                    ddof=1,
                )
            )
            if len(array) > 1
            else np.nan
        ),
        "median": float(
            np.median(array)
        ),
        "q25": q25,
        "q75": q75,
        "iqr": float(
            q75 - q25
        ),
        "minimum": float(
            np.min(array)
        ),
        "maximum": float(
            np.max(array)
        ),
    }


def build_openml_predictive_task_table(
    master: pd.DataFrame,
) -> pd.DataFrame:
    """
    Publication-oriented task-level predictive results.
    """
    columns = [
        "task_index",
        "task_id",
        "dataset_name",
        "feature_type",
        "n_samples",
        "n_features",
        "main_auroc",
        "agnam_auroc",
        "random_pair_auroc",
        "no_isr_auroc",
        "single_run_auroc",
        "ebm_auroc",
        "catboost_auroc",
        "main_auprc",
        "agnam_auprc",
        "random_pair_auprc",
        "no_isr_auprc",
        "single_run_auprc",
        "ebm_auprc",
        "catboost_auprc",
        "delta_agnam_main_auroc",
        "delta_agnam_main_auprc",
        "delta_agnam_random_auroc",
        "delta_agnam_ebm_auroc",
        "delta_agnam_catboost_auroc",
    ]

    missing = [
        column
        for column in columns
        if column not in master.columns
    ]

    if missing:
        raise ValueError(
            "Missing predictive publication "
            f"columns: {missing}"
        )

    return (
        master[
            columns
        ]
        .sort_values(
            "task_index"
        )
        .reset_index(
            drop=True
        )
    )


def build_openml_model_summary(
    master: pd.DataFrame,
) -> pd.DataFrame:
    """
    Cross-dataset descriptive summary for every prespecified model
    and available predictive metric.
    """
    rows = []

    for (
        model_name,
        metrics,
    ) in MODEL_METRIC_COLUMNS.items():
        for (
            metric_name,
            column,
        ) in metrics.items():
            if column not in master.columns:
                raise ValueError(
                    f"Missing model metric column: {column}"
                )

            summary = _finite_summary(
                master[
                    column
                ]
            )

            rows.append(
                {
                    "model": model_name,
                    "metric": metric_name,
                    **summary,
                }
            )

    return pd.DataFrame(rows)


def build_openml_structure_task_table(
    master: pd.DataFrame,
) -> pd.DataFrame:
    """
    Task-level real-world structural diagnostics.

    No ground-truth interaction accuracy is implied.
    """
    required = [
        "task_index",
        "task_id",
        "dataset_name",
        "feature_type",
        "n_features",
        "candidate_k",
        "mean_pairwise_jaccard",
        "n_selection_stable",
        "n_isr_retained",
        "isr_sparsification",
        "max_decomposition_error",
    ]

    missing = [
        column
        for column in required
        if column not in master.columns
    ]

    if missing:
        raise ValueError(
            "Missing structure publication "
            f"columns: {missing}"
        )

    frame = master[
        required
    ].copy()

    frame[
        "selection_empty"
    ] = (
        pd.to_numeric(
            frame[
                "n_selection_stable"
            ],
            errors="raise",
        )
        == 0
    )

    frame[
        "isr_empty"
    ] = (
        pd.to_numeric(
            frame[
                "n_isr_retained"
            ],
            errors="raise",
        )
        == 0
    )

    frame[
        "retained_fraction_of_candidate_k"
    ] = (
        pd.to_numeric(
            frame[
                "n_isr_retained"
            ],
            errors="raise",
        )
        /
        pd.to_numeric(
            frame[
                "candidate_k"
            ],
            errors="raise",
        )
    )

    return (
        frame
        .sort_values(
            "task_index"
        )
        .reset_index(
            drop=True
        )
    )


def build_openml_sparsity_performance_table(
    master: pd.DataFrame,
) -> pd.DataFrame:
    """
    Dataset-level sparsification versus predictive-change table.
    """
    required = [
        "task_index",
        "task_id",
        "dataset_name",
        "feature_type",
        "n_selection_stable",
        "n_isr_retained",
        "isr_sparsification",
        "main_auroc",
        "agnam_auroc",
        "no_isr_auroc",
        "single_run_auroc",
    ]

    missing = [
        column
        for column in required
        if column not in master.columns
    ]

    if missing:
        raise ValueError(
            "Missing sparsity-performance "
            f"columns: {missing}"
        )

    frame = master[
        required
    ].copy()

    frame[
        "agnam_minus_main_auroc"
    ] = (
        frame[
            "agnam_auroc"
        ]
        - frame[
            "main_auroc"
        ]
    )

    frame[
        "agnam_minus_no_isr_auroc"
    ] = (
        frame[
            "agnam_auroc"
        ]
        - frame[
            "no_isr_auroc"
        ]
    )

    frame[
        "agnam_minus_single_run_auroc"
    ] = (
        frame[
            "agnam_auroc"
        ]
        - frame[
            "single_run_auroc"
        ]
    )

    return (
        frame
        .sort_values(
            "task_index"
        )
        .reset_index(
            drop=True
        )
    )


def build_interaction_count_distribution(
    master: pd.DataFrame,
) -> pd.DataFrame:
    """
    Distribution of final ISR-retained interaction counts across the
    28 real-world tasks.
    """
    counts = (
        pd.to_numeric(
            master[
                "n_isr_retained"
            ],
            errors="raise",
        )
        .astype(int)
    )

    distribution = (
        counts
        .value_counts()
        .sort_index()
    )

    rows = []

    total = len(counts)

    for (
        interaction_count,
        n_datasets,
    ) in distribution.items():
        rows.append(
            {
                "n_isr_retained": int(
                    interaction_count
                ),
                "n_datasets": int(
                    n_datasets
                ),
                "dataset_fraction": float(
                    n_datasets
                    / total
                ),
            }
        )

    return pd.DataFrame(rows)


def build_feature_type_summary(
    master: pd.DataFrame,
) -> pd.DataFrame:
    """
    Descriptive summary by locked feature-type stratum.

    No inferential subgroup test is performed.
    """
    rows = []

    for feature_type in (
        "numeric",
        "mixed",
        "categorical",
    ):
        frame = master[
            master[
                "feature_type"
            ]
            == feature_type
        ]

        if len(frame) == 0:
            continue

        metrics = {
            "agnam_minus_main_auroc": (
                frame[
                    "agnam_auroc"
                ]
                - frame[
                    "main_auroc"
                ]
            ),
            "agnam_minus_ebm_auroc": (
                frame[
                    "agnam_auroc"
                ]
                - frame[
                    "ebm_auroc"
                ]
            ),
            "agnam_minus_catboost_auroc": (
                frame[
                    "agnam_auroc"
                ]
                - frame[
                    "catboost_auroc"
                ]
            ),
            "agnam_minus_no_isr_auroc": (
                frame[
                    "agnam_auroc"
                ]
                - frame[
                    "no_isr_auroc"
                ]
            ),
            "mean_pairwise_jaccard": (
                frame[
                    "mean_pairwise_jaccard"
                ]
            ),
            "n_selection_stable": (
                frame[
                    "n_selection_stable"
                ]
            ),
            "n_isr_retained": (
                frame[
                    "n_isr_retained"
                ]
            ),
            "isr_sparsification": (
                frame[
                    "isr_sparsification"
                ]
            ),
        }

        for (
            metric,
            values,
        ) in metrics.items():
            summary = _finite_summary(
                values
            )

            rows.append(
                {
                    "feature_type": feature_type,
                    "metric": metric,
                    **summary,
                }
            )

    return pd.DataFrame(rows)


def select_openml_case_studies(
    master: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select one illustrative real-world case per feature-type group.

    Locked non-performance-based rule:

    For each feature type, select the earliest registry task with at
    least one ISR-retained interaction.

    If no task in a feature type retains an interaction, select the
    earliest registry task in that feature type.

    Predictive metrics are not used in this rule.
    """
    required = {
        "task_index",
        "task_id",
        "dataset_name",
        "feature_type",
        "n_isr_retained",
    }

    missing = (
        required
        - set(
            master.columns
        )
    )

    if missing:
        raise ValueError(
            "Missing case-study selection "
            f"columns: {sorted(missing)}"
        )

    rows = []

    for feature_type in (
        "numeric",
        "mixed",
        "categorical",
    ):
        group = (
            master[
                master[
                    "feature_type"
                ]
                == feature_type
            ]
            .sort_values(
                "task_index"
            )
            .reset_index(
                drop=True
            )
        )

        if len(group) == 0:
            raise RuntimeError(
                f"No {feature_type} dataset exists "
                "in the locked registry."
            )

        eligible = group[
            pd.to_numeric(
                group[
                    "n_isr_retained"
                ],
                errors="raise",
            )
            > 0
        ]

        if len(eligible) > 0:
            selected = eligible.iloc[
                0
            ]

            selection_basis = (
                "earliest_registry_task_"
                "with_isr_interaction"
            )

        else:
            selected = group.iloc[
                0
            ]

            selection_basis = (
                "earliest_registry_task_"
                "fallback_no_isr_interaction"
            )

        rows.append(
            {
                "feature_type": feature_type,
                "registry_position": int(
                    selected[
                        "task_index"
                    ]
                )
                + 1,
                "task_id": int(
                    selected[
                        "task_id"
                    ]
                ),
                "dataset_name": str(
                    selected[
                        "dataset_name"
                    ]
                ),
                "n_isr_retained": int(
                    selected[
                        "n_isr_retained"
                    ]
                ),
                "selection_basis": (
                    selection_basis
                ),
                "predictive_performance_used": False,
            }
        )

    return pd.DataFrame(rows)


def _sha256_file(
    path: Path,
) -> str:
    digest = sha256()

    with path.open(
        "rb"
    ) as handle:
        while True:
            chunk = handle.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def build_openml_freeze_manifest(
    *,
    master: pd.DataFrame,
    results_root: str | Path,
    statistical_protocol: (
        RealWorldStatisticalProtocol
        | None
    ) = None,
) -> dict:
    """
    Build a cryptographic manifest of the completed 28-task evidence.
    """
    if statistical_protocol is None:
        statistical_protocol = (
            RealWorldStatisticalProtocol()
        )

    results_root = Path(
        results_root
    )

    validate_openml_master(
        master
    )

    completion = openml_completion_table(
        master
    )

    if not bool(
        completion[
            "complete"
        ].all()
    ):
        raise RuntimeError(
            "OpenML benchmark cannot be frozen "
            "before 28/28 completion."
        )

    file_entries = []

    required_paths = [
        results_root
        / "master_results.csv"
    ]

    for task in OPENML_TASKS:
        required_paths.extend(
            [
                results_root
                / f"task_{task.task_id}.csv",
                results_root
                / (
                    f"task_{task.task_id}"
                    "_pairs.csv"
                ),
            ]
        )

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Freeze source missing: {path}"
            )

        file_entries.append(
            {
                "path": str(
                    path.as_posix()
                ),
                "bytes": int(
                    path.stat().st_size
                ),
                "sha256": _sha256_file(
                    path
                ),
            }
        )

    comparisons = (
        analyze_openml_comparisons(
            master,
            statistical_protocol=(
                statistical_protocol
            ),
        )
    )

    primary = (
        comparisons[
            comparisons[
                "family"
            ]
            == "primary"
        ]
        .iloc[
            0
        ]
    )

    return {
        "benchmark_id": (
            "agnam_openml_primary_v1"
        ),
        "status": "FROZEN",
        "n_locked_tasks": 28,
        "n_complete_tasks": 28,
        "primary_metric": "AUROC",
        "primary_comparison": (
            "AG-NAM vs Main NAM"
        ),
        "primary_mean_difference": float(
            primary[
                "mean_difference"
            ]
        ),
        "primary_p_value": float(
            primary[
                "p_raw"
            ]
        ),
        "primary_reject": bool(
            primary[
                "reject"
            ]
        ),
        "bootstrap_resamples": int(
            statistical_protocol
            .bootstrap_resamples
        ),
        "bootstrap_seed": int(
            statistical_protocol
            .bootstrap_seed
        ),
        "alpha": float(
            statistical_protocol
            .alpha
        ),
        "evidence_files": file_entries,
    }


def export_openml_publication_tables(
    *,
    results_root: str | Path = (
        "results/openml/benchmark"
    ),
    output_root: str | Path = (
        "results/openml/publication"
    ),
    freeze_manifest_path: str | Path = (
        "configs/openml_benchmark_freeze.json"
    ),
    statistical_protocol: (
        RealWorldStatisticalProtocol
        | None
    ) = None,
) -> OpenMLPublicationExportPaths:
    """
    Export final publication-ready OpenML benchmark tables.

    Requires all 28 locked tasks.
    """
    if statistical_protocol is None:
        statistical_protocol = (
            RealWorldStatisticalProtocol()
        )

    results_root = Path(
        results_root
    )

    output_root = Path(
        output_root
    )

    freeze_manifest_path = Path(
        freeze_manifest_path
    )

    master_path = (
        results_root
        / "master_results.csv"
    )

    if not master_path.exists():
        raise FileNotFoundError(
            "OpenML master results "
            f"not found: {master_path}"
        )

    master = pd.read_csv(
        master_path
    )

    validate_openml_master(
        master
    )

    completion = (
        openml_completion_table(
            master
        )
    )

    if not bool(
        completion[
            "complete"
        ].all()
    ):
        raise RuntimeError(
            "Publication export requires "
            "28/28 locked OpenML tasks."
        )

    comparisons = (
        analyze_openml_comparisons(
            master,
            statistical_protocol=(
                statistical_protocol
            ),
        )
    )

    if comparisons[
        "p_raw"
    ].isna().any():
        raise RuntimeError(
            "Confirmatory OpenML inference "
            "is incomplete."
        )

    predictive = (
        build_openml_predictive_task_table(
            master
        )
    )

    model_summary = (
        build_openml_model_summary(
            master
        )
    )

    structure = (
        build_openml_structure_task_table(
            master
        )
    )

    feature_summary = (
        build_feature_type_summary(
            master
        )
    )

    sparsity_performance = (
        build_openml_sparsity_performance_table(
            master
        )
    )

    count_distribution = (
        build_interaction_count_distribution(
            master
        )
    )

    case_studies = (
        select_openml_case_studies(
            master
        )
    )

    diagnostics = (
        build_openml_diagnostic_summary(
            master
        )
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictive_path = (
        output_root
        / "table_openml_predictive_by_task.csv"
    )

    model_summary_path = (
        output_root
        / "table_openml_model_summary.csv"
    )

    inference_path = (
        output_root
        / "table_openml_inference.csv"
    )

    structure_path = (
        output_root
        / "table_openml_structure_by_task.csv"
    )

    feature_summary_path = (
        output_root
        / "table_openml_feature_type_summary.csv"
    )

    sparsity_path = (
        output_root
        / "table_openml_sparsity_performance.csv"
    )

    count_path = (
        output_root
        / (
            "table_openml_interaction_"
            "count_distribution.csv"
        )
    )

    case_study_path = (
        output_root
        / "table_openml_case_study_selection.csv"
    )

    diagnostic_path = (
        output_root
        / "table_openml_diagnostic_summary.csv"
    )

    completion_path = (
        output_root
        / "table_openml_completion.csv"
    )

    predictive.to_csv(
        predictive_path,
        index=False,
    )

    model_summary.to_csv(
        model_summary_path,
        index=False,
    )

    comparisons.to_csv(
        inference_path,
        index=False,
    )

    structure.to_csv(
        structure_path,
        index=False,
    )

    feature_summary.to_csv(
        feature_summary_path,
        index=False,
    )

    sparsity_performance.to_csv(
        sparsity_path,
        index=False,
    )

    count_distribution.to_csv(
        count_path,
        index=False,
    )

    case_studies.to_csv(
        case_study_path,
        index=False,
    )

    diagnostics.to_csv(
        diagnostic_path,
        index=False,
    )

    completion.to_csv(
        completion_path,
        index=False,
    )

    manifest = (
        build_openml_freeze_manifest(
            master=master,
            results_root=results_root,
            statistical_protocol=(
                statistical_protocol
            ),
        )
    )

    freeze_manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    freeze_manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return OpenMLPublicationExportPaths(
        predictive_by_task=(
            predictive_path
        ),
        model_summary=(
            model_summary_path
        ),
        inference_table=(
            inference_path
        ),
        structure_by_task=(
            structure_path
        ),
        feature_type_summary=(
            feature_summary_path
        ),
        sparsity_performance=(
            sparsity_path
        ),
        interaction_count_distribution=(
            count_path
        ),
        case_study_selection=(
            case_study_path
        ),
        diagnostic_summary=(
            diagnostic_path
        ),
        completion_table=(
            completion_path
        ),
        freeze_manifest_json=(
            freeze_manifest_path
        ),
    )