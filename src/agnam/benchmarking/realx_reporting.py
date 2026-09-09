from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use(
    "Agg"
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agnam.benchmarking.realx_batch import (
    DEFAULT_OUTPUT_ROOT,
    _read_protocol_csv,
)
from agnam.benchmarking.realx_freeze import (
    DEFAULT_MANIFEST_PATH,
    validate_realx_evidence_freeze,
)
from agnam.benchmarking.realx_statistics import (
    ANALYSIS_SCHEMA_VERSION as PRIMARY_ANALYSIS_SCHEMA_VERSION,
    DEFAULT_PRIMARY_DATASET_PATH,
    DEFAULT_PRIMARY_RESULT_PATH,
    PRIMARY_CONDITION,
    PRIMARY_METRIC,
    PRIMARY_TEST,
)
from agnam.benchmarking.realx_secondary import (
    DEFAULT_DATASET_STRENGTH_PATH,
    DEFAULT_FEATURE_TYPE_SUMMARY_PATH,
    DEFAULT_PREDICTIVE_COMPARATOR_PATH,
    DEFAULT_SECONDARY_MANIFEST_PATH,
    DEFAULT_STRENGTH_SUMMARY_PATH,
    SECONDARY_ANALYSIS_SCHEMA_VERSION,
    STRENGTH_ORDER,
)


REPORTING_SCHEMA_VERSION = (
    "realx_publication_reporting_v1"
)

EXPECTED_DATASETS = 9
EXPECTED_DATASET_STRENGTH_ROWS = 36

DEFAULT_PUBLICATION_DIRECTORY = (
    DEFAULT_OUTPUT_ROOT
    / "publication"
)

DEFAULT_PUBLICATION_TABLE_DIRECTORY = (
    DEFAULT_PUBLICATION_DIRECTORY
    / "tables"
)

DEFAULT_PUBLICATION_FIGURE_DIRECTORY = (
    DEFAULT_PUBLICATION_DIRECTORY
    / "figures"
)

DEFAULT_PRIMARY_DATASET_TABLE = (
    DEFAULT_PUBLICATION_TABLE_DIRECTORY
    / "table_realx_primary_by_dataset.csv"
)

DEFAULT_PRIMARY_SUMMARY_TABLE = (
    DEFAULT_PUBLICATION_TABLE_DIRECTORY
    / "table_realx_primary_summary.csv"
)

DEFAULT_STRENGTH_TABLE = (
    DEFAULT_PUBLICATION_TABLE_DIRECTORY
    / "table_realx_strength_descriptive.csv"
)

DEFAULT_COMPARATOR_TABLE = (
    DEFAULT_PUBLICATION_TABLE_DIRECTORY
    / "table_realx_predictive_comparators.csv"
)

DEFAULT_FEATURE_TYPE_TABLE = (
    DEFAULT_PUBLICATION_TABLE_DIRECTORY
    / "table_realx_feature_type_descriptive.csv"
)

DEFAULT_MAIN_FIGURE_PNG = (
    DEFAULT_PUBLICATION_FIGURE_DIRECTORY
    / "figure_realx_main.png"
)

DEFAULT_MAIN_FIGURE_PDF = (
    DEFAULT_PUBLICATION_FIGURE_DIRECTORY
    / "figure_realx_main.pdf"
)

DEFAULT_MAIN_FIGURE_SVG = (
    DEFAULT_PUBLICATION_FIGURE_DIRECTORY
    / "figure_realx_main.svg"
)

DEFAULT_TABLE_MANIFEST = (
    DEFAULT_PUBLICATION_DIRECTORY
    / "table_export_manifest.json"
)

DEFAULT_FIGURE_MANIFEST = (
    DEFAULT_PUBLICATION_DIRECTORY
    / "figure_export_manifest.json"
)


@dataclass(frozen=True)
class ReportingGate:
    frozen_evidence_sha256: str
    phase6e_git_commit: str

    primary_wilcoxon_statistic: float
    primary_p_value: float
    primary_reject_null: bool


def _read_json(
    path: str | Path,
) -> dict[str, Any]:
    path = Path(
        path
    )

    if not path.exists():
        raise RuntimeError(
            f"Required reporting input is missing: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(
            handle
        )


def _write_json_atomic(
    payload: dict[str, Any],
    path: str | Path,
) -> None:
    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = (
        path.parent
        / (
            path.name
            + ".tmp"
        )
    )

    with temporary.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )

        handle.write(
            "\n"
        )

    temporary.replace(
        path
    )


def _write_csv_atomic(
    table: pd.DataFrame,
    path: str | Path,
) -> None:
    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = (
        path.parent
        / (
            path.name
            + ".tmp"
        )
    )

    table.to_csv(
        temporary,
        index=False,
    )

    temporary.replace(
        path
    )


def validate_reporting_metadata(
    *,
    frozen_evidence_sha256: str,
    phase6e_git_commit: str,
    primary_payload: dict[str, Any],
    secondary_payload: dict[str, Any],
) -> ReportingGate:
    if str(
        primary_payload.get(
            "analysis_schema_version"
        )
    ) != PRIMARY_ANALYSIS_SCHEMA_VERSION:
        raise RuntimeError(
            "Unexpected primary-analysis schema."
        )

    if str(
        secondary_payload.get(
            "analysis_schema_version"
        )
    ) != SECONDARY_ANALYSIS_SCHEMA_VERSION:
        raise RuntimeError(
            "Unexpected secondary-analysis schema."
        )

    if str(
        primary_payload.get(
            "frozen_evidence_sha256"
        )
    ) != str(
        frozen_evidence_sha256
    ):
        raise RuntimeError(
            "Primary result does not match "
            "the current frozen evidence SHA-256."
        )

    if str(
        secondary_payload.get(
            "frozen_evidence_sha256"
        )
    ) != str(
        frozen_evidence_sha256
    ):
        raise RuntimeError(
            "Secondary result does not match "
            "the current frozen evidence SHA-256."
        )

    if str(
        primary_payload.get(
            "phase6e_git_commit"
        )
    ) != str(
        phase6e_git_commit
    ):
        raise RuntimeError(
            "Primary analysis does not reference "
            "the locked Phase 6E commit."
        )

    if str(
        primary_payload.get(
            "primary_condition"
        )
    ) != PRIMARY_CONDITION:
        raise RuntimeError(
            "Primary condition differs from "
            "the locked Real-X plan."
        )

    if str(
        primary_payload.get(
            "primary_metric"
        )
    ) != PRIMARY_METRIC:
        raise RuntimeError(
            "Primary metric differs from "
            "the locked Real-X plan."
        )

    if str(
        primary_payload.get(
            "primary_test"
        )
    ) != PRIMARY_TEST:
        raise RuntimeError(
            "Primary test differs from "
            "the locked Real-X plan."
        )

    if int(
        primary_payload.get(
            "n_datasets",
            -1,
        )
    ) != EXPECTED_DATASETS:
        raise RuntimeError(
            "Primary reporting requires exactly nine datasets."
        )

    if str(
        secondary_payload.get(
            "analysis_status"
        )
    ) != "SECONDARY_DESCRIPTIVE_COMPLETE":
        raise RuntimeError(
            "Secondary analysis is not complete."
        )

    if str(
        secondary_payload.get(
            "inference_scope"
        )
    ) != "descriptive_only":
        raise RuntimeError(
            "Secondary inference scope must remain descriptive_only."
        )

    if int(
        secondary_payload.get(
            "secondary_confirmatory_tests",
            -1,
        )
    ) != 0:
        raise RuntimeError(
            "Publication reporting refuses secondary "
            "confirmatory tests."
        )

    if int(
        secondary_payload.get(
            "dataset_strength_rows",
            -1,
        )
    ) != EXPECTED_DATASET_STRENGTH_ROWS:
        raise RuntimeError(
            "Secondary analysis must contain "
            "36 dataset-strength rows."
        )

    return ReportingGate(
        frozen_evidence_sha256=str(
            frozen_evidence_sha256
        ),

        phase6e_git_commit=str(
            phase6e_git_commit
        ),

        primary_wilcoxon_statistic=float(
            primary_payload[
                "wilcoxon_statistic"
            ]
        ),

        primary_p_value=float(
            primary_payload[
                "p_value"
            ]
        ),

        primary_reject_null=bool(
            primary_payload[
                "reject_null"
            ]
        ),
    )


def validate_reporting_gate(
    *,
    freeze_manifest_path: str | Path = (
        DEFAULT_MANIFEST_PATH
    ),
    primary_result_path: str | Path = (
        DEFAULT_PRIMARY_RESULT_PATH
    ),
    secondary_manifest_path: str | Path = (
        DEFAULT_SECONDARY_MANIFEST_PATH
    ),
    repo_root: str | Path = ".",
) -> ReportingGate:
    freeze = (
        validate_realx_evidence_freeze(
            manifest_path=(
                freeze_manifest_path
            ),
            repo_root=(
                repo_root
            ),
        )
    )

    primary_payload = (
        _read_json(
            primary_result_path
        )
    )

    secondary_payload = (
        _read_json(
            secondary_manifest_path
        )
    )

    return validate_reporting_metadata(
        frozen_evidence_sha256=(
            freeze
            .aggregate_evidence_sha256
        ),
        phase6e_git_commit=(
            freeze
            .phase6e_git_commit
        ),
        primary_payload=(
            primary_payload
        ),
        secondary_payload=(
            secondary_payload
        ),
    )


def load_reporting_sources():
    gate = (
        validate_reporting_gate()
    )

    primary_dataset = (
        _read_protocol_csv(
            DEFAULT_PRIMARY_DATASET_PATH
        )
    )

    dataset_strength = (
        _read_protocol_csv(
            DEFAULT_DATASET_STRENGTH_PATH
        )
    )

    strength_summary = (
        _read_protocol_csv(
            DEFAULT_STRENGTH_SUMMARY_PATH
        )
    )

    comparator_summary = (
        _read_protocol_csv(
            DEFAULT_PREDICTIVE_COMPARATOR_PATH
        )
    )

    feature_type_summary = (
        _read_protocol_csv(
            DEFAULT_FEATURE_TYPE_SUMMARY_PATH
        )
    )

    primary_result = (
        _read_json(
            DEFAULT_PRIMARY_RESULT_PATH
        )
    )

    if len(
        primary_dataset
    ) != EXPECTED_DATASETS:
        raise RuntimeError(
            "Primary dataset table must contain nine rows."
        )

    if len(
        dataset_strength
    ) != EXPECTED_DATASET_STRENGTH_ROWS:
        raise RuntimeError(
            "Dataset-strength table must contain 36 rows."
        )

    return (
        primary_dataset,
        dataset_strength,
        strength_summary,
        comparator_summary,
        feature_type_summary,
        primary_result,
        gate,
    )


def build_primary_publication_table(
    primary_dataset: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "task_id",
        "dataset_name",
        "feature_type",
        "n_features",
        "interaction_pair_universe",
        "random_ranking_interaction_prevalence",
        "mean_interaction_auprc",
        "primary_difference",
    }

    missing = (
        required
        - set(
            primary_dataset.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Primary dataset table is missing: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    if len(
        primary_dataset
    ) != EXPECTED_DATASETS:
        raise RuntimeError(
            "Primary publication table requires nine datasets."
        )

    output = (
        primary_dataset[
            [
                "task_id",
                "dataset_name",
                "feature_type",
                "n_features",
                "interaction_pair_universe",
                "random_ranking_interaction_prevalence",
                "mean_interaction_auprc",
                "primary_difference",
            ]
        ]
        .copy()
        .sort_values(
            "task_id"
        )
        .reset_index(
            drop=True
        )
    )

    output.columns = [
        "Task ID",
        "Dataset",
        "Predictor regime",
        "Features",
        "Pair universe",
        "Random-ranking prevalence",
        "Mean interaction AUPRC",
        "AUPRC minus random prevalence",
    ]

    return output


def build_primary_summary_table(
    primary_result: dict[str, Any],
) -> pd.DataFrame:
    record = {
        "Experimental unit": (
            primary_result[
                "experimental_unit"
            ]
        ),

        "Condition": (
            primary_result[
                "primary_condition"
            ]
        ),

        "Lambda": float(
            primary_result[
                "interaction_coefficient"
            ]
        ),

        "Datasets": int(
            primary_result[
                "n_datasets"
            ]
        ),

        "Realizations per dataset": int(
            primary_result[
                "n_realizations_per_dataset"
            ]
        ),

        "Wilcoxon statistic": float(
            primary_result[
                "wilcoxon_statistic"
            ]
        ),

        "One-sided p-value": float(
            primary_result[
                "p_value"
            ]
        ),

        "Reject H0": bool(
            primary_result[
                "reject_null"
            ]
        ),

        "Mean difference": float(
            primary_result[
                "mean_dataset_difference"
            ]
        ),

        "Median difference": float(
            primary_result[
                "median_dataset_difference"
            ]
        ),

        "95% CI lower": float(
            primary_result[
                "bootstrap_ci_lower"
            ]
        ),

        "95% CI upper": float(
            primary_result[
                "bootstrap_ci_upper"
            ]
        ),

        "Positive datasets": int(
            primary_result[
                "positive_dataset_count"
            ]
        ),

        "Zero datasets": int(
            primary_result[
                "zero_dataset_count"
            ]
        ),

        "Negative datasets": int(
            primary_result[
                "negative_dataset_count"
            ]
        ),
    }

    return pd.DataFrame(
        [
            record
        ]
    )


def build_strength_publication_table(
    strength_summary: pd.DataFrame,
) -> pd.DataFrame:
    metrics = (
        "mean_interaction_auprc",
        "isr_sparsification_fraction",
        "delta_auroc",
        "delta_auprc",
    )

    output = (
        strength_summary.loc[
            strength_summary[
                "metric"
            ].isin(
                metrics
            ),
            [
                "strength_name",
                "metric",
                "n",
                "mean",
                "median",
                "q25",
                "q75",
                "minimum",
                "maximum",
            ],
        ]
        .copy()
    )

    order = {
        strength: index
        for (
            index,
            strength,
        ) in enumerate(
            STRENGTH_ORDER
        )
    }

    output[
        "_strength_order"
    ] = output[
        "strength_name"
    ].map(
        order
    )

    output = (
        output
        .sort_values(
            [
                "_strength_order",
                "metric",
            ]
        )
        .drop(
            columns=[
                "_strength_order"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    output.columns = [
        "Strength",
        "Metric",
        "n datasets",
        "Mean",
        "Median",
        "Q25",
        "Q75",
        "Minimum",
        "Maximum",
    ]

    return output


def build_predictive_publication_table(
    comparator_summary: pd.DataFrame,
) -> pd.DataFrame:
    forbidden = {
        "p_value",
        "pvalue",
        "wilcoxon_statistic",
        "reject_null",
    }

    if (
        forbidden
        & set(
            comparator_summary.columns
        )
    ):
        raise RuntimeError(
            "Secondary predictive reporting cannot "
            "contain confirmatory inferential columns."
        )

    required = [
        "strength_name",
        "comparator",
        "metric",
        "n",
        "mean",
        "median",
        "bootstrap_ci_lower",
        "bootstrap_ci_upper",
        "positive_dataset_count",
        "zero_dataset_count",
        "negative_dataset_count",
        "inference",
    ]

    output = (
        comparator_summary[
            required
        ]
        .copy()
    )

    if not (
        output[
            "inference"
        ]
        .astype(
            str
        )
        .eq(
            "descriptive_only"
        )
        .all()
    ):
        raise RuntimeError(
            "Predictive comparator table must remain descriptive."
        )

    output.columns = [
        "Strength",
        "Comparator",
        "Metric",
        "n datasets",
        "Mean difference",
        "Median difference",
        "95% CI lower",
        "95% CI upper",
        "Positive datasets",
        "Zero datasets",
        "Negative datasets",
        "Inference",
    ]

    return output


def build_feature_type_publication_table(
    feature_type_summary: pd.DataFrame,
) -> pd.DataFrame:
    required = [
        "feature_type",
        "strength_name",
        "metric",
        "n",
        "mean",
        "std",
        "median",
        "q25",
        "q75",
        "minimum",
        "maximum",
    ]

    output = (
        feature_type_summary[
            required
        ]
        .copy()
    )

    output.columns = [
        "Predictor regime",
        "Strength",
        "Metric",
        "n datasets",
        "Mean",
        "SD",
        "Median",
        "Q25",
        "Q75",
        "Minimum",
        "Maximum",
    ]

    return output


def extract_figure_series(
    dataset_strength: pd.DataFrame,
    *,
    metric: str,
    strengths: tuple[str, ...],
) -> dict[str, np.ndarray]:
    if metric not in (
        dataset_strength.columns
    ):
        raise RuntimeError(
            f"Figure metric is missing: {metric}"
        )

    output = {}

    for strength in strengths:
        subset = (
            dataset_strength.loc[
                dataset_strength[
                    "strength_name"
                ]
                .astype(
                    str
                )
                .eq(
                    strength
                )
            ]
            .sort_values(
                "task_id"
            )
        )

        if len(
            subset
        ) != EXPECTED_DATASETS:
            raise RuntimeError(
                f"Figure strength {strength} must "
                "contain exactly nine datasets."
            )

        values = pd.to_numeric(
            subset[
                metric
            ],
            errors="coerce",
        ).to_numpy(
            dtype=np.float64
        )

        output[
            strength
        ] = values

    return output


def _scatter_box(
    ax,
    *,
    series: dict[str, np.ndarray],
    strengths: tuple[str, ...],
    ylabel: str,
    title: str,
    zero_line: bool = False,
) -> None:
    positions = np.arange(
        1,
        len(
            strengths
        )
        + 1,
    )

    finite_groups = []

    for strength in strengths:
        values = np.asarray(
            series[
                strength
            ],
            dtype=np.float64,
        )

        finite_groups.append(
            values[
                np.isfinite(
                    values
                )
            ]
        )

    ax.boxplot(
        finite_groups,
        positions=positions,
        widths=0.52,
        showfliers=False,
        patch_artist=False,
        medianprops={
            "linewidth": 1.8,
        },
        boxprops={
            "linewidth": 1.2,
        },
        whiskerprops={
            "linewidth": 1.1,
        },
        capprops={
            "linewidth": 1.1,
        },
    )

    offsets = np.linspace(
        -0.12,
        0.12,
        EXPECTED_DATASETS,
    )

    for position, strength in zip(
        positions,
        strengths,
    ):
        values = np.asarray(
            series[
                strength
            ],
            dtype=np.float64,
        )

        finite = np.isfinite(
            values
        )

        ax.scatter(
            position
            + offsets[
                finite
            ],
            values[
                finite
            ],
            s=34,
            alpha=0.82,
            zorder=3,
        )

    if zero_line:
        ax.axhline(
            0.0,
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
        )

    ax.set_xticks(
        positions
    )

    ax.set_xticklabels(
        [
            item.capitalize()
            for item in strengths
        ]
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_title(
        title,
        loc="left",
        fontweight="bold",
    )

    ax.grid(
        axis="y",
        alpha=0.18,
    )


def generate_main_realx_figure(
    *,
    primary_dataset: pd.DataFrame,
    dataset_strength: pd.DataFrame,
    primary_result: dict[str, Any],
    png_path: str | Path = (
        DEFAULT_MAIN_FIGURE_PNG
    ),
    pdf_path: str | Path = (
        DEFAULT_MAIN_FIGURE_PDF
    ),
    svg_path: str | Path = (
        DEFAULT_MAIN_FIGURE_SVG
    ),
) -> tuple[
    Path,
    Path,
    Path,
]:
    png_path = Path(
        png_path
    )

    pdf_path = Path(
        pdf_path
    )

    svg_path = Path(
        svg_path
    )

    png_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    primary = (
        primary_dataset
        .copy()
        .sort_values(
            "primary_difference"
        )
        .reset_index(
            drop=True
        )
    )

    if len(
        primary
    ) != EXPECTED_DATASETS:
        raise RuntimeError(
            "Primary figure requires exactly nine datasets."
        )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(
            12.6,
            9.2,
        ),
        constrained_layout=True,
    )

    # -------------------------------------------------------------------------
    # Panel A — Prespecified primary endpoint
    # -------------------------------------------------------------------------

    ax = axes[
        0,
        0
    ]

    y = np.arange(
        len(
            primary
        )
    )

    differences = pd.to_numeric(
        primary[
            "primary_difference"
        ],
        errors="raise",
    ).to_numpy(
        dtype=np.float64
    )

    ax.scatter(
        differences,
        y,
        s=48,
        zorder=3,
    )

    ax.axvline(
        0.0,
        linestyle="--",
        linewidth=1.0,
        alpha=0.7,
    )

    ax.set_yticks(
        y
    )

    ax.set_yticklabels(
        primary[
            "dataset_name"
        ]
        .astype(
            str
        )
        .tolist()
    )

    ax.set_xlabel(
        "Interaction AUPRC − random-ranking prevalence"
    )

    ax.set_title(
        "A  Prespecified primary endpoint",
        loc="left",
        fontweight="bold",
    )

    ax.grid(
        axis="x",
        alpha=0.18,
    )

    primary_annotation = (
        f"W = {float(primary_result['wilcoxon_statistic']):.1f}\n"
        f"one-sided p = {float(primary_result['p_value']):.4g}\n"
        f"{int(primary_result['positive_dataset_count'])}/"
        f"{int(primary_result['n_datasets'])} datasets > 0"
    )

    ax.text(
        0.98,
        0.03,
        primary_annotation,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.5,
    )

    # -------------------------------------------------------------------------
    # Panel B — Interaction-ranking AUPRC
    # -------------------------------------------------------------------------

    auprc_series = extract_figure_series(
        dataset_strength,
        metric="mean_interaction_auprc",
        strengths=(
            "weak",
            "moderate",
            "strong",
        ),
    )

    _scatter_box(
        axes[
            0,
            1
        ],
        series=auprc_series,
        strengths=(
            "weak",
            "moderate",
            "strong",
        ),
        ylabel="Interaction-ranking AUPRC",
        title="B  Interaction recovery across signal strength",
    )

    # -------------------------------------------------------------------------
    # Panel C — ISR sparsification
    # -------------------------------------------------------------------------

    sparsification_series = (
        extract_figure_series(
            dataset_strength,
            metric=(
                "isr_sparsification_fraction"
            ),
            strengths=(
                "null",
                "weak",
                "moderate",
                "strong",
            ),
        )
    )

    _scatter_box(
        axes[
            1,
            0
        ],
        series=(
            sparsification_series
        ),
        strengths=(
            "null",
            "weak",
            "moderate",
            "strong",
        ),
        ylabel="ISR sparsification fraction",
        title="C  ISR filtering across signal strength",
    )

    axes[
        1,
        0
    ].set_ylim(
        -0.04,
        1.04,
    )

    # -------------------------------------------------------------------------
    # Panel D — Predictive consequence
    # -------------------------------------------------------------------------

    delta_series = extract_figure_series(
        dataset_strength,
        metric="delta_auroc",
        strengths=(
            "null",
            "weak",
            "moderate",
            "strong",
        ),
    )

    _scatter_box(
        axes[
            1,
            1
        ],
        series=delta_series,
        strengths=(
            "null",
            "weak",
            "moderate",
            "strong",
        ),
        ylabel="ΔAUROC (AG-NAM − Main NAM)",
        title="D  Predictive change across signal strength",
        zero_line=True,
    )

    fig.suptitle(
        (
            "Real-X semi-synthetic evaluation of "
            "interaction recovery, filtering, and prediction"
        ),
        fontsize=15,
        fontweight="bold",
    )

    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    fig.savefig(
        svg_path,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return (
        png_path,
        pdf_path,
        svg_path,
    )


def export_publication_tables():
    (
        primary_dataset,
        _,
        strength_summary,
        comparator_summary,
        feature_type_summary,
        primary_result,
        gate,
    ) = load_reporting_sources()

    primary_table = (
        build_primary_publication_table(
            primary_dataset
        )
    )

    primary_summary = (
        build_primary_summary_table(
            primary_result
        )
    )

    strength_table = (
        build_strength_publication_table(
            strength_summary
        )
    )

    comparator_table = (
        build_predictive_publication_table(
            comparator_summary
        )
    )

    feature_type_table = (
        build_feature_type_publication_table(
            feature_type_summary
        )
    )

    _write_csv_atomic(
        primary_table,
        DEFAULT_PRIMARY_DATASET_TABLE,
    )

    _write_csv_atomic(
        primary_summary,
        DEFAULT_PRIMARY_SUMMARY_TABLE,
    )

    _write_csv_atomic(
        strength_table,
        DEFAULT_STRENGTH_TABLE,
    )

    _write_csv_atomic(
        comparator_table,
        DEFAULT_COMPARATOR_TABLE,
    )

    _write_csv_atomic(
        feature_type_table,
        DEFAULT_FEATURE_TYPE_TABLE,
    )

    manifest = {
        "reporting_schema_version": (
            REPORTING_SCHEMA_VERSION
        ),

        "status": (
            "PUBLICATION_TABLES_EXPORTED"
        ),

        "frozen_evidence_sha256": (
            gate
            .frozen_evidence_sha256
        ),

        "phase6e_git_commit": (
            gate
            .phase6e_git_commit
        ),

        "secondary_inference": (
            "descriptive_only"
        ),

        "secondary_confirmatory_tests": 0,

        "tables": {
            "primary_by_dataset": str(
                DEFAULT_PRIMARY_DATASET_TABLE
            ),
            "primary_summary": str(
                DEFAULT_PRIMARY_SUMMARY_TABLE
            ),
            "strength_descriptive": str(
                DEFAULT_STRENGTH_TABLE
            ),
            "predictive_comparators": str(
                DEFAULT_COMPARATOR_TABLE
            ),
            "feature_type_descriptive": str(
                DEFAULT_FEATURE_TYPE_TABLE
            ),
        },
    }

    _write_json_atomic(
        manifest,
        DEFAULT_TABLE_MANIFEST,
    )

    return manifest


def export_publication_figure():
    (
        primary_dataset,
        dataset_strength,
        _,
        _,
        _,
        primary_result,
        gate,
    ) = load_reporting_sources()

    paths = (
        generate_main_realx_figure(
            primary_dataset=(
                primary_dataset
            ),
            dataset_strength=(
                dataset_strength
            ),
            primary_result=(
                primary_result
            ),
        )
    )

    manifest = {
        "reporting_schema_version": (
            REPORTING_SCHEMA_VERSION
        ),

        "status": (
            "PUBLICATION_FIGURE_EXPORTED"
        ),

        "frozen_evidence_sha256": (
            gate
            .frozen_evidence_sha256
        ),

        "phase6e_git_commit": (
            gate
            .phase6e_git_commit
        ),

        "secondary_inference": (
            "descriptive_only"
        ),

        "secondary_confirmatory_tests": 0,

        "figure": {
            "png": str(
                paths[
                    0
                ]
            ),
            "pdf": str(
                paths[
                    1
                ]
            ),
            "svg": str(
                paths[
                    2
                ]
            ),
        },

        "png_dpi": 600,
    }

    _write_json_atomic(
        manifest,
        DEFAULT_FIGURE_MANIFEST,
    )

    return manifest