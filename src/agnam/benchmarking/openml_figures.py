from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


# =============================================================================
# LOCKED CASE STUDIES
# =============================================================================


@dataclass(frozen=True)
class LockedCaseStudySpec:
    feature_type: str
    registry_position: int
    task_id: int
    dataset_name: str


LOCKED_CASE_STUDIES = (
    LockedCaseStudySpec(
        feature_type="numeric",
        registry_position=15,
        task_id=146819,
        dataset_name="climate-model-simulation-crashes",
    ),
    LockedCaseStudySpec(
        feature_type="mixed",
        registry_position=9,
        task_id=3021,
        dataset_name="sick",
    ),
    LockedCaseStudySpec(
        feature_type="categorical",
        registry_position=2,
        task_id=3,
        dataset_name="kr-vs-kp",
    ),
)


# =============================================================================
# DATA CONTAINERS
# =============================================================================


@dataclass
class CaseNetworkData:
    specification: LockedCaseStudySpec
    nodes: tuple[str, ...]
    edges: pd.DataFrame


@dataclass
class OpenMLFigureInputs:
    predictive: pd.DataFrame
    interaction_distribution: pd.DataFrame
    sparsity_performance: pd.DataFrame
    inference: pd.DataFrame
    case_study_selection: pd.DataFrame
    case_pair_tables: dict[int, pd.DataFrame]


@dataclass(frozen=True)
class FigureFileSet:
    png: Path
    pdf: Path
    svg: Path


@dataclass(frozen=True)
class OpenMLFigureExportPaths:
    benchmark_evidence: FigureFileSet
    locked_case_studies: FigureFileSet
    case_study_edges_csv: Path


# =============================================================================
# FROZEN-EVIDENCE VERIFICATION
# =============================================================================


def _sha256_file(
    path: Path,
) -> str:
    digest = sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def verify_openml_freeze_manifest(
    manifest_path: str | Path = (
        "configs/openml_benchmark_freeze.json"
    ),
    *,
    repo_root: str | Path = ".",
) -> pd.DataFrame:
    """
    Verify every file contained in the frozen OpenML evidence manifest.

    Verification checks:
    - file existence
    - byte size
    - SHA-256 hash
    """
    manifest_path = Path(
        manifest_path
    )

    repo_root = Path(
        repo_root
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Freeze manifest not found: "
            f"{manifest_path}"
        )

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    if (
        manifest.get("status")
        != "FROZEN"
    ):
        raise RuntimeError(
            "OpenML benchmark manifest "
            "is not marked FROZEN."
        )

    if int(
        manifest.get(
            "n_complete_tasks",
            -1,
        )
    ) != 28:
        raise RuntimeError(
            "Freeze manifest does not "
            "contain 28 completed tasks."
        )

    evidence_files = manifest.get(
        "evidence_files",
        [],
    )

    if len(
        evidence_files
    ) != 57:
        raise RuntimeError(
            "Expected 57 frozen evidence files, "
            f"found {len(evidence_files)}."
        )

    rows = []

    for item in evidence_files:
        source_path = Path(
            item["path"]
        )

        if not source_path.is_absolute():
            source_path = (
                repo_root
                / source_path
            )

        if not source_path.exists():
            raise FileNotFoundError(
                "Frozen evidence file is missing: "
                f"{source_path}"
            )

        observed_size = int(
            source_path.stat().st_size
        )

        expected_size = int(
            item["bytes"]
        )

        if (
            observed_size
            != expected_size
        ):
            raise RuntimeError(
                "Frozen evidence file size mismatch: "
                f"{source_path}"
            )

        observed_hash = (
            _sha256_file(
                source_path
            )
        )

        expected_hash = str(
            item["sha256"]
        )

        if (
            observed_hash
            != expected_hash
        ):
            raise RuntimeError(
                "Frozen evidence SHA-256 mismatch: "
                f"{source_path}"
            )

        rows.append(
            {
                "path": str(
                    source_path
                ),
                "bytes": (
                    observed_size
                ),
                "sha256": (
                    observed_hash
                ),
                "verified": True,
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# BOOLEAN / INPUT HELPERS
# =============================================================================


def _as_bool_series(
    values: pd.Series,
) -> pd.Series:
    if pd.api.types.is_bool_dtype(
        values
    ):
        return values.astype(
            bool
        )

    normalized = (
        values
        .astype(str)
        .str.strip()
        .str.casefold()
    )

    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "false": False,
        "0": False,
        "no": False,
    }

    converted = normalized.map(
        mapping
    )

    if converted.isna().any():
        bad_values = (
            normalized[
                converted.isna()
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "Unable to parse boolean values: "
            f"{bad_values}"
        )

    return converted.astype(
        bool
    )


def validate_locked_case_study_selection(
    selection: pd.DataFrame,
) -> None:
    """
    Confirm that case-study selection exactly matches the locked,
    non-performance-based rule.
    """
    required = {
        "feature_type",
        "registry_position",
        "task_id",
        "dataset_name",
        "n_isr_retained",
        "selection_basis",
        "predictive_performance_used",
    }

    missing = (
        required
        - set(
            selection.columns
        )
    )

    if missing:
        raise ValueError(
            "Case-study selection table is "
            "missing columns: "
            f"{sorted(missing)}"
        )

    if len(
        selection
    ) != 3:
        raise RuntimeError(
            "Exactly three locked case "
            "studies are required."
        )

    performance_used = (
        _as_bool_series(
            selection[
                "predictive_performance_used"
            ]
        )
    )

    if performance_used.any():
        raise RuntimeError(
            "Case-study selection must not "
            "use predictive performance."
        )

    for specification in (
        LOCKED_CASE_STUDIES
    ):
        matches = selection[
            pd.to_numeric(
                selection["task_id"],
                errors="raise",
            ).astype(int)
            == specification.task_id
        ]

        if len(
            matches
        ) != 1:
            raise RuntimeError(
                "Locked case study missing "
                "or duplicated for task "
                f"{specification.task_id}."
            )

        row = matches.iloc[
            0
        ]

        if (
            str(
                row["feature_type"]
            )
            != specification.feature_type
        ):
            raise RuntimeError(
                "Feature-type mismatch "
                "for task "
                f"{specification.task_id}."
            )

        if int(
            row["registry_position"]
        ) != specification.registry_position:
            raise RuntimeError(
                "Registry-position mismatch "
                "for task "
                f"{specification.task_id}."
            )

        if (
            str(
                row["dataset_name"]
            )
            != specification.dataset_name
        ):
            raise RuntimeError(
                "Dataset-name mismatch "
                "for task "
                f"{specification.task_id}."
            )

        if int(
            row["n_isr_retained"]
        ) <= 0:
            raise RuntimeError(
                "Locked case study must "
                "contain at least one "
                "ISR-retained interaction."
            )

        if (
            str(
                row["selection_basis"]
            )
            != (
                "earliest_registry_task_"
                "with_isr_interaction"
            )
        ):
            raise RuntimeError(
                "Unexpected case-study "
                "selection basis."
            )


# =============================================================================
# BENCHMARK FIGURE DATA
# =============================================================================


def build_primary_delta_plot_data(
    predictive: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "task_id",
        "dataset_name",
        "feature_type",
        "delta_agnam_main_auroc",
    }

    missing = (
        required
        - set(
            predictive.columns
        )
    )

    if missing:
        raise ValueError(
            "Predictive table is missing "
            f"columns: {sorted(missing)}"
        )

    frame = predictive[
        [
            "task_id",
            "dataset_name",
            "feature_type",
            "delta_agnam_main_auroc",
        ]
    ].copy()

    frame[
        "delta_agnam_main_auroc"
    ] = pd.to_numeric(
        frame[
            "delta_agnam_main_auroc"
        ],
        errors="raise",
    )

    frame = (
        frame
        .sort_values(
            [
                "delta_agnam_main_auroc",
                "task_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    frame[
        "display_rank"
    ] = np.arange(
        1,
        len(frame) + 1,
    )

    tolerance = 1e-12

    frame[
        "direction"
    ] = np.where(
        frame[
            "delta_agnam_main_auroc"
        ]
        > tolerance,
        "AG-NAM higher",
        np.where(
            frame[
                "delta_agnam_main_auroc"
            ]
            < -tolerance,
            "AG-NAM lower",
            "Tie",
        ),
    )

    return frame


def build_sparsity_plot_data(
    sparsity_performance: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "task_id",
        "dataset_name",
        "feature_type",
        "isr_sparsification",
        "agnam_minus_no_isr_auroc",
    }

    missing = (
        required
        - set(
            sparsity_performance.columns
        )
    )

    if missing:
        raise ValueError(
            "Sparsity-performance table is "
            "missing columns: "
            f"{sorted(missing)}"
        )

    frame = sparsity_performance[
        [
            "task_id",
            "dataset_name",
            "feature_type",
            "isr_sparsification",
            "agnam_minus_no_isr_auroc",
        ]
    ].copy()

    frame[
        "isr_sparsification"
    ] = pd.to_numeric(
        frame[
            "isr_sparsification"
        ],
        errors="coerce",
    )

    frame[
        "agnam_minus_no_isr_auroc"
    ] = pd.to_numeric(
        frame[
            "agnam_minus_no_isr_auroc"
        ],
        errors="coerce",
    )

    finite = (
        np.isfinite(
            frame[
                "isr_sparsification"
            ]
        )
        & np.isfinite(
            frame[
                "agnam_minus_no_isr_auroc"
            ]
        )
    )

    return (
        frame[
            finite
        ]
        .reset_index(
            drop=True
        )
    )


# =============================================================================
# CASE-STUDY NETWORK DATA
# =============================================================================


def build_case_network_data(
    pair_table: pd.DataFrame,
    specification: LockedCaseStudySpec,
) -> CaseNetworkData:
    required = {
        "feature_j",
        "feature_k",
        "selection_frequency",
        "isr_retained",
    }

    missing = (
        required
        - set(
            pair_table.columns
        )
    )

    if missing:
        raise ValueError(
            "Pair table is missing "
            f"columns: {sorted(missing)}"
        )

    retained_mask = (
        _as_bool_series(
            pair_table[
                "isr_retained"
            ]
        )
    )

    retained = (
        pair_table[
            retained_mask
        ]
        .copy()
    )

    if len(
        retained
    ) == 0:
        raise RuntimeError(
            "Locked case study has no "
            "ISR-retained interactions."
        )

    retained[
        "feature_j"
    ] = retained[
        "feature_j"
    ].astype(str)

    retained[
        "feature_k"
    ] = retained[
        "feature_k"
    ].astype(str)

    retained[
        "selection_frequency"
    ] = pd.to_numeric(
        retained[
            "selection_frequency"
        ],
        errors="raise",
    )

    if not (
        retained[
            "selection_frequency"
        ]
        .between(
            0.0,
            1.0,
        )
        .all()
    ):
        raise ValueError(
            "Selection frequencies must "
            "lie in [0, 1]."
        )

    retained = (
        retained
        .sort_values(
            [
                "selection_frequency",
                "feature_j",
                "feature_k",
            ],
            ascending=[
                False,
                True,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    nodes = sorted(
        set(
            retained["feature_j"]
        )
        | set(
            retained["feature_k"]
        )
    )

    return CaseNetworkData(
        specification=(
            specification
        ),
        nodes=tuple(
            nodes
        ),
        edges=(
            retained
        ),
    )


# =============================================================================
# LOAD FIGURE INPUTS
# =============================================================================


def load_openml_figure_inputs(
    *,
    publication_root: str | Path = (
        "results/openml/publication"
    ),
    benchmark_root: str | Path = (
        "results/openml/benchmark"
    ),
    manifest_path: str | Path = (
        "configs/openml_benchmark_freeze.json"
    ),
) -> OpenMLFigureInputs:
    """
    Load publication inputs only after frozen evidence verification.
    """
    verify_openml_freeze_manifest(
        manifest_path
    )

    publication_root = Path(
        publication_root
    )

    benchmark_root = Path(
        benchmark_root
    )

    predictive = pd.read_csv(
        publication_root
        / "table_openml_predictive_by_task.csv"
    )

    interaction_distribution = pd.read_csv(
        publication_root
        / (
            "table_openml_interaction_"
            "count_distribution.csv"
        )
    )

    sparsity_performance = pd.read_csv(
        publication_root
        / "table_openml_sparsity_performance.csv"
    )

    inference = pd.read_csv(
        publication_root
        / "table_openml_inference.csv"
    )

    case_selection = pd.read_csv(
        publication_root
        / "table_openml_case_study_selection.csv"
    )

    if len(
        predictive
    ) != 28:
        raise RuntimeError(
            "Expected 28 predictive "
            "OpenML task rows."
        )

    validate_locked_case_study_selection(
        case_selection
    )

    case_pair_tables = {}

    for specification in (
        LOCKED_CASE_STUDIES
    ):
        path = (
            benchmark_root
            / (
                f"task_"
                f"{specification.task_id}"
                "_pairs.csv"
            )
        )

        if not path.exists():
            raise FileNotFoundError(
                "Case-study pair file "
                f"not found: {path}"
            )

        case_pair_tables[
            specification.task_id
        ] = pd.read_csv(
            path
        )

    return OpenMLFigureInputs(
        predictive=(
            predictive
        ),
        interaction_distribution=(
            interaction_distribution
        ),
        sparsity_performance=(
            sparsity_performance
        ),
        inference=(
            inference
        ),
        case_study_selection=(
            case_selection
        ),
        case_pair_tables=(
            case_pair_tables
        ),
    )


def build_locked_case_study_edge_table(
    inputs: OpenMLFigureInputs,
) -> pd.DataFrame:
    rows = []

    for specification in (
        LOCKED_CASE_STUDIES
    ):
        network = build_case_network_data(
            inputs.case_pair_tables[
                specification.task_id
            ],
            specification,
        )

        for edge in network.edges.itertuples(
            index=False
        ):
            rows.append(
                {
                    "feature_type": (
                        specification.feature_type
                    ),
                    "registry_position": (
                        specification.registry_position
                    ),
                    "task_id": (
                        specification.task_id
                    ),
                    "dataset_name": (
                        specification.dataset_name
                    ),
                    "feature_j": str(
                        edge.feature_j
                    ),
                    "feature_k": str(
                        edge.feature_k
                    ),
                    "selection_frequency": float(
                        edge.selection_frequency
                    ),
                    "isr_retained": True,
                }
            )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# FIGURE 1
# =============================================================================


def _primary_inference_row(
    inference: pd.DataFrame,
) -> pd.Series:
    required = {
        "comparison_id",
        "mean_difference",
        "bootstrap_mean_ci_low",
        "bootstrap_mean_ci_high",
        "p_raw",
        "reject",
    }

    missing = (
        required
        - set(
            inference.columns
        )
    )

    if missing:
        raise ValueError(
            "Inference table is missing "
            f"columns: {sorted(missing)}"
        )

    rows = inference[
        inference[
            "comparison_id"
        ]
        == "agnam_vs_main_auroc"
    ]

    if len(
        rows
    ) != 1:
        raise RuntimeError(
            "Expected exactly one primary "
            "OpenML inference row."
        )

    return rows.iloc[
        0
    ]


def _panel_label(
    axis,
    label: str,
) -> None:
    axis.text(
        -0.12,
        1.06,
        label,
        transform=axis.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
    )


def create_benchmark_evidence_figure(
    inputs: OpenMLFigureInputs,
):
    """
    A: paired AUROC change
    B: final ISR interaction-count distribution
    C: sparsification-performance trade-off
    D: descriptive feature-type comparison
    """
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
        }
    ):
        figure, axes = plt.subplots(
            2,
            2,
            figsize=(
                13.2,
                9.4,
            ),
            constrained_layout=True,
        )

        axis_a = axes[
            0,
            0
        ]

        axis_b = axes[
            0,
            1
        ]

        axis_c = axes[
            1,
            0
        ]

        axis_d = axes[
            1,
            1
        ]

        # ---------------------------------------------------------------------
        # PANEL A
        # ---------------------------------------------------------------------

        delta = build_primary_delta_plot_data(
            inputs.predictive
        )

        positive = (
            delta["direction"]
            == "AG-NAM higher"
        )

        negative = (
            delta["direction"]
            == "AG-NAM lower"
        )

        ties = (
            delta["direction"]
            == "Tie"
        )

        axis_a.axhline(
            0.0,
            linewidth=1.0,
            linestyle="--",
            color="0.45",
        )

        axis_a.scatter(
            delta.loc[
                positive,
                "display_rank",
            ],
            delta.loc[
                positive,
                "delta_agnam_main_auroc",
            ],
            marker="^",
            s=34,
            label="AG-NAM higher",
        )

        axis_a.scatter(
            delta.loc[
                negative,
                "display_rank",
            ],
            delta.loc[
                negative,
                "delta_agnam_main_auroc",
            ],
            marker="v",
            s=34,
            label="AG-NAM lower",
        )

        axis_a.scatter(
            delta.loc[
                ties,
                "display_rank",
            ],
            delta.loc[
                ties,
                "delta_agnam_main_auroc",
            ],
            marker="o",
            s=26,
            facecolors="none",
            edgecolors="0.25",
            label="Tie",
        )

        primary = _primary_inference_row(
            inputs.inference
        )

        annotation = (
            "Mean ΔAUROC = "
            f"{float(primary['mean_difference']):+.4f}\n"
            "95% bootstrap CI = "
            f"[{float(primary['bootstrap_mean_ci_low']):+.4f}, "
            f"{float(primary['bootstrap_mean_ci_high']):+.4f}]\n"
            "Wilcoxon p = "
            f"{float(primary['p_raw']):.3f}"
        )

        axis_a.text(
            0.03,
            0.97,
            annotation,
            transform=axis_a.transAxes,
            va="top",
            ha="left",
            fontsize=8.5,
            bbox={
                "boxstyle": "round,pad=0.30",
                "facecolor": "white",
                "edgecolor": "0.75",
                "alpha": 0.92,
            },
        )

        axis_a.set_title(
            "Paired predictive change across 28 datasets"
        )

        axis_a.set_xlabel(
            "Datasets ordered by AG-NAM − Main NAM AUROC"
        )

        axis_a.set_ylabel(
            "ΔAUROC (AG-NAM − Main NAM)"
        )

        axis_a.set_xlim(
            0,
            len(delta) + 1,
        )

        axis_a.legend(
            loc="lower right",
            frameon=False,
        )

        _panel_label(
            axis_a,
            "A",
        )

        # ---------------------------------------------------------------------
        # PANEL B
        # ---------------------------------------------------------------------

        distribution = (
            inputs
            .interaction_distribution
            .copy()
        )

        distribution[
            "n_isr_retained"
        ] = pd.to_numeric(
            distribution[
                "n_isr_retained"
            ],
            errors="raise",
        ).astype(int)

        distribution[
            "n_datasets"
        ] = pd.to_numeric(
            distribution[
                "n_datasets"
            ],
            errors="raise",
        ).astype(int)

        axis_b.bar(
            distribution[
                "n_isr_retained"
            ],
            distribution[
                "n_datasets"
            ],
            width=0.70,
            edgecolor="black",
            linewidth=0.7,
        )

        for row in distribution.itertuples(
            index=False
        ):
            axis_b.text(
                int(
                    row.n_isr_retained
                ),
                int(
                    row.n_datasets
                )
                + 0.25,
                str(
                    int(
                        row.n_datasets
                    )
                ),
                ha="center",
                va="bottom",
                fontsize=8.5,
            )

        axis_b.set_title(
            "Final interaction-set sparsity"
        )

        axis_b.set_xlabel(
            "Number of ISR-retained interactions"
        )

        axis_b.set_ylabel(
            "Number of datasets"
        )

        axis_b.set_xticks(
            distribution[
                "n_isr_retained"
            ].tolist()
        )

        _panel_label(
            axis_b,
            "B",
        )

        # ---------------------------------------------------------------------
        # PANEL C
        # ---------------------------------------------------------------------

        sparsity = build_sparsity_plot_data(
            inputs.sparsity_performance
        )

        axis_c.axhline(
            0.0,
            linewidth=1.0,
            linestyle="--",
            color="0.45",
        )

        axis_c.scatter(
            sparsity[
                "isr_sparsification"
            ],
            sparsity[
                "agnam_minus_no_isr_auroc"
            ],
            s=40,
            alpha=0.85,
        )

        axis_c.set_title(
            "ISR sparsification–performance trade-off"
        )

        axis_c.set_xlabel(
            "ISR sparsification fraction"
        )

        axis_c.set_ylabel(
            "ΔAUROC (AG-NAM − No-ISR)"
        )

        axis_c.set_xlim(
            -0.03,
            1.03,
        )

        axis_c.text(
            0.03,
            0.05,
            (
                f"n = {len(sparsity)} datasets "
                "with defined sparsification"
            ),
            transform=axis_c.transAxes,
            fontsize=8.5,
            va="bottom",
        )

        _panel_label(
            axis_c,
            "C",
        )

        # ---------------------------------------------------------------------
        # PANEL D
        # ---------------------------------------------------------------------

        feature_types = [
            "numeric",
            "mixed",
            "categorical",
        ]

        feature_labels = [
            "Numeric",
            "Mixed",
            "Categorical",
        ]

        grouped_values = []

        for feature_type in feature_types:
            values = inputs.predictive.loc[
                inputs.predictive[
                    "feature_type"
                ]
                == feature_type,
                "delta_agnam_main_auroc",
            ]

            values = pd.to_numeric(
                values,
                errors="raise",
            ).to_numpy(
                dtype=np.float64
            )

            grouped_values.append(
                values
            )

        axis_d.axhline(
            0.0,
            linewidth=1.0,
            linestyle="--",
            color="0.45",
        )

        axis_d.boxplot(
            grouped_values,
            positions=[
                1,
                2,
                3,
            ],
            widths=0.50,
            showfliers=False,
        )

        for position, values in enumerate(
            grouped_values,
            start=1,
        ):
            if len(
                values
            ) <= 1:
                jitter = np.zeros(
                    len(values)
                )

            else:
                jitter = np.linspace(
                    -0.10,
                    0.10,
                    len(values),
                )

            axis_d.scatter(
                np.full(
                    len(values),
                    position,
                    dtype=np.float64,
                )
                + jitter,
                values,
                s=30,
                alpha=0.80,
            )

        axis_d.set_title(
            "Descriptive effect by predictor type"
        )

        axis_d.set_xlabel(
            "OpenML predictor regime"
        )

        axis_d.set_ylabel(
            "ΔAUROC (AG-NAM − Main NAM)"
        )

        axis_d.set_xticks(
            [
                1,
                2,
                3,
            ],
            labels=(
                feature_labels
            ),
        )

        _panel_label(
            axis_d,
            "D",
        )

        return figure


# =============================================================================
# FIGURE 2 — FINAL PUBLICATION TYPOGRAPHY
# =============================================================================


# These overrides affect only visual line breaking.
# The underlying feature identifiers are NOT modified.
_DISPLAY_LABEL_OVERRIDES = {
    "bckgrnd_vdc1": "bckgrnd_\nvdc1",
    "convect_corr": "convect_\ncorr",
    "vconst_corr": "vconst_\ncorr",
    "query_on_thyroxine": "query_on_\nthyroxine",
    "referral_source": "referral_\nsource",
}


def _wrapped_label(
    label: str,
) -> str:
    """
    Return a publication-friendly display form of a feature label.

    Important:
    only visual line breaks are introduced. The original feature name
    remains unchanged in all data structures and exported tables.
    """
    label = str(
        label
    )

    return _DISPLAY_LABEL_OVERRIDES.get(
        label,
        label,
    )


def _manual_case_positions(
    task_id: int,
    nodes,
):
    """
    Deterministic publication layouts.

    Coordinates affect presentation only.
    Interaction identities and frequencies remain unchanged.
    """
    nodes = set(
        nodes
    )

    if task_id == 146819:
        coordinates = {
            "bckgrnd_vdc1": (
                -1.0,
                0.55,
            ),
            "convect_corr": (
                1.0,
                0.55,
            ),
            "vconst_2": (
                -1.0,
                -0.55,
            ),
            "vconst_corr": (
                1.0,
                -0.55,
            ),
        }

    elif task_id == 3021:
        coordinates = {
            "query_on_thyroxine": (
                -1.20,
                0.65,
            ),
            "T3": (
                -0.15,
                0.65,
            ),
            "sex": (
                0.95,
                0.65,
            ),
            "referral_source": (
                -1.20,
                -0.55,
            ),
            "FTI": (
                -0.15,
                -0.55,
            ),
            "TSH": (
                0.95,
                -0.55,
            ),
        }

    elif task_id == 3:
        coordinates = {
            "wknck": (
                0.0,
                0.0,
            ),
            "wkpos": (
                1.20,
                0.0,
            ),
            "bkxbq": (
                0.60,
                1.00,
            ),
            "bkxcr": (
                -0.75,
                1.15,
            ),
            "wkovl": (
                -1.20,
                -0.15,
            ),
            "qxmsq": (
                -0.10,
                -1.10,
            ),
            "katri": (
                2.20,
                -0.15,
            ),
        }

    else:
        graph = nx.Graph()

        graph.add_nodes_from(
            sorted(nodes)
        )

        return nx.spring_layout(
            graph,
            seed=42,
        )

    missing = (
        nodes
        - set(
            coordinates
        )
    )

    if missing:
        raise RuntimeError(
            "Manual case-study layout is "
            "missing nodes for task "
            f"{task_id}: {sorted(missing)}"
        )

    return {
        node: (
            coordinates[
                node
            ]
        )
        for node in nodes
    }


def _draw_case_network(
    axis,
    network: CaseNetworkData,
    panel_label: str,
) -> None:
    specification = (
        network.specification
    )

    graph = nx.Graph()

    graph.add_nodes_from(
        network.nodes
    )

    for edge in network.edges.itertuples(
        index=False
    ):
        graph.add_edge(
            str(
                edge.feature_j
            ),
            str(
                edge.feature_k
            ),
            selection_frequency=float(
                edge.selection_frequency
            ),
        )

    positions = _manual_case_positions(
        specification.task_id,
        graph.nodes,
    )

    frequencies = [
        float(
            graph.edges[
                edge
            ][
                "selection_frequency"
            ]
        )
        for edge in graph.edges
    ]

    widths = [
        1.3
        + 4.0
        * frequency
        for frequency in frequencies
    ]

    if specification.task_id == 146819:
        node_size = 2700
        node_font_size = 8.2

    elif specification.task_id == 3021:
        node_size = 2750
        node_font_size = 8.2

    else:
        node_size = 2200
        node_font_size = 8.5

    nx.draw_networkx_nodes(
        graph,
        positions,
        ax=axis,
        node_size=node_size,
        node_color="white",
        edgecolors="black",
        linewidths=1.15,
    )

    nx.draw_networkx_edges(
        graph,
        positions,
        ax=axis,
        width=widths,
        edge_color="0.38",
        alpha=0.88,
    )

    node_labels = {
        node: _wrapped_label(
            node
        )
        for node in graph.nodes
    }

    nx.draw_networkx_labels(
        graph,
        positions,
        labels=node_labels,
        ax=axis,
        font_size=node_font_size,
    )

    edge_labels = {
        (
            str(
                edge.feature_j
            ),
            str(
                edge.feature_k
            ),
        ): (
            f"{float(edge.selection_frequency):.2f}"
        )
        for edge in network.edges.itertuples(
            index=False
        )
    }

    nx.draw_networkx_edge_labels(
        graph,
        positions,
        edge_labels=edge_labels,
        ax=axis,
        font_size=8.0,
        rotate=False,
        label_pos=0.50,
        bbox={
            "boxstyle": "round,pad=0.16",
            "facecolor": "white",
            "edgecolor": "0.80",
            "linewidth": 0.5,
            "alpha": 0.96,
        },
    )

    axis.set_title(
        (
            f"{specification.dataset_name}\n"
            f"{specification.feature_type.capitalize()} | "
            f"{len(network.edges)} retained interactions"
        ),
        fontsize=10.5,
        pad=10,
    )

    axis.text(
        -0.05,
        1.03,
        panel_label,
        transform=axis.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
    )

    x_values = np.asarray(
        [
            position[
                0
            ]
            for position in positions.values()
        ],
        dtype=np.float64,
    )

    y_values = np.asarray(
        [
            position[
                1
            ]
            for position in positions.values()
        ],
        dtype=np.float64,
    )

    x_span = max(
        float(
            np.ptp(
                x_values
            )
        ),
        1.0,
    )

    y_span = max(
        float(
            np.ptp(
                y_values
            )
        ),
        1.0,
    )

    axis.set_xlim(
        float(
            np.min(
                x_values
            )
        )
        - 0.25
        * x_span,
        float(
            np.max(
                x_values
            )
        )
        + 0.25
        * x_span,
    )

    axis.set_ylim(
        float(
            np.min(
                y_values
            )
        )
        - 0.30
        * y_span,
        float(
            np.max(
                y_values
            )
        )
        + 0.30
        * y_span,
    )

    axis.set_axis_off()


def create_locked_case_study_figure(
    inputs: OpenMLFigureInputs,
):
    """
    Three locked, performance-independent real-world interaction
    networks.

    Only ISR-retained edges are shown.

    Edge labels:
        selection frequency across B=5 discovery runs.
    """
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "figure.titlesize": 13,
        }
    ):
        figure, axes = plt.subplots(
            1,
            3,
            figsize=(
                15.8,
                5.3,
            ),
        )

        panel_labels = (
            "A",
            "B",
            "C",
        )

        for (
            axis,
            panel_label,
            specification,
        ) in zip(
            axes,
            panel_labels,
            LOCKED_CASE_STUDIES,
        ):
            network = build_case_network_data(
                inputs.case_pair_tables[
                    specification.task_id
                ],
                specification,
            )

            _draw_case_network(
                axis,
                network,
                panel_label,
            )

        figure.suptitle(
            (
                "ISR-retained interaction structures "
                "across three locked OpenML case studies"
            ),
            fontsize=13,
            y=0.985,
        )

        figure.subplots_adjust(
            top=0.82,
            bottom=0.05,
            left=0.03,
            right=0.98,
            wspace=0.24,
        )

        return figure


# =============================================================================
# SAVE FIGURES
# =============================================================================


def save_figure_family(
    figure,
    *,
    output_root: str | Path,
    stem: str,
) -> FigureFileSet:
    output_root = Path(
        output_root
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    png_path = (
        output_root
        / f"{stem}.png"
    )

    pdf_path = (
        output_root
        / f"{stem}.pdf"
    )

    svg_path = (
        output_root
        / f"{stem}.svg"
    )

    figure.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )

    figure.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor="white",
    )

    figure.savefig(
        svg_path,
        bbox_inches="tight",
        facecolor="white",
    )

    return FigureFileSet(
        png=png_path,
        pdf=pdf_path,
        svg=svg_path,
    )


# =============================================================================
# PUBLIC FIGURE GENERATION
# =============================================================================


def generate_openml_publication_figures(
    *,
    publication_root: str | Path = (
        "results/openml/publication"
    ),
    benchmark_root: str | Path = (
        "results/openml/benchmark"
    ),
    manifest_path: str | Path = (
        "configs/openml_benchmark_freeze.json"
    ),
    output_root: str | Path = (
        "results/openml/publication/figures"
    ),
) -> OpenMLFigureExportPaths:
    """
    Generate publication figures exclusively from verified frozen
    OpenML evidence.

    No model fitting occurs.
    """
    inputs = load_openml_figure_inputs(
        publication_root=(
            publication_root
        ),
        benchmark_root=(
            benchmark_root
        ),
        manifest_path=(
            manifest_path
        ),
    )

    benchmark_figure = (
        create_benchmark_evidence_figure(
            inputs
        )
    )

    benchmark_paths = save_figure_family(
        benchmark_figure,
        output_root=(
            output_root
        ),
        stem=(
            "figure_openml_"
            "benchmark_evidence"
        ),
    )

    plt.close(
        benchmark_figure
    )

    case_figure = (
        create_locked_case_study_figure(
            inputs
        )
    )

    case_paths = save_figure_family(
        case_figure,
        output_root=(
            output_root
        ),
        stem=(
            "figure_openml_"
            "locked_case_studies"
        ),
    )

    plt.close(
        case_figure
    )

    case_edges = (
        build_locked_case_study_edge_table(
            inputs
        )
    )

    case_edges_path = (
        Path(
            publication_root
        )
        / (
            "table_openml_locked_"
            "case_study_edges.csv"
        )
    )

    case_edges.to_csv(
        case_edges_path,
        index=False,
    )

    return OpenMLFigureExportPaths(
        benchmark_evidence=(
            benchmark_paths
        ),
        locked_case_studies=(
            case_paths
        ),
        case_study_edges_csv=(
            case_edges_path
        ),
    )