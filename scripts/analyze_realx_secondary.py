from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from agnam.benchmarking.realx_freeze import (
    validate_realx_evidence_freeze,
)
from agnam.benchmarking.realx_secondary import (
    DEFAULT_DATASET_STRENGTH_PATH,
    DEFAULT_FEATURE_TYPE_SUMMARY_PATH,
    DEFAULT_PREDICTIVE_COMPARATOR_PATH,
    DEFAULT_SECONDARY_DIRECTORY,
    DEFAULT_SECONDARY_MANIFEST_PATH,
    DEFAULT_STRENGTH_SUMMARY_PATH,
    SECONDARY_ANALYSIS_SCHEMA_VERSION,
    analyze_frozen_realx_secondary,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run prespecified secondary/descriptive "
            "Real-X analyses on frozen evidence."
        )
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Deterministically regenerate secondary outputs "
            "from unchanged frozen evidence."
        ),
    )

    return parser


def write_csv_atomic(
    table: pd.DataFrame,
    path: Path,
):
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


def write_json_atomic(
    payload,
    path: Path,
):
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


def ensure_output_policy(
    *,
    overwrite: bool,
):
    paths = (
        DEFAULT_DATASET_STRENGTH_PATH,
        DEFAULT_STRENGTH_SUMMARY_PATH,
        DEFAULT_PREDICTIVE_COMPARATOR_PATH,
        DEFAULT_FEATURE_TYPE_SUMMARY_PATH,
        DEFAULT_SECONDARY_MANIFEST_PATH,
    )

    existing = [
        path
        for path in paths
        if path.exists()
    ]

    if (
        existing
        and not overwrite
    ):
        raise RuntimeError(
            "Secondary Real-X outputs already exist. "
            "Use --overwrite only for deterministic regeneration "
            "from unchanged frozen evidence. Existing: "
            + ", ".join(
                str(
                    path
                )
                for path in existing
            )
        )


def print_strength_metric(
    table: pd.DataFrame,
    *,
    metric: str,
    title: str,
):
    subset = (
        table.loc[
            table[
                "metric"
            ]
            .astype(
                str
            )
            .eq(
                metric
            ),
            [
                "strength_name",
                "n",
                "mean",
                "median",
                "q25",
                "q75",
            ],
        ]
        .copy()
    )

    print(
        title
    )

    print(
        subset.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print()


def main():
    args = (
        build_parser()
        .parse_args()
    )

    ensure_output_policy(
        overwrite=(
            args.overwrite
        )
    )

    print(
        "=============================================="
    )

    print(
        "REAL-X SECONDARY / DESCRIPTIVE ANALYSIS"
    )

    print(
        "=============================================="
    )

    print(
        "No secondary confirmatory tests are performed."
    )

    print(
        "Validating frozen evidence..."
    )

    initial_freeze = (
        validate_realx_evidence_freeze()
    )

    print(
        "Freeze status: PASS"
    )

    print(
        "Frozen evidence SHA-256:",
        initial_freeze
        .aggregate_evidence_sha256,
    )

    print()

    (
        dataset_strength,
        strength_summary,
        predictive_comparator,
        feature_type_summary,
        freeze_validation,
        primary_lock,
    ) = (
        analyze_frozen_realx_secondary()
    )

    DEFAULT_SECONDARY_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv_atomic(
        dataset_strength,
        DEFAULT_DATASET_STRENGTH_PATH,
    )

    write_csv_atomic(
        strength_summary,
        DEFAULT_STRENGTH_SUMMARY_PATH,
    )

    write_csv_atomic(
        predictive_comparator,
        DEFAULT_PREDICTIVE_COMPARATOR_PATH,
    )

    write_csv_atomic(
        feature_type_summary,
        DEFAULT_FEATURE_TYPE_SUMMARY_PATH,
    )

    manifest = {
        "analysis_schema_version": (
            SECONDARY_ANALYSIS_SCHEMA_VERSION
        ),

        "analysis_status": (
            "SECONDARY_DESCRIPTIVE_COMPLETE"
        ),

        "inference_scope": (
            "descriptive_only"
        ),

        "frozen_evidence_sha256": (
            freeze_validation
            .aggregate_evidence_sha256
        ),

        "phase6e_git_commit": (
            freeze_validation
            .phase6e_git_commit
        ),

        "primary_analysis_completed": True,

        "primary_result_evidence_sha256": (
            primary_lock
            .frozen_evidence_sha256
        ),

        "dataset_strength_rows": int(
            len(
                dataset_strength
            )
        ),

        "dataset_strength_table": str(
            DEFAULT_DATASET_STRENGTH_PATH
        ),

        "strength_summary": str(
            DEFAULT_STRENGTH_SUMMARY_PATH
        ),

        "predictive_comparator_summary": str(
            DEFAULT_PREDICTIVE_COMPARATOR_PATH
        ),

        "feature_type_strength_summary": str(
            DEFAULT_FEATURE_TYPE_SUMMARY_PATH
        ),

        "secondary_confirmatory_tests": 0,

        "bootstrap_usage": (
            "descriptive_comparator_mean_CI_only"
        ),
    }

    write_json_atomic(
        manifest,
        DEFAULT_SECONDARY_MANIFEST_PATH,
    )

    final_freeze = (
        validate_realx_evidence_freeze()
    )

    if (
        final_freeze
        .aggregate_evidence_sha256
        != initial_freeze
        .aggregate_evidence_sha256
    ):
        raise RuntimeError(
            "Frozen evidence changed during "
            "secondary descriptive analysis."
        )

    print(
        "DATASET × STRENGTH ROWS:",
        len(
            dataset_strength
        ),
    )

    print()

    print_strength_metric(
        strength_summary,
        metric="mean_interaction_auprc",
        title=(
            "INTERACTION-RANKING AUPRC BY STRENGTH "
            "(dataset-level descriptive summary)"
        ),
    )

    print_strength_metric(
        strength_summary,
        metric="isr_sparsification_fraction",
        title=(
            "ISR SPARSIFICATION BY STRENGTH "
            "(dataset-level descriptive summary)"
        ),
    )

    print_strength_metric(
        strength_summary,
        metric="delta_auroc",
        title=(
            "AG-NAM - MAIN NAM AUROC BY STRENGTH "
            "(dataset-level descriptive summary)"
        ),
    )

    print_strength_metric(
        strength_summary,
        metric="delta_auprc",
        title=(
            "AG-NAM - MAIN NAM AUPRC BY STRENGTH "
            "(dataset-level descriptive summary)"
        ),
    )

    print(
        "PREDICTIVE COMPARATOR DELTAS"
    )

    print(
        "----------------------------------------------"
    )

    comparator_display = (
        predictive_comparator.loc[
            predictive_comparator[
                "metric"
            ]
            .astype(
                str
            )
            .str
            .endswith(
                "_auroc"
            ),
            [
                "strength_name",
                "comparator",
                "metric",
                "mean",
                "median",
                "bootstrap_ci_lower",
                "bootstrap_ci_upper",
                "positive_dataset_count",
                "negative_dataset_count",
            ],
        ]
    )

    print(
        comparator_display.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print()

    print(
        "FEATURE-TYPE DESCRIPTIVE SUMMARY"
    )

    print(
        "----------------------------------------------"
    )

    print(
        feature_type_summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print()

    print(
        "Frozen evidence unchanged:",
        (
            final_freeze
            .aggregate_evidence_sha256
            ==
            initial_freeze
            .aggregate_evidence_sha256
        ),
    )

    print()

    print(
        "Secondary confirmatory tests: 0"
    )

    print(
        "STATUS: REAL-X SECONDARY DESCRIPTIVE ANALYSIS COMPLETE"
    )

    print(
        "NO ADDITIONAL CONFIRMATORY HYPOTHESIS FAMILY WAS INTRODUCED"
    )


if __name__ == "__main__":
    main()