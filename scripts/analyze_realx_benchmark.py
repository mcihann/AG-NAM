from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from agnam.benchmarking.realx_freeze import (
    validate_realx_evidence_freeze,
)
from agnam.benchmarking.realx_statistics import (
    ANALYSIS_SCHEMA_VERSION,
    DEFAULT_ANALYSIS_MANIFEST_PATH,
    DEFAULT_PRIMARY_DATASET_PATH,
    DEFAULT_PRIMARY_REALIZATION_PATH,
    DEFAULT_PRIMARY_RESULT_PATH,
    DEFAULT_STATISTICS_DIRECTORY,
    analyze_frozen_realx_primary,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run the prespecified confirmatory Real-X "
            "analysis on cryptographically frozen evidence."
        )
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Overwrite existing Phase 6F-B analysis outputs. "
            "Frozen evidence itself is never modified."
        ),
    )

    return parser


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


def ensure_outputs_available(
    *,
    overwrite: bool,
):
    paths = (
        DEFAULT_PRIMARY_REALIZATION_PATH,
        DEFAULT_PRIMARY_DATASET_PATH,
        DEFAULT_PRIMARY_RESULT_PATH,
        DEFAULT_ANALYSIS_MANIFEST_PATH,
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
            "Phase 6F-B analysis outputs already exist. "
            "Use --overwrite only to deterministically regenerate "
            "analysis outputs from the unchanged frozen evidence. "
            "Existing: "
            + ", ".join(
                str(
                    path
                )
                for path in existing
            )
        )


def main():
    args = (
        build_parser()
        .parse_args()
    )

    ensure_outputs_available(
        overwrite=(
            args.overwrite
        )
    )

    print(
        "=============================================="
    )

    print(
        "REAL-X CONFIRMATORY STATISTICAL ANALYSIS"
    )

    print(
        "=============================================="
    )

    print(
        "Validating frozen evidence before opening results..."
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
        realization_table,
        dataset_table,
        result,
        freeze_validation,
    ) = analyze_frozen_realx_primary()

    DEFAULT_STATISTICS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv_atomic(
        realization_table,
        DEFAULT_PRIMARY_REALIZATION_PATH,
    )

    write_csv_atomic(
        dataset_table,
        DEFAULT_PRIMARY_DATASET_PATH,
    )

    write_json_atomic(
        result.to_dict(),
        DEFAULT_PRIMARY_RESULT_PATH,
    )

    analysis_manifest = {
        "analysis_schema_version": (
            ANALYSIS_SCHEMA_VERSION
        ),

        "analysis_status": (
            "CONFIRMATORY_COMPLETE"
        ),

        "frozen_evidence_sha256": (
            freeze_validation
            .aggregate_evidence_sha256
        ),

        "phase6e_git_commit": (
            freeze_validation
            .phase6e_git_commit
        ),

        "primary_realization_table": str(
            DEFAULT_PRIMARY_REALIZATION_PATH
        ),

        "primary_dataset_table": str(
            DEFAULT_PRIMARY_DATASET_PATH
        ),

        "primary_result": str(
            DEFAULT_PRIMARY_RESULT_PATH
        ),

        "primary_condition": (
            result.primary_condition
        ),

        "primary_metric": (
            result.primary_metric
        ),

        "primary_test": (
            result.primary_test
        ),

        "alpha": (
            result.alpha
        ),

        "multiplicity_adjustment": (
            result
            .multiplicity_adjustment
        ),
    }

    write_json_atomic(
        analysis_manifest,
        DEFAULT_ANALYSIS_MANIFEST_PATH,
    )

    # -------------------------------------------------------------------------
    # Revalidate frozen evidence after analysis output generation.
    # -------------------------------------------------------------------------

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
            "Frozen evidence changed during confirmatory analysis."
        )

    print(
        "PRIMARY DATASET-LEVEL VALUES"
    )

    print(
        "----------------------------------------------"
    )

    display_columns = [
        "task_id",
        "dataset_name",
        "n_features",
        "interaction_pair_universe",
        "random_ranking_interaction_prevalence",
        "mean_interaction_auprc",
        "primary_difference",
    ]

    print(
        dataset_table[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print()

    print(
        "=============================================="
    )

    print(
        "PRESPECIFIED PRIMARY RESULT"
    )

    print(
        "=============================================="
    )

    print(
        "Experimental unit:",
        result.experimental_unit,
    )

    print(
        "Primary condition:",
        result.primary_condition,
    )

    print(
        "Lambda:",
        result.interaction_coefficient,
    )

    print(
        "Primary metric:",
        result.primary_metric,
    )

    print(
        "Realization aggregation:",
        result.realization_aggregation,
    )

    print()

    print(
        "Datasets:",
        result.n_datasets,
    )

    print(
        "Realizations per dataset:",
        result.n_realizations_per_dataset,
    )

    print()

    print(
        "Wilcoxon statistic:",
        f"{result.wilcoxon_statistic:.6f}",
    )

    print(
        "One-sided p-value:",
        f"{result.p_value:.10g}",
    )

    print(
        "Non-zero dataset differences:",
        result.n_nonzero_differences,
    )

    print(
        "Alpha:",
        result.alpha,
    )

    print(
        "Multiplicity adjustment:",
        result.multiplicity_adjustment,
    )

    print(
        "Reject H0:",
        result.reject_null,
    )

    print()

    print(
        "Mean dataset-level difference:",
        f"{result.mean_dataset_difference:+.6f}",
    )

    print(
        "Median dataset-level difference:",
        f"{result.median_dataset_difference:+.6f}",
    )

    print(
        "95% percentile bootstrap CI for mean:",
        (
            f"[{result.bootstrap_ci_lower:+.6f}, "
            f"{result.bootstrap_ci_upper:+.6f}]"
        ),
    )

    print(
        "Bootstrap resamples:",
        result.bootstrap_resamples,
    )

    print(
        "Bootstrap seed:",
        result.bootstrap_seed,
    )

    print()

    print(
        "Dataset signs:",
        (
            f"{result.positive_dataset_count} positive / "
            f"{result.zero_dataset_count} zero / "
            f"{result.negative_dataset_count} negative"
        ),
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
        "STATUS: REAL-X PRIMARY CONFIRMATORY ANALYSIS COMPLETE"
    )

    print(
        "SECONDARY REAL-X ANALYSES REMAIN DESCRIPTIVE ONLY"
    )


if __name__ == "__main__":
    main()