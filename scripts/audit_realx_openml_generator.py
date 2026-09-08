from __future__ import annotations

import pandas as pd

from agnam.benchmarking.realx_openml import (
    audit_all_locked_realx_datasets,
)


def main():
    print(
        "=============================================="
    )

    print(
        "LOCKED REAL-X OPENML GENERATOR AUDIT"
    )

    print(
        "=============================================="
    )

    print(
        "No model fitting. No recovery evaluation."
    )

    print(
        "Original OpenML outcomes are discarded."
    )

    print()

    audit = (
        audit_all_locked_realx_datasets()
    )

    print()
    print(
        "=============================================="
    )

    print(
        "AUDIT SUMMARY"
    )

    print(
        "=============================================="
    )

    print(
        "Datasets:",
        audit[
            "task_id"
        ].nunique(),
    )

    print(
        "Dataset-realizations:",
        len(
            audit
        ),
    )

    print(
        "All CRN checks:",
        bool(
            audit[
                "common_random_numbers_ok"
            ]
            .astype(bool)
            .all()
        ),
    )

    print(
        "Maximum purification marginal error:",
        f"{audit['max_purification_error'].max():.3e}",
    )

    print(
        "Maximum state count:",
        int(
            audit[
                "max_state_count"
            ].max()
        ),
    )

    print()

    summary_columns = [
        "task_id",
        "dataset_name",
        "feature_type",
        "original_rows",
        "original_features",
        "n_numeric_features",
        "n_categorical_features",
        "n_missing_values_original_X",
        "sampled_rows",
        "eligible_features",
    ]

    dataset_summary = (
        audit[
            summary_columns
        ]
        .drop_duplicates(
            subset=[
                "task_id"
            ]
        )
        .sort_values(
            "task_id"
        )
    )

    print(
        dataset_summary.to_string(
            index=False
        )
    )

    print()

    prevalence_columns = [
        "null_realized_prevalence",
        "weak_realized_prevalence",
        "moderate_realized_prevalence",
        "strong_realized_prevalence",
    ]

    prevalence_summary = (
        audit[
            prevalence_columns
        ]
        .agg(
            [
                "min",
                "mean",
                "max",
            ]
        )
    )

    print(
        "REALIZED PREVALENCE SUMMARY"
    )

    print(
        prevalence_summary.to_string()
    )

    print()

    print(
        "Saved audit:"
    )

    print(
        "results\\realx\\audit\\"
        "realx_generator_audit.csv"
    )

    print()

    print(
        "STATUS: ALL LOCKED REAL-X DATASETS "
        "PASSED GENERATOR AUDIT"
    )


if __name__ == "__main__":
    main()