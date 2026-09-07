from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import traceback

import pandas as pd

from agnam.benchmarking.openml_data import (
    audit_loaded_openml_task,
    load_openml_task,
)
from agnam.benchmarking.openml_protocol import (
    OPENML_TASKS,
    OpenMLBenchmarkProtocol,
)


def main():
    protocol = (
        OpenMLBenchmarkProtocol()
    )

    output_root = (
        Path(
            "results"
        )
        / "openml"
        / "audit"
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit_rows = []

    failure_rows = []

    print(
        "========================================"
    )

    print(
        "OPENML PRE-BENCHMARK AUDIT"
    )

    print(
        "========================================"
    )

    print(
        f"Locked tasks: "
        f"{len(OPENML_TASKS)}"
    )

    for index, task_spec in enumerate(
        OPENML_TASKS,
        start=1,
    ):
        print(
            f"\n[{index:02d}/"
            f"{len(OPENML_TASKS):02d}] "
            f"Task {task_spec.task_id} — "
            f"{task_spec.dataset_name}"
        )

        try:
            loaded = (
                load_openml_task(
                    task_spec,
                    protocol=(
                        protocol
                    ),
                )
            )

            audit = (
                audit_loaded_openml_task(
                    loaded
                )
            )

            audit_rows.append(
                asdict(
                    audit
                )
            )

            print(
                f"  samples: "
                f"{audit.n_samples}"
            )

            print(
                f"  predictors: "
                f"{audit.n_predictors} "
                f"("
                f"{audit.n_numeric_predictors} numeric, "
                f"{audit.n_categorical_predictors} categorical"
                f")"
            )

            print(
                f"  split: "
                f"{audit.n_development} development / "
                f"{audit.n_test} test"
            )

            print(
                f"  positive label: "
                f"{audit.positive_label}"
            )

            print(
                f"  development positive fraction: "
                f"{audit.development_positive_fraction:.4f}"
            )

            print(
                f"  missing fraction: "
                f"{audit.missing_fraction:.6f}"
            )

            print(
                f"  audit: "
                f"{'PASS' if audit.audit_pass else 'FAIL'}"
            )

        except Exception as exception:
            failure_rows.append(
                {
                    "task_id": (
                        task_spec.task_id
                    ),
                    "dataset_name": (
                        task_spec.dataset_name
                    ),
                    "exception_type": (
                        exception
                        .__class__
                        .__name__
                    ),
                    "error_message": str(
                        exception
                    ),
                    "traceback": (
                        traceback.format_exc()
                    ),
                }
            )

            print(
                "  audit: ERROR"
            )

            print(
                f"  {exception.__class__.__name__}: "
                f"{exception}"
            )

    audit_frame = pd.DataFrame(
        audit_rows
    )

    failure_frame = pd.DataFrame(
        failure_rows,
        columns=[
            "task_id",
            "dataset_name",
            "exception_type",
            "error_message",
            "traceback",
        ],
    )

    audit_path = (
        output_root
        / "openml_task_audit.csv"
    )

    failure_path = (
        output_root
        / "openml_task_audit_failures.csv"
    )

    audit_frame.to_csv(
        audit_path,
        index=False,
    )

    failure_frame.to_csv(
        failure_path,
        index=False,
    )

    passed = (
        int(
            audit_frame[
                "audit_pass"
            ].sum()
        )
        if (
            len(
                audit_frame
            ) > 0
            and "audit_pass"
            in audit_frame.columns
        )
        else 0
    )

    failed_audit = (
        int(
            (
                ~audit_frame[
                    "audit_pass"
                ]
            ).sum()
        )
        if (
            len(
                audit_frame
            ) > 0
            and "audit_pass"
            in audit_frame.columns
        )
        else 0
    )

    print(
        "\n========================================"
    )

    print(
        "OPENML AUDIT SUMMARY"
    )

    print(
        "========================================"
    )

    print(
        f"Locked tasks: "
        f"{len(OPENML_TASKS)}"
    )

    print(
        f"Successfully loaded: "
        f"{len(audit_frame)}"
    )

    print(
        f"Audit PASS: "
        f"{passed}"
    )

    print(
        f"Audit FAIL: "
        f"{failed_audit}"
    )

    print(
        f"Load ERROR: "
        f"{len(failure_frame)}"
    )

    print(
        "\nAudit table:"
    )

    print(
        audit_path
    )

    print(
        "\nFailure table:"
    )

    print(
        failure_path
    )

    if (
        len(
            failure_frame
        )
        > 0
        or failed_audit
        > 0
        or passed
        != len(
            OPENML_TASKS
        )
    ):
        raise RuntimeError(
            "OpenML pre-benchmark audit did not "
            "pass for all 28 locked tasks."
        )

    print(
        "\nSTATUS: ALL 28 LOCKED TASKS PASSED"
    )


if __name__ == "__main__":
    main()