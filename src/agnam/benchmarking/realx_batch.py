from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from agnam.benchmarking.realx_benchmark_engine import (
    SCHEMA_VERSION,
    run_realx_benchmark_spec,
)
from agnam.benchmarking.realx_protocol import (
    RealXProtocol,
    RealXRunSpec,
    build_realx_schedule,
)
from agnam.benchmarking.realx_runner import (
    realx_candidate_k,
)


DEFAULT_OUTPUT_ROOT = Path(
    "results/realx/benchmark"
)

RUN_DIRECTORY_NAME = "runs"


REQUIRED_CHECKPOINT_COLUMNS = {
    "schema_version",
    "benchmark_id",
    "status",
    "run_id",
    "task_id",
    "dataset_name",
    "feature_type",
    "realization",
    "strength_name",
    "interaction_coefficient",
    "n_samples",
    "n_features",
    "positive_fraction",
    "expected_prevalence",
    "train_size",
    "validation_size",
    "test_size",
    "discovery_dev_size",
    "discovery_base_seed",
    "random_pair_seed",
    "final_model_seed",
    "n_true_interactions",
    "expected_candidate_k",
    "observed_candidate_k",
    "template_true_pairs",
    "true_pairs",
    "selection_pairs",
    "isr_pairs",
    "oracle_pairs",
    "random_pairs",
    "no_isr_pairs",
    "single_run_pairs",
    "mean_pairwise_jaccard",
    "mean_interaction_auprc",
    "std_interaction_auprc",
    "n_selection_stable",
    "selection_precision",
    "selection_recall",
    "selection_f1",
    "n_isr_retained",
    "isr_precision",
    "isr_recall",
    "isr_f1",
    "false_positive_reduction",
    "main_auroc",
    "main_auprc",
    "main_balanced_accuracy",
    "main_f1",
    "agnam_auroc",
    "agnam_auprc",
    "agnam_balanced_accuracy",
    "agnam_f1",
    "delta_auroc",
    "delta_auprc",
    "delta_balanced_accuracy",
    "delta_f1",
    "oracle_auroc",
    "oracle_auprc",
    "random_pair_auroc",
    "random_pair_auprc",
    "no_isr_auroc",
    "no_isr_auprc",
    "single_run_auroc",
    "single_run_auprc",
    "max_decomposition_error",
    "test_integrity_ok",
    "publication_eligible",
    "runtime_seconds",
}


@dataclass(frozen=True)
class BatchPlanItem:
    run_spec: RealXRunSpec
    checkpoint_path: Path

    state: str
    reason: str


@dataclass(frozen=True)
class BatchSummary:
    selected: int

    completed_now: int
    skipped_existing: int

    invalid_existing: int
    failed: int

    planned: int

    total_complete: int


# =============================================================================
# CSV I/O
# =============================================================================


def _read_protocol_csv(
    path: str | Path,
) -> pd.DataFrame:
    """
    Read a benchmark CSV without allowing pandas to reinterpret
    protocol-level string labels such as "null" as missing values.

    Pandas normally includes the literal string "null" in its default
    NA vocabulary. In Real-X, however, "null" is a prespecified signal
    strength and must remain a literal categorical label.

    Actual NaN values written by pandas are emitted as empty CSV cells.
    We therefore disable the default NA vocabulary and declare only an
    empty field as missing.
    """
    return pd.read_csv(
        path,
        keep_default_na=False,
        na_values=[
            "",
        ],
    )


# =============================================================================
# CHECKPOINT PATHS
# =============================================================================


def checkpoint_filename(
    run_spec: RealXRunSpec,
) -> str:
    return (
        f"task_{run_spec.task_id}_"
        f"r{run_spec.realization:02d}_"
        f"{run_spec.strength_name}.csv"
    )


def checkpoint_path(
    run_spec: RealXRunSpec,
    *,
    output_root: str | Path = (
        DEFAULT_OUTPUT_ROOT
    ),
) -> Path:
    output_root = Path(
        output_root
    )

    return (
        output_root
        / RUN_DIRECTORY_NAME
        / checkpoint_filename(
            run_spec
        )
    )


# =============================================================================
# CHECKPOINT VALIDATION
# =============================================================================


def _parse_bool(
    value,
) -> bool:
    if isinstance(
        value,
        bool,
    ):
        return value

    if isinstance(
        value,
        np.bool_,
    ):
        return bool(
            value
        )

    normalized = str(
        value
    ).strip().lower()

    if normalized in {
        "true",
        "1",
        "yes",
    }:
        return True

    if normalized in {
        "false",
        "0",
        "no",
    }:
        return False

    raise ValueError(
        f"Unable to parse boolean value: {value!r}"
    )


def validate_checkpoint(
    path: str | Path,
    run_spec: RealXRunSpec,
    *,
    protocol: RealXProtocol | None = None,
) -> tuple[
    bool,
    str,
]:
    if protocol is None:
        protocol = (
            RealXProtocol()
        )

    path = Path(
        path
    )

    if not path.exists():
        return (
            False,
            "missing",
        )

    try:
        table = _read_protocol_csv(
            path
        )

    except Exception as exc:
        return (
            False,
            f"read_error:{type(exc).__name__}",
        )

    if len(
        table
    ) != 1:
        return (
            False,
            "checkpoint_must_contain_exactly_one_row",
        )

    missing_columns = (
        REQUIRED_CHECKPOINT_COLUMNS
        - set(
            table.columns
        )
    )

    if missing_columns:
        return (
            False,
            "missing_columns:"
            + ",".join(
                sorted(
                    missing_columns
                )
            ),
        )

    row = table.iloc[
        0
    ]

    try:
        if str(
            row[
                "schema_version"
            ]
        ) != SCHEMA_VERSION:
            return (
                False,
                "schema_version_mismatch",
            )

        if str(
            row[
                "benchmark_id"
            ]
        ) != protocol.benchmark_id:
            return (
                False,
                "benchmark_id_mismatch",
            )

        if str(
            row[
                "status"
            ]
        ) != "COMPLETE":
            return (
                False,
                "status_not_complete",
            )

        if str(
            row[
                "run_id"
            ]
        ) != run_spec.run_id:
            return (
                False,
                "run_id_mismatch",
            )

        if int(
            row[
                "task_id"
            ]
        ) != run_spec.task_id:
            return (
                False,
                "task_id_mismatch",
            )

        if str(
            row[
                "dataset_name"
            ]
        ) != run_spec.dataset_name:
            return (
                False,
                "dataset_name_mismatch",
            )

        if str(
            row[
                "feature_type"
            ]
        ) != run_spec.feature_type:
            return (
                False,
                "feature_type_mismatch",
            )

        if int(
            row[
                "realization"
            ]
        ) != run_spec.realization:
            return (
                False,
                "realization_mismatch",
            )

        if str(
            row[
                "strength_name"
            ]
        ) != run_spec.strength_name:
            return (
                False,
                "strength_mismatch",
            )

        if not np.isclose(
            float(
                row[
                    "interaction_coefficient"
                ]
            ),
            float(
                run_spec
                .interaction_coefficient
            ),
            atol=1e-12,
            rtol=0.0,
        ):
            return (
                False,
                "interaction_coefficient_mismatch",
            )

        if not _parse_bool(
            row[
                "publication_eligible"
            ]
        ):
            return (
                False,
                "publication_eligible_false",
            )

        if not _parse_bool(
            row[
                "test_integrity_ok"
            ]
        ):
            return (
                False,
                "test_integrity_failed",
            )

        expected_k = realx_candidate_k(
            int(
                row[
                    "n_features"
                ]
            )
        )

        if int(
            row[
                "expected_candidate_k"
            ]
        ) != expected_k:
            return (
                False,
                "expected_candidate_k_invalid",
            )

        if int(
            row[
                "observed_candidate_k"
            ]
        ) != expected_k:
            return (
                False,
                "observed_candidate_k_invalid",
            )

        if (
            float(
                row[
                    "max_decomposition_error"
                ]
            )
            > 1e-6
        ):
            return (
                False,
                "decomposition_error_exceeds_tolerance",
            )

        expected_true_count = (
            0
            if (
                run_spec
                .strength_name
                == "null"
            )
            else 3
        )

        if int(
            row[
                "n_true_interactions"
            ]
        ) != expected_true_count:
            return (
                False,
                "true_interaction_count_mismatch",
            )

    except Exception as exc:
        return (
            False,
            f"validation_error:{type(exc).__name__}",
        )

    return (
        True,
        "valid",
    )


# =============================================================================
# ATOMIC CHECKPOINT WRITING
# =============================================================================


def write_checkpoint_atomic(
    record: dict,
    path: str | Path,
) -> None:
    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        path.parent
        / (
            path.name
            + ".tmp"
        )
    )

    pd.DataFrame(
        [
            record
        ]
    ).to_csv(
        temporary_path,
        index=False,
    )

    os.replace(
        temporary_path,
        path,
    )


# =============================================================================
# LOCKED RUN SELECTION
# =============================================================================


def select_run_specs(
    *,
    protocol: RealXProtocol | None = None,
    run_id: str | None = None,
    task_id: int | None = None,
    realization: int | None = None,
    strength: str | None = None,
    start: int | None = None,
    end: int | None = None,
) -> tuple[
    RealXRunSpec,
    ...,
]:
    if protocol is None:
        protocol = (
            RealXProtocol()
        )

    schedule = list(
        build_realx_schedule(
            protocol
        )
    )

    if run_id is not None:
        schedule = [
            item
            for item in schedule
            if item.run_id
            == run_id
        ]

    if task_id is not None:
        schedule = [
            item
            for item in schedule
            if item.task_id
            == task_id
        ]

    if realization is not None:
        schedule = [
            item
            for item in schedule
            if item.realization
            == realization
        ]

    if strength is not None:
        schedule = [
            item
            for item in schedule
            if item.strength_name
            == strength
        ]

    if start is not None:
        if start < 1:
            raise ValueError(
                "--start is one-based and must be >= 1."
            )

    if end is not None:
        if end < 1:
            raise ValueError(
                "--end is one-based and must be >= 1."
            )

    if (
        start is not None
        and end is not None
        and end < start
    ):
        raise ValueError(
            "--end cannot be smaller than --start."
        )

    start_index = (
        0
        if start is None
        else start - 1
    )

    end_index = (
        len(
            schedule
        )
        if end is None
        else end
    )

    schedule = schedule[
        start_index:
        end_index
    ]

    return tuple(
        schedule
    )


# =============================================================================
# BATCH PLANNING
# =============================================================================


def build_batch_plan(
    specs: Iterable[
        RealXRunSpec
    ],
    *,
    output_root: str | Path = (
        DEFAULT_OUTPUT_ROOT
    ),
    protocol: RealXProtocol | None = None,
    force: bool = False,
) -> tuple[
    BatchPlanItem,
    ...,
]:
    if protocol is None:
        protocol = (
            RealXProtocol()
        )

    plan = []

    for run_spec in specs:
        path = checkpoint_path(
            run_spec,
            output_root=output_root,
        )

        if force:
            plan.append(
                BatchPlanItem(
                    run_spec=run_spec,
                    checkpoint_path=path,
                    state="PLANNED",
                    reason="force",
                )
            )

            continue

        if not path.exists():
            plan.append(
                BatchPlanItem(
                    run_spec=run_spec,
                    checkpoint_path=path,
                    state="PLANNED",
                    reason="missing",
                )
            )

            continue

        valid, reason = (
            validate_checkpoint(
                path,
                run_spec,
                protocol=protocol,
            )
        )

        if valid:
            plan.append(
                BatchPlanItem(
                    run_spec=run_spec,
                    checkpoint_path=path,
                    state="SKIP",
                    reason="validated_checkpoint_exists",
                )
            )

        else:
            plan.append(
                BatchPlanItem(
                    run_spec=run_spec,
                    checkpoint_path=path,
                    state="PLANNED_INVALID",
                    reason=reason,
                )
            )

    return tuple(
        plan
    )


# =============================================================================
# AGGREGATE OUTPUTS
# =============================================================================


def _all_schedule_rows(
    *,
    protocol: RealXProtocol,
    output_root: Path,
) -> pd.DataFrame:
    records = []

    for run_spec in (
        build_realx_schedule(
            protocol
        )
    ):
        path = checkpoint_path(
            run_spec,
            output_root=output_root,
        )

        valid, reason = (
            validate_checkpoint(
                path,
                run_spec,
                protocol=protocol,
            )
        )

        if valid:
            state = "COMPLETE"

        elif path.exists():
            state = "INVALID"

        else:
            state = "MISSING"

        records.append(
            {
                "run_id": run_spec.run_id,
                "task_id": run_spec.task_id,
                "dataset_name": (
                    run_spec.dataset_name
                ),
                "feature_type": (
                    run_spec.feature_type
                ),
                "realization": (
                    run_spec.realization
                ),
                "strength_name": (
                    run_spec.strength_name
                ),
                "state": state,
                "validation_reason": reason,
                "checkpoint": str(
                    path
                ),
            }
        )

    return pd.DataFrame(
        records
    )


def refresh_aggregate_outputs(
    *,
    protocol: RealXProtocol | None = None,
    output_root: str | Path = (
        DEFAULT_OUTPUT_ROOT
    ),
) -> int:
    if protocol is None:
        protocol = (
            RealXProtocol()
        )

    output_root = Path(
        output_root
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    progress = (
        _all_schedule_rows(
            protocol=protocol,
            output_root=output_root,
        )
    )

    progress.to_csv(
        output_root
        / "benchmark_progress.csv",
        index=False,
    )

    complete_records = []

    schedule_lookup = {
        item.run_id: (
            position,
            item,
        )
        for (
            position,
            item,
        ) in enumerate(
            build_realx_schedule(
                protocol
            )
        )
    }

    for (
        run_id,
        (
            position,
            run_spec,
        ),
    ) in schedule_lookup.items():
        path = checkpoint_path(
            run_spec,
            output_root=output_root,
        )

        valid, _ = (
            validate_checkpoint(
                path,
                run_spec,
                protocol=protocol,
            )
        )

        if not valid:
            continue

        table = _read_protocol_csv(
            path
        )

        row = table.iloc[
            0
        ].to_dict()

        row[
            "_schedule_position"
        ] = position

        complete_records.append(
            row
        )

    if complete_records:
        master = pd.DataFrame(
            complete_records
        )

        master = (
            master
            .sort_values(
                "_schedule_position"
            )
            .drop(
                columns=[
                    "_schedule_position"
                ]
            )
            .reset_index(
                drop=True
            )
        )

    else:
        master = pd.DataFrame()

    master.to_csv(
        output_root
        / "master_results.csv",
        index=False,
    )

    return len(
        master
    )


# =============================================================================
# FAILURE LOG
# =============================================================================


def append_failure(
    *,
    run_spec: RealXRunSpec,
    exc: Exception,
    output_root: str | Path,
) -> None:
    output_root = Path(
        output_root
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    failure_path = (
        output_root
        / "failures.csv"
    )

    record = {
        "timestamp_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "run_id": run_spec.run_id,
        "task_id": run_spec.task_id,
        "dataset_name": (
            run_spec.dataset_name
        ),
        "realization": (
            run_spec.realization
        ),
        "strength_name": (
            run_spec.strength_name
        ),
        "exception_type": (
            type(
                exc
            ).__name__
        ),
        "message": str(
            exc
        ),
    }

    new_row = pd.DataFrame(
        [
            record
        ]
    )

    if failure_path.exists():
        previous = (
            _read_protocol_csv(
                failure_path
            )
        )

        combined = pd.concat(
            [
                previous,
                new_row,
            ],
            ignore_index=True,
        )

    else:
        combined = (
            new_row
        )

    combined.to_csv(
        failure_path,
        index=False,
    )


# =============================================================================
# BATCH EXECUTION
# =============================================================================


def run_realx_batch(
    specs: Iterable[
        RealXRunSpec
    ],
    *,
    protocol: RealXProtocol | None = None,
    output_root: str | Path = (
        DEFAULT_OUTPUT_ROOT
    ),
    dry_run: bool = False,
    force: bool = False,
    continue_on_error: bool = False,
    verbose: bool = True,
) -> BatchSummary:
    if protocol is None:
        protocol = (
            RealXProtocol()
        )

    output_root = Path(
        output_root
    )

    specs = tuple(
        specs
    )

    plan = build_batch_plan(
        specs,
        output_root=output_root,
        protocol=protocol,
        force=force,
    )

    planned_items = [
        item
        for item in plan
        if item.state
        in {
            "PLANNED",
            "PLANNED_INVALID",
        }
    ]

    skipped_existing = sum(
        item.state
        == "SKIP"
        for item in plan
    )

    invalid_existing = sum(
        item.state
        == "PLANNED_INVALID"
        for item in plan
    )

    if verbose:
        print(
            "=============================================="
        )

        print(
            "RESUMABLE REAL-X BENCHMARK"
        )

        print(
            "=============================================="
        )

        print(
            f"Selected: {len(plan)}"
        )

        print(
            f"Dry run: {dry_run}"
        )

        print(
            f"Validated existing: "
            f"{skipped_existing}"
        )

        print(
            f"Invalid existing: "
            f"{invalid_existing}"
        )

        print(
            f"Planned: "
            f"{len(planned_items)}"
        )

    if dry_run:
        if verbose:
            for (
                position,
                item,
            ) in enumerate(
                plan,
                start=1,
            ):
                print(
                    f"[{position}/{len(plan)}] "
                    f"{item.run_spec.run_id} "
                    f"-> {item.state} "
                    f"({item.reason})"
                )

            print()

            print(
                "DRY-RUN SUMMARY"
            )

            print(
                f"Selected: {len(plan)}"
            )

            print(
                f"Skipped existing: "
                f"{skipped_existing}"
            )

            print(
                f"Invalid existing: "
                f"{invalid_existing}"
            )

            print(
                f"Planned: "
                f"{len(planned_items)}"
            )

        return BatchSummary(
            selected=len(
                plan
            ),
            completed_now=0,
            skipped_existing=(
                skipped_existing
            ),
            invalid_existing=(
                invalid_existing
            ),
            failed=0,
            planned=len(
                planned_items
            ),
            total_complete=(
                skipped_existing
            ),
        )

    completed_now = 0
    failed = 0

    for (
        position,
        item,
    ) in enumerate(
        plan,
        start=1,
    ):
        run_spec = (
            item.run_spec
        )

        if item.state == "SKIP":
            if verbose:
                print(
                    f"\n[{position}/{len(plan)}] "
                    f"{run_spec.run_id}"
                )

                print(
                    "  status: SKIP "
                    "(validated checkpoint exists)"
                )

            continue

        if verbose:
            print(
                f"\n[{position}/{len(plan)}] "
                f"{run_spec.run_id}"
            )

        try:
            result = (
                run_realx_benchmark_spec(
                    run_spec,
                    realx_protocol=protocol,
                    verbose=verbose,
                )
            )

            write_checkpoint_atomic(
                result.to_record(),
                item.checkpoint_path,
            )

            valid, reason = (
                validate_checkpoint(
                    item.checkpoint_path,
                    run_spec,
                    protocol=protocol,
                )
            )

            if not valid:
                raise RuntimeError(
                    "Fresh checkpoint failed validation: "
                    f"{reason}"
                )

            completed_now += 1

            total_complete = (
                refresh_aggregate_outputs(
                    protocol=protocol,
                    output_root=output_root,
                )
            )

            if verbose:
                print(
                    "  checkpoint:",
                    item.checkpoint_path,
                )

                print(
                    "  total complete:",
                    total_complete,
                    "/ 180",
                )

        except Exception as exc:
            failed += 1

            append_failure(
                run_spec=run_spec,
                exc=exc,
                output_root=output_root,
            )

            if verbose:
                print(
                    "  status: FAILED"
                )

                print(
                    "  error:",
                    repr(
                        exc
                    ),
                )

            if not continue_on_error:
                raise

    total_complete = (
        refresh_aggregate_outputs(
            protocol=protocol,
            output_root=output_root,
        )
    )

    if verbose:
        print()

        print(
            "=============================================="
        )

        print(
            "REAL-X BATCH SUMMARY"
        )

        print(
            "=============================================="
        )

        print(
            f"Selected: {len(plan)}"
        )

        print(
            f"Completed now: "
            f"{completed_now}"
        )

        print(
            f"Skipped existing: "
            f"{skipped_existing}"
        )

        print(
            f"Invalid existing: "
            f"{invalid_existing}"
        )

        print(
            f"Failed: {failed}"
        )

        print(
            f"Total benchmark complete: "
            f"{total_complete}/180"
        )

    return BatchSummary(
        selected=len(
            plan
        ),
        completed_now=(
            completed_now
        ),
        skipped_existing=(
            skipped_existing
        ),
        invalid_existing=(
            invalid_existing
        ),
        failed=failed,
        planned=len(
            planned_items
        ),
        total_complete=(
            total_complete
        ),
    )