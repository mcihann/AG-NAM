from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import traceback

import numpy as np
import pandas as pd
import torch

from agnam.benchmarking.external_baselines import (
    ExternalBaselineProtocol,
    ExternalBaselineExecutionSpec,
    build_external_baseline_schedule,
)
from agnam.benchmarking.openml_protocol import (
    OPENML_TASKS,
    OpenMLBenchmarkProtocol,
    OpenMLExecutionSpec,
    build_openml_schedule,
)
from agnam.benchmarking.openml_runner import (
    SingleOpenMLBenchmarkResult,
    build_real_world_pair_audit,
    run_single_openml_benchmark,
)
from agnam.benchmarking.synthetic_runner import (
    SyntheticModelConfig,
)


_TASK_RECORD_PATTERN = re.compile(
    r"^task_(\d+)\.csv$"
)


_PAIR_AUDIT_COLUMNS = [
    "feature_j",
    "feature_k",
    "selection_count",
    "selection_frequency",
    "selection_stable",
    "isr_retained",
    "isr_score",
    "random_control",
    "single_run_top_k",
]


@dataclass(frozen=True)
class OpenMLBatchExecutionSummary:
    selected: int
    completed_now: int
    adopted_existing: int
    skipped_existing: int
    failed: int
    planned_only: int
    adoptable_planned: int


def task_record_path(
    results_root: str | Path,
    execution_spec: OpenMLExecutionSpec,
) -> Path:
    return (
        Path(
            results_root
        )
        / (
            f"task_"
            f"{execution_spec.task_id}.csv"
        )
    )


def task_pairs_path(
    results_root: str | Path,
    execution_spec: OpenMLExecutionSpec,
) -> Path:
    return (
        Path(
            results_root
        )
        / (
            f"task_"
            f"{execution_spec.task_id}"
            f"_pairs.csv"
        )
    )


def master_results_path(
    results_root: str | Path,
) -> Path:
    return (
        Path(
            results_root
        )
        / "master_results.csv"
    )


def progress_path(
    results_root: str | Path,
) -> Path:
    return (
        Path(
            results_root
        )
        / "benchmark_progress.csv"
    )


def failures_path(
    results_root: str | Path,
) -> Path:
    return (
        Path(
            results_root
        )
        / "failures.csv"
    )


def _execution_lookup(
    benchmark_protocol: (
        OpenMLBenchmarkProtocol
        | None
    ) = None,
) -> dict[
    int,
    OpenMLExecutionSpec,
]:
    if benchmark_protocol is None:
        benchmark_protocol = (
            OpenMLBenchmarkProtocol()
        )

    return {
        specification.task_id: (
            specification
        )
        for specification in (
            build_openml_schedule(
                benchmark_protocol
            )
        )
    }


def _external_lookup(
    baseline_protocol: (
        ExternalBaselineProtocol
        | None
    ) = None,
) -> dict[
    int,
    ExternalBaselineExecutionSpec,
]:
    if baseline_protocol is None:
        baseline_protocol = (
            ExternalBaselineProtocol()
        )

    return {
        specification.task_id: (
            specification
        )
        for specification in (
            build_external_baseline_schedule(
                baseline_protocol
            )
        )
    }


def select_openml_schedule(
    *,
    benchmark_protocol: (
        OpenMLBenchmarkProtocol
        | None
    ) = None,
    start: int = 1,
    end: int | None = None,
    task_id: int | None = None,
) -> tuple[
    OpenMLExecutionSpec,
    ...
]:
    """
    Select locked OpenML benchmark tasks.

    start/end refer to human-facing registry positions 1..28 and are
    inclusive.

    task_id selects exactly one locked OpenML task.
    """
    if benchmark_protocol is None:
        benchmark_protocol = (
            OpenMLBenchmarkProtocol()
        )

    schedule = (
        build_openml_schedule(
            benchmark_protocol
        )
    )

    if task_id is not None:
        matches = tuple(
            specification
            for specification
            in schedule
            if specification.task_id
            == task_id
        )

        if len(
            matches
        ) != 1:
            raise ValueError(
                f"Task {task_id} is not in the "
                "locked OpenML registry."
            )

        return matches

    if end is None:
        end = len(
            schedule
        )

    if start < 1:
        raise ValueError(
            "start must be at least 1."
        )

    if end < start:
        raise ValueError(
            "end must be >= start."
        )

    if end > len(
        schedule
    ):
        raise ValueError(
            "end exceeds the locked "
            "OpenML registry size."
        )

    return tuple(
        schedule[
            start - 1:
            end
        ]
    )


def _checkpoint_metadata(
    execution_spec: OpenMLExecutionSpec,
    external_spec: ExternalBaselineExecutionSpec,
    *,
    benchmark_protocol: OpenMLBenchmarkProtocol,
    baseline_protocol: ExternalBaselineProtocol,
) -> dict:
    """
    Metadata required to verify benchmark identity and seed integrity.
    """
    return {
        "task_index": (
            execution_spec.task_index
        ),
        "discovery_base_seed": (
            execution_spec
            .discovery_base_seed
        ),
        "discovery_seeds": (
            "|".join(
                str(
                    seed
                )
                for seed in (
                    execution_spec
                    .discovery_seeds
                )
            )
        ),
        "random_pair_seed": (
            execution_spec
            .random_pair_seed
        ),
        "final_split_seed": (
            execution_spec
            .final_split_seed
        ),
        "final_model_seed": (
            execution_spec
            .final_model_seed
        ),
        "ebm_seed": (
            external_spec.ebm_seed
        ),
        "catboost_seed": (
            external_spec
            .catboost_seed
        ),
        "openml_repeat": (
            benchmark_protocol.repeat
        ),
        "openml_fold": (
            benchmark_protocol.fold
        ),
        "openml_sample": (
            benchmark_protocol.sample
        ),
        "selection_threshold": (
            benchmark_protocol
            .selection_threshold
        ),
        "isr_threshold": (
            benchmark_protocol
            .isr_threshold
        ),
        "n_discovery_runs": (
            benchmark_protocol
            .n_discovery_runs
        ),
        "residual_crossfit_folds": (
            benchmark_protocol
            .residual_crossfit_folds
        ),
        "ebm_version": (
            baseline_protocol
            .ebm_version
        ),
        "catboost_version": (
            baseline_protocol
            .catboost_version
        ),
        "benchmark_protocol": (
            "agnam_openml_primary_v1"
        ),
    }


def enrich_checkpoint_frame(
    frame: pd.DataFrame,
    execution_spec: OpenMLExecutionSpec,
    external_spec: ExternalBaselineExecutionSpec,
    *,
    benchmark_protocol: OpenMLBenchmarkProtocol,
    baseline_protocol: ExternalBaselineProtocol,
) -> pd.DataFrame:
    """
    Add locked benchmark-identity metadata to one task result.
    """
    if len(
        frame
    ) != 1:
        raise ValueError(
            "A task checkpoint must contain "
            "exactly one row."
        )

    enriched = frame.copy()

    metadata = (
        _checkpoint_metadata(
            execution_spec,
            external_spec,
            benchmark_protocol=(
                benchmark_protocol
            ),
            baseline_protocol=(
                baseline_protocol
            ),
        )
    )

    for (
        column,
        value,
    ) in metadata.items():
        enriched[
            column
        ] = value

    return enriched


def validate_openml_checkpoint(
    path: str | Path,
    execution_spec: OpenMLExecutionSpec,
    external_spec: ExternalBaselineExecutionSpec,
    *,
    benchmark_protocol: (
        OpenMLBenchmarkProtocol
        | None
    ) = None,
    baseline_protocol: (
        ExternalBaselineProtocol
        | None
    ) = None,
) -> bool:
    """
    Validate an existing OpenML benchmark checkpoint.

    Missing checkpoint:
        False

    Existing but inconsistent checkpoint:
        raises RuntimeError
    """
    if benchmark_protocol is None:
        benchmark_protocol = (
            OpenMLBenchmarkProtocol()
        )

    if baseline_protocol is None:
        baseline_protocol = (
            ExternalBaselineProtocol()
        )

    path = Path(
        path
    )

    if not path.exists():
        return False

    frame = pd.read_csv(
        path
    )

    if len(
        frame
    ) != 1:
        raise RuntimeError(
            f"Checkpoint must contain exactly "
            f"one row: {path}"
        )

    row = frame.iloc[
        0
    ]

    required = (
        _checkpoint_metadata(
            execution_spec,
            external_spec,
            benchmark_protocol=(
                benchmark_protocol
            ),
            baseline_protocol=(
                baseline_protocol
            ),
        )
    )

    if "task_id" not in (
        frame.columns
    ):
        raise RuntimeError(
            f"Checkpoint missing task_id: {path}"
        )

    if int(
        row[
            "task_id"
        ]
    ) != execution_spec.task_id:
        raise RuntimeError(
            f"Checkpoint task ID mismatch: "
            f"{path}"
        )

    expected_integer_fields = {
        "task_index": (
            execution_spec.task_index
        ),
        "discovery_base_seed": (
            execution_spec
            .discovery_base_seed
        ),
        "random_pair_seed": (
            execution_spec
            .random_pair_seed
        ),
        "final_split_seed": (
            execution_spec
            .final_split_seed
        ),
        "final_model_seed": (
            execution_spec
            .final_model_seed
        ),
        "ebm_seed": (
            external_spec.ebm_seed
        ),
        "catboost_seed": (
            external_spec
            .catboost_seed
        ),
        "openml_repeat": (
            benchmark_protocol.repeat
        ),
        "openml_fold": (
            benchmark_protocol.fold
        ),
        "openml_sample": (
            benchmark_protocol.sample
        ),
        "n_discovery_runs": (
            benchmark_protocol
            .n_discovery_runs
        ),
        "residual_crossfit_folds": (
            benchmark_protocol
            .residual_crossfit_folds
        ),
    }

    for (
        column,
        expected,
    ) in (
        expected_integer_fields
        .items()
    ):
        if column not in (
            frame.columns
        ):
            raise RuntimeError(
                f"Checkpoint missing {column}: "
                f"{path}"
            )

        actual = int(
            row[
                column
            ]
        )

        if actual != expected:
            raise RuntimeError(
                f"Checkpoint mismatch for "
                f"{column}: expected "
                f"{expected}, found "
                f"{actual}. File: {path}"
            )

    expected_float_fields = {
        "selection_threshold": (
            benchmark_protocol
            .selection_threshold
        ),
        "isr_threshold": (
            benchmark_protocol
            .isr_threshold
        ),
    }

    for (
        column,
        expected,
    ) in (
        expected_float_fields
        .items()
    ):
        if column not in (
            frame.columns
        ):
            raise RuntimeError(
                f"Checkpoint missing {column}: "
                f"{path}"
            )

        actual = float(
            row[
                column
            ]
        )

        if not np.isclose(
            actual,
            expected,
            atol=1e-12,
            rtol=0.0,
        ):
            raise RuntimeError(
                f"Checkpoint mismatch for "
                f"{column}: expected "
                f"{expected}, found "
                f"{actual}. File: {path}"
            )

    expected_string_fields = {
        "discovery_seeds": (
            required[
                "discovery_seeds"
            ]
        ),
        "ebm_version": (
            baseline_protocol
            .ebm_version
        ),
        "catboost_version": (
            baseline_protocol
            .catboost_version
        ),
        "benchmark_protocol": (
            "agnam_openml_primary_v1"
        ),
    }

    for (
        column,
        expected,
    ) in (
        expected_string_fields
        .items()
    ):
        if column not in (
            frame.columns
        ):
            raise RuntimeError(
                f"Checkpoint missing {column}: "
                f"{path}"
            )

        actual = str(
            row[
                column
            ]
        )

        if actual != str(
            expected
        ):
            raise RuntimeError(
                f"Checkpoint mismatch for "
                f"{column}: expected "
                f"{expected}, found "
                f"{actual}. File: {path}"
            )

    return True


def _atomic_write_frame(
    frame: pd.DataFrame,
    path: str | Path,
) -> Path:
    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = (
        path.with_name(
            path.name
            + f".tmp-{os.getpid()}"
        )
    )

    try:
        frame.to_csv(
            temporary,
            index=False,
        )

        os.replace(
            temporary,
            path,
        )

    finally:
        if temporary.exists():
            temporary.unlink()

    return path


def _pair_audit_frame(
    result: SingleOpenMLBenchmarkResult,
) -> pd.DataFrame:
    frame = (
        build_real_world_pair_audit(
            result
        )
    )

    if len(
        frame.columns
    ) == 0:
        frame = pd.DataFrame(
            columns=(
                _PAIR_AUDIT_COLUMNS
            )
        )

    return frame


def write_openml_outputs(
    result: SingleOpenMLBenchmarkResult,
    *,
    results_root: str | Path,
    execution_spec: OpenMLExecutionSpec,
    external_spec: ExternalBaselineExecutionSpec,
    benchmark_protocol: OpenMLBenchmarkProtocol,
    baseline_protocol: ExternalBaselineProtocol,
) -> tuple[
    Path,
    Path,
]:
    """
    Atomically persist one completed OpenML task.

    The main task CSV is written last and acts as the completion
    checkpoint.
    """
    record_path = (
        task_record_path(
            results_root,
            execution_spec,
        )
    )

    pair_path = (
        task_pairs_path(
            results_root,
            execution_spec,
        )
    )

    if record_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing "
            f"checkpoint: {record_path}"
        )

    if pair_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing "
            f"pair audit: {pair_path}"
        )

    record_frame = (
        pd.DataFrame(
            [
                asdict(
                    result.record
                )
            ]
        )
    )

    record_frame = (
        enrich_checkpoint_frame(
            record_frame,
            execution_spec,
            external_spec,
            benchmark_protocol=(
                benchmark_protocol
            ),
            baseline_protocol=(
                baseline_protocol
            ),
        )
    )

    pair_frame = (
        _pair_audit_frame(
            result
        )
    )

    record_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    record_temporary = (
        record_path.with_name(
            record_path.name
            + f".tmp-{os.getpid()}"
        )
    )

    pair_temporary = (
        pair_path.with_name(
            pair_path.name
            + f".tmp-{os.getpid()}"
        )
    )

    pair_written = False

    try:
        record_frame.to_csv(
            record_temporary,
            index=False,
        )

        pair_frame.to_csv(
            pair_temporary,
            index=False,
        )

        os.replace(
            pair_temporary,
            pair_path,
        )

        pair_written = True

        os.replace(
            record_temporary,
            record_path,
        )

    except Exception:
        if (
            pair_written
            and pair_path.exists()
            and not record_path.exists()
        ):
            pair_path.unlink()

        raise

    finally:
        if record_temporary.exists():
            record_temporary.unlink()

        if pair_temporary.exists():
            pair_temporary.unlink()

    return (
        record_path,
        pair_path,
    )


def canonical_checkpoint_available(
    canonical_root: str | Path,
    execution_spec: OpenMLExecutionSpec,
) -> bool:
    root = Path(
        canonical_root
    )

    record_path = (
        root
        / (
            f"task_"
            f"{execution_spec.task_id}.csv"
        )
    )

    pair_path = (
        root
        / (
            f"task_"
            f"{execution_spec.task_id}"
            f"_pairs.csv"
        )
    )

    return (
        record_path.exists()
        and pair_path.exists()
    )


def adopt_canonical_checkpoint(
    *,
    canonical_root: str | Path,
    results_root: str | Path,
    execution_spec: OpenMLExecutionSpec,
    external_spec: ExternalBaselineExecutionSpec,
    benchmark_protocol: OpenMLBenchmarkProtocol,
    baseline_protocol: ExternalBaselineProtocol,
) -> bool:
    """
    Adopt a previously completed canonical single-task result into the
    resumable benchmark without retraining the model.

    The record is enriched with locked seed/protocol metadata.
    """
    canonical_root = Path(
        canonical_root
    )

    target_record = (
        task_record_path(
            results_root,
            execution_spec,
        )
    )

    target_pairs = (
        task_pairs_path(
            results_root,
            execution_spec,
        )
    )

    if target_record.exists():
        return False

    source_record = (
        canonical_root
        / (
            f"task_"
            f"{execution_spec.task_id}.csv"
        )
    )

    source_pairs = (
        canonical_root
        / (
            f"task_"
            f"{execution_spec.task_id}"
            f"_pairs.csv"
        )
    )

    if not (
        source_record.exists()
        and source_pairs.exists()
    ):
        return False

    record_frame = pd.read_csv(
        source_record
    )

    if len(
        record_frame
    ) != 1:
        raise RuntimeError(
            "Canonical checkpoint must contain "
            "exactly one result row."
        )

    row = (
        record_frame.iloc[
            0
        ]
    )

    if int(
        row[
            "task_id"
        ]
    ) != execution_spec.task_id:
        raise RuntimeError(
            "Canonical task ID does not match "
            "the locked task."
        )

    task_spec = (
        OPENML_TASKS[
            execution_spec
            .task_index
        ]
    )

    if (
        "n_samples"
        in record_frame.columns
        and int(
            row[
                "n_samples"
            ]
        )
        != task_spec.expected_n_samples
    ):
        raise RuntimeError(
            "Canonical checkpoint sample count "
            "does not match the locked registry."
        )

    if (
        "n_features"
        in record_frame.columns
        and int(
            row[
                "n_features"
            ]
        )
        != task_spec.expected_n_predictors
    ):
        raise RuntimeError(
            "Canonical checkpoint feature count "
            "does not match the locked registry."
        )

    record_frame = (
        enrich_checkpoint_frame(
            record_frame,
            execution_spec,
            external_spec,
            benchmark_protocol=(
                benchmark_protocol
            ),
            baseline_protocol=(
                baseline_protocol
            ),
        )
    )

    try:
        pair_frame = (
            pd.read_csv(
                source_pairs
            )
        )

    except pd.errors.EmptyDataError:
        pair_frame = (
            pd.DataFrame(
                columns=(
                    _PAIR_AUDIT_COLUMNS
                )
            )
        )

    target_record.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if target_pairs.exists():
        raise FileExistsError(
            f"Target pair audit already exists: "
            f"{target_pairs}"
        )

    record_temporary = (
        target_record.with_name(
            target_record.name
            + f".tmp-{os.getpid()}"
        )
    )

    pair_temporary = (
        target_pairs.with_name(
            target_pairs.name
            + f".tmp-{os.getpid()}"
        )
    )

    pair_written = False

    try:
        record_frame.to_csv(
            record_temporary,
            index=False,
        )

        pair_frame.to_csv(
            pair_temporary,
            index=False,
        )

        os.replace(
            pair_temporary,
            target_pairs,
        )

        pair_written = True

        os.replace(
            record_temporary,
            target_record,
        )

    except Exception:
        if (
            pair_written
            and target_pairs.exists()
            and not target_record.exists()
        ):
            target_pairs.unlink()

        raise

    finally:
        if record_temporary.exists():
            record_temporary.unlink()

        if pair_temporary.exists():
            pair_temporary.unlink()

    return True


def _primary_task_files(
    results_root: str | Path,
) -> list[
    Path
]:
    root = Path(
        results_root
    )

    if not root.exists():
        return []

    files = []

    for path in (
        root.glob(
            "task_*.csv"
        )
    ):
        if (
            _TASK_RECORD_PATTERN
            .match(
                path.name
            )
        ):
            files.append(
                path
            )

    return files


def rebuild_openml_master(
    results_root: str | Path,
) -> pd.DataFrame:
    """
    Rebuild master_results.csv only from immutable task checkpoints.
    """
    frames = []

    for path in (
        _primary_task_files(
            results_root
        )
    ):
        frame = pd.read_csv(
            path
        )

        if len(
            frame
        ) != 1:
            raise RuntimeError(
                f"Invalid task checkpoint: "
                f"{path}"
            )

        frames.append(
            frame
        )

    if len(
        frames
    ) == 0:
        master = pd.DataFrame()

    else:
        master = pd.concat(
            frames,
            ignore_index=True,
        )

        master = (
            master
            .sort_values(
                by=[
                    "task_index"
                ]
            )
            .reset_index(
                drop=True
            )
        )

    _atomic_write_frame(
        master,
        master_results_path(
            results_root
        ),
    )

    return master


def rebuild_progress_table(
    results_root: str | Path,
    *,
    benchmark_protocol: (
        OpenMLBenchmarkProtocol
        | None
    ) = None,
    baseline_protocol: (
        ExternalBaselineProtocol
        | None
    ) = None,
) -> pd.DataFrame:
    if benchmark_protocol is None:
        benchmark_protocol = (
            OpenMLBenchmarkProtocol()
        )

    if baseline_protocol is None:
        baseline_protocol = (
            ExternalBaselineProtocol()
        )

    external_lookup = (
        _external_lookup(
            baseline_protocol
        )
    )

    rows = []

    for specification in (
        build_openml_schedule(
            benchmark_protocol
        )
    ):
        path = (
            task_record_path(
                results_root,
                specification,
            )
        )

        completed = False

        if path.exists():
            completed = (
                validate_openml_checkpoint(
                    path,
                    specification,
                    external_lookup[
                        specification
                        .task_id
                    ],
                    benchmark_protocol=(
                        benchmark_protocol
                    ),
                    baseline_protocol=(
                        baseline_protocol
                    ),
                )
            )

        rows.append(
            {
                "registry_position": (
                    specification
                    .task_index
                    + 1
                ),
                "task_id": (
                    specification
                    .task_id
                ),
                "dataset_name": (
                    specification
                    .dataset_name
                ),
                "complete": bool(
                    completed
                ),
                "checkpoint": (
                    str(
                        path
                    )
                    if completed
                    else ""
                ),
            }
        )

    frame = pd.DataFrame(
        rows
    )

    _atomic_write_frame(
        frame,
        progress_path(
            results_root
        ),
    )

    return frame


def record_failure(
    results_root: str | Path,
    execution_spec: OpenMLExecutionSpec,
    exception: Exception,
) -> Path:
    path = failures_path(
        results_root
    )

    columns = [
        "task_id",
        "dataset_name",
        "task_index",
        "timestamp_utc",
        "exception_type",
        "error_message",
        "traceback",
    ]

    if path.exists():
        frame = pd.read_csv(
            path
        )

    else:
        frame = pd.DataFrame(
            columns=columns
        )

    if len(
        frame
    ) > 0:
        keep = (
            pd.to_numeric(
                frame[
                    "task_id"
                ],
                errors="coerce",
            )
            != execution_spec.task_id
        )

        frame = frame[
            keep
        ].copy()

    new_row = {
        "task_id": (
            execution_spec.task_id
        ),
        "dataset_name": (
            execution_spec
            .dataset_name
        ),
        "task_index": (
            execution_spec.task_index
        ),
        "timestamp_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
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

    frame = pd.concat(
        [
            frame,
            pd.DataFrame(
                [
                    new_row
                ]
            ),
        ],
        ignore_index=True,
    )

    _atomic_write_frame(
        frame,
        path,
    )

    return path


def clear_failure(
    results_root: str | Path,
    execution_spec: OpenMLExecutionSpec,
) -> None:
    path = failures_path(
        results_root
    )

    if not path.exists():
        return

    frame = pd.read_csv(
        path
    )

    if len(
        frame
    ) == 0:
        return

    keep = (
        pd.to_numeric(
            frame[
                "task_id"
            ],
            errors="coerce",
        )
        != execution_spec.task_id
    )

    updated = frame[
        keep
    ].copy()

    _atomic_write_frame(
        updated,
        path,
    )


def run_openml_batch(
    *,
    benchmark_protocol: (
        OpenMLBenchmarkProtocol
        | None
    ) = None,
    baseline_protocol: (
        ExternalBaselineProtocol
        | None
    ) = None,
    model_config: (
        SyntheticModelConfig
        | None
    ) = None,
    start: int = 1,
    end: int | None = None,
    task_id: int | None = None,
    results_root: str | Path = (
        "results/openml/benchmark"
    ),
    canonical_root: str | Path = (
        "results/openml/single"
    ),
    device: (
        torch.device
        | None
    ) = None,
    dry_run: bool = False,
    stop_on_error: bool = False,
    verbose_tasks: bool = True,
) -> OpenMLBatchExecutionSummary:
    """
    Execute a resumable subset of the locked 28-task benchmark.
    """
    if benchmark_protocol is None:
        benchmark_protocol = (
            OpenMLBenchmarkProtocol()
        )

    if baseline_protocol is None:
        baseline_protocol = (
            ExternalBaselineProtocol()
        )

    if model_config is None:
        model_config = (
            SyntheticModelConfig()
        )

    selected = (
        select_openml_schedule(
            benchmark_protocol=(
                benchmark_protocol
            ),
            start=(
                start
            ),
            end=(
                end
            ),
            task_id=(
                task_id
            ),
        )
    )

    external_lookup = (
        _external_lookup(
            baseline_protocol
        )
    )

    results_root = Path(
        results_root
    )

    canonical_root = Path(
        canonical_root
    )

    results_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    completed_now = 0
    adopted_existing = 0
    skipped_existing = 0
    failed = 0
    planned_only = 0
    adoptable_planned = 0

    print(
        "========================================"
    )

    print(
        "RESUMABLE OPENML BENCHMARK"
    )

    print(
        "========================================"
    )

    print(
        f"Selected tasks: "
        f"{len(selected)}"
    )

    print(
        f"Dry run: "
        f"{dry_run}"
    )

    for position, specification in enumerate(
        selected,
        start=1,
    ):
        external_spec = (
            external_lookup[
                specification
                .task_id
            ]
        )

        record_path = (
            task_record_path(
                results_root,
                specification,
            )
        )

        pair_path = (
            task_pairs_path(
                results_root,
                specification,
            )
        )

        print(
            f"\n[{position}/"
            f"{len(selected)}] "
            f"Task "
            f"{specification.task_id} — "
            f"{specification.dataset_name}"
        )

        try:
            if record_path.exists():
                validate_openml_checkpoint(
                    record_path,
                    specification,
                    external_spec,
                    benchmark_protocol=(
                        benchmark_protocol
                    ),
                    baseline_protocol=(
                        baseline_protocol
                    ),
                )

                skipped_existing += 1

                print(
                    "  status: SKIP "
                    "(validated checkpoint exists)"
                )

                continue

            canonical_available = (
                canonical_checkpoint_available(
                    canonical_root,
                    specification,
                )
            )

            if dry_run:
                if canonical_available:
                    adoptable_planned += 1

                    print(
                        "  status: ADOPT "
                        "(canonical result available)"
                    )

                else:
                    planned_only += 1

                    print(
                        "  status: PLAN"
                    )

                continue

            if canonical_available:
                adopted = (
                    adopt_canonical_checkpoint(
                        canonical_root=(
                            canonical_root
                        ),
                        results_root=(
                            results_root
                        ),
                        execution_spec=(
                            specification
                        ),
                        external_spec=(
                            external_spec
                        ),
                        benchmark_protocol=(
                            benchmark_protocol
                        ),
                        baseline_protocol=(
                            baseline_protocol
                        ),
                    )
                )

                if adopted:
                    validate_openml_checkpoint(
                        record_path,
                        specification,
                        external_spec,
                        benchmark_protocol=(
                            benchmark_protocol
                        ),
                        baseline_protocol=(
                            baseline_protocol
                        ),
                    )

                    adopted_existing += 1

                    print(
                        "  status: ADOPTED"
                    )

                    clear_failure(
                        results_root,
                        specification,
                    )

                    rebuild_openml_master(
                        results_root
                    )

                    rebuild_progress_table(
                        results_root,
                        benchmark_protocol=(
                            benchmark_protocol
                        ),
                        baseline_protocol=(
                            baseline_protocol
                        ),
                    )

                    continue

            if pair_path.exists():
                raise RuntimeError(
                    "Incomplete prior task output "
                    "detected: pair audit exists "
                    "without primary checkpoint."
                )

            if device is None:
                from agnam.utils.reproducibility import (
                    get_device,
                )

                current_device = (
                    get_device()
                )

            else:
                current_device = (
                    device
                )

            print(
                "  status: RUN"
            )

            print(
                f"  device: "
                f"{current_device}"
            )

            result = (
                run_single_openml_benchmark(
                    task_id=(
                        specification
                        .task_id
                    ),
                    benchmark_protocol=(
                        benchmark_protocol
                    ),
                    baseline_protocol=(
                        baseline_protocol
                    ),
                    model_config=(
                        model_config
                    ),
                    device=(
                        current_device
                    ),
                    verbose=(
                        verbose_tasks
                    ),
                )
            )

            write_openml_outputs(
                result,
                results_root=(
                    results_root
                ),
                execution_spec=(
                    specification
                ),
                external_spec=(
                    external_spec
                ),
                benchmark_protocol=(
                    benchmark_protocol
                ),
                baseline_protocol=(
                    baseline_protocol
                ),
            )

            clear_failure(
                results_root,
                specification,
            )

            completed_now += 1

            print(
                "  status: COMPLETE"
            )

            print(
                f"  checkpoint: "
                f"{record_path}"
            )

            rebuild_openml_master(
                results_root
            )

            rebuild_progress_table(
                results_root,
                benchmark_protocol=(
                    benchmark_protocol
                ),
                baseline_protocol=(
                    baseline_protocol
                ),
            )

        except Exception as exception:
            failed += 1

            print(
                "  status: FAILED"
            )

            print(
                f"  "
                f"{exception.__class__.__name__}: "
                f"{exception}"
            )

            if not dry_run:
                record_failure(
                    results_root,
                    specification,
                    exception,
                )

            if stop_on_error:
                raise

    if not dry_run:
        master = (
            rebuild_openml_master(
                results_root
            )
        )

        progress = (
            rebuild_progress_table(
                results_root,
                benchmark_protocol=(
                    benchmark_protocol
                ),
                baseline_protocol=(
                    baseline_protocol
                ),
            )
        )

        total_complete = int(
            progress[
                "complete"
            ].sum()
        )

    else:
        master = pd.DataFrame()
        total_complete = 0

    summary = (
        OpenMLBatchExecutionSummary(
            selected=len(
                selected
            ),
            completed_now=(
                completed_now
            ),
            adopted_existing=(
                adopted_existing
            ),
            skipped_existing=(
                skipped_existing
            ),
            failed=(
                failed
            ),
            planned_only=(
                planned_only
            ),
            adoptable_planned=(
                adoptable_planned
            ),
        )
    )

    print(
        "\n========================================"
    )

    print(
        "OPENML BATCH SUMMARY"
    )

    print(
        "========================================"
    )

    print(
        f"Selected: "
        f"{summary.selected}"
    )

    print(
        f"Completed now: "
        f"{summary.completed_now}"
    )

    print(
        f"Adopted existing: "
        f"{summary.adopted_existing}"
    )

    print(
        f"Skipped existing: "
        f"{summary.skipped_existing}"
    )

    print(
        f"Failed: "
        f"{summary.failed}"
    )

    if dry_run:
        print(
            f"Adoptable: "
            f"{summary.adoptable_planned}"
        )

        print(
            f"Planned: "
            f"{summary.planned_only}"
        )

    else:
        print(
            f"Total benchmark complete: "
            f"{total_complete}/28"
        )

        print(
            f"Master results: "
            f"{master_results_path(results_root)}"
        )

        print(
            f"Progress table: "
            f"{progress_path(results_root)}"
        )

        print(
            f"Failure log: "
            f"{failures_path(results_root)}"
        )

    return summary