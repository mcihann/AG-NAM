from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import traceback
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from agnam.benchmarking.synthetic_protocol import (
    SYNTHETIC_SCENARIOS,
    SyntheticBenchmarkProtocol,
    SyntheticRealizationSpec,
    build_synthetic_schedule,
)
from agnam.benchmarking.synthetic_runner import (
    SingleSyntheticBenchmarkResult,
    SyntheticModelConfig,
    run_single_synthetic_benchmark,
)


_REALIZATION_PATTERN = re.compile(
    r"^realization_(\d{2})\.csv$"
)


SUMMARY_METRICS = (
    "mean_pairwise_jaccard",
    "mean_interaction_auprc",
    "selection_precision",
    "selection_recall",
    "selection_f1",
    "isr_precision",
    "isr_recall",
    "isr_f1",
    "false_positive_reduction",
    "n_selection_stable",
    "n_isr_retained",
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
    "runtime_seconds",
)


@dataclass(frozen=True)
class BatchExecutionSummary:
    """
    Summary of one batch-run invocation.
    """

    selected: int
    completed_now: int
    skipped_existing: int
    failed: int
    planned_only: int


def realization_record_path(
    results_root: str | Path,
    spec: SyntheticRealizationSpec,
) -> Path:
    root = Path(
        results_root
    )

    return (
        root
        / spec.scenario
        / (
            f"realization_"
            f"{spec.realization_number:02d}.csv"
        )
    )


def realization_pairs_path(
    results_root: str | Path,
    spec: SyntheticRealizationSpec,
) -> Path:
    root = Path(
        results_root
    )

    return (
        root
        / spec.scenario
        / (
            f"realization_"
            f"{spec.realization_number:02d}"
            f"_pairs.csv"
        )
    )


def master_results_path(
    results_root: str | Path,
) -> Path:
    return (
        Path(results_root)
        / "master_results.csv"
    )


def scenario_summary_path(
    results_root: str | Path,
) -> Path:
    return (
        Path(results_root)
        / "scenario_summary.csv"
    )


def failures_path(
    results_root: str | Path,
) -> Path:
    return (
        Path(results_root)
        / "failures.csv"
    )


def select_schedule(
    *,
    protocol: SyntheticBenchmarkProtocol | None = None,
    scenario: str = "ALL",
    start: int = 1,
    end: int | None = None,
) -> tuple[
    SyntheticRealizationSpec,
    ...
]:
    """
    Select a deterministic subset of the primary benchmark schedule.

    start/end use human-facing realization numbers and are inclusive.

    Examples
    --------
    scenario="S1", start=1, end=20
        -> all 20 S1 realizations

    scenario="ALL", start=1, end=2
        -> realizations 1 and 2 for each of S1-S4
    """
    if protocol is None:
        protocol = (
            SyntheticBenchmarkProtocol()
        )

    scenario = scenario.upper()

    valid_scenarios = {
        *protocol.scenarios,
        "ALL",
    }

    if scenario not in valid_scenarios:
        raise ValueError(
            f"scenario must be one of "
            f"{sorted(valid_scenarios)}."
        )

    if end is None:
        end = (
            protocol.n_realizations
        )

    if start < 1:
        raise ValueError(
            "start must be at least 1."
        )

    if end < start:
        raise ValueError(
            "end must be >= start."
        )

    if end > protocol.n_realizations:
        raise ValueError(
            "end exceeds the number of "
            "realizations in the protocol."
        )

    schedule = (
        build_synthetic_schedule(
            protocol
        )
    )

    selected = tuple(
        spec
        for spec in schedule
        if (
            (
                scenario == "ALL"
                or spec.scenario
                == scenario
            )
            and start
            <= spec.realization_number
            <= end
        )
    )

    return selected


def validate_completed_realization(
    path: str | Path,
    spec: SyntheticRealizationSpec,
) -> bool:
    """
    Validate an existing realization checkpoint.

    Returns False when no checkpoint exists.

    If a checkpoint exists but its identifying fields disagree with
    the locked schedule, an exception is raised rather than silently
    overwriting the file.
    """
    path = Path(
        path
    )

    if not path.exists():
        return False

    frame = pd.read_csv(
        path
    )

    if len(frame) != 1:
        raise RuntimeError(
            f"Existing realization file must contain exactly "
            f"one row: {path}"
        )

    row = frame.iloc[
        0
    ]

    expected_strings = {
        "scenario": (
            spec.scenario
        ),
    }

    expected_integers = {
        "realization_index": (
            spec.realization_index
        ),
        "dataset_seed": (
            spec.dataset_seed
        ),
        "outer_split_seed": (
            spec.outer_split_seed
        ),
        "final_split_seed": (
            spec.final_split_seed
        ),
        "discovery_base_seed": (
            spec.discovery_base_seed
        ),
        "random_pair_seed": (
            spec.random_pair_seed
        ),
        "final_model_seed": (
            spec.final_model_seed
        ),
    }

    for field, expected in (
        expected_strings.items()
    ):
        if field not in frame.columns:
            raise RuntimeError(
                f"Existing checkpoint is missing "
                f"required field '{field}': {path}"
            )

        actual = str(
            row[field]
        )

        if actual != expected:
            raise RuntimeError(
                f"Checkpoint identity mismatch for "
                f"{field}: expected {expected}, "
                f"found {actual}. File: {path}"
            )

    for field, expected in (
        expected_integers.items()
    ):
        if field not in frame.columns:
            raise RuntimeError(
                f"Existing checkpoint is missing "
                f"required field '{field}': {path}"
            )

        actual = int(
            row[field]
        )

        if actual != expected:
            raise RuntimeError(
                f"Checkpoint identity mismatch for "
                f"{field}: expected {expected}, "
                f"found {actual}. File: {path}"
            )

    return True


def build_pair_audit_frame(
    result: SingleSyntheticBenchmarkResult,
) -> pd.DataFrame:
    """
    Build the per-realization pair-membership audit table.
    """
    true_pairs = set(
        result.true_pairs
    )

    selection_pairs = set(
        result.selection_pairs
    )

    isr_pairs = set(
        result.isr_pairs
    )

    random_pairs = set(
        result.random_pairs
    )

    single_run_pairs = set(
        result.single_run_pairs
    )

    all_pairs = set()

    all_pairs.update(
        true_pairs
    )

    all_pairs.update(
        selection_pairs
    )

    all_pairs.update(
        isr_pairs
    )

    all_pairs.update(
        random_pairs
    )

    all_pairs.update(
        single_run_pairs
    )

    rows = []

    for pair in sorted(
        all_pairs
    ):
        rows.append(
            {
                "feature_j": (
                    pair[0]
                ),
                "feature_k": (
                    pair[1]
                ),
                "is_true": (
                    pair
                    in true_pairs
                ),
                "oracle_control": (
                    pair
                    in true_pairs
                ),
                "selection_stable": (
                    pair
                    in selection_pairs
                ),
                "no_isr_control": (
                    pair
                    in selection_pairs
                ),
                "isr_retained": (
                    pair
                    in isr_pairs
                ),
                "random_control": (
                    pair
                    in random_pairs
                ),
                "single_run_top_k": (
                    pair
                    in single_run_pairs
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def _atomic_write_frame(
    frame: pd.DataFrame,
    path: str | Path,
) -> Path:
    """
    Atomically replace a derived CSV file.

    Used for master, summary and failure logs. Individual realization
    checkpoints use a separate no-overwrite writer.
    """
    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(
        path.name
        + f".tmp-{os.getpid()}"
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


def write_realization_outputs(
    result: SingleSyntheticBenchmarkResult,
    *,
    results_root: str | Path,
    spec: SyntheticRealizationSpec,
) -> tuple[
    Path,
    Path,
]:
    """
    Write one realization checkpoint and its pair audit.

    Existing realization outputs are never overwritten.

    The main realization CSV is written last and therefore acts as
    the primary completion checkpoint.
    """
    record_path = (
        realization_record_path(
            results_root,
            spec,
        )
    )

    pair_path = (
        realization_pairs_path(
            results_root,
            spec,
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

    record_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    record_frame = pd.DataFrame(
        [
            asdict(
                result.record
            )
        ]
    )

    pair_frame = (
        build_pair_audit_frame(
            result
        )
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

        # Pair audit first.
        os.replace(
            pair_temporary,
            pair_path,
        )

        pair_written = True

        # Main checkpoint last.
        os.replace(
            record_temporary,
            record_path,
        )

    except Exception:
        # If this call created a pair audit but failed before the
        # main completion checkpoint, remove that incomplete output
        # so the realization can be rerun cleanly.
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


def _realization_record_files(
    results_root: str | Path,
) -> list[Path]:
    """
    Return only primary realization CSV files.

    Pair-audit CSV files are intentionally excluded.
    """
    root = Path(
        results_root
    )

    files = []

    for scenario in (
        SYNTHETIC_SCENARIOS
    ):
        scenario_dir = (
            root
            / scenario
        )

        if not scenario_dir.exists():
            continue

        for path in (
            scenario_dir.glob(
                "realization_*.csv"
            )
        ):
            if _REALIZATION_PATTERN.match(
                path.name
            ):
                files.append(
                    path
                )

    return files


def rebuild_master_table(
    results_root: str | Path,
) -> pd.DataFrame:
    """
    Rebuild master_results.csv entirely from completed realization
    checkpoint files.
    """
    files = (
        _realization_record_files(
            results_root
        )
    )

    frames = []

    for path in files:
        frame = pd.read_csv(
            path
        )

        if len(frame) != 1:
            raise RuntimeError(
                f"Invalid realization checkpoint: {path}"
            )

        frames.append(
            frame
        )

    if len(frames) == 0:
        master = pd.DataFrame()

    else:
        master = pd.concat(
            frames,
            ignore_index=True,
        )

        scenario_order = {
            scenario: index
            for index, scenario
            in enumerate(
                SYNTHETIC_SCENARIOS
            )
        }

        master[
            "_scenario_order"
        ] = (
            master[
                "scenario"
            ]
            .map(
                scenario_order
            )
        )

        master = (
            master
            .sort_values(
                by=[
                    "_scenario_order",
                    "realization_index",
                ]
            )
            .drop(
                columns=[
                    "_scenario_order"
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


def summarize_master_table(
    master: pd.DataFrame,
) -> pd.DataFrame:
    """
    Produce long-form scenario-level descriptive statistics.

    The primary benchmark requires at least:
        mean
        standard deviation
        median
        interquartile range

    q25 and q75 are stored explicitly.
    """
    columns = [
        "scenario",
        "metric",
        "n",
        "mean",
        "std",
        "median",
        "q25",
        "q75",
        "iqr",
    ]

    if len(master) == 0:
        return pd.DataFrame(
            columns=columns
        )

    rows = []

    for scenario in (
        SYNTHETIC_SCENARIOS
    ):
        scenario_frame = master[
            master[
                "scenario"
            ]
            == scenario
        ]

        if len(
            scenario_frame
        ) == 0:
            continue

        for metric in (
            SUMMARY_METRICS
        ):
            if metric not in (
                scenario_frame.columns
            ):
                continue

            values = pd.to_numeric(
                scenario_frame[
                    metric
                ],
                errors="coerce",
            ).dropna()

            if len(values) == 0:
                continue

            q25 = float(
                values.quantile(
                    0.25
                )
            )

            q75 = float(
                values.quantile(
                    0.75
                )
            )

            rows.append(
                {
                    "scenario": (
                        scenario
                    ),
                    "metric": (
                        metric
                    ),
                    "n": int(
                        len(values)
                    ),
                    "mean": float(
                        values.mean()
                    ),
                    "std": (
                        float(
                            values.std(
                                ddof=1
                            )
                        )
                        if len(values) > 1
                        else np.nan
                    ),
                    "median": float(
                        values.median()
                    ),
                    "q25": q25,
                    "q75": q75,
                    "iqr": float(
                        q75
                        - q25
                    ),
                }
            )

    return pd.DataFrame(
        rows,
        columns=columns,
    )


def rebuild_scenario_summary(
    results_root: str | Path,
    *,
    master: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Rebuild scenario_summary.csv.
    """
    if master is None:
        master = (
            rebuild_master_table(
                results_root
            )
        )

    summary = (
        summarize_master_table(
            master
        )
    )

    _atomic_write_frame(
        summary,
        scenario_summary_path(
            results_root
        ),
    )

    return summary


def record_failure(
    results_root: str | Path,
    spec: SyntheticRealizationSpec,
    exception: Exception,
) -> Path:
    """
    Record one failed realization.

    A later failure for the same realization replaces its previous
    failure entry so failures.csv contains the most recent state.
    """
    path = failures_path(
        results_root
    )

    if path.exists():
        frame = pd.read_csv(
            path
        )

    else:
        frame = pd.DataFrame(
            columns=[
                "scenario",
                "realization_index",
                "realization_number",
                "dataset_seed",
                "outer_split_seed",
                "final_split_seed",
                "discovery_base_seed",
                "random_pair_seed",
                "final_model_seed",
                "timestamp_utc",
                "exception_type",
                "error_message",
                "traceback",
            ]
        )

    if len(frame) > 0:
        keep = ~(
            (
                frame[
                    "scenario"
                ]
                == spec.scenario
            )
            & (
                pd.to_numeric(
                    frame[
                        "realization_index"
                    ],
                    errors="coerce",
                )
                == spec.realization_index
            )
        )

        frame = frame[
            keep
        ].copy()

    new_row = {
        "scenario": (
            spec.scenario
        ),
        "realization_index": (
            spec.realization_index
        ),
        "realization_number": (
            spec.realization_number
        ),
        "dataset_seed": (
            spec.dataset_seed
        ),
        "outer_split_seed": (
            spec.outer_split_seed
        ),
        "final_split_seed": (
            spec.final_split_seed
        ),
        "discovery_base_seed": (
            spec.discovery_base_seed
        ),
        "random_pair_seed": (
            spec.random_pair_seed
        ),
        "final_model_seed": (
            spec.final_model_seed
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
        "error_message": (
            str(
                exception
            )
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
    spec: SyntheticRealizationSpec,
) -> None:
    """
    Remove a stale failure entry after successful completion.
    """
    path = failures_path(
        results_root
    )

    if not path.exists():
        return

    frame = pd.read_csv(
        path
    )

    if len(frame) == 0:
        return

    keep = ~(
        (
            frame[
                "scenario"
            ]
            == spec.scenario
        )
        & (
            pd.to_numeric(
                frame[
                    "realization_index"
                ],
                errors="coerce",
            )
            == spec.realization_index
        )
    )

    updated = frame[
        keep
    ].copy()

    _atomic_write_frame(
        updated,
        path,
    )


def run_synthetic_batch(
    *,
    protocol: SyntheticBenchmarkProtocol | None = None,
    model_config: SyntheticModelConfig | None = None,
    scenario: str = "ALL",
    start: int = 1,
    end: int | None = None,
    results_root: str | Path = (
        "results/synthetic"
    ),
    device: torch.device | None = None,
    dry_run: bool = False,
    stop_on_error: bool = False,
    verbose_realizations: bool = True,
) -> BatchExecutionSummary:
    """
    Execute a resumable subset of the locked synthetic benchmark.

    Completed realization checkpoints are validated and skipped.

    Failed realizations are logged to failures.csv and execution
    continues unless stop_on_error=True.

    Master and scenario-summary tables are rebuilt after every
    successful realization.
    """
    if protocol is None:
        protocol = (
            SyntheticBenchmarkProtocol()
        )

    if model_config is None:
        model_config = (
            SyntheticModelConfig()
        )

    selected_specs = (
        select_schedule(
            protocol=protocol,
            scenario=scenario,
            start=start,
            end=end,
        )
    )

    root = Path(
        results_root
    )

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    completed_now = 0
    skipped_existing = 0
    failed = 0
    planned_only = 0

    print(
        "========================================"
    )

    print(
        "RESUMABLE SYNTHETIC BENCHMARK"
    )

    print(
        "========================================"
    )

    print(
        f"Selected realizations: "
        f"{len(selected_specs)}"
    )

    print(
        f"Scenario filter: "
        f"{scenario.upper()}"
    )

    print(
        f"Realization range: "
        f"{start}.."
        f"{end if end is not None else protocol.n_realizations}"
    )

    print(
        f"Dry run: "
        f"{dry_run}"
    )

    for position, spec in enumerate(
        selected_specs,
        start=1,
    ):
        record_path = (
            realization_record_path(
                root,
                spec,
            )
        )

        pair_path = (
            realization_pairs_path(
                root,
                spec,
            )
        )

        label = (
            f"{spec.scenario} "
            f"R{spec.realization_number:02d}"
        )

        print(
            f"\n[{position}/"
            f"{len(selected_specs)}] "
            f"{label}"
        )

        try:
            completed = (
                validate_completed_realization(
                    record_path,
                    spec,
                )
            )

            if completed:
                skipped_existing += 1

                print(
                    "  status: SKIP "
                    "(validated checkpoint exists)"
                )

                continue

            if pair_path.exists():
                raise RuntimeError(
                    "Incomplete prior output detected: "
                    f"pair audit exists without primary "
                    f"checkpoint: {pair_path}"
                )

            if dry_run:
                planned_only += 1

                print(
                    "  status: PLAN"
                )

                print(
                    f"  dataset_seed="
                    f"{spec.dataset_seed}"
                )

                print(
                    f"  outer_split_seed="
                    f"{spec.outer_split_seed}"
                )

                print(
                    f"  discovery_seeds="
                    f"{spec.discovery_seeds}"
                )

                continue

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
                run_single_synthetic_benchmark(
                    spec=spec,
                    protocol=protocol,
                    model_config=(
                        model_config
                    ),
                    device=(
                        current_device
                    ),
                    verbose=(
                        verbose_realizations
                    ),
                )
            )

            write_realization_outputs(
                result,
                results_root=root,
                spec=spec,
            )

            clear_failure(
                root,
                spec,
            )

            completed_now += 1

            print(
                "  status: COMPLETE"
            )

            print(
                f"  checkpoint: "
                f"{record_path}"
            )

            master = (
                rebuild_master_table(
                    root
                )
            )

            rebuild_scenario_summary(
                root,
                master=master,
            )

        except Exception as exception:
            failed += 1

            print(
                "  status: FAILED"
            )

            print(
                f"  {exception.__class__.__name__}: "
                f"{exception}"
            )

            if not dry_run:
                record_failure(
                    root,
                    spec,
                    exception,
                )

            if stop_on_error:
                raise

    # Rebuild derived tables at the end even when every selected
    # realization was skipped.
    if not dry_run:
        master = (
            rebuild_master_table(
                root
            )
        )

        rebuild_scenario_summary(
            root,
            master=master,
        )

    summary = (
        BatchExecutionSummary(
            selected=len(
                selected_specs
            ),
            completed_now=(
                completed_now
            ),
            skipped_existing=(
                skipped_existing
            ),
            failed=failed,
            planned_only=(
                planned_only
            ),
        )
    )

    print(
        "\n========================================"
    )

    print(
        "BATCH SUMMARY"
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
        f"Skipped existing: "
        f"{summary.skipped_existing}"
    )

    print(
        f"Failed: "
        f"{summary.failed}"
    )

    if dry_run:
        print(
            f"Planned: "
            f"{summary.planned_only}"
        )

    else:
        print(
            f"Master results: "
            f"{master_results_path(root)}"
        )

        print(
            f"Scenario summary: "
            f"{scenario_summary_path(root)}"
        )

        print(
            f"Failure log: "
            f"{failures_path(root)}"
        )

    return summary