from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable

import numpy as np
import pandas as pd

from agnam.benchmarking.realx_batch import (
    DEFAULT_OUTPUT_ROOT,
    REQUIRED_CHECKPOINT_COLUMNS,
    RUN_DIRECTORY_NAME,
    _read_protocol_csv,
    checkpoint_path,
    validate_checkpoint,
)
from agnam.benchmarking.realx_protocol import (
    RealXProtocol,
    build_realx_schedule,
)


FREEZE_SCHEMA_VERSION = (
    "realx_evidence_freeze_v1"
)

EXPECTED_RUN_COUNT = 180

EXPECTED_EVIDENCE_FILE_COUNT = (
    EXPECTED_RUN_COUNT + 2
)

DEFAULT_FREEZE_DIRECTORY = (
    DEFAULT_OUTPUT_ROOT
    / "freeze"
)

DEFAULT_MANIFEST_PATH = (
    DEFAULT_FREEZE_DIRECTORY
    / "realx_evidence_manifest.json"
)

DEFAULT_FILE_TABLE_PATH = (
    DEFAULT_FREEZE_DIRECTORY
    / "realx_evidence_files.csv"
)


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass(frozen=True)
class EvidenceFileRecord:
    role: str
    relative_path: str

    sha256: str
    size_bytes: int

    run_id: str | None = None
    task_id: int | None = None
    dataset_name: str | None = None
    feature_type: str | None = None
    realization: int | None = None
    strength_name: str | None = None


@dataclass(frozen=True)
class GitProvenance:
    commit: str
    branch: str
    subject: str

    tracked_worktree_clean: bool


@dataclass(frozen=True)
class EvidenceAudit:
    benchmark_id: str

    run_checkpoint_count: int
    master_result_rows: int
    progress_rows: int

    evidence_file_count: int

    failure_log_present: bool
    failure_log_rows: int

    schema_versions: tuple[
        str,
        ...,
    ]

    file_records: tuple[
        EvidenceFileRecord,
        ...,
    ]

    aggregate_evidence_sha256: str


@dataclass(frozen=True)
class FreezeValidationResult:
    valid: bool

    evidence_file_count: int
    run_checkpoint_count: int

    aggregate_evidence_sha256: str

    phase6e_git_commit: str


# =============================================================================
# SHA-256
# =============================================================================


def file_sha256(
    path: str | Path,
    *,
    chunk_size: int = (
        1024 * 1024
    ),
) -> str:
    path = Path(
        path
    )

    digest = sha256()

    with path.open(
        "rb"
    ) as handle:
        while True:
            chunk = handle.read(
                chunk_size
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def _coerce_file_record(
    record: EvidenceFileRecord | dict,
) -> EvidenceFileRecord:
    if isinstance(
        record,
        EvidenceFileRecord,
    ):
        return record

    return EvidenceFileRecord(
        role=str(
            record[
                "role"
            ]
        ),
        relative_path=str(
            record[
                "relative_path"
            ]
        ),
        sha256=str(
            record[
                "sha256"
            ]
        ),
        size_bytes=int(
            record[
                "size_bytes"
            ]
        ),
        run_id=(
            None
            if record.get(
                "run_id"
            ) is None
            else str(
                record[
                    "run_id"
                ]
            )
        ),
        task_id=(
            None
            if record.get(
                "task_id"
            ) is None
            else int(
                record[
                    "task_id"
                ]
            )
        ),
        dataset_name=(
            None
            if record.get(
                "dataset_name"
            ) is None
            else str(
                record[
                    "dataset_name"
                ]
            )
        ),
        feature_type=(
            None
            if record.get(
                "feature_type"
            ) is None
            else str(
                record[
                    "feature_type"
                ]
            )
        ),
        realization=(
            None
            if record.get(
                "realization"
            ) is None
            else int(
                record[
                    "realization"
                ]
            )
        ),
        strength_name=(
            None
            if record.get(
                "strength_name"
            ) is None
            else str(
                record[
                    "strength_name"
                ]
            )
        ),
    )


def aggregate_evidence_sha256(
    records: Iterable[
        EvidenceFileRecord | dict
    ],
) -> str:
    normalized = sorted(
        (
            _coerce_file_record(
                record
            )
            for record in records
        ),
        key=lambda item: (
            item.relative_path
        ),
    )

    digest = sha256()

    for record in normalized:
        line = (
            f"{record.role}\t"
            f"{record.relative_path}\t"
            f"{record.size_bytes}\t"
            f"{record.sha256}\n"
        )

        digest.update(
            line.encode(
                "utf-8"
            )
        )

    return digest.hexdigest()


# =============================================================================
# GIT PROVENANCE
# =============================================================================


def _git_command(
    *arguments: str,
    repo_root: str | Path = ".",
) -> str:
    result = subprocess.run(
        [
            "git",
            *arguments,
        ],
        cwd=Path(
            repo_root
        ),
        check=True,
        capture_output=True,
        text=True,
    )

    return result.stdout.strip()


def get_git_provenance(
    *,
    repo_root: str | Path = ".",
) -> GitProvenance:
    commit = _git_command(
        "rev-parse",
        "HEAD",
        repo_root=repo_root,
    )

    branch = _git_command(
        "rev-parse",
        "--abbrev-ref",
        "HEAD",
        repo_root=repo_root,
    )

    subject = _git_command(
        "log",
        "-1",
        "--pretty=%s",
        repo_root=repo_root,
    )

    tracked_status = _git_command(
        "status",
        "--porcelain",
        "--untracked-files=no",
        repo_root=repo_root,
    )

    tracked_worktree_clean = (
        tracked_status
        == ""
    )

    if not tracked_worktree_clean:
        raise RuntimeError(
            "Tracked repository files changed after "
            "the locked Phase 6E commit. Evidence "
            "freeze is prohibited."
        )

    return GitProvenance(
        commit=commit,
        branch=branch,
        subject=subject,
        tracked_worktree_clean=True,
    )


def verify_git_commit_exists(
    commit: str,
    *,
    repo_root: str | Path = ".",
) -> None:
    subprocess.run(
        [
            "git",
            "cat-file",
            "-e",
            f"{commit}^{{commit}}",
        ],
        cwd=Path(
            repo_root
        ),
        check=True,
        capture_output=True,
        text=True,
    )


# =============================================================================
# HELPERS
# =============================================================================


def expected_checkpoint_filenames(
    protocol: RealXProtocol | None = None,
) -> tuple[
    str,
    ...,
]:
    if protocol is None:
        protocol = (
            RealXProtocol()
        )

    return tuple(
        (
            f"task_{run.task_id}_"
            f"r{run.realization:02d}_"
            f"{run.strength_name}.csv"
        )
        for run in (
            build_realx_schedule(
                protocol
            )
        )
    )


def _bool_series_all_true(
    values: pd.Series,
) -> bool:
    normalized = (
        values
        .astype(
            str
        )
        .str
        .strip()
        .str
        .lower()
    )

    return bool(
        normalized.isin(
            [
                "true",
                "1",
            ]
        ).all()
    )


def _relative_to_output_root(
    path: Path,
    output_root: Path,
) -> str:
    return (
        path
        .resolve()
        .relative_to(
            output_root.resolve()
        )
        .as_posix()
    )


def _make_file_record(
    *,
    role: str,
    path: Path,
    output_root: Path,
    run_spec=None,
) -> EvidenceFileRecord:
    return EvidenceFileRecord(
        role=role,
        relative_path=(
            _relative_to_output_root(
                path,
                output_root,
            )
        ),
        sha256=file_sha256(
            path
        ),
        size_bytes=int(
            path.stat().st_size
        ),
        run_id=(
            None
            if run_spec is None
            else run_spec.run_id
        ),
        task_id=(
            None
            if run_spec is None
            else run_spec.task_id
        ),
        dataset_name=(
            None
            if run_spec is None
            else run_spec.dataset_name
        ),
        feature_type=(
            None
            if run_spec is None
            else run_spec.feature_type
        ),
        realization=(
            None
            if run_spec is None
            else run_spec.realization
        ),
        strength_name=(
            None
            if run_spec is None
            else run_spec.strength_name
        ),
    )


# =============================================================================
# MASTER/CHECKPOINT CONSISTENCY
# =============================================================================


def _assert_master_matches_checkpoints(
    *,
    master: pd.DataFrame,
    checkpoint_tables: list[
        pd.DataFrame
    ],
) -> None:
    checkpoint_master = pd.concat(
        checkpoint_tables,
        ignore_index=True,
    )

    required_columns = sorted(
        REQUIRED_CHECKPOINT_COLUMNS
    )

    missing_master = (
        set(
            required_columns
        )
        - set(
            master.columns
        )
    )

    if missing_master:
        raise RuntimeError(
            "master_results.csv is missing checkpoint columns: "
            + ", ".join(
                sorted(
                    missing_master
                )
            )
        )

    missing_checkpoint = (
        set(
            required_columns
        )
        - set(
            checkpoint_master.columns
        )
    )

    if missing_checkpoint:
        raise RuntimeError(
            "Checkpoint aggregate is missing columns: "
            + ", ".join(
                sorted(
                    missing_checkpoint
                )
            )
        )

    master_compare = (
        master[
            required_columns
        ]
        .sort_values(
            "run_id"
        )
        .reset_index(
            drop=True
        )
    )

    checkpoint_compare = (
        checkpoint_master[
            required_columns
        ]
        .sort_values(
            "run_id"
        )
        .reset_index(
            drop=True
        )
    )

    try:
        pd.testing.assert_frame_equal(
            master_compare,
            checkpoint_compare,
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )

    except AssertionError as exc:
        raise RuntimeError(
            "master_results.csv does not exactly "
            "represent the validated per-run checkpoints."
        ) from exc


# =============================================================================
# EVIDENCE AUDIT
# =============================================================================


def audit_benchmark_evidence(
    *,
    output_root: str | Path = (
        DEFAULT_OUTPUT_ROOT
    ),
    protocol: RealXProtocol | None = None,
) -> EvidenceAudit:
    if protocol is None:
        protocol = (
            RealXProtocol()
        )

    output_root = Path(
        output_root
    )

    schedule = tuple(
        build_realx_schedule(
            protocol
        )
    )

    if len(
        schedule
    ) != EXPECTED_RUN_COUNT:
        raise RuntimeError(
            "Locked Real-X schedule must contain "
            f"{EXPECTED_RUN_COUNT} runs."
        )

    run_ids = [
        run.run_id
        for run in schedule
    ]

    if len(
        run_ids
    ) != len(
        set(
            run_ids
        )
    ):
        raise RuntimeError(
            "Locked Real-X schedule contains duplicate run IDs."
        )

    runs_directory = (
        output_root
        / RUN_DIRECTORY_NAME
    )

    if not runs_directory.exists():
        raise RuntimeError(
            "Real-X benchmark run directory does not exist."
        )

    actual_run_files = sorted(
        runs_directory.glob(
            "*.csv"
        )
    )

    if len(
        actual_run_files
    ) != EXPECTED_RUN_COUNT:
        raise RuntimeError(
            "Real-X evidence freeze requires exactly "
            f"{EXPECTED_RUN_COUNT} run checkpoint CSV files; "
            f"found {len(actual_run_files)}."
        )

    actual_names = {
        path.name
        for path in actual_run_files
    }

    expected_names = set(
        expected_checkpoint_filenames(
            protocol
        )
    )

    if actual_names != expected_names:
        missing = sorted(
            expected_names
            - actual_names
        )

        unexpected = sorted(
            actual_names
            - expected_names
        )

        raise RuntimeError(
            "Checkpoint filename set differs from "
            "the locked schedule. "
            f"Missing={missing}; "
            f"Unexpected={unexpected}."
        )

    file_records: list[
        EvidenceFileRecord
    ] = []

    checkpoint_tables: list[
        pd.DataFrame
    ] = []

    for run_spec in schedule:
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

        if not valid:
            raise RuntimeError(
                f"Checkpoint {run_spec.run_id} "
                f"failed final evidence validation: {reason}"
            )

        table = _read_protocol_csv(
            path
        )

        if len(
            table
        ) != 1:
            raise RuntimeError(
                f"Checkpoint {run_spec.run_id} "
                "must contain exactly one row."
            )

        checkpoint_tables.append(
            table
        )

        file_records.append(
            _make_file_record(
                role="checkpoint",
                path=path,
                output_root=output_root,
                run_spec=run_spec,
            )
        )

    # -------------------------------------------------------------------------
    # Master results
    # -------------------------------------------------------------------------

    master_path = (
        output_root
        / "master_results.csv"
    )

    if not master_path.exists():
        raise RuntimeError(
            "master_results.csv is missing."
        )

    master = _read_protocol_csv(
        master_path
    )

    if len(
        master
    ) != EXPECTED_RUN_COUNT:
        raise RuntimeError(
            "master_results.csv must contain "
            f"{EXPECTED_RUN_COUNT} rows."
        )

    if (
        master[
            "run_id"
        ]
        .nunique()
        != EXPECTED_RUN_COUNT
    ):
        raise RuntimeError(
            "master_results.csv does not contain "
            "180 unique run IDs."
        )

    if set(
        master[
            "run_id"
        ].astype(
            str
        )
    ) != set(
        run_ids
    ):
        raise RuntimeError(
            "master_results.csv run IDs differ "
            "from the locked Real-X schedule."
        )

    if not _bool_series_all_true(
        master[
            "publication_eligible"
        ]
    ):
        raise RuntimeError(
            "At least one master result is not "
            "publication eligible."
        )

    if not _bool_series_all_true(
        master[
            "test_integrity_ok"
        ]
    ):
        raise RuntimeError(
            "At least one master result failed "
            "untouched-test integrity."
        )

    if not (
        master[
            "status"
        ]
        .astype(
            str
        )
        .eq(
            "COMPLETE"
        )
        .all()
    ):
        raise RuntimeError(
            "At least one master result is not COMPLETE."
        )

    if not np.all(
        master[
            "expected_candidate_k"
        ].to_numpy(
            dtype=np.int64
        )
        ==
        master[
            "observed_candidate_k"
        ].to_numpy(
            dtype=np.int64
        )
    ):
        raise RuntimeError(
            "Candidate-K mismatch detected in master results."
        )

    if (
        pd.to_numeric(
            master[
                "max_decomposition_error"
            ],
            errors="raise",
        )
        .max()
        > 1e-6
    ):
        raise RuntimeError(
            "Exact decomposition tolerance was exceeded."
        )

    _assert_master_matches_checkpoints(
        master=master,
        checkpoint_tables=(
            checkpoint_tables
        ),
    )

    file_records.append(
        _make_file_record(
            role="master_results",
            path=master_path,
            output_root=output_root,
        )
    )

    # -------------------------------------------------------------------------
    # Progress table
    # -------------------------------------------------------------------------

    progress_path = (
        output_root
        / "benchmark_progress.csv"
    )

    if not progress_path.exists():
        raise RuntimeError(
            "benchmark_progress.csv is missing."
        )

    progress = _read_protocol_csv(
        progress_path
    )

    if len(
        progress
    ) != EXPECTED_RUN_COUNT:
        raise RuntimeError(
            "benchmark_progress.csv must contain "
            f"{EXPECTED_RUN_COUNT} rows."
        )

    if set(
        progress[
            "run_id"
        ].astype(
            str
        )
    ) != set(
        run_ids
    ):
        raise RuntimeError(
            "benchmark_progress.csv run IDs differ "
            "from the locked schedule."
        )

    if not (
        progress[
            "state"
        ]
        .astype(
            str
        )
        .eq(
            "COMPLETE"
        )
        .all()
    ):
        raise RuntimeError(
            "benchmark_progress.csv contains "
            "non-COMPLETE runs."
        )

    file_records.append(
        _make_file_record(
            role="benchmark_progress",
            path=progress_path,
            output_root=output_root,
        )
    )

    # -------------------------------------------------------------------------
    # Failure log is audit metadata, not evidentiary model output
    # -------------------------------------------------------------------------

    failure_path = (
        output_root
        / "failures.csv"
    )

    if failure_path.exists():
        failure_log_present = True

        failures = _read_protocol_csv(
            failure_path
        )

        failure_log_rows = int(
            len(
                failures
            )
        )

    else:
        failure_log_present = False
        failure_log_rows = 0

    # -------------------------------------------------------------------------
    # Schema audit
    # -------------------------------------------------------------------------

    schema_versions = tuple(
        sorted(
            {
                str(
                    value
                )
                for value in (
                    master[
                        "schema_version"
                    ]
                    .unique()
                    .tolist()
                )
            }
        )
    )

    if schema_versions != (
        "realx_benchmark_v1",
    ):
        raise RuntimeError(
            "Unexpected Real-X benchmark schema version(s): "
            f"{schema_versions}"
        )

    if len(
        file_records
    ) != EXPECTED_EVIDENCE_FILE_COUNT:
        raise RuntimeError(
            "Evidence file count mismatch. "
            f"Expected {EXPECTED_EVIDENCE_FILE_COUNT}, "
            f"found {len(file_records)}."
        )

    aggregate_hash = (
        aggregate_evidence_sha256(
            file_records
        )
    )

    return EvidenceAudit(
        benchmark_id=(
            protocol.benchmark_id
        ),
        run_checkpoint_count=(
            EXPECTED_RUN_COUNT
        ),
        master_result_rows=int(
            len(
                master
            )
        ),
        progress_rows=int(
            len(
                progress
            )
        ),
        evidence_file_count=int(
            len(
                file_records
            )
        ),
        failure_log_present=(
            failure_log_present
        ),
        failure_log_rows=(
            failure_log_rows
        ),
        schema_versions=(
            schema_versions
        ),
        file_records=tuple(
            file_records
        ),
        aggregate_evidence_sha256=(
            aggregate_hash
        ),
    )


# =============================================================================
# ATOMIC WRITERS
# =============================================================================


def _write_json_atomic(
    payload: dict[
        str,
        Any,
    ],
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


# =============================================================================
# FREEZE
# =============================================================================


def freeze_realx_benchmark(
    *,
    output_root: str | Path = (
        DEFAULT_OUTPUT_ROOT
    ),
    freeze_directory: str | Path = (
        DEFAULT_FREEZE_DIRECTORY
    ),
    repo_root: str | Path = ".",
    overwrite: bool = False,
) -> dict[
    str,
    Any,
]:
    output_root = Path(
        output_root
    )

    freeze_directory = Path(
        freeze_directory
    )

    manifest_path = (
        freeze_directory
        / DEFAULT_MANIFEST_PATH.name
    )

    file_table_path = (
        freeze_directory
        / DEFAULT_FILE_TABLE_PATH.name
    )

    if (
        manifest_path.exists()
        and not overwrite
    ):
        raise RuntimeError(
            "A Real-X evidence freeze manifest already exists. "
            "Use validation instead of silently overwriting it."
        )

    # HEAD should still be the Phase 6E execution lock at this point.
    git = get_git_provenance(
        repo_root=repo_root
    )

    audit = audit_benchmark_evidence(
        output_root=output_root,
    )

    created_at = (
        datetime.now(
            timezone.utc
        )
        .isoformat()
    )

    files_payload = [
        asdict(
            record
        )
        for record in (
            audit.file_records
        )
    ]

    manifest = {
        "freeze_schema_version": (
            FREEZE_SCHEMA_VERSION
        ),
        "freeze_status": "FROZEN",

        "created_at_utc": (
            created_at
        ),

        "benchmark_id": (
            audit.benchmark_id
        ),

        "phase6e_git_commit": (
            git.commit
        ),
        "phase6e_git_branch": (
            git.branch
        ),
        "phase6e_git_subject": (
            git.subject
        ),
        "tracked_worktree_clean": (
            git.tracked_worktree_clean
        ),

        "run_checkpoint_count": (
            audit.run_checkpoint_count
        ),
        "master_result_rows": (
            audit.master_result_rows
        ),
        "progress_rows": (
            audit.progress_rows
        ),

        "evidence_file_count": (
            audit.evidence_file_count
        ),

        "expected_run_count": (
            EXPECTED_RUN_COUNT
        ),

        "expected_evidence_file_count": (
            EXPECTED_EVIDENCE_FILE_COUNT
        ),

        "benchmark_schema_versions": list(
            audit.schema_versions
        ),

        "aggregate_evidence_sha256": (
            audit
            .aggregate_evidence_sha256
        ),

        "failure_log_present": (
            audit.failure_log_present
        ),
        "failure_log_rows": (
            audit.failure_log_rows
        ),

        "evidence_roles": {
            "checkpoint": (
                EXPECTED_RUN_COUNT
            ),
            "master_results": 1,
            "benchmark_progress": 1,
        },

        "files": (
            files_payload
        ),
    }

    file_table = pd.DataFrame(
        files_payload
    )

    _write_csv_atomic(
        file_table,
        file_table_path,
    )

    _write_json_atomic(
        manifest,
        manifest_path,
    )

    return manifest


# =============================================================================
# MANIFEST FILE VERIFICATION
# =============================================================================


def verify_manifest_files(
    manifest: dict[
        str,
        Any,
    ],
    *,
    output_root: str | Path = (
        DEFAULT_OUTPUT_ROOT
    ),
) -> str:
    output_root = Path(
        output_root
    )

    records = tuple(
        _coerce_file_record(
            item
        )
        for item in (
            manifest[
                "files"
            ]
        )
    )

    for record in records:
        path = (
            output_root
            / Path(
                record.relative_path
            )
        )

        if not path.exists():
            raise RuntimeError(
                "Frozen evidence file is missing: "
                f"{record.relative_path}"
            )

        current_size = int(
            path.stat().st_size
        )

        if current_size != (
            record.size_bytes
        ):
            raise RuntimeError(
                "Frozen evidence file size changed: "
                f"{record.relative_path}"
            )

        current_hash = (
            file_sha256(
                path
            )
        )

        if current_hash != (
            record.sha256
        ):
            raise RuntimeError(
                "Frozen evidence file SHA-256 changed: "
                f"{record.relative_path}"
            )

    aggregate_hash = (
        aggregate_evidence_sha256(
            records
        )
    )

    if aggregate_hash != str(
        manifest[
            "aggregate_evidence_sha256"
        ]
    ):
        raise RuntimeError(
            "Aggregate evidence SHA-256 does not "
            "match the freeze manifest."
        )

    return aggregate_hash


# =============================================================================
# FREEZE VALIDATION
# =============================================================================


def validate_realx_evidence_freeze(
    *,
    manifest_path: str | Path = (
        DEFAULT_MANIFEST_PATH
    ),
    output_root: str | Path = (
        DEFAULT_OUTPUT_ROOT
    ),
    repo_root: str | Path = ".",
) -> FreezeValidationResult:
    manifest_path = Path(
        manifest_path
    )

    if not manifest_path.exists():
        raise RuntimeError(
            "Real-X evidence freeze manifest does not exist."
        )

    with manifest_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        manifest = json.load(
            handle
        )

    if str(
        manifest.get(
            "freeze_schema_version"
        )
    ) != FREEZE_SCHEMA_VERSION:
        raise RuntimeError(
            "Unexpected evidence freeze schema version."
        )

    if str(
        manifest.get(
            "freeze_status"
        )
    ) != "FROZEN":
        raise RuntimeError(
            "Evidence manifest is not marked FROZEN."
        )

    if int(
        manifest.get(
            "run_checkpoint_count",
            -1,
        )
    ) != EXPECTED_RUN_COUNT:
        raise RuntimeError(
            "Frozen run checkpoint count is not 180."
        )

    if int(
        manifest.get(
            "evidence_file_count",
            -1,
        )
    ) != EXPECTED_EVIDENCE_FILE_COUNT:
        raise RuntimeError(
            "Frozen evidence file count is not 182."
        )

    phase6e_git_commit = str(
        manifest[
            "phase6e_git_commit"
        ]
    )

    verify_git_commit_exists(
        phase6e_git_commit,
        repo_root=repo_root,
    )

    manifest_hash = (
        verify_manifest_files(
            manifest,
            output_root=output_root,
        )
    )

    # Re-run the full semantic audit independently of hashes.
    current_audit = (
        audit_benchmark_evidence(
            output_root=output_root,
        )
    )

    if (
        current_audit
        .aggregate_evidence_sha256
        != manifest_hash
    ):
        raise RuntimeError(
            "Current audited evidence set differs "
            "from the frozen aggregate hash."
        )

    if (
        current_audit
        .run_checkpoint_count
        != EXPECTED_RUN_COUNT
    ):
        raise RuntimeError(
            "Current evidence audit does not contain "
            "180 valid checkpoints."
        )

    return FreezeValidationResult(
        valid=True,
        evidence_file_count=(
            current_audit
            .evidence_file_count
        ),
        run_checkpoint_count=(
            current_audit
            .run_checkpoint_count
        ),
        aggregate_evidence_sha256=(
            current_audit
            .aggregate_evidence_sha256
        ),
        phase6e_git_commit=(
            phase6e_git_commit
        ),
    )