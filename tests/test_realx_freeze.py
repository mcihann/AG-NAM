from dataclasses import asdict
from hashlib import sha256
import json

import pytest

from agnam.benchmarking.realx_freeze import (
    DEFAULT_FREEZE_DIRECTORY,
    DEFAULT_MANIFEST_PATH,
    EXPECTED_EVIDENCE_FILE_COUNT,
    EXPECTED_RUN_COUNT,
    EvidenceFileRecord,
    _write_json_atomic,
    aggregate_evidence_sha256,
    expected_checkpoint_filenames,
    file_sha256,
    verify_manifest_files,
)


def test_file_sha256_matches_known_digest(
    tmp_path,
):
    path = (
        tmp_path
        / "sample.txt"
    )

    payload = b"AG-NAM Real-X freeze"

    path.write_bytes(
        payload
    )

    expected = (
        sha256(
            payload
        )
        .hexdigest()
    )

    assert (
        file_sha256(
            path
        )
        == expected
    )


def test_aggregate_hash_is_order_independent():
    first = EvidenceFileRecord(
        role="checkpoint",
        relative_path="runs/a.csv",
        sha256="a" * 64,
        size_bytes=10,
    )

    second = EvidenceFileRecord(
        role="checkpoint",
        relative_path="runs/b.csv",
        sha256="b" * 64,
        size_bytes=20,
    )

    assert (
        aggregate_evidence_sha256(
            [
                first,
                second,
            ]
        )
        ==
        aggregate_evidence_sha256(
            [
                second,
                first,
            ]
        )
    )


def test_aggregate_hash_changes_when_evidence_changes():
    original = EvidenceFileRecord(
        role="checkpoint",
        relative_path="runs/a.csv",
        sha256="a" * 64,
        size_bytes=10,
    )

    changed = EvidenceFileRecord(
        role="checkpoint",
        relative_path="runs/a.csv",
        sha256="b" * 64,
        size_bytes=10,
    )

    assert (
        aggregate_evidence_sha256(
            [
                original
            ]
        )
        !=
        aggregate_evidence_sha256(
            [
                changed
            ]
        )
    )


def test_expected_checkpoint_names_cover_locked_schedule():
    names = (
        expected_checkpoint_filenames()
    )

    assert len(
        names
    ) == EXPECTED_RUN_COUNT

    assert len(
        set(
            names
        )
    ) == EXPECTED_RUN_COUNT

    assert all(
        name.endswith(
            ".csv"
        )
        for name in names
    )


def _build_small_manifest(
    root,
):
    first_path = (
        root
        / "runs"
        / "a.csv"
    )

    second_path = (
        root
        / "master_results.csv"
    )

    first_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    first_path.write_text(
        "a,b\n1,2\n",
        encoding="utf-8",
    )

    second_path.write_text(
        "c,d\n3,4\n",
        encoding="utf-8",
    )

    records = [
        EvidenceFileRecord(
            role="checkpoint",
            relative_path=(
                "runs/a.csv"
            ),
            sha256=(
                file_sha256(
                    first_path
                )
            ),
            size_bytes=(
                first_path
                .stat()
                .st_size
            ),
        ),
        EvidenceFileRecord(
            role="master_results",
            relative_path=(
                "master_results.csv"
            ),
            sha256=(
                file_sha256(
                    second_path
                )
            ),
            size_bytes=(
                second_path
                .stat()
                .st_size
            ),
        ),
    ]

    manifest = {
        "aggregate_evidence_sha256": (
            aggregate_evidence_sha256(
                records
            )
        ),
        "files": [
            asdict(
                record
            )
            for record in records
        ],
    }

    return (
        manifest,
        first_path,
        second_path,
    )


def test_manifest_file_verification_accepts_unchanged_files(
    tmp_path,
):
    (
        manifest,
        _,
        _,
    ) = _build_small_manifest(
        tmp_path
    )

    observed = (
        verify_manifest_files(
            manifest,
            output_root=tmp_path,
        )
    )

    assert observed == (
        manifest[
            "aggregate_evidence_sha256"
        ]
    )


def test_manifest_file_verification_rejects_modified_file(
    tmp_path,
):
    (
        manifest,
        first_path,
        _,
    ) = _build_small_manifest(
        tmp_path
    )

    first_path.write_text(
        "a,b\n9,9\n",
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError
    ):
        verify_manifest_files(
            manifest,
            output_root=tmp_path,
        )


def test_manifest_file_verification_rejects_missing_file(
    tmp_path,
):
    (
        manifest,
        first_path,
        _,
    ) = _build_small_manifest(
        tmp_path
    )

    first_path.unlink()

    with pytest.raises(
        RuntimeError
    ):
        verify_manifest_files(
            manifest,
            output_root=tmp_path,
        )


def test_manifest_file_verification_rejects_size_mismatch(
    tmp_path,
):
    (
        manifest,
        first_path,
        _,
    ) = _build_small_manifest(
        tmp_path
    )

    first_path.write_text(
        "this file is now much longer",
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError
    ):
        verify_manifest_files(
            manifest,
            output_root=tmp_path,
        )


def test_atomic_json_writer_round_trips(
    tmp_path,
):
    path = (
        tmp_path
        / "manifest.json"
    )

    payload = {
        "status": "FROZEN",
        "count": 180,
    }

    _write_json_atomic(
        payload,
        path,
    )

    assert path.exists()

    assert not (
        tmp_path
        / "manifest.json.tmp"
    ).exists()

    loaded = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    assert loaded == payload


def test_freeze_constants_use_separate_directory():
    assert (
        DEFAULT_FREEZE_DIRECTORY.name
        == "freeze"
    )

    assert (
        DEFAULT_MANIFEST_PATH.parent
        == DEFAULT_FREEZE_DIRECTORY
    )

    assert (
        EXPECTED_EVIDENCE_FILE_COUNT
        == 182
    )