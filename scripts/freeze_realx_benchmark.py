from __future__ import annotations

import argparse

from agnam.benchmarking.realx_freeze import (
    DEFAULT_MANIFEST_PATH,
    freeze_realx_benchmark,
    validate_realx_evidence_freeze,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Cryptographically freeze and validate "
            "the completed 180-run Real-X benchmark."
        )
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Validate the existing freeze manifest "
            "without creating a new one."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Explicitly overwrite an existing manifest. "
            "Do not use after confirmatory analysis begins."
        ),
    )

    return parser


def print_validation(
    result,
):
    print(
        "=============================================="
    )

    print(
        "REAL-X EVIDENCE FREEZE VALIDATION"
    )

    print(
        "=============================================="
    )

    print(
        "Status: PASS"
    )

    print(
        "Run checkpoints:",
        result.run_checkpoint_count,
    )

    print(
        "Evidence files:",
        result.evidence_file_count,
    )

    print(
        "Aggregate SHA-256:",
        result.aggregate_evidence_sha256,
    )

    print(
        "Phase 6E Git commit:",
        result.phase6e_git_commit,
    )

    print()

    print(
        "STATUS: FROZEN EVIDENCE VALIDATED"
    )


def main():
    args = (
        build_parser()
        .parse_args()
    )

    if args.validate_only:
        result = (
            validate_realx_evidence_freeze()
        )

        print_validation(
            result
        )

        return

    manifest = (
        freeze_realx_benchmark(
            overwrite=args.overwrite,
        )
    )

    print(
        "=============================================="
    )

    print(
        "REAL-X EVIDENCE FREEZE"
    )

    print(
        "=============================================="
    )

    print(
        "Status:",
        manifest[
            "freeze_status"
        ],
    )

    print(
        "Run checkpoints:",
        manifest[
            "run_checkpoint_count"
        ],
    )

    print(
        "Master rows:",
        manifest[
            "master_result_rows"
        ],
    )

    print(
        "Progress rows:",
        manifest[
            "progress_rows"
        ],
    )

    print(
        "Evidence files:",
        manifest[
            "evidence_file_count"
        ],
    )

    print(
        "Benchmark schema:",
        manifest[
            "benchmark_schema_versions"
        ],
    )

    print(
        "Failure-log rows:",
        manifest[
            "failure_log_rows"
        ],
    )

    print(
        "Aggregate SHA-256:",
        manifest[
            "aggregate_evidence_sha256"
        ],
    )

    print(
        "Phase 6E Git commit:",
        manifest[
            "phase6e_git_commit"
        ],
    )

    print(
        "Phase 6E Git subject:",
        manifest[
            "phase6e_git_subject"
        ],
    )

    print()

    print(
        "Manifest:"
    )

    print(
        DEFAULT_MANIFEST_PATH
    )

    print()

    print(
        "Running immediate independent validation..."
    )

    result = (
        validate_realx_evidence_freeze()
    )

    print()

    print_validation(
        result
    )


if __name__ == "__main__":
    main()