from __future__ import annotations

from agnam.benchmarking.realx_freeze import (
    validate_realx_evidence_freeze,
)
from agnam.benchmarking.realx_reporting import (
    DEFAULT_COMPARATOR_TABLE,
    DEFAULT_FEATURE_TYPE_TABLE,
    DEFAULT_PRIMARY_DATASET_TABLE,
    DEFAULT_PRIMARY_SUMMARY_TABLE,
    DEFAULT_STRENGTH_TABLE,
    export_publication_tables,
)


def main():
    print(
        "=============================================="
    )

    print(
        "REAL-X PUBLICATION TABLE EXPORT"
    )

    print(
        "=============================================="
    )

    print(
        "Validating frozen evidence..."
    )

    before = (
        validate_realx_evidence_freeze()
    )

    print(
        "Freeze status: PASS"
    )

    manifest = (
        export_publication_tables()
    )

    after = (
        validate_realx_evidence_freeze()
    )

    unchanged = (
        before
        .aggregate_evidence_sha256
        ==
        after
        .aggregate_evidence_sha256
    )

    if not unchanged:
        raise RuntimeError(
            "Frozen evidence changed during publication export."
        )

    print()

    print(
        "Primary dataset table:"
    )

    print(
        DEFAULT_PRIMARY_DATASET_TABLE
    )

    print()

    print(
        "Primary summary table:"
    )

    print(
        DEFAULT_PRIMARY_SUMMARY_TABLE
    )

    print()

    print(
        "Strength descriptive table:"
    )

    print(
        DEFAULT_STRENGTH_TABLE
    )

    print()

    print(
        "Predictive comparator table:"
    )

    print(
        DEFAULT_COMPARATOR_TABLE
    )

    print()

    print(
        "Feature-type descriptive table:"
    )

    print(
        DEFAULT_FEATURE_TYPE_TABLE
    )

    print()

    print(
        "Frozen evidence unchanged:",
        unchanged,
    )

    print(
        "Secondary confirmatory tests:",
        manifest[
            "secondary_confirmatory_tests"
        ],
    )

    print()

    print(
        "STATUS: REAL-X PUBLICATION TABLES EXPORTED"
    )


if __name__ == "__main__":
    main()