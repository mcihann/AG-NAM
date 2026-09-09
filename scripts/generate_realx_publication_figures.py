from __future__ import annotations

from agnam.benchmarking.realx_freeze import (
    validate_realx_evidence_freeze,
)
from agnam.benchmarking.realx_reporting import (
    DEFAULT_MAIN_FIGURE_PDF,
    DEFAULT_MAIN_FIGURE_PNG,
    DEFAULT_MAIN_FIGURE_SVG,
    export_publication_figure,
)


def main():
    print(
        "=============================================="
    )

    print(
        "REAL-X PUBLICATION FIGURE GENERATION"
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
        export_publication_figure()
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
            "Frozen evidence changed during figure generation."
        )

    print()

    print(
        "PNG:"
    )

    print(
        DEFAULT_MAIN_FIGURE_PNG
    )

    print()

    print(
        "PDF:"
    )

    print(
        DEFAULT_MAIN_FIGURE_PDF
    )

    print()

    print(
        "SVG:"
    )

    print(
        DEFAULT_MAIN_FIGURE_SVG
    )

    print()

    print(
        "PNG DPI:",
        manifest[
            "png_dpi"
        ],
    )

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
        "STATUS: REAL-X PUBLICATION FIGURE GENERATED"
    )


if __name__ == "__main__":
    main()