from __future__ import annotations

import pandas as pd

from agnam.benchmarking.openml_figures import (
    LOCKED_CASE_STUDIES,
    generate_openml_publication_figures,
    verify_openml_freeze_manifest,
)


def main():
    print(
        "========================================"
    )

    print(
        "OPENML PUBLICATION FIGURE GENERATION"
    )

    print(
        "========================================"
    )

    verification = (
        verify_openml_freeze_manifest()
    )

    print(
        f"Frozen evidence verified: "
        f"{len(verification)}/57 files"
    )

    paths = (
        generate_openml_publication_figures()
    )

    print(
        "\nBENCHMARK EVIDENCE FIGURE"
    )

    print(
        "PNG:",
        paths.benchmark_evidence.png,
    )

    print(
        "PDF:",
        paths.benchmark_evidence.pdf,
    )

    print(
        "SVG:",
        paths.benchmark_evidence.svg,
    )

    print(
        "\nLOCKED CASE-STUDY FIGURE"
    )

    print(
        "PNG:",
        paths.locked_case_studies.png,
    )

    print(
        "PDF:",
        paths.locked_case_studies.pdf,
    )

    print(
        "SVG:",
        paths.locked_case_studies.svg,
    )

    print(
        "\nLOCKED CASE-STUDY EDGE TABLE"
    )

    print(
        paths.case_study_edges_csv
    )

    print(
        "\n========================================"
    )

    print(
        "LOCKED ISR-RETAINED CASE-STUDY EDGES"
    )

    print(
        "========================================"
    )

    edges = pd.read_csv(
        paths.case_study_edges_csv
    )

    print(
        edges.to_string(
            index=False
        )
    )

    print(
        "\n========================================"
    )

    print(
        "STATUS: FIGURES GENERATED FROM "
        "VERIFIED FROZEN EVIDENCE"
    )

    print(
        "========================================"
    )


if __name__ == "__main__":
    main()