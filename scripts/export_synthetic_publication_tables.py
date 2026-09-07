from __future__ import annotations

from agnam.benchmarking.reporting import (
    export_synthetic_publication_tables,
)


def main():
    paths = (
        export_synthetic_publication_tables()
    )

    print(
        "========================================"
    )

    print(
        "SYNTHETIC PUBLICATION TABLE EXPORT"
    )

    print(
        "========================================"
    )

    print(
        "Status: COMPLETE"
    )

    print(
        "\nPredictive table:"
    )

    print(
        paths.predictive_table
    )

    print(
        "\nStructure table:"
    )

    print(
        paths.structure_table
    )

    print(
        "\nInference table:"
    )

    print(
        paths.inference_table
    )

    print(
        "\nDiagnostic table:"
    )

    print(
        paths.diagnostic_table
    )

    print(
        "\nCompletion table:"
    )

    print(
        paths.completion_table
    )

    print(
        "\nManifest:"
    )

    print(
        paths.manifest
    )


if __name__ == "__main__":
    main()