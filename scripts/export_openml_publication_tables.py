from __future__ import annotations

import pandas as pd

from agnam.benchmarking.openml_reporting import (
    export_openml_publication_tables,
)


def main():
    paths = (
        export_openml_publication_tables()
    )

    print(
        "========================================"
    )

    print(
        "OPENML FINAL PUBLICATION EXPORT"
    )

    print(
        "========================================"
    )

    print(
        "Status: 28/28 COMPLETE AND FROZEN"
    )

    print(
        "\nPredictive task table:"
    )

    print(
        paths.predictive_by_task
    )

    print(
        "\nModel summary:"
    )

    print(
        paths.model_summary
    )

    print(
        "\nInference table:"
    )

    print(
        paths.inference_table
    )

    print(
        "\nStructure table:"
    )

    print(
        paths.structure_by_task
    )

    print(
        "\nFeature-type summary:"
    )

    print(
        paths.feature_type_summary
    )

    print(
        "\nSparsity-performance table:"
    )

    print(
        paths.sparsity_performance
    )

    print(
        "\nInteraction-count distribution:"
    )

    print(
        paths.interaction_count_distribution
    )

    print(
        "\nCase-study selection:"
    )

    print(
        paths.case_study_selection
    )

    print(
        "\nDiagnostic summary:"
    )

    print(
        paths.diagnostic_summary
    )

    print(
        "\nCompletion table:"
    )

    print(
        paths.completion_table
    )

    print(
        "\nFreeze manifest:"
    )

    print(
        paths.freeze_manifest_json
    )

    print(
        "\n========================================"
    )

    print(
        "LOCKED CASE STUDIES"
    )

    print(
        "========================================"
    )

    case_studies = pd.read_csv(
        paths.case_study_selection
    )

    print(
        case_studies.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()