import warnings

import pandas as pd

from agnam.benchmarking.statistics import (
    StatisticalAnalysisProtocol,
    analyze_paired_comparisons,
)
from agnam.benchmarking.synthetic_protocol import (
    build_synthetic_schedule,
)


def make_master(
    *,
    scenarios,
    n_realizations,
):
    rows = []

    for specification in (
        build_synthetic_schedule()
    ):
        if (
            specification.scenario
            not in scenarios
        ):
            continue

        if (
            specification.realization_number
            > n_realizations
        ):
            continue

        index = (
            specification
            .realization_index
        )

        main_auroc = (
            0.70
            + 0.001
            * index
        )

        agnam_auroc = (
            main_auroc
            + 0.10
        )

        main_auprc = (
            0.69
            + 0.001
            * index
        )

        agnam_auprc = (
            main_auprc
            + 0.10
        )

        rows.append(
            {
                "scenario": (
                    specification
                    .scenario
                ),
                "realization_index": (
                    specification
                    .realization_index
                ),
                "dataset_seed": (
                    specification
                    .dataset_seed
                ),
                "outer_split_seed": (
                    specification
                    .outer_split_seed
                ),
                "final_split_seed": (
                    specification
                    .final_split_seed
                ),
                "discovery_base_seed": (
                    specification
                    .discovery_base_seed
                ),
                "random_pair_seed": (
                    specification
                    .random_pair_seed
                ),
                "final_model_seed": (
                    specification
                    .final_model_seed
                ),
                "main_auroc": (
                    main_auroc
                ),
                "agnam_auroc": (
                    agnam_auroc
                ),
                "main_auprc": (
                    main_auprc
                ),
                "agnam_auprc": (
                    agnam_auprc
                ),
                "random_pair_auroc": (
                    main_auroc
                    + 0.01
                ),
                "random_pair_auprc": (
                    main_auprc
                    + 0.01
                ),
                "single_run_auroc": (
                    agnam_auroc
                    - 0.01
                ),
                "single_run_auprc": (
                    agnam_auprc
                    - 0.01
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def test_partial_inference_uses_nullable_boolean_dtype():
    master = make_master(
        scenarios=(
            "S1",
        ),
        n_realizations=20,
    )

    result = (
        analyze_paired_comparisons(
            master,
            statistical_protocol=(
                StatisticalAnalysisProtocol(
                    bootstrap_resamples=1000
                )
            ),
        )
    )

    assert (
        str(
            result[
                "reject_holm"
            ].dtype
        )
        == "boolean"
    )

    assert result[
        "reject_holm"
    ].isna().all()


def test_complete_inference_emits_no_incompatible_dtype_warning():
    master = make_master(
        scenarios=(
            "S1",
            "S2",
            "S3",
            "S4",
        ),
        n_realizations=20,
    )

    with warnings.catch_warnings(
        record=True
    ) as captured:
        warnings.simplefilter(
            "always"
        )

        result = (
            analyze_paired_comparisons(
                master,
                statistical_protocol=(
                    StatisticalAnalysisProtocol(
                        bootstrap_resamples=1000
                    )
                ),
            )
        )

    incompatible = [
        warning
        for warning in captured
        if (
            issubclass(
                warning.category,
                FutureWarning,
            )
            and "incompatible dtype"
            in str(
                warning.message
            ).lower()
        )
    ]

    assert incompatible == []

    assert (
        str(
            result[
                "reject_holm"
            ].dtype
        )
        == "boolean"
    )

    assert result[
        "reject_holm"
    ].notna().all()