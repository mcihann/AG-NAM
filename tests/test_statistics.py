import numpy as np
import pandas as pd
import pytest

from agnam.benchmarking.statistics import (
    PRIMARY_COMPARISONS,
    StatisticalAnalysisProtocol,
    analyze_paired_comparisons,
    benchmark_completion_table,
    build_diagnostic_summary,
    compute_paired_comparison,
    holm_adjust,
    paired_bootstrap_ci,
    paired_rank_biserial,
    validate_master_against_schedule,
)
from agnam.benchmarking.synthetic_protocol import (
    build_synthetic_schedule,
)


def make_master_rows(
    *,
    scenarios=("S1",),
    n_realizations=20,
):
    schedule = (
        build_synthetic_schedule()
    )

    rows = []

    for specification in schedule:
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

        realization = (
            specification
            .realization_index
        )

        main_auroc = (
            0.70
            + 0.001
            * realization
        )

        agnam_auroc = (
            main_auroc
            + 0.10
            + 0.0002
            * realization
        )

        main_auprc = (
            0.69
            + 0.001
            * realization
        )

        agnam_auprc = (
            main_auprc
            + 0.11
            + 0.0002
            * realization
        )

        rows.append(
            {
                "scenario": (
                    specification.scenario
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
                "candidate_k": 19,
                "n_selection_stable": 16,
                "n_isr_retained": 4,
                "main_auroc": (
                    main_auroc
                ),
                "agnam_auroc": (
                    agnam_auroc
                ),
                "oracle_auroc": (
                    agnam_auroc
                    + 0.03
                ),
                "random_pair_auroc": (
                    main_auroc
                    + 0.01
                ),
                "no_isr_auroc": (
                    agnam_auroc
                    - 0.001
                ),
                "single_run_auroc": (
                    agnam_auroc
                    - 0.01
                ),
                "main_auprc": (
                    main_auprc
                ),
                "agnam_auprc": (
                    agnam_auprc
                ),
                "random_pair_auprc": (
                    main_auprc
                    + 0.01
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


def test_statistical_protocol_defaults_are_locked():
    protocol = (
        StatisticalAnalysisProtocol()
    )

    assert np.isclose(
        protocol.alpha,
        0.05,
    )

    assert (
        protocol.bootstrap_resamples
        == 10_000
    )

    assert (
        protocol.bootstrap_seed
        == 20_260_902
    )

    assert (
        protocol.primary_metric
        == "auroc"
    )


def test_paired_bootstrap_is_deterministic():
    differences = np.array(
        [
            0.01,
            0.02,
            0.03,
            0.04,
            0.05,
        ]
    )

    first = paired_bootstrap_ci(
        differences,
        statistic="mean",
        resamples=2000,
        seed=123,
    )

    second = paired_bootstrap_ci(
        differences,
        statistic="mean",
        resamples=2000,
        seed=123,
    )

    assert first == second

    assert (
        first[0]
        <= differences.mean()
        <= first[1]
    )


def test_rank_biserial_is_one_for_all_positive():
    differences = np.array(
        [
            0.1,
            0.2,
            0.3,
            0.4,
        ]
    )

    result = (
        paired_rank_biserial(
            differences
        )
    )

    assert np.isclose(
        result,
        1.0,
    )


def test_zero_difference_comparison_is_handled():
    frame = pd.DataFrame(
        {
            "agnam_auroc": [
                0.8,
                0.8,
                0.8,
            ],
            "main_auroc": [
                0.8,
                0.8,
                0.8,
            ],
        }
    )

    result = compute_paired_comparison(
        frame=frame,
        comparison=(
            PRIMARY_COMPARISONS[0]
        ),
        scenario="S1",
        statistical_protocol=(
            StatisticalAnalysisProtocol(
                bootstrap_resamples=1000
            )
        ),
    )

    assert np.isclose(
        result[
            "mean_difference"
        ],
        0.0,
    )

    assert np.isclose(
        result[
            "p_raw"
        ],
        1.0,
    )

    assert np.isclose(
        result[
            "rank_biserial"
        ],
        0.0,
    )

    assert (
        result[
            "wilcoxon_effective_n"
        ]
        == 0
    )


def test_holm_adjustment_known_values():
    p_values = np.array(
        [
            0.01,
            0.04,
            0.03,
        ]
    )

    adjusted = holm_adjust(
        p_values
    )

    expected = np.array(
        [
            0.03,
            0.06,
            0.06,
        ]
    )

    np.testing.assert_allclose(
        adjusted,
        expected,
        atol=1e-12,
        rtol=0.0,
    )


def test_completion_detects_complete_s1_only():
    master = make_master_rows(
        scenarios=(
            "S1",
        ),
        n_realizations=20,
    )

    completion = (
        benchmark_completion_table(
            master
        )
    )

    lookup = {
        row.scenario: row
        for row in (
            completion.itertuples()
        )
    }

    assert (
        lookup[
            "S1"
        ].completed
        == 20
    )

    assert bool(
        lookup[
            "S1"
        ].is_complete
    )

    assert (
        lookup[
            "S2"
        ].completed
        == 0
    )

    assert not bool(
        lookup[
            "S2"
        ].is_complete
    )


def test_master_validation_rejects_seed_mismatch():
    master = make_master_rows(
        scenarios=(
            "S1",
        ),
        n_realizations=1,
    )

    master.loc[
        0,
        "dataset_seed",
    ] = 999999

    with pytest.raises(
        ValueError
    ):
        validate_master_against_schedule(
            master
        )


def test_holm_inference_is_deferred_for_partial_benchmark():
    master = make_master_rows(
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

    assert len(
        result
    ) == 6

    assert result[
        "p_holm"
    ].isna().all()

    assert not result[
        "family_complete"
    ].any()


def test_full_benchmark_enables_holm_inference():
    master = make_master_rows(
        scenarios=(
            "S1",
            "S2",
            "S3",
            "S4",
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

    primary = result[
        result[
            "family"
        ]
        == "primary"
    ]

    secondary = result[
        result[
            "family"
        ]
        == "secondary"
    ]

    assert len(
        primary
    ) == 4

    assert len(
        secondary
    ) == 20

    assert primary[
        "p_holm"
    ].notna().all()

    assert secondary[
        "p_holm"
    ].notna().all()

    assert primary[
        "family_complete"
    ].all()

    assert secondary[
        "family_complete"
    ].all()


def test_diagnostic_summary_computes_sparsification():
    master = make_master_rows(
        scenarios=(
            "S1",
        ),
        n_realizations=2,
    )

    summary = (
        build_diagnostic_summary(
            master
        )
    )

    row = summary[
        (
            summary[
                "scenario"
            ]
            == "S1"
        )
        & (
            summary[
                "metric"
            ]
            == (
                "isr_sparsification_"
                "vs_selection"
            )
        )
    ].iloc[
        0
    ]

    expected = (
        1.0
        - 4.0
        / 16.0
    )

    assert np.isclose(
        row[
            "mean"
        ],
        expected,
    )

    oracle_row = summary[
        (
            summary[
                "scenario"
            ]
            == "S1"
        )
        & (
            summary[
                "metric"
            ]
            == "oracle_gap_auroc"
        )
    ].iloc[
        0
    ]

    assert np.isclose(
        oracle_row[
            "mean"
        ],
        0.03,
    )