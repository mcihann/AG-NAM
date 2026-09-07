import numpy as np
import pandas as pd
import pytest

from agnam.benchmarking.external_baselines import (
    build_external_baseline_schedule,
)
from agnam.benchmarking.openml_protocol import (
    OPENML_TASKS,
    build_openml_schedule,
)
from agnam.benchmarking.openml_statistics import (
    PRIMARY_COMPARISON,
    SECONDARY_COMPARISONS,
    RealWorldStatisticalProtocol,
    analyze_openml_comparisons,
    build_openml_diagnostic_summary,
    openml_completion_table,
    validate_openml_master,
)


def make_master(
    n_tasks: int,
) -> pd.DataFrame:
    benchmark_schedule = (
        build_openml_schedule()
    )

    external_lookup = {
        specification.task_id: (
            specification
        )
        for specification in (
            build_external_baseline_schedule()
        )
    }

    rows = []

    for specification in (
        benchmark_schedule[
            :n_tasks
        ]
    ):
        external = (
            external_lookup[
                specification.task_id
            ]
        )

        index = (
            specification.task_index
        )

        task = (
            OPENML_TASKS[
                index
            ]
        )

        main_auroc = (
            0.70
            + index
            * 0.001
        )

        agnam_auroc = (
            main_auroc
            + 0.03
        )

        main_auprc = (
            0.68
            + index
            * 0.001
        )

        agnam_auprc = (
            main_auprc
            + 0.03
        )

        rows.append(
            {
                "task_id": (
                    specification.task_id
                ),
                "task_index": (
                    specification.task_index
                ),
                "dataset_name": (
                    specification.dataset_name
                ),
                "feature_type": (
                    task.feature_type
                ),
                "discovery_base_seed": (
                    specification
                    .discovery_base_seed
                ),
                "random_pair_seed": (
                    specification
                    .random_pair_seed
                ),
                "final_split_seed": (
                    specification
                    .final_split_seed
                ),
                "final_model_seed": (
                    specification
                    .final_model_seed
                ),
                "ebm_seed": (
                    external.ebm_seed
                ),
                "catboost_seed": (
                    external.catboost_seed
                ),
                "openml_repeat": 0,
                "openml_fold": 0,
                "openml_sample": 0,
                "selection_threshold": 0.60,
                "isr_threshold": 0.60,
                "n_discovery_runs": 5,
                "residual_crossfit_folds": 5,
                "ebm_version": "0.7.8",
                "catboost_version": "1.2.10",
                "benchmark_protocol": (
                    "agnam_openml_primary_v1"
                ),
                "mean_pairwise_jaccard": (
                    0.30
                ),
                "n_selection_stable": 10,
                "n_isr_retained": 4,
                "isr_sparsification": 0.60,
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
                "ebm_auroc": (
                    agnam_auroc
                    - 0.005
                ),
                "ebm_auprc": (
                    agnam_auprc
                    - 0.005
                ),
                "catboost_auroc": (
                    agnam_auroc
                    + 0.005
                ),
                "catboost_auprc": (
                    agnam_auprc
                    + 0.005
                ),
                "random_pair_auroc": (
                    main_auroc
                    + 0.005
                ),
                "random_pair_auprc": (
                    main_auprc
                    + 0.005
                ),
                "single_run_auroc": (
                    agnam_auroc
                    - 0.002
                ),
                "single_run_auprc": (
                    agnam_auprc
                    - 0.002
                ),
                "no_isr_auroc": (
                    agnam_auroc
                    - 0.0005
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def test_real_world_statistical_defaults_are_locked():
    protocol = (
        RealWorldStatisticalProtocol()
    )

    assert (
        protocol.alpha
        == 0.05
    )

    assert (
        protocol.bootstrap_resamples
        == 10_000
    )

    assert (
        protocol.bootstrap_seed
        == 20_260_907
    )

    assert (
        protocol.primary_metric
        == "AUROC"
    )


def test_real_world_comparison_family_is_locked():
    assert (
        PRIMARY_COMPARISON
        .comparison_id
        == "agnam_vs_main_auroc"
    )

    assert len(
        SECONDARY_COMPARISONS
    ) == 9

    assert all(
        "no_isr"
        not in (
            comparison
            .comparison_id
        )
        for comparison in (
            SECONDARY_COMPARISONS
        )
    )


def test_partial_master_passes_schedule_validation():
    master = (
        make_master(
            3
        )
    )

    validate_openml_master(
        master
    )


def test_seed_mismatch_is_rejected():
    master = (
        make_master(
            1
        )
    )

    master.loc[
        0,
        "final_model_seed",
    ] = 999999

    with pytest.raises(
        ValueError
    ):
        validate_openml_master(
            master
        )


def test_completion_table_reports_28_tasks():
    master = (
        make_master(
            1
        )
    )

    completion = (
        openml_completion_table(
            master
        )
    )

    assert len(
        completion
    ) == 28

    assert int(
        completion[
            "complete"
        ].sum()
    ) == 1


def test_partial_benchmark_defers_confirmatory_inference():
    master = (
        make_master(
            5
        )
    )

    results = (
        analyze_openml_comparisons(
            master,
            statistical_protocol=(
                RealWorldStatisticalProtocol(
                    bootstrap_resamples=1000
                )
            ),
        )
    )

    assert len(
        results
    ) == 10

    assert results[
        "p_raw"
    ].isna().all()

    assert results[
        "p_adjusted"
    ].isna().all()

    assert results[
        "reject"
    ].isna().all()

    assert not results[
        "benchmark_complete"
    ].any()


def test_complete_benchmark_enables_primary_and_secondary_inference():
    master = (
        make_master(
            28
        )
    )

    results = (
        analyze_openml_comparisons(
            master,
            statistical_protocol=(
                RealWorldStatisticalProtocol(
                    bootstrap_resamples=1000
                )
            ),
        )
    )

    primary = (
        results[
            results[
                "family"
            ]
            == "primary"
        ]
    )

    secondary = (
        results[
            results[
                "family"
            ]
            == "secondary"
        ]
    )

    assert len(
        primary
    ) == 1

    assert len(
        secondary
    ) == 9

    assert primary[
        "p_raw"
    ].notna().all()

    assert primary[
        "p_adjusted"
    ].notna().all()

    assert secondary[
        "p_raw"
    ].notna().all()

    assert secondary[
        "p_adjusted"
    ].notna().all()

    assert results[
        "reject"
    ].notna().all()

    assert results[
        "benchmark_complete"
    ].all()

    assert np.isclose(
        float(
            primary.iloc[
                0
            ][
                "p_raw"
            ]
        ),
        float(
            primary.iloc[
                0
            ][
                "p_adjusted"
            ]
        ),
    )


def test_diagnostic_summary_includes_all_group():
    master = (
        make_master(
            28
        )
    )

    diagnostics = (
        build_openml_diagnostic_summary(
            master
        )
    )

    assert "ALL" in set(
        diagnostics[
            "group"
        ]
    )

    assert (
        "isr_sparsification"
        in set(
            diagnostics[
                "metric"
            ]
        )
    )

    assert (
        "agnam_minus_no_isr_auroc"
        in set(
            diagnostics[
                "metric"
            ]
        )
    )