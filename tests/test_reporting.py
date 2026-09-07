import pandas as pd
import pytest

from agnam.benchmarking.reporting import (
    build_predictive_publication_table,
    build_structure_publication_table,
    export_synthetic_publication_tables,
)
from agnam.benchmarking.statistics import (
    StatisticalAnalysisProtocol,
    analyze_paired_comparisons,
)
from agnam.benchmarking.synthetic_protocol import (
    build_synthetic_schedule,
)


def make_complete_master():
    rows = []

    for specification in (
        build_synthetic_schedule()
    ):
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
            + 0.11
        )

        rows.append(
            {
                "scenario": (
                    specification.scenario
                ),
                "realization_index": (
                    specification.realization_index
                ),
                "dataset_seed": (
                    specification.dataset_seed
                ),
                "outer_split_seed": (
                    specification.outer_split_seed
                ),
                "final_split_seed": (
                    specification.final_split_seed
                ),
                "discovery_base_seed": (
                    specification.discovery_base_seed
                ),
                "random_pair_seed": (
                    specification.random_pair_seed
                ),
                "final_model_seed": (
                    specification.final_model_seed
                ),
                "candidate_k": 20,
                "mean_pairwise_jaccard": 0.30,
                "mean_interaction_auprc": 0.35,
                "selection_precision": 0.20,
                "selection_recall": 0.60,
                "selection_f1": 0.30,
                "isr_precision": 0.60,
                "isr_recall": 0.60,
                "isr_f1": 0.60,
                "false_positive_reduction": 0.80,
                "n_selection_stable": 10,
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


def make_comparisons(
    master,
):
    return (
        analyze_paired_comparisons(
            master,
            statistical_protocol=(
                StatisticalAnalysisProtocol(
                    bootstrap_resamples=1000
                )
            ),
        )
    )


def test_predictive_publication_table_has_four_scenarios():
    master = (
        make_complete_master()
    )

    comparisons = (
        make_comparisons(
            master
        )
    )

    table = (
        build_predictive_publication_table(
            master,
            comparisons,
        )
    )

    assert len(
        table
    ) == 4

    assert tuple(
        table[
            "scenario"
        ].tolist()
    ) == (
        "S1",
        "S2",
        "S3",
        "S4",
    )

    assert table[
        "primary_reject_holm"
    ].all()


def test_structure_publication_table_calculates_sparsification():
    master = (
        make_complete_master()
    )

    table = (
        build_structure_publication_table(
            master
        )
    )

    expected = (
        1.0
        - 4.0
        / 10.0
    )

    assert all(
        abs(
            value
            - expected
        )
        < 1e-12
        for value in (
            table[
                "sparsification_vs_selection_mean"
            ]
        )
    )


def test_publication_export_rejects_incomplete_benchmark(
    tmp_path,
):
    master = (
        make_complete_master()
        .query(
            "scenario == 'S1'"
        )
        .copy()
    )

    results_root = (
        tmp_path
        / "results"
    )

    results_root.mkdir(
        parents=True
    )

    master.to_csv(
        results_root
        / "master_results.csv",
        index=False,
    )

    with pytest.raises(
        RuntimeError
    ):
        export_synthetic_publication_tables(
            results_root=(
                results_root
            ),
            output_root=(
                tmp_path
                / "publication"
            ),
            statistical_protocol=(
                StatisticalAnalysisProtocol(
                    bootstrap_resamples=1000
                )
            ),
        )


def test_publication_export_writes_expected_files(
    tmp_path,
):
    master = (
        make_complete_master()
    )

    results_root = (
        tmp_path
        / "results"
    )

    output_root = (
        tmp_path
        / "publication"
    )

    results_root.mkdir(
        parents=True
    )

    master.to_csv(
        results_root
        / "master_results.csv",
        index=False,
    )

    paths = (
        export_synthetic_publication_tables(
            results_root=(
                results_root
            ),
            output_root=(
                output_root
            ),
            statistical_protocol=(
                StatisticalAnalysisProtocol(
                    bootstrap_resamples=1000
                )
            ),
        )
    )

    assert (
        paths.predictive_table.exists()
    )

    assert (
        paths.structure_table.exists()
    )

    assert (
        paths.inference_table.exists()
    )

    assert (
        paths.diagnostic_table.exists()
    )

    assert (
        paths.completion_table.exists()
    )

    assert (
        paths.manifest.exists()
    )


def test_exported_inference_contains_all_24_prespecified_tests(
    tmp_path,
):
    master = (
        make_complete_master()
    )

    results_root = (
        tmp_path
        / "results"
    )

    output_root = (
        tmp_path
        / "publication"
    )

    results_root.mkdir(
        parents=True
    )

    master.to_csv(
        results_root
        / "master_results.csv",
        index=False,
    )

    paths = (
        export_synthetic_publication_tables(
            results_root=(
                results_root
            ),
            output_root=(
                output_root
            ),
            statistical_protocol=(
                StatisticalAnalysisProtocol(
                    bootstrap_resamples=1000
                )
            ),
        )
    )

    inference = pd.read_csv(
        paths.inference_table
    )

    assert len(
        inference
    ) == 24

    assert inference[
        "p_holm"
    ].notna().all()