import json

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
from agnam.benchmarking.openml_reporting import (
    build_feature_type_summary,
    build_interaction_count_distribution,
    build_openml_model_summary,
    build_openml_predictive_task_table,
    build_openml_sparsity_performance_table,
    build_openml_structure_task_table,
    export_openml_publication_tables,
    select_openml_case_studies,
)
from agnam.benchmarking.openml_statistics import (
    RealWorldStatisticalProtocol,
)


def make_complete_master():
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
        benchmark_schedule
    ):
        task = OPENML_TASKS[
            specification.task_index
        ]

        external = (
            external_lookup[
                specification.task_id
            ]
        )

        index = (
            specification.task_index
        )

        main_auroc = (
            0.70
            + 0.001
            * index
        )

        agnam_auroc = (
            main_auroc
            + 0.02
        )

        main_auprc = (
            0.65
            + 0.001
            * index
        )

        agnam_auprc = (
            main_auprc
            + 0.02
        )

        n_selection = (
            0
            if index == 0
            else 4
        )

        n_isr = (
            0
            if index == 0
            else 2
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
                "n_samples": (
                    task.expected_n_samples
                ),
                "n_features": (
                    task.expected_n_predictors
                ),
                "candidate_k": 5,
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
                "mean_pairwise_jaccard": 0.30,
                "n_selection_stable": (
                    n_selection
                ),
                "n_isr_retained": (
                    n_isr
                ),
                "isr_sparsification": (
                    np.nan
                    if n_selection == 0
                    else 0.50
                ),
                "max_decomposition_error": (
                    1e-6
                ),
                "main_auroc": (
                    main_auroc
                ),
                "agnam_auroc": (
                    agnam_auroc
                ),
                "random_pair_auroc": (
                    main_auroc
                    + 0.005
                ),
                "no_isr_auroc": (
                    agnam_auroc
                    - 0.001
                ),
                "single_run_auroc": (
                    agnam_auroc
                    - 0.002
                ),
                "ebm_auroc": (
                    agnam_auroc
                    - 0.003
                ),
                "catboost_auroc": (
                    agnam_auroc
                    + 0.003
                ),
                "main_auprc": (
                    main_auprc
                ),
                "agnam_auprc": (
                    agnam_auprc
                ),
                "random_pair_auprc": (
                    main_auprc
                    + 0.005
                ),
                "no_isr_auprc": (
                    agnam_auprc
                    - 0.001
                ),
                "single_run_auprc": (
                    agnam_auprc
                    - 0.002
                ),
                "ebm_auprc": (
                    agnam_auprc
                    - 0.003
                ),
                "catboost_auprc": (
                    agnam_auprc
                    + 0.003
                ),
                "main_balanced_accuracy": 0.70,
                "main_f1": 0.70,
                "agnam_balanced_accuracy": 0.72,
                "agnam_f1": 0.72,
                "ebm_balanced_accuracy": 0.71,
                "ebm_f1": 0.71,
                "catboost_balanced_accuracy": 0.73,
                "catboost_f1": 0.73,
                "delta_agnam_main_auroc": (
                    agnam_auroc
                    - main_auroc
                ),
                "delta_agnam_main_auprc": (
                    agnam_auprc
                    - main_auprc
                ),
                "delta_agnam_random_auroc": (
                    agnam_auroc
                    - (
                        main_auroc
                        + 0.005
                    )
                ),
                "delta_agnam_ebm_auroc": 0.003,
                "delta_agnam_catboost_auroc": -0.003,
            }
        )

    return pd.DataFrame(
        rows
    )


def write_complete_checkpoint_set(
    root,
    master,
):
    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    master.to_csv(
        root
        / "master_results.csv",
        index=False,
    )

    for row in (
        master.itertuples(
            index=False
        )
    ):
        pd.DataFrame(
            [
                row._asdict()
            ]
        ).to_csv(
            root
            / f"task_{row.task_id}.csv",
            index=False,
        )

        pd.DataFrame(
            [
                {
                    "feature_j": "x1",
                    "feature_k": "x2",
                    "selection_count": 3,
                    "selection_frequency": 0.6,
                    "selection_stable": True,
                    "isr_retained": (
                        int(
                            row.n_isr_retained
                        )
                        > 0
                    ),
                    "isr_score": 0.8,
                    "random_control": False,
                    "single_run_top_k": True,
                }
            ]
        ).to_csv(
            root
            / (
                f"task_{row.task_id}"
                "_pairs.csv"
            ),
            index=False,
        )


def test_predictive_task_table_contains_28_rows():
    master = (
        make_complete_master()
    )

    table = (
        build_openml_predictive_task_table(
            master
        )
    )

    assert len(
        table
    ) == 28

    assert (
        "delta_agnam_main_auroc"
        in table.columns
    )


def test_model_summary_contains_all_locked_models():
    master = (
        make_complete_master()
    )

    table = (
        build_openml_model_summary(
            master
        )
    )

    models = set(
        table[
            "model"
        ]
    )

    assert {
        "Main NAM",
        "AG-NAM",
        "Random-Pair NAM",
        "No-ISR AG-NAM",
        "Single-Run AG-NAM",
        "EBM",
        "CatBoost",
    }.issubset(
        models
    )


def test_structure_table_marks_empty_isr_sets():
    master = (
        make_complete_master()
    )

    table = (
        build_openml_structure_task_table(
            master
        )
    )

    first = (
        table.iloc[
            0
        ]
    )

    assert bool(
        first[
            "selection_empty"
        ]
    )

    assert bool(
        first[
            "isr_empty"
        ]
    )


def test_sparsity_performance_table_calculates_deltas():
    master = (
        make_complete_master()
    )

    table = (
        build_openml_sparsity_performance_table(
            master
        )
    )

    assert np.allclose(
        table[
            "agnam_minus_main_auroc"
        ],
        0.02,
    )


def test_interaction_count_distribution_sums_to_28():
    master = (
        make_complete_master()
    )

    table = (
        build_interaction_count_distribution(
            master
        )
    )

    assert int(
        table[
            "n_datasets"
        ].sum()
    ) == 28


def test_feature_type_summary_contains_three_groups():
    master = (
        make_complete_master()
    )

    table = (
        build_feature_type_summary(
            master
        )
    )

    assert set(
        table[
            "feature_type"
        ]
    ) == {
        "numeric",
        "mixed",
        "categorical",
    }


def test_case_study_selection_uses_no_predictive_metric():
    master = (
        make_complete_master()
    )

    selected = (
        select_openml_case_studies(
            master
        )
    )

    assert len(
        selected
    ) == 3

    assert not selected[
        "predictive_performance_used"
    ].any()

    assert (
        selected[
            "n_isr_retained"
        ]
        > 0
    ).all()


def test_incomplete_benchmark_cannot_be_exported(
    tmp_path,
):
    master = (
        make_complete_master()
        .iloc[
            :27
        ]
        .copy()
    )

    results_root = (
        tmp_path
        / "benchmark"
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
        export_openml_publication_tables(
            results_root=(
                results_root
            ),
            output_root=(
                tmp_path
                / "publication"
            ),
            freeze_manifest_path=(
                tmp_path
                / "freeze.json"
            ),
            statistical_protocol=(
                RealWorldStatisticalProtocol(
                    bootstrap_resamples=1000
                )
            ),
        )


def test_complete_export_writes_freeze_manifest(
    tmp_path,
):
    master = (
        make_complete_master()
    )

    results_root = (
        tmp_path
        / "benchmark"
    )

    output_root = (
        tmp_path
        / "publication"
    )

    freeze_path = (
        tmp_path
        / "freeze.json"
    )

    write_complete_checkpoint_set(
        results_root,
        master,
    )

    paths = (
        export_openml_publication_tables(
            results_root=(
                results_root
            ),
            output_root=(
                output_root
            ),
            freeze_manifest_path=(
                freeze_path
            ),
            statistical_protocol=(
                RealWorldStatisticalProtocol(
                    bootstrap_resamples=1000
                )
            ),
        )
    )

    assert (
        paths.predictive_by_task
        .exists()
    )

    assert (
        paths.inference_table
        .exists()
    )

    assert (
        paths.case_study_selection
        .exists()
    )

    assert (
        paths.freeze_manifest_json
        .exists()
    )

    manifest = json.loads(
        freeze_path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        manifest[
            "status"
        ]
        == "FROZEN"
    )

    assert (
        manifest[
            "n_complete_tasks"
        ]
        == 28
    )

    # 1 master + 28 task records + 28 pair audits
    assert len(
        manifest[
            "evidence_files"
        ]
    ) == 57

    assert all(
        len(
            item[
                "sha256"
            ]
        )
        == 64
        for item in (
            manifest[
                "evidence_files"
            ]
        )
    )