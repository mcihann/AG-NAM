import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from agnam.benchmarking.openml_figures import (
    LOCKED_CASE_STUDIES,
    LockedCaseStudySpec,
    build_case_network_data,
    build_primary_delta_plot_data,
    build_sparsity_plot_data,
    save_figure_family,
    validate_locked_case_study_selection,
    verify_openml_freeze_manifest,
)


def locked_selection_frame():
    return pd.DataFrame(
        [
            {
                "feature_type": "numeric",
                "registry_position": 15,
                "task_id": 146819,
                "dataset_name": (
                    "climate-model-simulation-crashes"
                ),
                "n_isr_retained": 2,
                "selection_basis": (
                    "earliest_registry_task_"
                    "with_isr_interaction"
                ),
                "predictive_performance_used": False,
            },
            {
                "feature_type": "mixed",
                "registry_position": 9,
                "task_id": 3021,
                "dataset_name": "sick",
                "n_isr_retained": 4,
                "selection_basis": (
                    "earliest_registry_task_"
                    "with_isr_interaction"
                ),
                "predictive_performance_used": False,
            },
            {
                "feature_type": "categorical",
                "registry_position": 2,
                "task_id": 3,
                "dataset_name": "kr-vs-kp",
                "n_isr_retained": 7,
                "selection_basis": (
                    "earliest_registry_task_"
                    "with_isr_interaction"
                ),
                "predictive_performance_used": False,
            },
        ]
    )


def test_locked_case_study_identities_are_fixed():
    assert tuple(
        specification.task_id
        for specification
        in LOCKED_CASE_STUDIES
    ) == (
        146819,
        3021,
        3,
    )

    assert tuple(
        specification.feature_type
        for specification
        in LOCKED_CASE_STUDIES
    ) == (
        "numeric",
        "mixed",
        "categorical",
    )


def test_locked_case_study_selection_is_accepted():
    selection = (
        locked_selection_frame()
    )

    validate_locked_case_study_selection(
        selection
    )


def test_performance_based_case_study_selection_is_rejected():
    selection = (
        locked_selection_frame()
    )

    selection.loc[
        0,
        "predictive_performance_used",
    ] = True

    with pytest.raises(
        RuntimeError
    ):
        validate_locked_case_study_selection(
            selection
        )


def test_case_network_uses_only_isr_retained_edges():
    pair_table = pd.DataFrame(
        [
            {
                "feature_j": "A",
                "feature_k": "B",
                "selection_frequency": 0.8,
                "isr_retained": True,
            },
            {
                "feature_j": "A",
                "feature_k": "C",
                "selection_frequency": 0.6,
                "isr_retained": True,
            },
            {
                "feature_j": "B",
                "feature_k": "C",
                "selection_frequency": 1.0,
                "isr_retained": False,
            },
        ]
    )

    specification = (
        LockedCaseStudySpec(
            feature_type="numeric",
            registry_position=1,
            task_id=123,
            dataset_name="toy",
        )
    )

    network = (
        build_case_network_data(
            pair_table,
            specification,
        )
    )

    assert len(
        network.edges
    ) == 2

    assert set(
        network.nodes
    ) == {
        "A",
        "B",
        "C",
    }

    assert not (
        (
            network.edges[
                "feature_j"
            ]
            == "B"
        )
        & (
            network.edges[
                "feature_k"
            ]
            == "C"
        )
    ).any()


def test_primary_delta_data_is_sorted():
    frame = pd.DataFrame(
        [
            {
                "task_id": 1,
                "dataset_name": "A",
                "feature_type": "numeric",
                "delta_agnam_main_auroc": 0.02,
            },
            {
                "task_id": 2,
                "dataset_name": "B",
                "feature_type": "mixed",
                "delta_agnam_main_auroc": -0.01,
            },
            {
                "task_id": 3,
                "dataset_name": "C",
                "feature_type": "categorical",
                "delta_agnam_main_auroc": 0.0,
            },
        ]
    )

    result = (
        build_primary_delta_plot_data(
            frame
        )
    )

    assert result[
        "delta_agnam_main_auroc"
    ].tolist() == [
        -0.01,
        0.0,
        0.02,
    ]

    assert result[
        "display_rank"
    ].tolist() == [
        1,
        2,
        3,
    ]

    assert result[
        "direction"
    ].tolist() == [
        "AG-NAM lower",
        "Tie",
        "AG-NAM higher",
    ]


def test_sparsity_plot_excludes_undefined_rows():
    frame = pd.DataFrame(
        [
            {
                "task_id": 1,
                "dataset_name": "A",
                "feature_type": "numeric",
                "isr_sparsification": 0.5,
                "agnam_minus_no_isr_auroc": 0.01,
            },
            {
                "task_id": 2,
                "dataset_name": "B",
                "feature_type": "mixed",
                "isr_sparsification": np.nan,
                "agnam_minus_no_isr_auroc": 0.0,
            },
            {
                "task_id": 3,
                "dataset_name": "C",
                "feature_type": "categorical",
                "isr_sparsification": 0.7,
                "agnam_minus_no_isr_auroc": np.nan,
            },
        ]
    )

    result = (
        build_sparsity_plot_data(
            frame
        )
    )

    assert len(
        result
    ) == 1

    assert int(
        result.iloc[
            0
        ][
            "task_id"
        ]
    ) == 1


def test_figure_family_writes_png_pdf_svg(
    tmp_path,
):
    figure = plt.figure(
        figsize=(
            3,
            2,
        )
    )

    axis = figure.add_subplot(
        111
    )

    axis.plot(
        [
            0,
            1,
        ],
        [
            0,
            1,
        ],
    )

    paths = (
        save_figure_family(
            figure,
            output_root=(
                tmp_path
            ),
            stem="test_figure",
        )
    )

    plt.close(
        figure
    )

    assert paths.png.exists()
    assert paths.pdf.exists()
    assert paths.svg.exists()

    assert (
        paths.png.stat()
        .st_size
        > 0
    )

    assert (
        paths.pdf.stat()
        .st_size
        > 0
    )

    assert (
        paths.svg.stat()
        .st_size
        > 0
    )


def test_freeze_manifest_verification_accepts_intact_file(
    tmp_path,
):
    evidence = (
        tmp_path
        / "evidence.csv"
    )

    evidence.write_text(
        "a,b\n1,2\n",
        encoding="utf-8",
    )

    import hashlib

    digest = hashlib.sha256(
        evidence.read_bytes()
    ).hexdigest()

    entries = [
        {
            "path": str(
                evidence
            ),
            "bytes": int(
                evidence.stat()
                .st_size
            ),
            "sha256": digest,
        }
        for _ in range(
            57
        )
    ]

    manifest = {
        "status": "FROZEN",
        "n_complete_tasks": 28,
        "evidence_files": entries,
    }

    manifest_path = (
        tmp_path
        / "freeze.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest
        ),
        encoding="utf-8",
    )

    result = (
        verify_openml_freeze_manifest(
            manifest_path
        )
    )

    assert len(
        result
    ) == 57

    assert result[
        "verified"
    ].all()


def test_freeze_manifest_detects_tampering(
    tmp_path,
):
    evidence = (
        tmp_path
        / "evidence.csv"
    )

    evidence.write_text(
        "a,b\n1,2\n",
        encoding="utf-8",
    )

    import hashlib

    digest = hashlib.sha256(
        evidence.read_bytes()
    ).hexdigest()

    entries = [
        {
            "path": str(
                evidence
            ),
            "bytes": int(
                evidence.stat()
                .st_size
            ),
            "sha256": digest,
        }
        for _ in range(
            57
        )
    ]

    manifest = {
        "status": "FROZEN",
        "n_complete_tasks": 28,
        "evidence_files": entries,
    }

    manifest_path = (
        tmp_path
        / "freeze.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest
        ),
        encoding="utf-8",
    )

    evidence.write_text(
        "a,b\n9,9\n",
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError
    ):
        verify_openml_freeze_manifest(
            manifest_path
        )