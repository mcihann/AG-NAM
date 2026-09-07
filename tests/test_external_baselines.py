from agnam.benchmarking.external_baselines import (
    ExternalBaselineProtocol,
    build_external_baseline_schedule,
    catboost_locked_parameters,
    ebm_locked_parameters,
    external_interaction_budget,
)


def test_external_baseline_versions_are_locked():
    protocol = (
        ExternalBaselineProtocol()
    )

    assert (
        protocol.ebm_package
        == "interpret-core"
    )

    assert (
        protocol.ebm_version
        == "0.7.8"
    )

    assert (
        protocol.catboost_package
        == "catboost"
    )

    assert (
        protocol.catboost_version
        == "1.2.10"
    )


def test_external_baseline_schedule_contains_28_tasks():
    schedule = (
        build_external_baseline_schedule()
    )

    assert len(
        schedule
    ) == 28

    assert len(
        {
            item.ebm_seed
            for item in schedule
        }
    ) == 28

    assert len(
        {
            item.catboost_seed
            for item in schedule
        }
    ) == 28


def test_external_interaction_budget_matches_locked_rule():
    assert (
        external_interaction_budget(
            5
        )
        == 5
    )

    assert (
        external_interaction_budget(
            10
        )
        == 5
    )

    assert (
        external_interaction_budget(
            20
        )
        == 19
    )

    assert (
        external_interaction_budget(
            50
        )
        == 20
    )

    assert (
        external_interaction_budget(
            72
        )
        == 20
    )


def test_ebm_parameters_are_explicitly_locked():
    parameters = (
        ebm_locked_parameters(
            n_features=20,
            seed=123,
        )
    )

    assert (
        parameters[
            "interactions"
        ]
        == 19
    )

    assert (
        parameters[
            "validation_size"
        ]
        == 0.15
    )

    assert (
        parameters[
            "outer_bags"
        ]
        == 14
    )

    assert (
        parameters[
            "inner_bags"
        ]
        == 0
    )

    assert (
        parameters[
            "max_rounds"
        ]
        == 50_000
    )

    assert (
        parameters[
            "random_state"
        ]
        == 123
    )


def test_catboost_is_locked_to_cpu():
    parameters = (
        catboost_locked_parameters(
            seed=456
        )
    )

    assert (
        parameters[
            "task_type"
        ]
        == "CPU"
    )

    assert (
        parameters[
            "random_seed"
        ]
        == 456
    )

    assert (
        parameters[
            "iterations"
        ]
        == 2000
    )

    assert (
        parameters[
            "learning_rate"
        ]
        == 0.03
    )

    assert (
        parameters[
            "depth"
        ]
        == 6
    )

    assert (
        parameters[
            "loss_function"
        ]
        == "Logloss"
    )

    assert (
        parameters[
            "eval_metric"
        ]
        == "AUC"
    )


def test_secondary_inferential_family_contains_nine_tests():
    protocol = (
        ExternalBaselineProtocol()
    )

    assert len(
        protocol
        .secondary_comparisons
    ) == 9

    assert (
        "agnam_vs_ebm_auroc"
        in protocol
        .secondary_comparisons
    )

    assert (
        "agnam_vs_catboost_auroc"
        in protocol
        .secondary_comparisons
    )

    assert (
        "agnam_vs_single_run_auroc"
        in protocol
        .secondary_comparisons
    )


def test_no_isr_is_not_in_confirmatory_secondary_family():
    protocol = (
        ExternalBaselineProtocol()
    )

    assert all(
        "no_isr"
        not in comparison
        for comparison
        in protocol
        .secondary_comparisons
    )