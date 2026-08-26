import numpy as np
import pandas as pd

from agnam.data.synthetic import (
    generate_s1,
    generate_s2,
    generate_s3,
    generate_s4,
    generate_synthetic,
)


def test_s1_dimensions_and_ground_truth():
    dataset = generate_s1(
        seed=42,
        n=1000,
    )

    assert dataset.X.shape == (1000, 20)
    assert dataset.y.shape == (1000,)

    assert dataset.true_interactions == (
        ("x3", "x4"),
        ("x5", "x6"),
        ("x8", "x9"),
    )


def test_s2_dimensions_and_ground_truth():
    dataset = generate_s2(
        seed=42,
        n=1000,
    )

    assert dataset.X.shape == (1000, 30)

    assert dataset.true_interactions == (
        ("x2", "x5"),
        ("x7", "x9"),
        ("x11", "x12"),
        ("x15", "x18"),
    )


def test_s3_proxy_features_are_correlated():
    dataset = generate_s3(
        seed=42,
        n=5000,
    )

    corr_x1_x11 = np.corrcoef(
        dataset.X["x1"],
        dataset.X["x11"],
    )[0, 1]

    corr_x2_x12 = np.corrcoef(
        dataset.X["x2"],
        dataset.X["x12"],
    )[0, 1]

    assert corr_x1_x11 > 0.80
    assert corr_x2_x12 > 0.80

    assert dataset.true_interactions == (
        ("x1", "x2"),
        ("x5", "x6"),
    )


def test_s4_prevalence_is_near_target():
    dataset = generate_s4(
        seed=42,
        n=10000,
    )

    prevalence = dataset.y.mean()

    assert 0.17 <= prevalence <= 0.23


def test_all_targets_are_binary():
    for scenario in ("S1", "S2", "S3", "S4"):
        dataset = generate_synthetic(
            scenario=scenario,
            seed=42,
            n=1000,
        )

        assert set(dataset.y.unique()).issubset({0, 1})


def test_generation_is_reproducible():
    first = generate_synthetic(
        scenario="S2",
        seed=123,
        n=500,
    )

    second = generate_synthetic(
        scenario="S2",
        seed=123,
        n=500,
    )

    pd.testing.assert_frame_equal(
        first.X,
        second.X,
    )

    pd.testing.assert_series_equal(
        first.y,
        second.y,
    )

    np.testing.assert_allclose(
        first.logits,
        second.logits,
    )


def test_component_arrays_have_correct_length():
    dataset = generate_s1(
        seed=42,
        n=750,
    )

    for values in dataset.main_effects.values():
        assert len(values) == 750
        assert np.isfinite(values).all()

    for values in dataset.interaction_effects.values():
        assert len(values) == 750
        assert np.isfinite(values).all()


def test_unknown_scenario_raises_error():
    try:
        generate_synthetic(
            scenario="S99",
            seed=42,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Unknown synthetic scenarios must raise ValueError."
        )