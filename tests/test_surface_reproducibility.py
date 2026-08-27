import numpy as np

from agnam.interpretation.surface_reproducibility import (
    compute_isr,
    diagonal_purified_vector,
    purify_interaction_grid,
)


def test_purification_removes_pure_additive_structure():
    x = np.array(
        [-2.0, -1.0, 0.0, 1.0, 2.0]
    )

    a = x ** 2
    b = 2.0 * x

    raw = (
        a[:, None]
        + b[None, :]
        + 4.0
    )

    purified = (
        purify_interaction_grid(
            raw
        )
    )

    np.testing.assert_allclose(
        purified,
        0.0,
        atol=1e-12,
    )


def test_purification_preserves_centered_product_interaction():
    x = np.array(
        [-1.0, 0.0, 1.0]
    )

    z = np.array(
        [-2.0, 0.0, 2.0]
    )

    raw = np.outer(
        x,
        z,
    )

    purified = (
        purify_interaction_grid(
            raw
        )
    )

    np.testing.assert_allclose(
        purified,
        raw,
        atol=1e-12,
    )


def test_diagonal_vector_has_correct_shape():
    raw = np.arange(
        25,
        dtype=float,
    ).reshape(
        5,
        5,
    )

    vector = (
        diagonal_purified_vector(
            raw
        )
    )

    assert vector.shape == (
        5,
    )


def test_identical_surfaces_have_isr_one():
    first = np.array(
        [-1.0, 0.2, 0.7, 1.4]
    )

    second = first.copy()

    third = first.copy()

    result = compute_isr(
        (
            first,
            second,
            third,
        )
    )

    assert np.isclose(
        result.isr,
        1.0,
    )


def test_zero_variance_surface_has_zero_agreement():
    first = np.ones(
        10
    )

    second = np.linspace(
        -1.0,
        1.0,
        10,
    )

    result = compute_isr(
        (
            first,
            second,
        )
    )

    assert np.isclose(
        result.isr,
        0.0,
    )


def test_opposite_surfaces_have_negative_isr():
    first = np.array(
        [-2.0, -1.0, 0.0, 1.0, 2.0]
    )

    second = -first

    result = compute_isr(
        (
            first,
            second,
        )
    )

    assert np.isclose(
        result.isr,
        -1.0,
    )