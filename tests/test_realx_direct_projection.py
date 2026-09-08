import numpy as np

from agnam.benchmarking.realx_generator import (
    interaction_marginal_error,
    interaction_support_degrees_of_freedom,
    purify_two_way_surface,
)


def test_direct_projection_handles_highly_imbalanced_sparse_support():
    rng = np.random.default_rng(
        26090801
    )

    raw = rng.normal(
        size=(
            8,
            8,
        )
    )

    weights = np.asarray(
        [
            [1800, 0,    0,   0,   0,  0, 0, 1],
            [0,    900,  0,   0,   0,  0, 2, 0],
            [0,    0,    400, 0,   0,  3, 0, 0],
            [0,    0,    0,   200, 4,  0, 0, 0],
            [0,    0,    0,   5,   90, 0, 0, 0],
            [0,    0,    6,   0,   0, 40, 0, 0],
            [0,    7,    0,   0,   0,  0, 20, 0],
            [8,    0,    0,   0,   0,  0, 0, 10],
        ],
        dtype=np.float64,
    )

    support_rows = []
    support_columns = []

    for row in range(
        weights.shape[
            0
        ]
    ):
        for column in range(
            weights.shape[
                1
            ]
        ):
            count = int(
                weights[
                    row,
                    column,
                ]
            )

            support_rows.extend(
                [
                    row
                ]
                * count
            )

            support_columns.extend(
                [
                    column
                ]
                * count
            )

    interaction_df = (
        interaction_support_degrees_of_freedom(
            np.asarray(
                support_rows,
                dtype=np.int64,
            ),
            np.asarray(
                support_columns,
                dtype=np.int64,
            ),
            n_states_a=8,
            n_states_b=8,
        )
    )

    assert (
        interaction_df
        >= 1
    )

    (
        purified,
        error,
        passes,
    ) = purify_two_way_surface(
        raw,
        weights,
        tolerance=1e-10,
        max_iterations=1000,
    )

    assert (
        passes
        in {
            1,
            2,
        }
    )

    assert (
        error
        <= 1e-10
    )

    assert (
        interaction_marginal_error(
            purified,
            weights,
        )
        <= 1e-10
    )


def test_direct_projection_handles_disconnected_support():
    raw = np.asarray(
        [
            [1.2, -0.4, 0.0, 0.0],
            [0.7,  2.1, 0.0, 0.0],
            [0.0,  0.0, 1.5, -1.0],
            [0.0,  0.0, 0.2,  0.9],
        ],
        dtype=np.float64,
    )

    weights = np.asarray(
        [
            [20.0, 5.0, 0.0, 0.0],
            [3.0,  9.0, 0.0, 0.0],
            [0.0,  0.0, 7.0, 4.0],
            [0.0,  0.0, 2.0, 8.0],
        ],
        dtype=np.float64,
    )

    (
        purified,
        error,
        _,
    ) = purify_two_way_surface(
        raw,
        weights,
        tolerance=1e-10,
        max_iterations=1000,
    )

    assert (
        error
        <= 1e-10
    )

    assert (
        interaction_marginal_error(
            purified,
            weights,
        )
        <= 1e-10
    )

    assert np.std(
        purified[
            weights > 0.0
        ]
    ) > 0.0


def test_direct_projection_is_deterministic():
    rng = np.random.default_rng(
        1234
    )

    raw = rng.normal(
        size=(
            6,
            7,
        )
    )

    weights = rng.integers(
        0,
        20,
        size=(
            6,
            7,
        ),
    ).astype(
        np.float64
    )

    # Guarantee a sufficiently connected observed support.
    weights[
        0,
        :
    ] += 1.0

    weights[
        :,
        0
    ] += 1.0

    first = purify_two_way_surface(
        raw,
        weights,
        tolerance=1e-10,
        max_iterations=1000,
    )[
        0
    ]

    second = purify_two_way_surface(
        raw,
        weights,
        tolerance=1e-10,
        max_iterations=1000,
    )[
        0
    ]

    assert np.array_equal(
        first,
        second,
    )


def test_direct_projection_removes_purely_additive_surface():
    row_effect = np.asarray(
        [
            -1.0,
            0.4,
            1.7,
        ],
        dtype=np.float64,
    )

    column_effect = np.asarray(
        [
            -0.5,
            0.2,
            0.8,
            1.4,
        ],
        dtype=np.float64,
    )

    raw = (
        2.5
        + row_effect[
            :,
            None,
        ]
        + column_effect[
            None,
            :,
        ]
    )

    weights = np.asarray(
        [
            [10, 4, 8, 2],
            [3, 15, 5, 9],
            [7, 2, 13, 6],
        ],
        dtype=np.float64,
    )

    (
        purified,
        error,
        _,
    ) = purify_two_way_surface(
        raw,
        weights,
        tolerance=1e-10,
        max_iterations=1000,
    )

    assert (
        error
        <= 1e-10
    )

    assert np.max(
        np.abs(
            purified[
                weights > 0
            ]
        )
    ) <= 1e-10