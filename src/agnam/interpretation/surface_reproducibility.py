from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np


@dataclass(frozen=True)
class SurfaceAgreement:
    n_surfaces: int
    pairwise_correlations: tuple[
        float,
        ...
    ]
    isr: float


def purify_interaction_grid(
    raw_grid: np.ndarray,
) -> np.ndarray:
    """
    Empirical functional-ANOVA purification.

    raw_grid[i, m] represents:

        h(x_j_i, x_k_m)

    Returns the purified interaction contribution matrix:

        h_jk
        - E_k[h_jk | x_j]
        - E_j[h_jk | x_k]
        + E_jk[h_jk]
    """
    raw_grid = np.asarray(
        raw_grid,
        dtype=np.float64,
    )

    if raw_grid.ndim != 2:
        raise ValueError(
            "raw_grid must be two-dimensional."
        )

    if raw_grid.shape[0] != (
        raw_grid.shape[1]
    ):
        raise ValueError(
            "raw_grid must be square."
        )

    if not np.isfinite(
        raw_grid
    ).all():
        raise ValueError(
            "raw_grid must contain finite values."
        )

    row_means = raw_grid.mean(
        axis=1,
        keepdims=True,
    )

    column_means = raw_grid.mean(
        axis=0,
        keepdims=True,
    )

    grand_mean = raw_grid.mean()

    return (
        raw_grid
        - row_means
        - column_means
        + grand_mean
    )


def diagonal_purified_vector(
    raw_grid: np.ndarray,
) -> np.ndarray:
    """
    Return purified interaction contributions evaluated at the
    original paired reference observations.
    """
    purified = (
        purify_interaction_grid(
            raw_grid
        )
    )

    return np.diag(
        purified
    ).copy()


def _safe_pearson(
    first: np.ndarray,
    second: np.ndarray,
    variance_tolerance: float = 1e-12,
) -> float:
    first = np.asarray(
        first,
        dtype=np.float64,
    )

    second = np.asarray(
        second,
        dtype=np.float64,
    )

    if first.shape != second.shape:
        raise ValueError(
            "Surface vectors must have identical shapes."
        )

    if first.ndim != 1:
        raise ValueError(
            "Surface vectors must be one-dimensional."
        )

    if (
        np.std(first)
        < variance_tolerance
        or np.std(second)
        < variance_tolerance
    ):
        return 0.0

    correlation = float(
        np.corrcoef(
            first,
            second,
        )[0, 1]
    )

    if not np.isfinite(
        correlation
    ):
        return 0.0

    return correlation


def compute_isr(
    purified_vectors: tuple[
        np.ndarray,
        ...
    ],
) -> SurfaceAgreement:
    """
    Interaction Surface Reproducibility.

    ISR is the mean pairwise Pearson correlation between purified
    interaction vectors learned in the runs where the pair was selected.
    """
    if len(
        purified_vectors
    ) < 2:
        raise ValueError(
            "At least two interaction surfaces are required for ISR."
        )

    reference_shape = (
        purified_vectors[0].shape
    )

    for vector in (
        purified_vectors
    ):
        if vector.shape != (
            reference_shape
        ):
            raise ValueError(
                "All purified vectors must share the same shape."
            )

    correlations: list[
        float
    ] = []

    for first, second in combinations(
        purified_vectors,
        2,
    ):
        correlations.append(
            _safe_pearson(
                first,
                second,
            )
        )

    return SurfaceAgreement(
        n_surfaces=len(
            purified_vectors
        ),
        pairwise_correlations=tuple(
            correlations
        ),
        isr=float(
            np.mean(
                correlations
            )
        ),
    )