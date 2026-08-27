from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Optional

import numpy as np
import torch

from agnam.data.preprocessing import TransformedTabular
from agnam.models.pairwise_interaction import PairwiseResidualNetwork
from agnam.utils.reproducibility import get_device


@dataclass(frozen=True)
class SurfaceAgreement:
    n_surfaces: int
    pairwise_correlations: tuple[float, ...]
    isr: float


def purify_interaction_grid(
    raw_grid: np.ndarray,
) -> np.ndarray:
    """
    Empirical functional-ANOVA purification.

    raw_grid[i, m] represents:

        h(x_j_i, x_k_m)

    Returns:

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

    if raw_grid.shape[0] != raw_grid.shape[1]:
        raise ValueError(
            "raw_grid must be square."
        )

    if not np.isfinite(raw_grid).all():
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
    Purified contributions evaluated at the original
    paired reference observations.
    """
    purified = purify_interaction_grid(
        raw_grid
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
        np.std(first) < variance_tolerance
        or np.std(second) < variance_tolerance
    ):
        return 0.0

    correlation = float(
        np.corrcoef(
            first,
            second,
        )[0, 1]
    )

    if not np.isfinite(correlation):
        return 0.0

    return correlation


def compute_isr(
    purified_vectors: tuple[
        np.ndarray,
        ...
    ],
) -> SurfaceAgreement:
    """
    Mean pairwise Pearson agreement between purified
    interaction surfaces.
    """
    if len(purified_vectors) < 2:
        raise ValueError(
            "At least two interaction surfaces are required for ISR."
        )

    reference_shape = (
        purified_vectors[0].shape
    )

    for vector in purified_vectors:
        if vector.shape != reference_shape:
            raise ValueError(
                "All purified vectors must share the same shape."
            )

    correlations: list[float] = []

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


@torch.no_grad()
def evaluate_pairwise_raw_grid(
    model: PairwiseResidualNetwork,
    reference_data: TransformedTabular,
    *,
    batch_size: int = 8192,
    device: Optional[
        torch.device
    ] = None,
) -> np.ndarray:
    """
    Evaluate h_jk(x_j_i, x_k_m) for all M x M combinations
    in a common transformed reference support.

    The first feature receives observation i.
    The second feature receives observation m.
    """
    if len(reference_data) == 0:
        raise ValueError(
            "reference_data cannot be empty."
        )

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be positive."
        )

    if device is None:
        device = get_device()

    model = model.to(
        device
    )

    model.eval()

    m = len(
        reference_data
    )

    n_numeric = (
        reference_data
        .numeric
        .shape[1]
    )

    n_categorical = (
        reference_data
        .categorical
        .shape[1]
    )

    total = m * m

    predictions = np.empty(
        total,
        dtype=np.float64,
    )

    for start in range(
        0,
        total,
        batch_size,
    ):
        stop = min(
            start + batch_size,
            total,
        )

        flat_indices = np.arange(
            start,
            stop,
            dtype=np.int64,
        )

        row_indices = (
            flat_indices // m
        )

        column_indices = (
            flat_indices % m
        )

        current_batch = (
            stop - start
        )

        numeric = np.zeros(
            (
                current_batch,
                n_numeric,
            ),
            dtype=np.float32,
        )

        numeric_missing = np.zeros(
            (
                current_batch,
                n_numeric,
            ),
            dtype=np.float32,
        )

        categorical = np.zeros(
            (
                current_batch,
                n_categorical,
            ),
            dtype=np.int64,
        )

        source_indices = (
            row_indices,
            column_indices,
        )

        for (
            spec,
            source
        ) in zip(
            model.feature_specs,
            source_indices,
        ):
            if spec.kind == "numeric":
                numeric[
                    :,
                    spec.transformed_index,
                ] = (
                    reference_data
                    .numeric[
                        source,
                        spec.transformed_index,
                    ]
                )

                numeric_missing[
                    :,
                    spec.transformed_index,
                ] = (
                    reference_data
                    .numeric_missing[
                        source,
                        spec.transformed_index,
                    ]
                )

            else:
                categorical[
                    :,
                    spec.transformed_index,
                ] = (
                    reference_data
                    .categorical[
                        source,
                        spec.transformed_index,
                    ]
                )

        prediction = model(
            numeric=torch.from_numpy(
                numeric
            ).to(
                device
            ),
            numeric_missing=torch.from_numpy(
                numeric_missing
            ).to(
                device
            ),
            categorical=torch.from_numpy(
                categorical
            ).to(
                device
            ),
        )

        predictions[
            start:stop
        ] = (
            prediction
            .detach()
            .cpu()
            .numpy()
        )

    return predictions.reshape(
        m,
        m,
    )