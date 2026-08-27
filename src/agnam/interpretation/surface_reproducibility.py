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

    Purified interaction:

        h_jk
        - E_k[h_jk | x_j]
        - E_j[h_jk | x_k]
        + E_jk[h_jk]

    The empirical projection is defined relative to the product
    of the marginal reference distributions.
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

    grand_mean = float(
        raw_grid.mean()
    )

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
    interaction vectors learned in the runs where a pair was selected.
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

    for vector in purified_vectors:
        if vector.shape != reference_shape:
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


def _validate_cross_grid_inputs(
    model: PairwiseResidualNetwork,
    row_data: TransformedTabular,
    column_data: TransformedTabular,
) -> None:
    if len(row_data) == 0:
        raise ValueError(
            "row_data cannot be empty."
        )

    if len(column_data) == 0:
        raise ValueError(
            "column_data cannot be empty."
        )

    if tuple(
        row_data.feature_specs
    ) != tuple(
        column_data.feature_specs
    ):
        raise ValueError(
            "row_data and column_data must use identical "
            "feature specifications."
        )

    available_names = {
        spec.name
        for spec
        in row_data.feature_specs
    }

    for spec in model.feature_specs:
        if spec.name not in available_names:
            raise ValueError(
                f"Pairwise feature '{spec.name}' "
                "is absent from transformed data."
            )


@torch.no_grad()
def evaluate_pairwise_cross_grid(
    model: PairwiseResidualNetwork,
    row_data: TransformedTabular,
    column_data: TransformedTabular,
    *,
    batch_size: int = 8192,
    device: Optional[
        torch.device
    ] = None,
) -> np.ndarray:
    """
    Evaluate a pairwise network on a rectangular cross-product.

    For model features (j, k):

        grid[i, m] =
            h_jk(
                x_j from row_data[i],
                x_k from column_data[m]
            )

    This operation is used by the empirical functional-ANOVA
    projection.

    Returns
    -------
    np.ndarray
        Shape [n_rows, n_columns].
    """
    if batch_size <= 0:
        raise ValueError(
            "batch_size must be positive."
        )

    _validate_cross_grid_inputs(
        model=model,
        row_data=row_data,
        column_data=column_data,
    )

    if device is None:
        device = get_device()

    model = model.to(
        device
    )

    model.eval()

    n_rows = len(
        row_data
    )

    n_columns = len(
        column_data
    )

    n_numeric = (
        row_data
        .numeric
        .shape[1]
    )

    n_categorical = (
        row_data
        .categorical
        .shape[1]
    )

    total = (
        n_rows
        * n_columns
    )

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
            flat_indices
            // n_columns
        )

        column_indices = (
            flat_indices
            % n_columns
        )

        current_batch = (
            stop
            - start
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

        first_spec = (
            model.feature_specs[0]
        )

        second_spec = (
            model.feature_specs[1]
        )

        if first_spec.kind == "numeric":
            numeric[
                :,
                first_spec.transformed_index,
            ] = (
                row_data
                .numeric[
                    row_indices,
                    first_spec.transformed_index,
                ]
            )

            numeric_missing[
                :,
                first_spec.transformed_index,
            ] = (
                row_data
                .numeric_missing[
                    row_indices,
                    first_spec.transformed_index,
                ]
            )

        else:
            categorical[
                :,
                first_spec.transformed_index,
            ] = (
                row_data
                .categorical[
                    row_indices,
                    first_spec.transformed_index,
                ]
            )

        if second_spec.kind == "numeric":
            numeric[
                :,
                second_spec.transformed_index,
            ] = (
                column_data
                .numeric[
                    column_indices,
                    second_spec.transformed_index,
                ]
            )

            numeric_missing[
                :,
                second_spec.transformed_index,
            ] = (
                column_data
                .numeric_missing[
                    column_indices,
                    second_spec.transformed_index,
                ]
            )

        else:
            categorical[
                :,
                second_spec.transformed_index,
            ] = (
                column_data
                .categorical[
                    column_indices,
                    second_spec.transformed_index,
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
        n_rows,
        n_columns,
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
    Evaluate:

        h_jk(x_j_i, x_k_m)

    for every pair of observations in one common reference support.

    Returns
    -------
    np.ndarray
        Square matrix of shape [M, M].
    """
    return evaluate_pairwise_cross_grid(
        model=model,
        row_data=reference_data,
        column_data=reference_data,
        batch_size=batch_size,
        device=device,
    )