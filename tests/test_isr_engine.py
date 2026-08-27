import numpy as np
import pandas as pd
import torch
from torch import nn

from agnam.data.preprocessing import (
    FeatureSpec,
    TabularPreprocessor,
)
from agnam.interpretation.surface_reproducibility import (
    diagonal_purified_vector,
    evaluate_pairwise_raw_grid,
)
from agnam.training.isr import (
    sample_reference_indices,
)


class ProductPairModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_specs = (
            FeatureSpec(
                name="x1",
                kind="numeric",
                original_index=0,
                transformed_index=0,
            ),
            FeatureSpec(
                name="x2",
                kind="numeric",
                original_index=1,
                transformed_index=1,
            ),
        )

    def forward(
        self,
        numeric,
        numeric_missing,
        categorical,
    ):
        return (
            numeric[:, 0]
            * numeric[:, 1]
        )


def make_numeric_reference():
    X = pd.DataFrame(
        {
            "x1": [
                -2.0,
                -1.0,
                0.0,
                1.0,
                2.0,
            ],
            "x2": [
                -1.5,
                -0.5,
                0.0,
                0.5,
                1.5,
            ],
        }
    )

    preprocessor = (
        TabularPreprocessor()
    )

    return (
        preprocessor
        .fit_transform(X)
    )


def test_reference_sampling_is_reproducible():
    first = sample_reference_indices(
        1000,
        reference_size=100,
        seed=2026,
    )

    second = sample_reference_indices(
        1000,
        reference_size=100,
        seed=2026,
    )

    np.testing.assert_array_equal(
        first,
        second,
    )

    assert len(
        np.unique(first)
    ) == 100


def test_reference_size_is_capped_by_dataset():
    indices = sample_reference_indices(
        25,
        reference_size=512,
        seed=2026,
    )

    assert len(indices) == 25

    assert len(
        np.unique(indices)
    ) == 25


def test_pairwise_raw_grid_matches_product_function():
    transformed = (
        make_numeric_reference()
    )

    model = ProductPairModel()

    raw_grid = (
        evaluate_pairwise_raw_grid(
            model=model,
            reference_data=(
                transformed
            ),
            batch_size=4,
            device=torch.device(
                "cpu"
            ),
        )
    )

    expected = np.outer(
        transformed.numeric[
            :,
            0,
        ],
        transformed.numeric[
            :,
            1,
        ],
    )

    np.testing.assert_allclose(
        raw_grid,
        expected,
        atol=1e-7,
        rtol=1e-6,
    )


def test_raw_grid_is_batch_size_invariant():
    transformed = (
        make_numeric_reference()
    )

    model = ProductPairModel()

    first = (
        evaluate_pairwise_raw_grid(
            model=model,
            reference_data=(
                transformed
            ),
            batch_size=3,
            device=torch.device(
                "cpu"
            ),
        )
    )

    second = (
        evaluate_pairwise_raw_grid(
            model=model,
            reference_data=(
                transformed
            ),
            batch_size=100,
            device=torch.device(
                "cpu"
            ),
        )
    )

    np.testing.assert_allclose(
        first,
        second,
        atol=1e-7,
        rtol=1e-6,
    )


def test_product_function_survives_purification():
    transformed = (
        make_numeric_reference()
    )

    model = ProductPairModel()

    raw_grid = (
        evaluate_pairwise_raw_grid(
            model=model,
            reference_data=(
                transformed
            ),
            batch_size=10,
            device=torch.device(
                "cpu"
            ),
        )
    )

    purified = (
        diagonal_purified_vector(
            raw_grid
        )
    )

    expected_grid = np.outer(
        transformed.numeric[
            :,
            0,
        ],
        transformed.numeric[
            :,
            1,
        ],
    )

    expected = np.diag(
        expected_grid
    )

    np.testing.assert_allclose(
        purified,
        expected,
        atol=1e-6,
        rtol=1e-6,
    )