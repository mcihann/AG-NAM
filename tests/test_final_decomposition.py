import numpy as np
import pandas as pd
import torch

from agnam.data.preprocessing import (
    TabularPreprocessor,
)
from agnam.interpretation.final_decomposition import (
    compute_purified_agnam_decomposition,
)
from agnam.models.agnam import AGNAM


def make_data():
    return pd.DataFrame(
        {
            "x1": [
                -1.5,
                -1.0,
                -0.5,
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
            ],
            "x2": [
                1.0,
                0.5,
                -0.5,
                -1.0,
                1.5,
                0.0,
                -1.5,
                2.0,
            ],
            "category": [
                "A",
                "A",
                "B",
                "B",
                "C",
                "A",
                "C",
                "B",
            ],
        }
    )


def prepare_model():
    X = make_data()

    preprocessor = (
        TabularPreprocessor()
    )

    transformed = (
        preprocessor
        .fit_transform(X)
    )

    torch.manual_seed(
        42
    )

    model = AGNAM(
        feature_specs=(
            transformed.feature_specs
        ),
        interaction_pairs=[
            ("x1", "x2"),
            ("x2", "category"),
        ],
        main_hidden_dim=8,
        main_depth=1,
        main_dropout=0.0,
        categorical_embedding_dim=4,
        interaction_embedding_dim=4,
        interaction_hidden_dim=8,
        interaction_depth=1,
        interaction_dropout=0.0,
    )

    return (
        model,
        transformed,
    )


def raw_logits(
    model,
    transformed,
):
    numeric = torch.from_numpy(
        transformed.numeric
    )

    missing = torch.from_numpy(
        transformed.numeric_missing
    )

    categorical = torch.from_numpy(
        transformed.categorical
    )

    model.eval()

    with torch.no_grad():
        output = model(
            numeric=numeric,
            numeric_missing=missing,
            categorical=categorical,
        )

    return (
        output.logits
        .detach()
        .cpu()
    )


def test_purified_decomposition_shapes():
    model, transformed = (
        prepare_model()
    )

    result = (
        compute_purified_agnam_decomposition(
            model=model,
            transformed=transformed,
            reference_data=transformed,
            query_batch_size=3,
            surface_batch_size=16,
            device=torch.device(
                "cpu"
            ),
        )
    )

    assert result.logits.shape == (
        8,
    )

    assert (
        result
        .main_contributions
        .shape
        == (
            8,
            3,
        )
    )

    assert (
        result
        .interaction_contributions
        .shape
        == (
            8,
            2,
        )
    )

    assert result.baseline.ndim == 0

    assert result.reference_size == 8


def test_purification_preserves_raw_logits():
    model, transformed = (
        prepare_model()
    )

    original = raw_logits(
        model,
        transformed,
    )

    purified = (
        compute_purified_agnam_decomposition(
            model=model,
            transformed=transformed,
            reference_data=transformed,
            query_batch_size=4,
            surface_batch_size=32,
            device=torch.device(
                "cpu"
            ),
        )
    )

    assert torch.allclose(
        original,
        purified.logits,
        atol=1e-5,
        rtol=1e-5,
    )

    assert (
        purified
        .max_logit_invariance_error
        < 1e-5
    )


def test_purified_decomposition_is_exactly_additive():
    model, transformed = (
        prepare_model()
    )

    purified = (
        compute_purified_agnam_decomposition(
            model=model,
            transformed=transformed,
            reference_data=transformed,
            query_batch_size=4,
            surface_batch_size=32,
            device=torch.device(
                "cpu"
            ),
        )
    )

    reconstructed = (
        purified.baseline
        + purified
        .main_contributions
        .sum(dim=1)
        + purified
        .interaction_contributions
        .sum(dim=1)
    )

    assert torch.allclose(
        purified.logits,
        reconstructed,
        atol=1e-6,
        rtol=1e-6,
    )


def test_purification_is_query_batch_size_invariant():
    model, transformed = (
        prepare_model()
    )

    first = (
        compute_purified_agnam_decomposition(
            model=model,
            transformed=transformed,
            reference_data=transformed,
            query_batch_size=1,
            surface_batch_size=16,
            device=torch.device(
                "cpu"
            ),
        )
    )

    second = (
        compute_purified_agnam_decomposition(
            model=model,
            transformed=transformed,
            reference_data=transformed,
            query_batch_size=8,
            surface_batch_size=100,
            device=torch.device(
                "cpu"
            ),
        )
    )

    torch.testing.assert_close(
        first.logits,
        second.logits,
        atol=1e-6,
        rtol=1e-6,
    )

    torch.testing.assert_close(
        first.main_contributions,
        second.main_contributions,
        atol=1e-6,
        rtol=1e-6,
    )

    torch.testing.assert_close(
        first.interaction_contributions,
        second.interaction_contributions,
        atol=1e-6,
        rtol=1e-6,
    )


def test_zero_interaction_model_is_supported():
    X = make_data()

    preprocessor = (
        TabularPreprocessor()
    )

    transformed = (
        preprocessor
        .fit_transform(X)
    )

    model = AGNAM(
        feature_specs=(
            transformed.feature_specs
        ),
        interaction_pairs=[],
        main_hidden_dim=8,
        main_depth=1,
        main_dropout=0.0,
    )

    original = raw_logits(
        model,
        transformed,
    )

    result = (
        compute_purified_agnam_decomposition(
            model=model,
            transformed=transformed,
            reference_data=transformed,
            query_batch_size=4,
            surface_batch_size=16,
            device=torch.device(
                "cpu"
            ),
        )
    )

    assert (
        result
        .interaction_contributions
        .shape
        == (
            8,
            0,
        )
    )

    torch.testing.assert_close(
        original,
        result.logits,
        atol=1e-6,
        rtol=1e-6,
    )

    assert (
        result
        .max_logit_invariance_error
        < 1e-6
    )