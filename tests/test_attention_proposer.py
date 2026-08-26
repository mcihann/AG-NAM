import numpy as np
import pandas as pd
import pytest
import torch

from agnam.data.preprocessing import TabularPreprocessor
from agnam.models.attention_proposer import (
    ResidualAttentionProposer,
)


def make_mixed_data():
    return pd.DataFrame(
        {
            "age": [
                22.0,
                35.0,
                np.nan,
                49.0,
                61.0,
            ],
            "bmi": [
                20.0,
                25.0,
                30.0,
                28.0,
                33.0,
            ],
            "sex": [
                "F",
                "M",
                "F",
                "M",
                None,
            ],
            "group": [
                "A",
                "A",
                "B",
                "C",
                "B",
            ],
        }
    )


def to_torch(transformed):
    return (
        torch.from_numpy(
            transformed.numeric
        ),
        torch.from_numpy(
            transformed.numeric_missing
        ),
        torch.from_numpy(
            transformed.categorical
        ),
    )


def test_proposer_output_shapes():
    X = make_mixed_data()

    preprocessor = TabularPreprocessor()

    transformed = (
        preprocessor.fit_transform(X)
    )

    model = ResidualAttentionProposer(
        feature_specs=transformed.feature_specs,
        d_model=32,
        n_heads=4,
        n_layers=2,
        dropout=0.0,
    )

    numeric, missing, categorical = (
        to_torch(transformed)
    )

    output = model(
        numeric,
        missing,
        categorical,
    )

    assert output.residual_prediction.shape == (
        5,
    )

    assert output.contextual_tokens.shape == (
        5,
        4,
        32,
    )

    assert len(
        output.attention_maps
    ) == 2

    for attention in output.attention_maps:
        assert attention.shape == (
            5,
            4,
            4,
            4,
        )


def test_attention_rows_sum_to_one():
    X = make_mixed_data()

    preprocessor = TabularPreprocessor()
    transformed = (
        preprocessor.fit_transform(X)
    )

    model = ResidualAttentionProposer(
        feature_specs=transformed.feature_specs,
        d_model=32,
        n_heads=4,
        n_layers=2,
        dropout=0.0,
    )

    model.eval()

    numeric, missing, categorical = (
        to_torch(transformed)
    )

    output = model(
        numeric,
        missing,
        categorical,
    )

    for attention in output.attention_maps:
        row_sums = attention.sum(
            dim=-1
        )

        assert torch.allclose(
            row_sums,
            torch.ones_like(
                row_sums
            ),
            atol=1e-6,
            rtol=1e-6,
        )


def test_attention_maps_receive_residual_gradients():
    X = make_mixed_data()

    preprocessor = TabularPreprocessor()

    transformed = (
        preprocessor.fit_transform(X)
    )

    model = ResidualAttentionProposer(
        feature_specs=transformed.feature_specs,
        d_model=32,
        n_heads=4,
        n_layers=2,
        dropout=0.0,
    )

    numeric, missing, categorical = (
        to_torch(transformed)
    )

    output = model(
        numeric,
        missing,
        categorical,
    )

    gradients = torch.autograd.grad(
        outputs=output.residual_prediction.sum(),
        inputs=output.attention_maps,
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )

    assert len(
        gradients
    ) == 2

    for gradient in gradients:
        assert gradient is not None

        assert gradient.shape == (
            5,
            4,
            4,
            4,
        )

        assert torch.isfinite(
            gradient
        ).all()


def test_feature_order_is_preserved():
    X = make_mixed_data()

    preprocessor = TabularPreprocessor()
    transformed = (
        preprocessor.fit_transform(X)
    )

    model = ResidualAttentionProposer(
        feature_specs=transformed.feature_specs,
        d_model=16,
        n_heads=4,
        n_layers=1,
        dropout=0.0,
    )

    assert model.feature_names == (
        "age",
        "bmi",
        "sex",
        "group",
    )


def test_all_numeric_data():
    X = pd.DataFrame(
        {
            "x1": [0.1, 0.2, 0.3, 0.4],
            "x2": [1.0, 2.0, 3.0, 4.0],
            "x3": [2.0, 1.0, 4.0, 3.0],
        }
    )

    preprocessor = TabularPreprocessor()
    transformed = (
        preprocessor.fit_transform(X)
    )

    model = ResidualAttentionProposer(
        feature_specs=transformed.feature_specs,
        d_model=16,
        n_heads=4,
        n_layers=1,
        dropout=0.0,
    )

    numeric, missing, categorical = (
        to_torch(transformed)
    )

    output = model(
        numeric,
        missing,
        categorical,
    )

    assert output.residual_prediction.shape == (
        4,
    )

    assert output.attention_maps[
        0
    ].shape == (
        4,
        4,
        3,
        3,
    )


def test_all_categorical_data():
    X = pd.DataFrame(
        {
            "a": ["x", "y", "z", "x"],
            "b": ["m", "n", "m", "n"],
            "c": ["u", "u", "v", "v"],
        }
    )

    preprocessor = TabularPreprocessor(
        numeric_features=[],
        categorical_features=[
            "a",
            "b",
            "c",
        ],
    )

    transformed = (
        preprocessor.fit_transform(X)
    )

    model = ResidualAttentionProposer(
        feature_specs=transformed.feature_specs,
        d_model=16,
        n_heads=4,
        n_layers=1,
        dropout=0.0,
    )

    numeric, missing, categorical = (
        to_torch(transformed)
    )

    output = model(
        numeric,
        missing,
        categorical,
    )

    assert output.residual_prediction.shape == (
        4,
    )

    assert output.attention_maps[
        0
    ].shape == (
        4,
        4,
        3,
        3,
    )


def test_invalid_head_configuration_raises():
    X = make_mixed_data()

    preprocessor = TabularPreprocessor()

    transformed = (
        preprocessor.fit_transform(X)
    )

    with pytest.raises(ValueError):
        ResidualAttentionProposer(
            feature_specs=transformed.feature_specs,
            d_model=30,
            n_heads=4,
            n_layers=2,
        )