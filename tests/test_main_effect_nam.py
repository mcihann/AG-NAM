import numpy as np
import pandas as pd
import torch

from agnam.data.preprocessing import TabularPreprocessor
from agnam.models.main_effect_nam import MainEffectNAM


def make_mixed_data():
    return pd.DataFrame(
        {
            "age": [
                23.0,
                35.0,
                np.nan,
                52.0,
                41.0,
            ],
            "bmi": [
                21.0,
                27.0,
                30.0,
                33.0,
                25.0,
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
                "B",
                "C",
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


def test_output_shapes():
    X = make_mixed_data()

    preprocessor = TabularPreprocessor()
    transformed = preprocessor.fit_transform(X)

    model = MainEffectNAM(
        feature_specs=transformed.feature_specs,
        hidden_dim=16,
        depth=2,
        dropout=0.0,
        categorical_embedding_dim=8,
    )

    numeric, missing, categorical = to_torch(
        transformed
    )

    output = model(
        numeric,
        missing,
        categorical,
    )

    assert output.logits.shape == (5,)
    assert output.main_contributions.shape == (
        5,
        4,
    )

    assert output.baseline.ndim == 0


def test_exact_additive_decomposition():
    X = make_mixed_data()

    preprocessor = TabularPreprocessor()
    transformed = preprocessor.fit_transform(X)

    model = MainEffectNAM(
        feature_specs=transformed.feature_specs,
        hidden_dim=16,
        depth=2,
        dropout=0.0,
        categorical_embedding_dim=8,
    )

    numeric, missing, categorical = to_torch(
        transformed
    )

    output = model(
        numeric,
        missing,
        categorical,
    )

    reconstructed = (
        output.baseline
        + output.main_contributions.sum(dim=1)
    )

    assert torch.allclose(
        output.logits,
        reconstructed,
        atol=1e-7,
        rtol=1e-6,
    )


def test_contributions_follow_original_feature_order():
    X = make_mixed_data()

    preprocessor = TabularPreprocessor()
    transformed = preprocessor.fit_transform(X)

    model = MainEffectNAM(
        feature_specs=transformed.feature_specs,
        hidden_dim=8,
        depth=1,
        dropout=0.0,
        categorical_embedding_dim=4,
    )

    assert model.feature_names == (
        "age",
        "bmi",
        "sex",
        "group",
    )


def test_centering_preserves_predictions_exactly():
    X = make_mixed_data()

    preprocessor = TabularPreprocessor()
    transformed = preprocessor.fit_transform(X)

    model = MainEffectNAM(
        feature_specs=transformed.feature_specs,
        hidden_dim=16,
        depth=2,
        dropout=0.0,
        categorical_embedding_dim=8,
    )

    numeric, missing, categorical = to_torch(
        transformed
    )

    model.eval()

    with torch.no_grad():
        before = model(
            numeric,
            missing,
            categorical,
        )

        raw = model.raw_contributions(
            numeric,
            missing,
            categorical,
        )

        offsets = raw.mean(dim=0)

        model.set_centering_offsets(
            offsets
        )

        after = model(
            numeric,
            missing,
            categorical,
        )

    assert torch.allclose(
        before.logits,
        after.logits,
        atol=1e-6,
        rtol=1e-6,
    )

    assert torch.allclose(
        after.main_contributions.mean(dim=0),
        torch.zeros(
            model.n_features
        ),
        atol=1e-6,
        rtol=1e-6,
    )


def test_clear_centering_resets_offsets():
    X = make_mixed_data()

    preprocessor = TabularPreprocessor()
    transformed = preprocessor.fit_transform(X)

    model = MainEffectNAM(
        feature_specs=transformed.feature_specs,
        hidden_dim=8,
        depth=1,
        dropout=0.0,
        categorical_embedding_dim=4,
    )

    model.set_centering_offsets(
        torch.ones(model.n_features)
    )

    model.clear_centering()

    assert torch.allclose(
        model.centering_offsets,
        torch.zeros(model.n_features),
    )


def test_gradients_flow_through_all_feature_networks():
    X = make_mixed_data()

    preprocessor = TabularPreprocessor()
    transformed = preprocessor.fit_transform(X)

    model = MainEffectNAM(
        feature_specs=transformed.feature_specs,
        hidden_dim=8,
        depth=1,
        dropout=0.0,
        categorical_embedding_dim=4,
    )

    numeric, missing, categorical = to_torch(
        transformed
    )

    output = model(
        numeric,
        missing,
        categorical,
    )

    loss = output.logits.pow(2).mean()

    loss.backward()

    for network in model.feature_networks:
        trainable_parameters = [
            parameter
            for parameter in network.parameters()
            if parameter.requires_grad
        ]

        assert trainable_parameters

        assert any(
            parameter.grad is not None
            for parameter in trainable_parameters
        )


def test_all_numeric_dataset():
    X = pd.DataFrame(
        {
            "x1": [0.1, 0.2, 0.3, 0.4],
            "x2": [1.0, 2.0, 3.0, 4.0],
        }
    )

    preprocessor = TabularPreprocessor()
    transformed = preprocessor.fit_transform(X)

    model = MainEffectNAM(
        feature_specs=transformed.feature_specs,
        hidden_dim=8,
        depth=1,
        dropout=0.0,
    )

    numeric, missing, categorical = to_torch(
        transformed
    )

    output = model(
        numeric,
        missing,
        categorical,
    )

    assert output.logits.shape == (4,)
    assert output.main_contributions.shape == (
        4,
        2,
    )


def test_all_categorical_dataset():
    X = pd.DataFrame(
        {
            "a": ["x", "y", "x", "z"],
            "b": ["m", "m", "n", "n"],
        }
    )

    preprocessor = TabularPreprocessor(
        numeric_features=[],
        categorical_features=[
            "a",
            "b",
        ],
    )

    transformed = preprocessor.fit_transform(X)

    model = MainEffectNAM(
        feature_specs=transformed.feature_specs,
        hidden_dim=8,
        depth=1,
        dropout=0.0,
        categorical_embedding_dim=4,
    )

    numeric, missing, categorical = to_torch(
        transformed
    )

    output = model(
        numeric,
        missing,
        categorical,
    )

    assert output.logits.shape == (4,)
    assert output.main_contributions.shape == (
        4,
        2,
    )