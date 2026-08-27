import numpy as np
import pandas as pd
import pytest
import torch

from agnam.data.preprocessing import (
    TabularPreprocessor,
)
from agnam.models.agnam import AGNAM


def make_mixed_data():
    return pd.DataFrame(
        {
            "age": [
                21.0,
                34.0,
                np.nan,
                48.0,
                57.0,
                63.0,
            ],
            "bmi": [
                20.0,
                24.0,
                29.0,
                32.0,
                27.0,
                31.0,
            ],
            "sex": [
                "F",
                "M",
                "F",
                "M",
                None,
                "F",
            ],
            "group": [
                "A",
                "A",
                "B",
                "B",
                "C",
                "C",
            ],
        }
    )


def to_torch(
    transformed,
):
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


def build_model():
    X = make_mixed_data()

    preprocessor = (
        TabularPreprocessor()
    )

    transformed = (
        preprocessor.fit_transform(X)
    )

    torch.manual_seed(
        42
    )

    model = AGNAM(
        feature_specs=(
            transformed.feature_specs
        ),
        interaction_pairs=[
            ("age", "bmi"),
            ("sex", "group"),
            ("age", "sex"),
        ],
        main_hidden_dim=16,
        main_depth=1,
        main_dropout=0.0,
        categorical_embedding_dim=8,
        interaction_embedding_dim=8,
        interaction_hidden_dim=16,
        interaction_depth=1,
        interaction_dropout=0.0,
    )

    return (
        model,
        transformed,
    )


def test_agnam_output_shapes():
    model, transformed = (
        build_model()
    )

    numeric, missing, categorical = (
        to_torch(
            transformed
        )
    )

    output = model(
        numeric=numeric,
        numeric_missing=missing,
        categorical=categorical,
    )

    assert output.logits.shape == (
        6,
    )

    assert (
        output
        .main_contributions
        .shape
        == (
            6,
            4,
        )
    )

    assert (
        output
        .interaction_contributions
        .shape
        == (
            6,
            3,
        )
    )

    assert output.baseline.ndim == 0


def test_exact_full_decomposition():
    model, transformed = (
        build_model()
    )

    numeric, missing, categorical = (
        to_torch(
            transformed
        )
    )

    output = model(
        numeric=numeric,
        numeric_missing=missing,
        categorical=categorical,
    )

    reconstructed = (
        output.baseline
        + output
        .main_contributions
        .sum(dim=1)
        + output
        .interaction_contributions
        .sum(dim=1)
    )

    assert torch.allclose(
        output.logits,
        reconstructed,
        atol=1e-7,
        rtol=1e-6,
    )


def test_interaction_pairs_are_normalized_by_feature_order():
    X = make_mixed_data()

    preprocessor = (
        TabularPreprocessor()
    )

    transformed = (
        preprocessor.fit_transform(X)
    )

    model = AGNAM(
        feature_specs=(
            transformed.feature_specs
        ),
        interaction_pairs=[
            ("bmi", "age"),
            ("group", "sex"),
        ],
        main_hidden_dim=8,
        main_depth=1,
        main_dropout=0.0,
        interaction_embedding_dim=4,
        interaction_hidden_dim=8,
        interaction_depth=1,
        interaction_dropout=0.0,
    )

    assert model.interaction_pairs == (
        ("age", "bmi"),
        ("sex", "group"),
    )


def test_duplicate_interaction_raises():
    X = make_mixed_data()

    preprocessor = (
        TabularPreprocessor()
    )

    transformed = (
        preprocessor.fit_transform(X)
    )

    with pytest.raises(
        ValueError
    ):
        AGNAM(
            feature_specs=(
                transformed.feature_specs
            ),
            interaction_pairs=[
                ("age", "bmi"),
                ("bmi", "age"),
            ],
        )


def test_unknown_interaction_feature_raises():
    X = make_mixed_data()

    preprocessor = (
        TabularPreprocessor()
    )

    transformed = (
        preprocessor.fit_transform(X)
    )

    with pytest.raises(
        ValueError
    ):
        AGNAM(
            feature_specs=(
                transformed.feature_specs
            ),
            interaction_pairs=[
                (
                    "age",
                    "does_not_exist",
                )
            ],
        )


def test_self_interaction_raises():
    X = make_mixed_data()

    preprocessor = (
        TabularPreprocessor()
    )

    transformed = (
        preprocessor.fit_transform(X)
    )

    with pytest.raises(
        ValueError
    ):
        AGNAM(
            feature_specs=(
                transformed.feature_specs
            ),
            interaction_pairs=[
                ("age", "age")
            ],
        )


def test_interaction_centering_preserves_predictions():
    model, transformed = (
        build_model()
    )

    numeric, missing, categorical = (
        to_torch(
            transformed
        )
    )

    model.eval()

    with torch.no_grad():
        before = model(
            numeric=numeric,
            numeric_missing=missing,
            categorical=categorical,
        )

        raw = (
            model
            .raw_interaction_contributions(
                numeric=numeric,
                numeric_missing=missing,
                categorical=categorical,
            )
        )

        offsets = raw.mean(
            dim=0
        )

        model.set_interaction_centering_offsets(
            offsets
        )

        after = model(
            numeric=numeric,
            numeric_missing=missing,
            categorical=categorical,
        )

    assert torch.allclose(
        before.logits,
        after.logits,
        atol=1e-6,
        rtol=1e-6,
    )

    assert torch.allclose(
        after
        .interaction_contributions
        .mean(dim=0),
        torch.zeros(
            model.n_interactions
        ),
        atol=1e-6,
        rtol=1e-6,
    )


def test_clear_centering_resets_all_offsets():
    model, transformed = (
        build_model()
    )

    model.main_effect_model.set_centering_offsets(
        torch.ones(
            model.n_features
        )
    )

    model.set_interaction_centering_offsets(
        torch.ones(
            model.n_interactions
        )
    )

    model.clear_centering()

    assert torch.allclose(
        model
        .main_effect_model
        .centering_offsets,
        torch.zeros(
            model.n_features
        ),
    )

    assert torch.allclose(
        model
        .interaction_centering_offsets,
        torch.zeros(
            model.n_interactions
        ),
    )


def test_no_interactions_is_supported():
    X = make_mixed_data()

    preprocessor = (
        TabularPreprocessor()
    )

    transformed = (
        preprocessor.fit_transform(X)
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

    numeric, missing, categorical = (
        to_torch(
            transformed
        )
    )

    output = model(
        numeric=numeric,
        numeric_missing=missing,
        categorical=categorical,
    )

    assert (
        output
        .interaction_contributions
        .shape
        == (
            6,
            0,
        )
    )

    reconstructed = (
        output.baseline
        + output
        .main_contributions
        .sum(dim=1)
    )

    assert torch.allclose(
        output.logits,
        reconstructed,
        atol=1e-7,
        rtol=1e-6,
    )


def test_gradients_reach_main_and_interaction_networks():
    model, transformed = (
        build_model()
    )

    numeric, missing, categorical = (
        to_torch(
            transformed
        )
    )

    output = model(
        numeric=numeric,
        numeric_missing=missing,
        categorical=categorical,
    )

    loss = (
        output.logits
        .pow(2)
        .mean()
    )

    loss.backward()

    main_parameters = [
        parameter
        for parameter
        in model
        .main_effect_model
        .parameters()
        if parameter.requires_grad
    ]

    interaction_parameters = [
        parameter
        for network
        in model.interaction_networks
        for parameter
        in network.parameters()
        if parameter.requires_grad
    ]

    assert any(
        parameter.grad is not None
        for parameter
        in main_parameters
    )

    assert any(
        parameter.grad is not None
        for parameter
        in interaction_parameters
    )


def test_final_model_contains_no_attention_modules():
    model, _ = build_model()

    module_class_names = {
        module
        .__class__
        .__name__
        .lower()
        for module
        in model.modules()
    }

    assert not any(
        "attention" in name
        for name
        in module_class_names
    )