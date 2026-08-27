import numpy as np
import torch
from sklearn.model_selection import (
    train_test_split,
)

from agnam.data.preprocessing import (
    TabularPreprocessor,
)
from agnam.data.synthetic import (
    generate_s1,
)
from agnam.models.agnam import (
    AGNAM,
)
from agnam.training.agnam_trainer import (
    AGNAMTrainingConfig,
    apply_agnam_centering,
    compute_agnam_binary_metrics,
    estimate_agnam_centering_offsets,
    predict_agnam_probabilities,
    train_agnam,
)


TRUE_S1_INTERACTIONS = (
    ("x3", "x4"),
    ("x5", "x6"),
    ("x8", "x9"),
)


def prepare_small_s1(
    seed: int = 42,
    n: int = 800,
):
    dataset = generate_s1(
        seed=seed,
        n=n,
    )

    indices = np.arange(
        len(dataset.X)
    )

    train_idx, val_idx = (
        train_test_split(
            indices,
            test_size=0.25,
            random_state=seed,
            stratify=dataset.y,
        )
    )

    X_train = (
        dataset.X
        .iloc[
            train_idx
        ]
        .reset_index(
            drop=True
        )
    )

    X_val = (
        dataset.X
        .iloc[
            val_idx
        ]
        .reset_index(
            drop=True
        )
    )

    y_train = (
        dataset.y
        .iloc[
            train_idx
        ]
        .to_numpy()
    )

    y_val = (
        dataset.y
        .iloc[
            val_idx
        ]
        .to_numpy()
    )

    preprocessor = (
        TabularPreprocessor()
    )

    train_data = (
        preprocessor
        .fit_transform(
            X_train
        )
    )

    val_data = (
        preprocessor
        .transform(
            X_val
        )
    )

    return (
        train_data,
        y_train,
        val_data,
        y_val,
    )


def make_small_model(
    feature_specs,
):
    return AGNAM(
        feature_specs=feature_specs,
        interaction_pairs=(
            TRUE_S1_INTERACTIONS
        ),
        main_hidden_dim=16,
        main_depth=1,
        main_dropout=0.0,
        interaction_embedding_dim=8,
        interaction_hidden_dim=16,
        interaction_depth=1,
        interaction_dropout=0.0,
    )


def test_agnam_binary_metrics_perfect_prediction():
    y = np.array(
        [0, 0, 1, 1]
    )

    probabilities = np.array(
        [
            0.05,
            0.10,
            0.90,
            0.95,
        ]
    )

    metrics = (
        compute_agnam_binary_metrics(
            y_true=y,
            probabilities=probabilities,
        )
    )

    assert np.isclose(
        metrics["auroc"],
        1.0,
    )

    assert np.isclose(
        metrics["auprc"],
        1.0,
    )

    assert np.isclose(
        metrics[
            "balanced_accuracy"
        ],
        1.0,
    )

    assert np.isclose(
        metrics["f1"],
        1.0,
    )


def test_prediction_probabilities_are_valid():
    (
        train_data,
        _,
        val_data,
        _,
    ) = prepare_small_s1(
        n=400
    )

    model = make_small_model(
        train_data.feature_specs
    )

    probabilities = (
        predict_agnam_probabilities(
            model=model,
            transformed=val_data,
            device=torch.device(
                "cpu"
            ),
        )
    )

    assert probabilities.shape == (
        len(val_data),
    )

    assert np.isfinite(
        probabilities
    ).all()

    assert np.all(
        probabilities >= 0.0
    )

    assert np.all(
        probabilities <= 1.0
    )


def test_centering_offsets_have_correct_shapes():
    (
        train_data,
        _,
        _,
        _,
    ) = prepare_small_s1(
        n=400
    )

    model = make_small_model(
        train_data.feature_specs
    )

    (
        main_offsets,
        interaction_offsets,
    ) = estimate_agnam_centering_offsets(
        model=model,
        transformed=train_data,
        device=torch.device(
            "cpu"
        ),
    )

    assert main_offsets.shape == (
        model.n_features,
    )

    assert interaction_offsets.shape == (
        model.n_interactions,
    )

    assert torch.isfinite(
        main_offsets
    ).all()

    assert torch.isfinite(
        interaction_offsets
    ).all()


def test_centering_preserves_final_predictions():
    (
        train_data,
        _,
        val_data,
        _,
    ) = prepare_small_s1(
        n=400
    )

    model = make_small_model(
        train_data.feature_specs
    )

    model.eval()

    before = (
        predict_agnam_probabilities(
            model=model,
            transformed=val_data,
            device=torch.device(
                "cpu"
            ),
        )
    )

    apply_agnam_centering(
        model=model,
        transformed=train_data,
        device=torch.device(
            "cpu"
        ),
    )

    after = (
        predict_agnam_probabilities(
            model=model,
            transformed=val_data,
            device=torch.device(
                "cpu"
            ),
        )
    )

    np.testing.assert_allclose(
        before,
        after,
        atol=1e-6,
        rtol=1e-6,
    )


def test_final_agnam_training_runs():
    (
        train_data,
        y_train,
        val_data,
        y_val,
    ) = prepare_small_s1(
        n=800
    )

    torch.manual_seed(
        42
    )

    model = make_small_model(
        train_data.feature_specs
    )

    config = (
        AGNAMTrainingConfig(
            learning_rate=1e-3,
            weight_decay=1e-5,
            batch_size=128,
            max_epochs=5,
            patience=5,
            seed=42,
            deterministic=True,
        )
    )

    result = train_agnam(
        model=model,
        train_data=train_data,
        train_y=y_train,
        val_data=val_data,
        val_y=y_val,
        config=config,
        device=torch.device(
            "cpu"
        ),
    )

    assert result.best_epoch >= 0

    assert np.isfinite(
        result.best_validation_auc
    )

    assert (
        0.0
        <= result.metrics["auroc"]
        <= 1.0
    )

    probabilities = (
        predict_agnam_probabilities(
            model=model,
            transformed=val_data,
            device=torch.device(
                "cpu"
            ),
        )
    )

    assert probabilities.shape == (
        len(y_val),
    )


def test_final_training_is_reproducible_on_cpu():
    (
        train_data,
        y_train,
        val_data,
        y_val,
    ) = prepare_small_s1(
        seed=123,
        n=600,
    )

    torch.manual_seed(
        999
    )

    model_1 = make_small_model(
        train_data.feature_specs
    )

    model_2 = make_small_model(
        train_data.feature_specs
    )

    model_2.load_state_dict(
        model_1.state_dict()
    )

    config = (
        AGNAMTrainingConfig(
            learning_rate=1e-3,
            weight_decay=1e-5,
            batch_size=128,
            max_epochs=3,
            patience=3,
            seed=123,
            deterministic=True,
        )
    )

    result_1 = train_agnam(
        model=model_1,
        train_data=train_data,
        train_y=y_train,
        val_data=val_data,
        val_y=y_val,
        config=config,
        device=torch.device(
            "cpu"
        ),
    )

    result_2 = train_agnam(
        model=model_2,
        train_data=train_data,
        train_y=y_train,
        val_data=val_data,
        val_y=y_val,
        config=config,
        device=torch.device(
            "cpu"
        ),
    )

    np.testing.assert_allclose(
        result_1.history[
            "train_loss"
        ],
        result_2.history[
            "train_loss"
        ],
        atol=1e-7,
        rtol=1e-6,
    )

    np.testing.assert_allclose(
        result_1.history[
            "val_auc"
        ],
        result_2.history[
            "val_auc"
        ],
        atol=1e-7,
        rtol=1e-6,
    )