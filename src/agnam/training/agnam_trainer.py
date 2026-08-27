from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

from agnam.data.preprocessing import (
    TransformedTabular,
)
from agnam.models.agnam import AGNAM
from agnam.utils.reproducibility import (
    get_device,
    seed_everything,
)


@dataclass
class AGNAMTrainingConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5

    batch_size: int = 256

    max_epochs: int = 300
    patience: int = 30
    min_delta: float = 1e-4

    seed: int = 42
    deterministic: bool = True


@dataclass
class AGNAMTrainingResult:
    best_epoch: int
    best_validation_auc: float

    history: dict[
        str,
        list[float],
    ]

    metrics: dict[
        str,
        float,
    ]


def _make_loader(
    transformed: TransformedTabular,
    y: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    y = np.asarray(
        y,
        dtype=np.float32,
    )

    if len(transformed) != len(y):
        raise ValueError(
            "Features and targets must have the same length."
        )

    if not np.all(
        np.isin(y, [0.0, 1.0])
    ):
        raise ValueError(
            "Targets must contain only 0 and 1."
        )

    dataset = TensorDataset(
        torch.from_numpy(
            transformed.numeric
        ),
        torch.from_numpy(
            transformed.numeric_missing
        ),
        torch.from_numpy(
            transformed.categorical
        ),
        torch.from_numpy(
            y
        ),
    )

    generator = torch.Generator()
    generator.manual_seed(
        seed
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=(
            generator
            if shuffle
            else None
        ),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def _safe_auc(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    if np.unique(
        y_true
    ).size < 2:
        return float(
            "nan"
        )

    return float(
        roc_auc_score(
            y_true,
            probabilities,
        )
    )


def compute_agnam_binary_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float = 0.50,
) -> dict[str, float]:
    y_true = np.asarray(
        y_true,
        dtype=np.int64,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    if y_true.shape != (
        probabilities.shape
    ):
        raise ValueError(
            "y_true and probabilities must have identical shapes."
        )

    predictions = (
        probabilities
        >= threshold
    ).astype(
        np.int64
    )

    return {
        "auroc": _safe_auc(
            y_true,
            probabilities,
        ),
        "auprc": float(
            average_precision_score(
                y_true,
                probabilities,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                predictions,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
    }


@torch.no_grad()
def predict_agnam_probabilities(
    model: AGNAM,
    transformed: TransformedTabular,
    *,
    batch_size: int = 1024,
    device: Optional[
        torch.device
    ] = None,
) -> np.ndarray:
    if device is None:
        device = get_device()

    model = model.to(
        device
    )

    model.eval()

    dummy_y = np.zeros(
        len(transformed),
        dtype=np.float32,
    )

    loader = _make_loader(
        transformed=transformed,
        y=dummy_y,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
    )

    probabilities: list[
        np.ndarray
    ] = []

    for (
        numeric,
        missing,
        categorical,
        _,
    ) in loader:
        numeric = numeric.to(
            device,
            non_blocking=True,
        )

        missing = missing.to(
            device,
            non_blocking=True,
        )

        categorical = categorical.to(
            device,
            non_blocking=True,
        )

        output = model(
            numeric=numeric,
            numeric_missing=missing,
            categorical=categorical,
        )

        batch_probabilities = torch.sigmoid(
            output.logits
        )

        probabilities.append(
            batch_probabilities
            .cpu()
            .numpy()
        )

    return np.concatenate(
        probabilities
    )


@torch.no_grad()
def estimate_agnam_centering_offsets(
    model: AGNAM,
    transformed: TransformedTabular,
    *,
    batch_size: int = 1024,
    device: Optional[
        torch.device
    ] = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    """
    Estimate training-distribution centering offsets for both:

        main effects
        pairwise interactions

    Applying these offsets does not change predictions.
    """
    if len(transformed) == 0:
        raise ValueError(
            "Cannot estimate centering offsets from an empty dataset."
        )

    if device is None:
        device = get_device()

    model = model.to(
        device
    )

    model.eval()

    dummy_y = np.zeros(
        len(transformed),
        dtype=np.float32,
    )

    loader = _make_loader(
        transformed=transformed,
        y=dummy_y,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
    )

    main_sum = torch.zeros(
        model.n_features,
        dtype=torch.float64,
        device=device,
    )

    interaction_sum = torch.zeros(
        model.n_interactions,
        dtype=torch.float64,
        device=device,
    )

    n_seen = 0

    for (
        numeric,
        missing,
        categorical,
        _,
    ) in loader:
        numeric = numeric.to(
            device,
            non_blocking=True,
        )

        missing = missing.to(
            device,
            non_blocking=True,
        )

        categorical = categorical.to(
            device,
            non_blocking=True,
        )

        raw_main = (
            model
            .main_effect_model
            .raw_contributions(
                numeric=numeric,
                numeric_missing=missing,
                categorical=categorical,
            )
        )

        raw_interactions = (
            model
            .raw_interaction_contributions(
                numeric=numeric,
                numeric_missing=missing,
                categorical=categorical,
            )
        )

        main_sum += (
            raw_main
            .sum(dim=0)
            .to(
                torch.float64
            )
        )

        if model.n_interactions > 0:
            interaction_sum += (
                raw_interactions
                .sum(dim=0)
                .to(
                    torch.float64
                )
            )

        n_seen += (
            numeric.shape[0]
        )

    if n_seen != len(
        transformed
    ):
        raise RuntimeError(
            "Centering pass did not process every observation."
        )

    main_offsets = (
        main_sum / n_seen
    ).to(
        dtype=(
            model
            .main_effect_model
            .centering_offsets
            .dtype
        )
    )

    interaction_offsets = (
        interaction_sum / n_seen
    ).to(
        dtype=(
            model
            .interaction_centering_offsets
            .dtype
        )
    )

    return (
        main_offsets,
        interaction_offsets,
    )


def apply_agnam_centering(
    model: AGNAM,
    transformed: TransformedTabular,
    *,
    batch_size: int = 1024,
    device: Optional[
        torch.device
    ] = None,
) -> None:
    """
    Center all main and interaction contributions on the
    training distribution while preserving logits exactly.
    """
    main_offsets, interaction_offsets = (
        estimate_agnam_centering_offsets(
            model=model,
            transformed=transformed,
            batch_size=batch_size,
            device=device,
        )
    )

    model.main_effect_model.set_centering_offsets(
        main_offsets
    )

    model.set_interaction_centering_offsets(
        interaction_offsets
    )


def train_agnam(
    model: AGNAM,
    train_data: TransformedTabular,
    train_y: np.ndarray,
    val_data: TransformedTabular,
    val_y: np.ndarray,
    config: AGNAMTrainingConfig,
    *,
    device: Optional[
        torch.device
    ] = None,
) -> AGNAMTrainingResult:
    """
    Train the final attention-free AG-NAM classifier.

    Training objective:

        BCEWithLogitsLoss

    Early stopping criterion:

        validation AUROC

    After best-state restoration, both main and interaction
    contributions are centered using training observations only.
    """
    seed_everything(
        seed=config.seed,
        deterministic=(
            config.deterministic
        ),
    )

    if device is None:
        device = get_device()

    train_y = np.asarray(
        train_y,
        dtype=np.float32,
    )

    val_y = np.asarray(
        val_y,
        dtype=np.float32,
    )

    if len(train_data) != len(
        train_y
    ):
        raise ValueError(
            "train_data and train_y lengths must match."
        )

    if len(val_data) != len(
        val_y
    ):
        raise ValueError(
            "val_data and val_y lengths must match."
        )

    model = model.to(
        device
    )

    # Training starts without post-hoc centering offsets.
    model.clear_centering()

    train_loader = _make_loader(
        transformed=train_data,
        y=train_y,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed,
    )

    criterion = (
        nn.BCEWithLogitsLoss()
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_auc": [],
    }

    best_epoch = -1

    best_auc = (
        -np.inf
    )

    best_state = None

    epochs_without_improvement = 0

    for epoch in range(
        config.max_epochs
    ):
        model.train()

        running_loss = 0.0
        n_train = 0

        for (
            numeric,
            missing,
            categorical,
            target,
        ) in train_loader:
            numeric = numeric.to(
                device,
                non_blocking=True,
            )

            missing = missing.to(
                device,
                non_blocking=True,
            )

            categorical = categorical.to(
                device,
                non_blocking=True,
            )

            target = target.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            output = model(
                numeric=numeric,
                numeric_missing=missing,
                categorical=categorical,
            )

            loss = criterion(
                output.logits,
                target,
            )

            loss.backward()

            optimizer.step()

            batch_n = (
                target.shape[0]
            )

            running_loss += (
                float(
                    loss.item()
                )
                * batch_n
            )

            n_train += (
                batch_n
            )

        train_loss = (
            running_loss
            / n_train
        )

        model.eval()

        val_probabilities = (
            predict_agnam_probabilities(
                model=model,
                transformed=val_data,
                batch_size=max(
                    config.batch_size,
                    1024,
                ),
                device=device,
            )
        )

        val_auc = _safe_auc(
            val_y,
            val_probabilities,
        )

        eps = 1e-7

        clipped = np.clip(
            val_probabilities,
            eps,
            1.0 - eps,
        )

        val_loss = float(
            -np.mean(
                val_y
                * np.log(
                    clipped
                )
                + (
                    1.0
                    - val_y
                )
                * np.log(
                    1.0
                    - clipped
                )
            )
        )

        history[
            "train_loss"
        ].append(
            train_loss
        )

        history[
            "val_loss"
        ].append(
            val_loss
        )

        history[
            "val_auc"
        ].append(
            val_auc
        )

        improved = (
            np.isfinite(
                val_auc
            )
            and (
                val_auc
                > best_auc
                + config.min_delta
            )
        )

        if improved:
            best_auc = (
                val_auc
            )

            best_epoch = (
                epoch
            )

            best_state = deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= config.patience
        ):
            break

    if best_state is None:
        raise RuntimeError(
            "AG-NAM training failed to produce a valid "
            "validation AUROC."
        )

    model.load_state_dict(
        best_state
    )

    # ---------------------------------------------------------
    # Post-training exact contribution centering
    # ---------------------------------------------------------

    apply_agnam_centering(
        model=model,
        transformed=train_data,
        batch_size=max(
            config.batch_size,
            1024,
        ),
        device=device,
    )

    final_probabilities = (
        predict_agnam_probabilities(
            model=model,
            transformed=val_data,
            batch_size=max(
                config.batch_size,
                1024,
            ),
            device=device,
        )
    )

    metrics = (
        compute_agnam_binary_metrics(
            y_true=val_y,
            probabilities=(
                final_probabilities
            ),
        )
    )

    return AGNAMTrainingResult(
        best_epoch=best_epoch,
        best_validation_auc=float(
            best_auc
        ),
        history=history,
        metrics=metrics,
    )