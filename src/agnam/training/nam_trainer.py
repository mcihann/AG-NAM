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
from torch.utils.data import DataLoader, TensorDataset

from agnam.data.preprocessing import TransformedTabular
from agnam.models.main_effect_nam import MainEffectNAM
from agnam.utils.reproducibility import get_device, seed_everything


@dataclass
class NAMTrainingConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    max_epochs: int = 300
    patience: int = 30
    min_delta: float = 1e-4
    seed: int = 42
    deterministic: bool = True


@dataclass
class NAMTrainingResult:
    best_epoch: int
    best_validation_auc: float
    history: dict[str, list[float]]
    metrics: dict[str, float]


def _to_tensor_dataset(
    transformed: TransformedTabular,
    y: np.ndarray,
) -> TensorDataset:
    y = np.asarray(y, dtype=np.float32)

    if len(transformed) != len(y):
        raise ValueError(
            "Transformed features and target must have the same length."
        )

    return TensorDataset(
        torch.from_numpy(transformed.numeric),
        torch.from_numpy(transformed.numeric_missing),
        torch.from_numpy(transformed.categorical),
        torch.from_numpy(y),
    )


def _make_loader(
    transformed: TransformedTabular,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = _to_tensor_dataset(
        transformed=transformed,
        y=y,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def _safe_auc(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")

    return float(
        roc_auc_score(
            y_true,
            probabilities,
        )
    )


def compute_binary_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)

    predictions = (
        probabilities >= threshold
    ).astype(int)

    metrics = {
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

    return metrics


@torch.no_grad()
def predict_probabilities(
    model: MainEffectNAM,
    transformed: TransformedTabular,
    batch_size: int = 1024,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    if device is None:
        device = get_device()

    model = model.to(device)
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

    probabilities: list[np.ndarray] = []

    for numeric, missing, categorical, _ in loader:
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
            batch_probabilities.cpu().numpy()
        )

    return np.concatenate(probabilities)


@torch.no_grad()
def estimate_centering_offsets(
    model: MainEffectNAM,
    transformed: TransformedTabular,
    batch_size: int = 1024,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if device is None:
        device = get_device()

    model = model.to(device)
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

    contribution_sum = torch.zeros(
        model.n_features,
        device=device,
    )

    n_seen = 0

    for numeric, missing, categorical, _ in loader:
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

        raw = model.raw_contributions(
            numeric=numeric,
            numeric_missing=missing,
            categorical=categorical,
        )

        contribution_sum += raw.sum(dim=0)
        n_seen += raw.shape[0]

    if n_seen == 0:
        raise ValueError(
            "Cannot estimate centering offsets from an empty dataset."
        )

    return contribution_sum / n_seen


def train_main_effect_nam(
    model: MainEffectNAM,
    train_data: TransformedTabular,
    train_y: np.ndarray,
    val_data: TransformedTabular,
    val_y: np.ndarray,
    config: NAMTrainingConfig,
    device: Optional[torch.device] = None,
) -> NAMTrainingResult:
    seed_everything(
        seed=config.seed,
        deterministic=config.deterministic,
    )

    if device is None:
        device = get_device()

    model = model.to(device)

    model.clear_centering()

    train_loader = _make_loader(
        transformed=train_data,
        y=train_y,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed,
    )

    criterion = nn.BCEWithLogitsLoss()

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
    best_auc = -np.inf
    best_state = None

    epochs_without_improvement = 0

    val_y = np.asarray(
        val_y,
        dtype=np.float32,
    )

    for epoch in range(config.max_epochs):
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

            batch_size = target.shape[0]

            running_loss += (
                float(loss.item())
                * batch_size
            )

            n_train += batch_size

        train_loss = (
            running_loss / n_train
        )

        model.eval()

        val_probabilities = predict_probabilities(
            model=model,
            transformed=val_data,
            batch_size=max(
                config.batch_size,
                1024,
            ),
            device=device,
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
                val_y * np.log(clipped)
                + (1.0 - val_y)
                * np.log(1.0 - clipped)
            )
        )

        history["train_loss"].append(
            train_loss
        )

        history["val_loss"].append(
            val_loss
        )

        history["val_auc"].append(
            val_auc
        )

        improved = (
            np.isfinite(val_auc)
            and (
                val_auc
                > best_auc + config.min_delta
            )
        )

        if improved:
            best_auc = val_auc
            best_epoch = epoch

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
            "Training failed to produce a valid validation AUROC."
        )

    model.load_state_dict(
        best_state
    )

    offsets = estimate_centering_offsets(
        model=model,
        transformed=train_data,
        batch_size=max(
            config.batch_size,
            1024,
        ),
        device=device,
    )

    model.set_centering_offsets(
        offsets
    )

    final_probabilities = predict_probabilities(
        model=model,
        transformed=val_data,
        batch_size=max(
            config.batch_size,
            1024,
        ),
        device=device,
    )

    metrics = compute_binary_metrics(
        y_true=val_y,
        probabilities=final_probabilities,
    )

    return NAMTrainingResult(
        best_epoch=best_epoch,
        best_validation_auc=float(
            best_auc
        ),
        history=history,
        metrics=metrics,
    )