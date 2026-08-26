from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from agnam.data.preprocessing import TransformedTabular
from agnam.models.attention_proposer import ResidualAttentionProposer
from agnam.utils.reproducibility import get_device, seed_everything


@dataclass
class ProposerTrainingConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    max_epochs: int = 200
    patience: int = 20
    min_delta: float = 1e-5
    seed: int = 42
    deterministic: bool = True


@dataclass
class ProposerTrainingResult:
    best_epoch: int
    best_validation_mse: float
    history: dict[str, list[float]]
    metrics: dict[str, float]


def _to_tensor_dataset(
    transformed: TransformedTabular,
    residuals: np.ndarray,
) -> TensorDataset:
    residuals = np.asarray(
        residuals,
        dtype=np.float32,
    )

    if len(transformed) != len(residuals):
        raise ValueError(
            "Features and residual targets must have the same length."
        )

    if not np.isfinite(residuals).all():
        raise ValueError(
            "Residual targets must be finite."
        )

    return TensorDataset(
        torch.from_numpy(transformed.numeric),
        torch.from_numpy(transformed.numeric_missing),
        torch.from_numpy(transformed.categorical),
        torch.from_numpy(residuals),
    )


def _make_loader(
    transformed: TransformedTabular,
    residuals: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = _to_tensor_dataset(
        transformed=transformed,
        residuals=residuals,
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


def compute_residual_metrics(
    residual_true: np.ndarray,
    residual_pred: np.ndarray,
) -> dict[str, float]:
    residual_true = np.asarray(
        residual_true,
        dtype=np.float64,
    )

    residual_pred = np.asarray(
        residual_pred,
        dtype=np.float64,
    )

    if residual_true.shape != residual_pred.shape:
        raise ValueError(
            "residual_true and residual_pred must have the same shape."
        )

    mse = float(
        mean_squared_error(
            residual_true,
            residual_pred,
        )
    )

    r2 = float(
        r2_score(
            residual_true,
            residual_pred,
        )
    )

    if (
        np.std(residual_true) < 1e-12
        or np.std(residual_pred) < 1e-12
    ):
        correlation = float("nan")
    else:
        correlation = float(
            np.corrcoef(
                residual_true,
                residual_pred,
            )[0, 1]
        )

    zero_baseline_mse = float(
        np.mean(
            residual_true ** 2
        )
    )

    return {
        "mse": mse,
        "r2": r2,
        "correlation": correlation,
        "zero_baseline_mse": zero_baseline_mse,
        "mse_improvement_over_zero": (
            zero_baseline_mse - mse
        ),
    }


@torch.no_grad()
def predict_residuals(
    model: ResidualAttentionProposer,
    transformed: TransformedTabular,
    batch_size: int = 1024,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    if device is None:
        device = get_device()

    model = model.to(device)
    model.eval()

    dummy_residuals = np.zeros(
        len(transformed),
        dtype=np.float32,
    )

    loader = _make_loader(
        transformed=transformed,
        residuals=dummy_residuals,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
    )

    predictions: list[np.ndarray] = []

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

        predictions.append(
            output.residual_prediction
            .cpu()
            .numpy()
        )

    return np.concatenate(
        predictions
    )


def train_residual_attention_proposer(
    model: ResidualAttentionProposer,
    train_data: TransformedTabular,
    train_residuals: np.ndarray,
    val_data: TransformedTabular,
    val_residuals: np.ndarray,
    config: ProposerTrainingConfig,
    device: Optional[torch.device] = None,
) -> ProposerTrainingResult:
    seed_everything(
        seed=config.seed,
        deterministic=config.deterministic,
    )

    if device is None:
        device = get_device()

    model = model.to(device)

    train_loader = _make_loader(
        transformed=train_data,
        residuals=train_residuals,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed,
    )

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history = {
        "train_mse": [],
        "val_mse": [],
    }

    best_epoch = -1
    best_mse = np.inf
    best_state = None

    epochs_without_improvement = 0

    val_residuals = np.asarray(
        val_residuals,
        dtype=np.float32,
    )

    for epoch in range(
        config.max_epochs
    ):
        model.train()

        running_loss = 0.0
        n_seen = 0

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
                output.residual_prediction,
                target,
            )

            loss.backward()

            optimizer.step()

            batch_size = target.shape[0]

            running_loss += (
                float(loss.item())
                * batch_size
            )

            n_seen += batch_size

        train_mse = (
            running_loss / n_seen
        )

        val_predictions = predict_residuals(
            model=model,
            transformed=val_data,
            batch_size=max(
                config.batch_size,
                1024,
            ),
            device=device,
        )

        val_mse = float(
            mean_squared_error(
                val_residuals,
                val_predictions,
            )
        )

        history["train_mse"].append(
            train_mse
        )

        history["val_mse"].append(
            val_mse
        )

        improved = (
            val_mse
            < best_mse - config.min_delta
        )

        if improved:
            best_mse = val_mse
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
            "Residual proposer training did not produce a valid model."
        )

    model.load_state_dict(
        best_state
    )

    final_predictions = predict_residuals(
        model=model,
        transformed=val_data,
        batch_size=max(
            config.batch_size,
            1024,
        ),
        device=device,
    )

    metrics = compute_residual_metrics(
        residual_true=val_residuals,
        residual_pred=final_predictions,
    )

    return ProposerTrainingResult(
        best_epoch=best_epoch,
        best_validation_mse=float(
            best_mse
        ),
        history=history,
        metrics=metrics,
    )