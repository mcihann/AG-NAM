from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from torch import nn
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

from agnam.data.preprocessing import (
    TransformedTabular,
)
from agnam.models.pairwise_interaction import (
    PairwiseResidualNetwork,
)
from agnam.utils.reproducibility import (
    get_device,
    seed_everything,
)


@dataclass
class PairwiseTrainingConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256

    max_epochs: int = 200
    patience: int = 20
    min_delta: float = 1e-5

    seed: int = 42
    deterministic: bool = True


@dataclass
class PairwiseTrainingResult:
    best_epoch: int
    best_validation_mse: float
    history: dict[str, list[float]]


def _make_loader(
    transformed: TransformedTabular,
    residuals: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    residuals = np.asarray(
        residuals,
        dtype=np.float32,
    )

    if len(transformed) != len(
        residuals
    ):
        raise ValueError(
            "Features and residual targets must have equal length."
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
            residuals
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


@torch.no_grad()
def predict_pairwise_residuals(
    model: PairwiseResidualNetwork,
    transformed: TransformedTabular,
    *,
    batch_size: int = 2048,
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

    dummy = np.zeros(
        len(transformed),
        dtype=np.float32,
    )

    loader = _make_loader(
        transformed=transformed,
        residuals=dummy,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
    )

    predictions: list[
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

        prediction = model(
            numeric=numeric,
            numeric_missing=missing,
            categorical=categorical,
        )

        predictions.append(
            prediction
            .cpu()
            .numpy()
        )

    return np.concatenate(
        predictions
    )


def train_pairwise_residual_network(
    model: PairwiseResidualNetwork,
    train_data: TransformedTabular,
    train_residuals: np.ndarray,
    val_data: TransformedTabular,
    val_residuals: np.ndarray,
    config: PairwiseTrainingConfig,
    *,
    device: Optional[
        torch.device
    ] = None,
) -> PairwiseTrainingResult:
    seed_everything(
        config.seed,
        deterministic=(
            config.deterministic
        ),
    )

    if device is None:
        device = get_device()

    model = model.to(
        device
    )

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

        total_loss = 0.0
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

            prediction = model(
                numeric=numeric,
                numeric_missing=missing,
                categorical=categorical,
            )

            loss = criterion(
                prediction,
                target,
            )

            loss.backward()
            optimizer.step()

            batch_n = target.shape[0]

            total_loss += (
                float(loss.item())
                * batch_n
            )

            n_seen += batch_n

        train_mse = (
            total_loss / n_seen
        )

        val_prediction = (
            predict_pairwise_residuals(
                model=model,
                transformed=val_data,
                batch_size=max(
                    config.batch_size,
                    1024,
                ),
                device=device,
            )
        )

        val_mse = float(
            mean_squared_error(
                val_residuals,
                val_prediction,
            )
        )

        history[
            "train_mse"
        ].append(
            train_mse
        )

        history[
            "val_mse"
        ].append(
            val_mse
        )

        if (
            val_mse
            < best_mse
            - config.min_delta
        ):
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
            "Pairwise residual training failed."
        )

    model.load_state_dict(
        best_state
    )

    return PairwiseTrainingResult(
        best_epoch=best_epoch,
        best_validation_mse=float(
            best_mse
        ),
        history=history,
    )