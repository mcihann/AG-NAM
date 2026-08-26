from __future__ import annotations

import numpy as np
import torch


def bernoulli_pseudo_residual_numpy(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    """
    Compute Bernoulli negative-gradient pseudo-residuals.

    For binary cross-entropy with logit eta:

        dL / d eta = p - y

    therefore:

        r = -dL/deta = y - p
    """
    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    if y_true.shape != probabilities.shape:
        raise ValueError(
            "y_true and probabilities must have the same shape."
        )

    if not np.all(
        np.isin(y_true, [0.0, 1.0])
    ):
        raise ValueError(
            "y_true must contain only binary labels 0 and 1."
        )

    if not np.isfinite(
        probabilities
    ).all():
        raise ValueError(
            "probabilities must contain only finite values."
        )

    if np.any(
        (probabilities < 0.0)
        | (probabilities > 1.0)
    ):
        raise ValueError(
            "probabilities must lie in [0, 1]."
        )

    return (
        y_true
        - probabilities
    ).astype(np.float32)


def bernoulli_pseudo_residual_torch(
    y_true: torch.Tensor,
    logits: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Bernoulli negative-gradient pseudo-residuals
    directly from logits.

    r = y - sigmoid(logit)
    """
    if y_true.shape != logits.shape:
        raise ValueError(
            "y_true and logits must have the same shape."
        )

    probabilities = torch.sigmoid(
        logits
    )

    return (
        y_true.to(
            dtype=probabilities.dtype
        )
        - probabilities
    )