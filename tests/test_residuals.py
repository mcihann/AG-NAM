import numpy as np
import pytest
import torch
from torch.nn import functional as F

from agnam.training.residuals import (
    bernoulli_pseudo_residual_numpy,
    bernoulli_pseudo_residual_torch,
)


def test_numpy_residual_matches_y_minus_p():
    y = np.array(
        [0, 1, 1, 0],
        dtype=float,
    )

    p = np.array(
        [0.2, 0.8, 0.3, 0.7],
        dtype=float,
    )

    residual = (
        bernoulli_pseudo_residual_numpy(
            y,
            p,
        )
    )

    expected = y - p

    np.testing.assert_allclose(
        residual,
        expected,
        atol=1e-7,
    )


def test_torch_residual_is_negative_bce_gradient():
    logits = torch.tensor(
        [-1.2, 0.3, 1.5, -0.4],
        dtype=torch.float64,
        requires_grad=True,
    )

    y = torch.tensor(
        [0.0, 1.0, 1.0, 0.0],
        dtype=torch.float64,
    )

    loss = F.binary_cross_entropy_with_logits(
        logits,
        y,
        reduction="sum",
    )

    gradient = torch.autograd.grad(
        loss,
        logits,
    )[0]

    residual = bernoulli_pseudo_residual_torch(
        y_true=y,
        logits=logits.detach(),
    )

    assert torch.allclose(
        residual,
        -gradient,
        atol=1e-10,
        rtol=1e-10,
    )


def test_residuals_are_bounded():
    y = np.array(
        [0, 0, 1, 1],
        dtype=float,
    )

    p = np.array(
        [0.0, 0.99, 0.01, 1.0],
        dtype=float,
    )

    residual = (
        bernoulli_pseudo_residual_numpy(
            y,
            p,
        )
    )

    assert np.all(
        residual >= -1.0
    )

    assert np.all(
        residual <= 1.0
    )


def test_invalid_probability_raises():
    y = np.array(
        [0, 1]
    )

    p = np.array(
        [0.2, 1.2]
    )

    with pytest.raises(ValueError):
        bernoulli_pseudo_residual_numpy(
            y,
            p,
        )