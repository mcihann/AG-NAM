import numpy as np
import torch

from agnam.utils.reproducibility import (
    get_device,
    seed_everything,
)


def test_seed_everything_reproduces_numpy():
    seed_everything(42)
    first = np.random.rand(10)

    seed_everything(42)
    second = np.random.rand(10)

    np.testing.assert_allclose(first, second)


def test_seed_everything_reproduces_torch():
    seed_everything(42)
    first = torch.rand(10)

    seed_everything(42)
    second = torch.rand(10)

    assert torch.allclose(first, second)


def test_get_device_returns_valid_device():
    device = get_device()

    assert device.type in {"cpu", "cuda"}


def test_cpu_can_be_requested_explicitly():
    device = get_device("cpu")

    assert device.type == "cpu"