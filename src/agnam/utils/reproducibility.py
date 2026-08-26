from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def seed_everything(
    seed: int,
    deterministic: bool = True,
) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""
    if seed < 0:
        raise ValueError("seed must be non-negative.")

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        try:
            torch.use_deterministic_algorithms(
                True,
                warn_only=True,
            )
        except AttributeError:
            pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_device(
    preferred: Optional[str] = None,
) -> torch.device:
    """Return the requested or automatically selected compute device."""
    if preferred is not None:
        device = torch.device(preferred)

        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not available."
            )

        return device

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")