from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


Interaction = tuple[int, int]


@dataclass
class SyntheticDataset:
    """Container for a synthetic AG-NAM benchmark dataset."""

    X: pd.DataFrame
    y: pd.Series
    logits: np.ndarray

    main_effects: dict[str, np.ndarray]
    interaction_effects: dict[tuple[str, str], np.ndarray]

    true_interactions: tuple[tuple[str, str], ...]
    metadata: dict


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def _calibrate_intercept(
    signal: np.ndarray,
    target_prevalence: float,
    max_iter: int = 100,
) -> float:
    """
    Find an intercept such that mean(sigmoid(intercept + signal))
    approximately equals target_prevalence.
    """
    if not 0.0 < target_prevalence < 1.0:
        raise ValueError("target_prevalence must be between 0 and 1.")

    low = -20.0
    high = 20.0

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        prevalence = _sigmoid(mid + signal).mean()

        if prevalence < target_prevalence:
            low = mid
        else:
            high = mid

    return 0.5 * (low + high)


def _sample_binary_target(
    signal: np.ndarray,
    rng: np.random.Generator,
    target_prevalence: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    if target_prevalence is None:
        intercept = 0.0
    else:
        intercept = _calibrate_intercept(
            signal=signal,
            target_prevalence=target_prevalence,
        )

    logits = intercept + signal
    probabilities = _sigmoid(logits)

    y = rng.binomial(
        n=1,
        p=probabilities,
        size=len(probabilities),
    ).astype(np.int64)

    return y, logits, intercept


def _feature_names(p: int) -> list[str]:
    return [f"x{i}" for i in range(1, p + 1)]


def _make_dataframe(X: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        X,
        columns=_feature_names(X.shape[1]),
    )


def generate_s1(
    seed: int,
    n: int = 5000,
) -> SyntheticDataset:
    """
    S1: Sparse strong pairwise interactions.

    Ground-truth interactions:
        x3 × x4
        x5 × x6
        x8 × x9
    """
    rng = np.random.default_rng(seed)

    p = 20
    X = rng.normal(size=(n, p))

    main_effects = {
        "x1": 1.5 * np.sin(X[:, 0]),
        "x2": 1.0 * (X[:, 1] ** 2 - 1.0),
        "x7": -1.2 * X[:, 6],
    }

    interaction_effects = {
        ("x3", "x4"): 2.0 * X[:, 2] * X[:, 3],
        ("x5", "x6"): 1.5 * np.sin(X[:, 4] * X[:, 5]),
        ("x8", "x9"): 1.2 * X[:, 7] * X[:, 8],
    }

    signal = (
        sum(main_effects.values())
        + sum(interaction_effects.values())
    )

    y, logits, intercept = _sample_binary_target(
        signal=signal,
        rng=rng,
    )

    return SyntheticDataset(
        X=_make_dataframe(X),
        y=pd.Series(y, name="target"),
        logits=logits,
        main_effects=main_effects,
        interaction_effects=interaction_effects,
        true_interactions=tuple(interaction_effects.keys()),
        metadata={
            "scenario": "S1",
            "description": "Sparse strong pairwise interactions",
            "seed": seed,
            "n": n,
            "p": p,
            "intercept": intercept,
            "target_prevalence_requested": None,
            "target_prevalence_observed": float(y.mean()),
        },
    )


def generate_s2(
    seed: int,
    n: int = 8000,
) -> SyntheticDataset:
    """
    S2: Multiple nonlinear pairwise interactions.

    The scenario contains interaction surfaces with different
    functional geometries.
    """
    rng = np.random.default_rng(seed)

    p = 30
    X = rng.normal(size=(n, p))

    main_effects = {
        "x1": 1.0 * np.sin(X[:, 0]),
        "x3": 0.8 * np.tanh(X[:, 2]),
        "x14": -0.9 * X[:, 13],
        "x21": 0.7 * (X[:, 20] ** 2 - 1.0),
    }

    interaction_effects = {
        ("x2", "x5"): 1.5 * np.sin(
            np.pi * X[:, 1] * X[:, 4]
        ),
        ("x7", "x9"): 1.2 * X[:, 6] * X[:, 8],
        ("x11", "x12"): 1.5 * np.tanh(
            X[:, 10] * X[:, 11]
        ),
        ("x15", "x18"): 1.2 * (
            np.tanh(2.0 * X[:, 14])
            * np.tanh(2.0 * X[:, 17])
        ),
    }

    signal = (
        sum(main_effects.values())
        + sum(interaction_effects.values())
    )

    y, logits, intercept = _sample_binary_target(
        signal=signal,
        rng=rng,
    )

    return SyntheticDataset(
        X=_make_dataframe(X),
        y=pd.Series(y, name="target"),
        logits=logits,
        main_effects=main_effects,
        interaction_effects=interaction_effects,
        true_interactions=tuple(interaction_effects.keys()),
        metadata={
            "scenario": "S2",
            "description": "Multiple nonlinear interactions",
            "seed": seed,
            "n": n,
            "p": p,
            "intercept": intercept,
            "target_prevalence_requested": None,
            "target_prevalence_observed": float(y.mean()),
        },
    )


def generate_s3(
    seed: int,
    n: int = 8000,
) -> SyntheticDataset:
    """
    S3: Correlated nuisance features.

    x11 and x12 are correlated proxies for x1 and x2.
    x13 and x14 are correlated proxies for x5 and x6.

    True interactions remain:
        x1 × x2
        x5 × x6
    """
    rng = np.random.default_rng(seed)

    p = 30
    X = rng.normal(size=(n, p))

    # Create correlated nuisance / proxy features.
    proxy_noise_scale = np.sqrt(1.0 - 0.85**2)

    X[:, 10] = (
        0.85 * X[:, 0]
        + proxy_noise_scale * rng.normal(size=n)
    )
    X[:, 11] = (
        0.85 * X[:, 1]
        + proxy_noise_scale * rng.normal(size=n)
    )
    X[:, 12] = (
        0.85 * X[:, 4]
        + proxy_noise_scale * rng.normal(size=n)
    )
    X[:, 13] = (
        0.85 * X[:, 5]
        + proxy_noise_scale * rng.normal(size=n)
    )

    main_effects = {
        "x3": 1.0 * np.sin(X[:, 2]),
        "x4": -0.8 * X[:, 3],
        "x7": 0.8 * np.tanh(X[:, 6]),
    }

    interaction_effects = {
        ("x1", "x2"): 1.6 * X[:, 0] * X[:, 1],
        ("x5", "x6"): 1.3 * np.sin(
            X[:, 4] * X[:, 5]
        ),
    }

    signal = (
        sum(main_effects.values())
        + sum(interaction_effects.values())
    )

    y, logits, intercept = _sample_binary_target(
        signal=signal,
        rng=rng,
    )

    return SyntheticDataset(
        X=_make_dataframe(X),
        y=pd.Series(y, name="target"),
        logits=logits,
        main_effects=main_effects,
        interaction_effects=interaction_effects,
        true_interactions=tuple(interaction_effects.keys()),
        metadata={
            "scenario": "S3",
            "description": "Correlated nuisance features",
            "seed": seed,
            "n": n,
            "p": p,
            "intercept": intercept,
            "proxy_correlation_target": 0.85,
            "proxy_pairs": [
                ("x1", "x11"),
                ("x2", "x12"),
                ("x5", "x13"),
                ("x6", "x14"),
            ],
            "target_prevalence_requested": None,
            "target_prevalence_observed": float(y.mean()),
        },
    )


def generate_s4(
    seed: int,
    n: int = 10000,
) -> SyntheticDataset:
    """
    S4: Weak interactions under high-dimensional noise
    and class imbalance.

    Target positive-class prevalence is approximately 20%.
    """
    rng = np.random.default_rng(seed)

    p = 50
    X = rng.normal(size=(n, p))

    main_effects = {
        "x1": 0.60 * np.sin(X[:, 0]),
        "x4": -0.55 * X[:, 3],
        "x10": 0.50 * np.tanh(X[:, 9]),
        "x21": 0.45 * (X[:, 20] ** 2 - 1.0),
    }

    interaction_effects = {
        ("x2", "x3"): 0.65 * X[:, 1] * X[:, 2],
        ("x6", "x8"): 0.60 * np.sin(
            X[:, 5] * X[:, 7]
        ),
        ("x12", "x15"): 0.55 * np.tanh(
            X[:, 11] * X[:, 14]
        ),
        ("x25", "x30"): 0.50 * (
            np.tanh(X[:, 24])
            * np.tanh(X[:, 29])
        ),
    }

    signal = (
        sum(main_effects.values())
        + sum(interaction_effects.values())
    )

    y, logits, intercept = _sample_binary_target(
        signal=signal,
        rng=rng,
        target_prevalence=0.20,
    )

    return SyntheticDataset(
        X=_make_dataframe(X),
        y=pd.Series(y, name="target"),
        logits=logits,
        main_effects=main_effects,
        interaction_effects=interaction_effects,
        true_interactions=tuple(interaction_effects.keys()),
        metadata={
            "scenario": "S4",
            "description": (
                "Weak interactions under high-dimensional noise "
                "and class imbalance"
            ),
            "seed": seed,
            "n": n,
            "p": p,
            "intercept": intercept,
            "target_prevalence_requested": 0.20,
            "target_prevalence_observed": float(y.mean()),
        },
    )


GENERATORS: dict[str, Callable[..., SyntheticDataset]] = {
    "S1": generate_s1,
    "S2": generate_s2,
    "S3": generate_s3,
    "S4": generate_s4,
}


def generate_synthetic(
    scenario: str,
    seed: int,
    n: int | None = None,
) -> SyntheticDataset:
    """
    Generate one predefined AG-NAM synthetic benchmark.

    Parameters
    ----------
    scenario:
        One of S1, S2, S3, or S4.

    seed:
        Dataset-generation seed.

    n:
        Optional sample-size override.
    """
    scenario = scenario.upper()

    if scenario not in GENERATORS:
        raise ValueError(
            f"Unknown scenario '{scenario}'. "
            f"Expected one of {tuple(GENERATORS)}."
        )

    generator = GENERATORS[scenario]

    if n is None:
        return generator(seed=seed)

    return generator(
        seed=seed,
        n=n,
    )