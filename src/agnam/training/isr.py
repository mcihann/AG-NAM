from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from agnam.data.preprocessing import (
    TabularPreprocessor,
)
from agnam.interpretation.interaction_metrics import (
    canonical_pair,
)
from agnam.interpretation.surface_reproducibility import (
    compute_isr,
    diagonal_purified_vector,
    evaluate_pairwise_raw_grid,
)
from agnam.models.pairwise_interaction import (
    PairwiseResidualNetwork,
)
from agnam.training.discovery import (
    InteractionDiscoveryRun,
    ReproducibleDiscoveryResult,
)
from agnam.training.pairwise_trainer import (
    PairwiseTrainingConfig,
    train_pairwise_residual_network,
)
from agnam.utils.reproducibility import (
    seed_everything,
)


Pair = tuple[str, str]


@dataclass(frozen=True)
class ISRConfig:
    reference_size: int = 512
    reference_seed: int = 2026

    isr_threshold: float = 0.60

    feature_embedding_dim: int = 16
    hidden_dim: int = 64
    depth: int = 2
    dropout: float = 0.10

    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    max_epochs: int = 200
    patience: int = 20
    min_delta: float = 1e-5

    surface_batch_size: int = 8192


@dataclass(frozen=True)
class PairSurfaceRunResult:
    run_index: int
    discovery_seed: int
    pairwise_seed: int

    best_epoch: int
    best_validation_mse: float

    purified_vector: np.ndarray


@dataclass(frozen=True)
class InteractionISRResult:
    feature_j: str
    feature_k: str

    selection_count: int
    selection_probability: float

    isr: float

    pairwise_correlations: tuple[
        float,
        ...
    ]

    surface_runs: tuple[
        PairSurfaceRunResult,
        ...
    ]

    final_accepted: bool


@dataclass(frozen=True)
class ISRDiscoveryResult:
    reference_indices: np.ndarray

    interactions: tuple[
        InteractionISRResult,
        ...
    ]

    @property
    def retained_interactions(
        self,
    ) -> tuple[
        InteractionISRResult,
        ...
    ]:
        return tuple(
            item
            for item in self.interactions
            if item.final_accepted
        )


def sample_reference_indices(
    n_samples: int,
    *,
    reference_size: int = 512,
    seed: int = 2026,
) -> np.ndarray:
    if n_samples <= 0:
        raise ValueError(
            "n_samples must be positive."
        )

    if reference_size <= 0:
        raise ValueError(
            "reference_size must be positive."
        )

    m = min(
        reference_size,
        n_samples,
    )

    rng = np.random.default_rng(
        seed
    )

    indices = rng.choice(
        n_samples,
        size=m,
        replace=False,
    )

    return np.asarray(
        indices,
        dtype=np.int64,
    )


def _selected_runs_for_pair(
    pair: Pair,
    runs: tuple[
        InteractionDiscoveryRun,
        ...
    ],
) -> tuple[
    InteractionDiscoveryRun,
    ...
]:
    pair = canonical_pair(
        *pair
    )

    selected = []

    for run in runs:
        run_pairs = {
            canonical_pair(
                *candidate
            )
            for candidate
            in run.top_k_pairs
        }

        if pair in run_pairs:
            selected.append(
                run
            )

    return tuple(
        selected
    )


def _pair_specs(
    transformed,
    pair: Pair,
):
    lookup = {
        spec.name: spec
        for spec
        in transformed.feature_specs
    }

    missing = [
        name
        for name in pair
        if name not in lookup
    ]

    if missing:
        raise ValueError(
            f"Unknown pair features: {missing}"
        )

    return (
        lookup[
            pair[0]
        ],
        lookup[
            pair[1]
        ],
    )


def _pairwise_seed(
    run_seed: int,
    pair_specs,
) -> int:
    indices = sorted(
        [
            int(
                pair_specs[0]
                .original_index
            ),
            int(
                pair_specs[1]
                .original_index
            ),
        ]
    )

    return int(
        run_seed
        + 100_000
        + 1_000 * indices[0]
        + indices[1]
    )


def evaluate_isr_for_discovery(
    X: pd.DataFrame,
    discovery: ReproducibleDiscoveryResult,
    *,
    numeric_features: Sequence[
        str
    ] | None = None,
    categorical_features: Sequence[
        str
    ] | None = None,
    config: ISRConfig | None = None,
    device: torch.device | None = None,
    verbose: bool = True,
) -> ISRDiscoveryResult:
    """
    Fit explicit pairwise residual models for every interaction
    passing selection reproducibility, purify their learned
    surfaces, and compute ISR.

    Ground-truth labels are not required.
    """
    if not isinstance(
        X,
        pd.DataFrame,
    ):
        raise TypeError(
            "X must be a pandas DataFrame."
        )

    if config is None:
        config = ISRConfig()

    reference_indices = (
        sample_reference_indices(
            len(X),
            reference_size=(
                config.reference_size
            ),
            seed=(
                config.reference_seed
            ),
        )
    )

    X_reference = (
        X.iloc[
            reference_indices
        ]
        .reset_index(
            drop=True
        )
    )

    results: list[
        InteractionISRResult
    ] = []

    accepted_candidates = (
        discovery
        .selection
        .accepted_interactions
    )

    if verbose:
        print(
            "\n=== ISR EVALUATION ==="
        )

        print(
            "Selection-stable candidates:",
            len(
                accepted_candidates
            ),
        )

        print(
            "Reference support:",
            len(
                X_reference
            ),
        )

    for candidate_index, candidate in enumerate(
        accepted_candidates,
        start=1,
    ):
        pair = canonical_pair(
            candidate.feature_j,
            candidate.feature_k,
        )

        selected_runs = (
            _selected_runs_for_pair(
                pair,
                discovery.runs,
            )
        )

        if len(selected_runs) != (
            candidate.selection_count
        ):
            raise RuntimeError(
                "Selected-run count disagrees with "
                "selection reproducibility summary."
            )

        if len(selected_runs) < 2:
            raise RuntimeError(
                "ISR requires at least two selected runs."
            )

        if verbose:
            print(
                f"\n[{candidate_index}/"
                f"{len(accepted_candidates)}] "
                f"{pair[0]} x {pair[1]} "
                f"(pi="
                f"{candidate.selection_probability:.2f}, "
                f"runs="
                f"{len(selected_runs)})"
            )

        surface_results: list[
            PairSurfaceRunResult
        ] = []

        purified_vectors: list[
            np.ndarray
        ] = []

        for run_position, run in enumerate(
            selected_runs,
            start=1,
        ):
            X_train = (
                X.iloc[
                    run.train_indices
                ]
                .reset_index(
                    drop=True
                )
            )

            X_validation = (
                X.iloc[
                    run.validation_indices
                ]
                .reset_index(
                    drop=True
                )
            )

            preprocessor = (
                TabularPreprocessor(
                    numeric_features=(
                        numeric_features
                    ),
                    categorical_features=(
                        categorical_features
                    ),
                )
            )

            train_data = (
                preprocessor
                .fit_transform(
                    X_train
                )
            )

            validation_data = (
                preprocessor
                .transform(
                    X_validation
                )
            )

            reference_data = (
                preprocessor
                .transform(
                    X_reference
                )
            )

            pair_specs = (
                _pair_specs(
                    train_data,
                    pair,
                )
            )

            pair_seed = (
                _pairwise_seed(
                    run.seed,
                    pair_specs,
                )
            )

            seed_everything(
                pair_seed,
                deterministic=True,
            )

            model = (
                PairwiseResidualNetwork(
                    feature_specs=(
                        pair_specs
                    ),
                    feature_embedding_dim=(
                        config
                        .feature_embedding_dim
                    ),
                    hidden_dim=(
                        config.hidden_dim
                    ),
                    depth=(
                        config.depth
                    ),
                    dropout=(
                        config.dropout
                    ),
                )
            )

            training_config = (
                PairwiseTrainingConfig(
                    learning_rate=(
                        config
                        .learning_rate
                    ),
                    weight_decay=(
                        config
                        .weight_decay
                    ),
                    batch_size=(
                        config
                        .batch_size
                    ),
                    max_epochs=(
                        config
                        .max_epochs
                    ),
                    patience=(
                        config
                        .patience
                    ),
                    min_delta=(
                        config
                        .min_delta
                    ),
                    seed=pair_seed,
                    deterministic=True,
                )
            )

            training_result = (
                train_pairwise_residual_network(
                    model=model,
                    train_data=train_data,
                    train_residuals=(
                        run.train_residuals
                    ),
                    val_data=(
                        validation_data
                    ),
                    val_residuals=(
                        run.validation_residuals
                    ),
                    config=(
                        training_config
                    ),
                    device=device,
                )
            )

            raw_grid = (
                evaluate_pairwise_raw_grid(
                    model=model,
                    reference_data=(
                        reference_data
                    ),
                    batch_size=(
                        config
                        .surface_batch_size
                    ),
                    device=device,
                )
            )

            purified_vector = (
                diagonal_purified_vector(
                    raw_grid
                )
            )

            purified_vectors.append(
                purified_vector
            )

            surface_results.append(
                PairSurfaceRunResult(
                    run_index=(
                        run.run_index
                    ),
                    discovery_seed=(
                        run.seed
                    ),
                    pairwise_seed=(
                        pair_seed
                    ),
                    best_epoch=(
                        training_result
                        .best_epoch
                    ),
                    best_validation_mse=(
                        training_result
                        .best_validation_mse
                    ),
                    purified_vector=(
                        purified_vector
                        .copy()
                    ),
                )
            )

            if verbose:
                print(
                    f"  run "
                    f"{run.run_index + 1}: "
                    f"epoch="
                    f"{training_result.best_epoch + 1}, "
                    f"val_MSE="
                    f"{training_result.best_validation_mse:.6f}"
                )

        agreement = compute_isr(
            tuple(
                purified_vectors
            )
        )

        final_accepted = bool(
            candidate.selection_probability
            >= discovery.selection.threshold
            and agreement.isr
            >= config.isr_threshold
        )

        if verbose:
            print(
                f"  ISR="
                f"{agreement.isr:.4f} "
                f"-> "
                f"{'RETAIN' if final_accepted else 'REJECT'}"
            )

        results.append(
            InteractionISRResult(
                feature_j=pair[0],
                feature_k=pair[1],
                selection_count=(
                    candidate
                    .selection_count
                ),
                selection_probability=(
                    candidate
                    .selection_probability
                ),
                isr=(
                    agreement.isr
                ),
                pairwise_correlations=(
                    agreement
                    .pairwise_correlations
                ),
                surface_runs=tuple(
                    surface_results
                ),
                final_accepted=(
                    final_accepted
                ),
            )
        )

    results.sort(
        key=lambda item: (
            -int(
                item.final_accepted
            ),
            -item.selection_probability,
            -item.isr,
            item.feature_j,
            item.feature_k,
        )
    )

    return ISRDiscoveryResult(
        reference_indices=(
            reference_indices
        ),
        interactions=tuple(
            results
        ),
    )