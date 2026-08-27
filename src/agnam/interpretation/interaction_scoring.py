from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from agnam.data.preprocessing import TransformedTabular
from agnam.models.attention_proposer import ResidualAttentionProposer
from agnam.utils.reproducibility import get_device


@dataclass(frozen=True)
class RankedInteraction:
    feature_j: str
    feature_k: str
    index_j: int
    index_k: int
    score: float
    rank: int


@dataclass
class InteractionScoreResult:
    directed_score_matrix: np.ndarray
    symmetric_score_matrix: np.ndarray
    layer_score_matrices: tuple[np.ndarray, ...]
    ranked_interactions: tuple[RankedInteraction, ...]
    feature_names: tuple[str, ...]
    n_samples: int


def candidate_count(n_features: int) -> int:
    """
    AG-NAM predefined candidate-set rule:

        P = p(p - 1) / 2
        K = clip(ceil(0.10 * P), 5, 20)
    """
    if n_features < 2:
        return 0

    n_pairs = n_features * (n_features - 1) // 2

    proposed = ceil(0.10 * n_pairs)
    proposed = max(5, proposed)
    proposed = min(20, proposed)

    return min(n_pairs, proposed)


def _make_feature_loader(
    transformed: TransformedTabular,
    batch_size: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(transformed.numeric),
        torch.from_numpy(transformed.numeric_missing),
        torch.from_numpy(transformed.categorical),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def rank_symmetric_interactions(
    symmetric_score_matrix: np.ndarray,
    feature_names: tuple[str, ...],
) -> tuple[RankedInteraction, ...]:
    matrix = np.asarray(
        symmetric_score_matrix,
        dtype=np.float64,
    )

    n_features = len(feature_names)

    if matrix.shape != (n_features, n_features):
        raise ValueError(
            "Score matrix shape does not match feature names."
        )

    pairs: list[tuple[int, int, float]] = []

    for j in range(n_features):
        for k in range(j + 1, n_features):
            pairs.append(
                (j, k, float(matrix[j, k]))
            )

    pairs.sort(
        key=lambda item: (
            -item[2],
            item[0],
            item[1],
        )
    )

    ranked = []

    for rank, (j, k, score) in enumerate(
        pairs,
        start=1,
    ):
        ranked.append(
            RankedInteraction(
                feature_j=feature_names[j],
                feature_k=feature_names[k],
                index_j=j,
                index_k=k,
                score=score,
                rank=rank,
            )
        )

    return tuple(ranked)


def compute_interaction_scores(
    model: ResidualAttentionProposer,
    transformed: TransformedTabular,
    *,
    batch_size: int = 256,
    device: Optional[torch.device] = None,
) -> InteractionScoreResult:
    """
    Compute residual-sensitive attention interaction scores.

    For each sample, layer and head:

        attribution_jk =
            |A_jk * d(r_hat) / d(A_jk)|

    Aggregation:
        1. absolute attribution
        2. mean across heads
        3. mean across samples
        4. mean across layers
        5. j->k / k->j symmetrization
        6. diagonal removal
    """
    if len(transformed) == 0:
        raise ValueError(
            "Cannot score interactions on an empty dataset."
        )

    if device is None:
        device = get_device()

    model = model.to(device)
    model.eval()

    n_features = model.n_features
    n_layers = model.n_layers

    layer_sums = [
        torch.zeros(
            (n_features, n_features),
            dtype=torch.float64,
            device=device,
        )
        for _ in range(n_layers)
    ]

    n_seen = 0

    loader = _make_feature_loader(
        transformed=transformed,
        batch_size=batch_size,
    )

    for numeric, missing, categorical in loader:
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

        model.zero_grad(set_to_none=True)

        output = model(
            numeric=numeric,
            numeric_missing=missing,
            categorical=categorical,
        )

        gradients = torch.autograd.grad(
            outputs=output.residual_prediction.sum(),
            inputs=output.attention_maps,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )

        batch_n = numeric.shape[0]

        for layer_index, (
            attention,
            gradient,
        ) in enumerate(
            zip(
                output.attention_maps,
                gradients,
            )
        ):
            attribution = torch.abs(
                attention * gradient
            )

            # [batch, heads, p, p] -> [batch, p, p]
            attribution = attribution.mean(
                dim=1
            )

            layer_sums[layer_index] += (
                attribution
                .sum(dim=0)
                .to(torch.float64)
            )

        n_seen += batch_n

    if n_seen != len(transformed):
        raise RuntimeError(
            "Interaction scorer did not process all observations."
        )

    layer_matrices = tuple(
        (
            layer_sum / n_seen
        )
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
        for layer_sum in layer_sums
    )

    directed = np.mean(
        np.stack(
            layer_matrices,
            axis=0,
        ),
        axis=0,
    )

    symmetric = 0.5 * (
        directed + directed.T
    )

    np.fill_diagonal(
        symmetric,
        0.0,
    )

    ranked = rank_symmetric_interactions(
        symmetric_score_matrix=symmetric,
        feature_names=model.feature_names,
    )

    return InteractionScoreResult(
        directed_score_matrix=directed,
        symmetric_score_matrix=symmetric,
        layer_score_matrices=layer_matrices,
        ranked_interactions=ranked,
        feature_names=model.feature_names,
        n_samples=n_seen,
    )