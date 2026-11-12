from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from agnam.data.preprocessing import TransformedTabular
from agnam.interpretation.interaction_scoring import (
    InteractionScoreResult,
    rank_symmetric_interactions,
)
from agnam.models.attention_proposer import ResidualAttentionProposer
from agnam.utils.reproducibility import get_device


AttentionScoreMode = Literal[
    "raw_attention",
    "gradient_only",
    "grad_x_attention",
]

ATTENTION_SCORE_MODES: tuple[AttentionScoreMode, ...] = (
    "raw_attention",
    "gradient_only",
    "grad_x_attention",
)


@dataclass(frozen=True)
class AttentionScoreAblationResult:
    """Interaction rankings from matched scoring rules on one proposer."""

    raw_attention: InteractionScoreResult
    gradient_only: InteractionScoreResult
    grad_x_attention: InteractionScoreResult

    def by_mode(
        self,
        mode: AttentionScoreMode,
    ) -> InteractionScoreResult:
        if mode == "raw_attention":
            return self.raw_attention
        if mode == "gradient_only":
            return self.gradient_only
        if mode == "grad_x_attention":
            return self.grad_x_attention
        raise ValueError(f"Unknown attention score mode: {mode}")


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


def _build_score_result(
    layer_sums: list[torch.Tensor],
    *,
    n_seen: int,
    feature_names: tuple[str, ...],
) -> InteractionScoreResult:
    layer_matrices = tuple(
        (layer_sum / n_seen)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
        for layer_sum in layer_sums
    )

    directed = np.mean(
        np.stack(layer_matrices, axis=0),
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
        feature_names=feature_names,
    )

    return InteractionScoreResult(
        directed_score_matrix=directed,
        symmetric_score_matrix=symmetric,
        layer_score_matrices=layer_matrices,
        ranked_interactions=ranked,
        feature_names=feature_names,
        n_samples=n_seen,
    )


def compute_attention_score_ablation(
    model: ResidualAttentionProposer,
    transformed: TransformedTabular,
    *,
    batch_size: int = 256,
    device: Optional[torch.device] = None,
) -> AttentionScoreAblationResult:
    """
    Compute three interaction-ranking rules on the same fitted proposer.

    The proposer parameters, scoring observations, attention tensors, and
    residual-prediction gradients are shared across all three rules:

        raw_attention:   |A_jk|
        gradient_only:   |d(r_hat) / d(A_jk)|
        grad_x_attention:|A_jk * d(r_hat) / d(A_jk)|

    This function is used only for reviewer-requested post-submission
    ablation analyses. It does not alter the locked primary AG-NAM scorer.
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

    mode_layer_sums: dict[
        AttentionScoreMode,
        list[torch.Tensor],
    ] = {
        mode: [
            torch.zeros(
                (n_features, n_features),
                dtype=torch.float64,
                device=device,
            )
            for _ in range(n_layers)
        ]
        for mode in ATTENTION_SCORE_MODES
    }

    loader = _make_feature_loader(
        transformed=transformed,
        batch_size=batch_size,
    )

    n_seen = 0

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
            attributions: dict[
                AttentionScoreMode,
                torch.Tensor,
            ] = {
                "raw_attention": torch.abs(attention),
                "gradient_only": torch.abs(gradient),
                "grad_x_attention": torch.abs(
                    attention * gradient
                ),
            }

            for mode, attribution in attributions.items():
                # [batch, heads, p, p] -> [batch, p, p]
                attribution = attribution.mean(dim=1)

                mode_layer_sums[mode][layer_index] += (
                    attribution
                    .sum(dim=0)
                    .to(torch.float64)
                )

        n_seen += batch_n

    if n_seen != len(transformed):
        raise RuntimeError(
            "Attention ablation scorer did not process all observations."
        )

    results = {
        mode: _build_score_result(
            mode_layer_sums[mode],
            n_seen=n_seen,
            feature_names=model.feature_names,
        )
        for mode in ATTENTION_SCORE_MODES
    }

    return AttentionScoreAblationResult(
        raw_attention=results["raw_attention"],
        gradient_only=results["gradient_only"],
        grad_x_attention=results["grad_x_attention"],
    )
