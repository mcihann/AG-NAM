from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from agnam.data.preprocessing import FeatureSpec


class NumericPairEncoder(nn.Module):
    """Encode one numerical feature for an explicit pairwise model."""

    def __init__(
        self,
        embedding_dim: int = 16,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(2, embedding_dim),
            nn.SiLU(),
            nn.Linear(
                embedding_dim,
                embedding_dim,
            ),
        )

    def forward(
        self,
        value: torch.Tensor,
        missing: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.stack(
            [value, missing],
            dim=-1,
        )

        return self.network(x)


class CategoricalPairEncoder(nn.Module):
    """Encode one categorical feature for an explicit pairwise model."""

    def __init__(
        self,
        n_categories: int,
        embedding_dim: int = 16,
    ) -> None:
        super().__init__()

        if n_categories < 2:
            raise ValueError(
                "n_categories must include <MISSING> and <UNK>."
            )

        self.embedding = nn.Embedding(
            num_embeddings=n_categories,
            embedding_dim=embedding_dim,
        )

    def forward(
        self,
        codes: torch.Tensor,
    ) -> torch.Tensor:
        return self.embedding(
            codes.long()
        )


class PairwiseResidualNetwork(nn.Module):
    """
    Explicit two-feature residual model:

        h_jk(x_j, x_k)

    The raw network is permitted to contain marginal structure.
    Functional-ANOVA purification is applied after fitting before
    interaction surfaces are compared across runs.
    """

    def __init__(
        self,
        feature_specs: Sequence[FeatureSpec],
        feature_embedding_dim: int = 16,
        hidden_dim: int = 64,
        depth: int = 2,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()

        self.feature_specs = tuple(
            feature_specs
        )

        if len(self.feature_specs) != 2:
            raise ValueError(
                "PairwiseResidualNetwork requires exactly two features."
            )

        if depth < 1:
            raise ValueError(
                "depth must be at least 1."
            )

        if feature_embedding_dim <= 0:
            raise ValueError(
                "feature_embedding_dim must be positive."
            )

        if hidden_dim <= 0:
            raise ValueError(
                "hidden_dim must be positive."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must satisfy 0 <= dropout < 1."
            )

        encoders: list[nn.Module] = []

        for spec in self.feature_specs:
            if spec.kind == "numeric":
                encoder = NumericPairEncoder(
                    embedding_dim=feature_embedding_dim,
                )

            elif spec.kind == "categorical":
                if spec.n_categories is None:
                    raise ValueError(
                        f"Categorical feature '{spec.name}' "
                        "requires n_categories."
                    )

                encoder = CategoricalPairEncoder(
                    n_categories=spec.n_categories,
                    embedding_dim=feature_embedding_dim,
                )

            else:
                raise ValueError(
                    f"Unknown feature kind: {spec.kind}"
                )

            encoders.append(
                encoder
            )

        self.encoders = nn.ModuleList(
            encoders
        )

        layers: list[nn.Module] = []

        current_dim = (
            2 * feature_embedding_dim
        )

        for _ in range(depth):
            layers.extend(
                [
                    nn.Linear(
                        current_dim,
                        hidden_dim,
                    ),
                    nn.SiLU(),
                ]
            )

            if dropout > 0.0:
                layers.append(
                    nn.Dropout(dropout)
                )

            current_dim = hidden_dim

        layers.append(
            nn.Linear(
                current_dim,
                1,
            )
        )

        self.network = nn.Sequential(
            *layers
        )

    @property
    def feature_names(
        self,
    ) -> tuple[str, str]:
        return (
            self.feature_specs[0].name,
            self.feature_specs[1].name,
        )

    def forward(
        self,
        numeric: torch.Tensor,
        numeric_missing: torch.Tensor,
        categorical: torch.Tensor,
    ) -> torch.Tensor:
        encoded: list[
            torch.Tensor
        ] = []

        for spec, encoder in zip(
            self.feature_specs,
            self.encoders,
        ):
            if spec.kind == "numeric":
                representation = encoder(
                    numeric[
                        :,
                        spec.transformed_index,
                    ],
                    numeric_missing[
                        :,
                        spec.transformed_index,
                    ],
                )

            else:
                representation = encoder(
                    categorical[
                        :,
                        spec.transformed_index,
                    ]
                )

            encoded.append(
                representation
            )

        pair_representation = torch.cat(
            encoded,
            dim=-1,
        )

        return (
            self.network(
                pair_representation
            )
            .squeeze(-1)
        )