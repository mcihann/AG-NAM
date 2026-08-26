from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from agnam.data.preprocessing import FeatureSpec


@dataclass
class NAMOutput:
    """
    Output of the main-effect neural additive model.

    logits:
        Binary-classification logits, shape [batch].

    main_contributions:
        Per-feature additive contributions in ORIGINAL feature order,
        shape [batch, n_features].

    baseline:
        Effective intercept after optional contribution centering.
    """

    logits: torch.Tensor
    main_contributions: torch.Tensor
    baseline: torch.Tensor


class FeatureMLP(nn.Module):
    """Small feature-specific MLP ending in one scalar contribution."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        depth: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        if depth < 1:
            raise ValueError("depth must be at least 1.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1.")

        layers: list[nn.Module] = []

        current_dim = input_dim

        for _ in range(depth):
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.SiLU(),
                ]
            )

            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))

            current_dim = hidden_dim

        layers.append(
            nn.Linear(current_dim, 1)
        )

        self.network = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class NumericMainEffect(nn.Module):
    """
    Main-effect network for one numerical predictor.

    Input:
        standardized value
        missingness indicator

    These remain one conceptual feature and produce one contribution.
    """

    def __init__(
        self,
        hidden_dim: int,
        depth: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.mlp = FeatureMLP(
            input_dim=2,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
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

        return self.mlp(x)


class CategoricalMainEffect(nn.Module):
    """Main-effect network for one categorical predictor."""

    def __init__(
        self,
        n_categories: int,
        embedding_dim: int,
        hidden_dim: int,
        depth: int,
        dropout: float,
    ) -> None:
        super().__init__()

        if n_categories < 2:
            raise ValueError(
                "n_categories must include at least "
                "<MISSING> and <UNK>."
            )

        self.embedding = nn.Embedding(
            num_embeddings=n_categories,
            embedding_dim=embedding_dim,
        )

        self.mlp = FeatureMLP(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
        )

    def forward(
        self,
        codes: torch.Tensor,
    ) -> torch.Tensor:
        embedded = self.embedding(
            codes.long()
        )

        return self.mlp(embedded)


class MainEffectNAM(nn.Module):
    """
    Intrinsically decomposable main-effect neural additive model.

    Final logit:

        eta(x) = baseline + sum_j contribution_j(x_j)

    Feature contributions are returned in the ORIGINAL predictor order.
    """

    def __init__(
        self,
        feature_specs: Sequence[FeatureSpec],
        hidden_dim: int = 64,
        depth: int = 2,
        dropout: float = 0.1,
        categorical_embedding_dim: int = 16,
    ) -> None:
        super().__init__()

        if len(feature_specs) == 0:
            raise ValueError(
                "At least one feature specification is required."
            )

        self.feature_specs = tuple(feature_specs)
        self.n_features = len(self.feature_specs)

        original_indices = [
            spec.original_index
            for spec in self.feature_specs
        ]

        expected_indices = list(
            range(self.n_features)
        )

        if original_indices != expected_indices:
            raise ValueError(
                "feature_specs must be supplied in original feature order."
            )

        modules: list[nn.Module] = []

        for spec in self.feature_specs:
            if spec.kind == "numeric":
                module = NumericMainEffect(
                    hidden_dim=hidden_dim,
                    depth=depth,
                    dropout=dropout,
                )

            elif spec.kind == "categorical":
                if spec.n_categories is None:
                    raise ValueError(
                        f"Categorical feature '{spec.name}' "
                        "requires n_categories."
                    )

                module = CategoricalMainEffect(
                    n_categories=spec.n_categories,
                    embedding_dim=categorical_embedding_dim,
                    hidden_dim=hidden_dim,
                    depth=depth,
                    dropout=dropout,
                )

            else:
                raise ValueError(
                    f"Unknown feature kind '{spec.kind}' "
                    f"for feature '{spec.name}'."
                )

            modules.append(module)

        self.feature_networks = nn.ModuleList(
            modules
        )

        self.intercept = nn.Parameter(
            torch.zeros(())
        )

        # These offsets are zero during ordinary training.
        # After fitting, they can be estimated on training data so that
        # every feature contribution has approximately zero training mean.
        #
        # Subtracting offsets from contributions and adding their sum
        # to the baseline preserves predictions exactly.
        self.register_buffer(
            "centering_offsets",
            torch.zeros(self.n_features),
        )

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(
            spec.name
            for spec in self.feature_specs
        )

    def clear_centering(self) -> None:
        self.centering_offsets.zero_()

    def set_centering_offsets(
        self,
        offsets: torch.Tensor,
    ) -> None:
        if offsets.shape != (
            self.n_features,
        ):
            raise ValueError(
                "offsets must have shape "
                f"({self.n_features},), got {tuple(offsets.shape)}."
            )

        offsets = offsets.to(
            device=self.centering_offsets.device,
            dtype=self.centering_offsets.dtype,
        )

        self.centering_offsets.copy_(
            offsets
        )

    def raw_contributions(
        self,
        numeric: torch.Tensor,
        numeric_missing: torch.Tensor,
        categorical: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute uncentered feature contributions.

        Returns
        -------
        torch.Tensor
            Shape [batch, n_original_features].
        """
        if numeric.ndim != 2:
            raise ValueError(
                "numeric must have shape [batch, n_numeric]."
            )

        if numeric_missing.shape != numeric.shape:
            raise ValueError(
                "numeric_missing must match numeric shape."
            )

        if categorical.ndim != 2:
            raise ValueError(
                "categorical must have shape "
                "[batch, n_categorical]."
            )

        batch_size = numeric.shape[0]

        if categorical.shape[0] != batch_size:
            raise ValueError(
                "numeric and categorical batch sizes must match."
            )

        contributions: list[torch.Tensor] = []

        for spec, network in zip(
            self.feature_specs,
            self.feature_networks,
        ):
            if spec.kind == "numeric":
                contribution = network(
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
                contribution = network(
                    categorical[
                        :,
                        spec.transformed_index,
                    ]
                )

            contributions.append(
                contribution
            )

        return torch.stack(
            contributions,
            dim=1,
        )

    def forward(
        self,
        numeric: torch.Tensor,
        numeric_missing: torch.Tensor,
        categorical: torch.Tensor,
    ) -> NAMOutput:
        raw = self.raw_contributions(
            numeric=numeric,
            numeric_missing=numeric_missing,
            categorical=categorical,
        )

        centered = (
            raw
            - self.centering_offsets.unsqueeze(0)
        )

        baseline = (
            self.intercept
            + self.centering_offsets.sum()
        )

        logits = (
            baseline
            + centered.sum(dim=1)
        )

        return NAMOutput(
            logits=logits,
            main_contributions=centered,
            baseline=baseline,
        )