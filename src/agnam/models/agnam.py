from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from agnam.data.preprocessing import FeatureSpec
from agnam.models.main_effect_nam import MainEffectNAM
from agnam.models.pairwise_interaction import PairwiseResidualNetwork


Pair = tuple[str, str]


@dataclass
class AGNAMOutput:
    """
    Output of the final attention-free AG-NAM predictor.

    logits:
        Binary-classification logits with shape [batch].

    main_contributions:
        Main-effect contributions in original predictor order,
        shape [batch, n_features].

    interaction_contributions:
        Explicit pairwise interaction contributions,
        shape [batch, n_interactions].

    baseline:
        Effective scalar baseline after optional centering.

    interaction_pairs:
        Pair names corresponding to interaction_contributions columns.
    """

    logits: torch.Tensor
    main_contributions: torch.Tensor
    interaction_contributions: torch.Tensor
    baseline: torch.Tensor
    interaction_pairs: tuple[Pair, ...]


class AGNAM(nn.Module):
    """
    Final attention-free AG-NAM predictor.

    The architecture contains:

        beta_0
        + explicit univariate main-effect networks
        + explicit pairwise interaction networks

    Attention is NOT present in this model.

    Interaction discovery is assumed to have already produced
    a fixed stable interaction set S*.
    """

    def __init__(
        self,
        feature_specs: Sequence[FeatureSpec],
        interaction_pairs: Sequence[Pair],
        *,
        main_hidden_dim: int = 64,
        main_depth: int = 2,
        main_dropout: float = 0.10,
        categorical_embedding_dim: int = 16,
        interaction_embedding_dim: int = 16,
        interaction_hidden_dim: int = 64,
        interaction_depth: int = 2,
        interaction_dropout: float = 0.10,
    ) -> None:
        super().__init__()

        self.feature_specs = tuple(
            feature_specs
        )

        if len(self.feature_specs) == 0:
            raise ValueError(
                "At least one feature specification is required."
            )

        original_indices = [
            spec.original_index
            for spec in self.feature_specs
        ]

        if original_indices != list(
            range(len(self.feature_specs))
        ):
            raise ValueError(
                "feature_specs must be supplied in original feature order."
            )

        feature_lookup = {
            spec.name: spec
            for spec in self.feature_specs
        }

        if len(feature_lookup) != len(
            self.feature_specs
        ):
            raise ValueError(
                "Feature names must be unique."
            )

        normalized_pairs: list[
            Pair
        ] = []

        pair_specs: list[
            tuple[
                FeatureSpec,
                FeatureSpec,
            ]
        ] = []

        seen_pairs: set[
            Pair
        ] = set()

        for raw_pair in interaction_pairs:
            if len(raw_pair) != 2:
                raise ValueError(
                    "Every interaction must contain exactly two features."
                )

            first_name = raw_pair[0]
            second_name = raw_pair[1]

            if first_name == second_name:
                raise ValueError(
                    "Self-interactions are not allowed."
                )

            if first_name not in feature_lookup:
                raise ValueError(
                    f"Unknown feature in interaction: {first_name}"
                )

            if second_name not in feature_lookup:
                raise ValueError(
                    f"Unknown feature in interaction: {second_name}"
                )

            first_spec = feature_lookup[
                first_name
            ]

            second_spec = feature_lookup[
                second_name
            ]

            if (
                first_spec.original_index
                <= second_spec.original_index
            ):
                ordered_specs = (
                    first_spec,
                    second_spec,
                )
            else:
                ordered_specs = (
                    second_spec,
                    first_spec,
                )

            normalized_pair = (
                ordered_specs[0].name,
                ordered_specs[1].name,
            )

            if normalized_pair in seen_pairs:
                raise ValueError(
                    f"Duplicate interaction pair: {normalized_pair}"
                )

            seen_pairs.add(
                normalized_pair
            )

            normalized_pairs.append(
                normalized_pair
            )

            pair_specs.append(
                ordered_specs
            )

        self.interaction_pairs = tuple(
            normalized_pairs
        )

        self.n_features = len(
            self.feature_specs
        )

        self.n_interactions = len(
            self.interaction_pairs
        )

        # -----------------------------------------------------
        # Main-effect component
        # -----------------------------------------------------

        self.main_effect_model = (
            MainEffectNAM(
                feature_specs=(
                    self.feature_specs
                ),
                hidden_dim=(
                    main_hidden_dim
                ),
                depth=main_depth,
                dropout=main_dropout,
                categorical_embedding_dim=(
                    categorical_embedding_dim
                ),
            )
        )

        # -----------------------------------------------------
        # Explicit pairwise interaction components
        # -----------------------------------------------------

        self.interaction_networks = (
            nn.ModuleList(
                [
                    PairwiseResidualNetwork(
                        feature_specs=specs,
                        feature_embedding_dim=(
                            interaction_embedding_dim
                        ),
                        hidden_dim=(
                            interaction_hidden_dim
                        ),
                        depth=(
                            interaction_depth
                        ),
                        dropout=(
                            interaction_dropout
                        ),
                    )
                    for specs in pair_specs
                ]
            )
        )

        # Post-training centering offsets.
        #
        # Subtracting each interaction's mean contribution and
        # adding their total mean to the baseline preserves
        # predictions exactly.
        self.register_buffer(
            "interaction_centering_offsets",
            torch.zeros(
                self.n_interactions
            ),
        )

    @property
    def feature_names(
        self,
    ) -> tuple[str, ...]:
        return tuple(
            spec.name
            for spec in self.feature_specs
        )

    def clear_centering(
        self,
    ) -> None:
        """
        Remove both main-effect and interaction centering offsets.
        """
        self.main_effect_model.clear_centering()

        self.interaction_centering_offsets.zero_()

    def set_interaction_centering_offsets(
        self,
        offsets: torch.Tensor,
    ) -> None:
        """
        Set post-training interaction contribution means.

        Predictions remain unchanged because the offset sum is
        transferred to the effective baseline.
        """
        if offsets.shape != (
            self.n_interactions,
        ):
            raise ValueError(
                "offsets must have shape "
                f"({self.n_interactions},), "
                f"got {tuple(offsets.shape)}."
            )

        offsets = offsets.to(
            device=(
                self
                .interaction_centering_offsets
                .device
            ),
            dtype=(
                self
                .interaction_centering_offsets
                .dtype
            ),
        )

        self.interaction_centering_offsets.copy_(
            offsets
        )

    def raw_interaction_contributions(
        self,
        numeric: torch.Tensor,
        numeric_missing: torch.Tensor,
        categorical: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute uncentered pairwise contributions.

        Returns
        -------
        torch.Tensor
            Shape [batch, n_interactions].
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

        if self.n_interactions == 0:
            return torch.zeros(
                (
                    batch_size,
                    0,
                ),
                dtype=numeric.dtype,
                device=numeric.device,
            )

        contributions = [
            network(
                numeric=numeric,
                numeric_missing=(
                    numeric_missing
                ),
                categorical=categorical,
            )
            for network
            in self.interaction_networks
        ]

        return torch.stack(
            contributions,
            dim=1,
        )

    def forward(
        self,
        numeric: torch.Tensor,
        numeric_missing: torch.Tensor,
        categorical: torch.Tensor,
    ) -> AGNAMOutput:
        """
        Compute exact additive AG-NAM decomposition.
        """
        main_output = (
            self.main_effect_model(
                numeric=numeric,
                numeric_missing=(
                    numeric_missing
                ),
                categorical=categorical,
            )
        )

        raw_interactions = (
            self.raw_interaction_contributions(
                numeric=numeric,
                numeric_missing=(
                    numeric_missing
                ),
                categorical=categorical,
            )
        )

        centered_interactions = (
            raw_interactions
            - self
            .interaction_centering_offsets
            .unsqueeze(0)
        )

        interaction_offset_sum = (
            self
            .interaction_centering_offsets
            .sum()
        )

        baseline = (
            main_output.baseline
            + interaction_offset_sum
        )

        logits = (
            baseline
            + main_output
            .main_contributions
            .sum(dim=1)
            + centered_interactions
            .sum(dim=1)
        )

        return AGNAMOutput(
            logits=logits,
            main_contributions=(
                main_output
                .main_contributions
            ),
            interaction_contributions=(
                centered_interactions
            ),
            baseline=baseline,
            interaction_pairs=(
                self.interaction_pairs
            ),
        )