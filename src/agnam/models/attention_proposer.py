from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Sequence

import torch
from torch import nn

from agnam.data.preprocessing import FeatureSpec


@dataclass
class ResidualAttentionOutput:
    """
    Output of the residual-targeted attention proposer.

    residual_prediction:
        Predicted cross-fitted pseudo-residual, shape [batch].

    attention_maps:
        Tuple containing one attention tensor per layer.
        Each tensor has shape:
        [batch, n_heads, n_features, n_features]

    contextual_tokens:
        Final contextualized feature tokens,
        shape [batch, n_features, d_model].
    """

    residual_prediction: torch.Tensor
    attention_maps: tuple[torch.Tensor, ...]
    contextual_tokens: torch.Tensor


class NumericTokenEncoder(nn.Module):
    """
    Encode one numerical predictor while preserving its feature identity.

    Input consists of:
        standardized value
        missingness indicator
    """

    def __init__(
        self,
        d_model: int,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
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


class CategoricalTokenEncoder(nn.Module):
    """Encode one categorical predictor as one feature token."""

    def __init__(
        self,
        n_categories: int,
        d_model: int,
    ) -> None:
        super().__init__()

        if n_categories < 2:
            raise ValueError(
                "n_categories must include at least "
                "<MISSING> and <UNK>."
            )

        self.embedding = nn.Embedding(
            num_embeddings=n_categories,
            embedding_dim=d_model,
        )

    def forward(
        self,
        codes: torch.Tensor,
    ) -> torch.Tensor:
        return self.embedding(
            codes.long()
        )


class FeatureTokenizer(nn.Module):
    """
    Convert original tabular predictors into one token per feature.

    No one-hot expansion is performed.
    """

    def __init__(
        self,
        feature_specs: Sequence[FeatureSpec],
        d_model: int = 64,
    ) -> None:
        super().__init__()

        if len(feature_specs) == 0:
            raise ValueError(
                "At least one feature specification is required."
            )

        if d_model <= 0:
            raise ValueError(
                "d_model must be positive."
            )

        self.feature_specs = tuple(feature_specs)
        self.n_features = len(self.feature_specs)
        self.d_model = d_model

        encoders: list[nn.Module] = []

        for spec in self.feature_specs:
            if spec.kind == "numeric":
                encoder = NumericTokenEncoder(
                    d_model=d_model,
                )

            elif spec.kind == "categorical":
                if spec.n_categories is None:
                    raise ValueError(
                        f"Categorical feature '{spec.name}' "
                        "requires n_categories."
                    )

                encoder = CategoricalTokenEncoder(
                    n_categories=spec.n_categories,
                    d_model=d_model,
                )

            else:
                raise ValueError(
                    f"Unknown feature kind: {spec.kind}"
                )

            encoders.append(encoder)

        self.encoders = nn.ModuleList(
            encoders
        )

        # Explicit feature-identity embedding.
        self.feature_identity = nn.Parameter(
            torch.empty(
                self.n_features,
                d_model,
            )
        )

        nn.init.normal_(
            self.feature_identity,
            mean=0.0,
            std=0.02,
        )

        self.norm = nn.LayerNorm(
            d_model
        )

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(
            spec.name
            for spec in self.feature_specs
        )

    def forward(
        self,
        numeric: torch.Tensor,
        numeric_missing: torch.Tensor,
        categorical: torch.Tensor,
    ) -> torch.Tensor:
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

        tokens: list[torch.Tensor] = []

        for spec, encoder in zip(
            self.feature_specs,
            self.encoders,
        ):
            if spec.kind == "numeric":
                token = encoder(
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
                token = encoder(
                    categorical[
                        :,
                        spec.transformed_index,
                    ]
                )

            tokens.append(token)

        token_tensor = torch.stack(
            tokens,
            dim=1,
        )

        token_tensor = (
            token_tensor
            + self.feature_identity.unsqueeze(0)
        )

        return self.norm(
            token_tensor
        )


class ExplicitMultiheadSelfAttention(nn.Module):
    """
    Multi-head self-attention with explicit attention probabilities.

    Attention probabilities returned by this layer are the SAME
    probabilities used to compute contextual representations.

    This is important because AG-NAM later computes gradients of the
    residual prediction with respect to these attention probabilities.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(
                "d_model must be divisible by n_heads."
            )

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = (
            d_model // n_heads
        )

        self.qkv = nn.Linear(
            d_model,
            3 * d_model,
        )

        self.output_projection = nn.Linear(
            d_model,
            d_model,
        )

        self.output_dropout = nn.Dropout(
            dropout
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        batch_size, n_features, _ = x.shape

        qkv = self.qkv(x)

        q, k, v = torch.chunk(
            qkv,
            chunks=3,
            dim=-1,
        )

        def reshape_heads(
            tensor: torch.Tensor,
        ) -> torch.Tensor:
            return (
                tensor
                .view(
                    batch_size,
                    n_features,
                    self.n_heads,
                    self.head_dim,
                )
                .transpose(1, 2)
            )

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)

        scores = torch.matmul(
            q,
            k.transpose(-2, -1),
        ) / sqrt(self.head_dim)

        attention = torch.softmax(
            scores,
            dim=-1,
        )

        contextual = torch.matmul(
            attention,
            v,
        )

        contextual = (
            contextual
            .transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                n_features,
                self.d_model,
            )
        )

        contextual = self.output_projection(
            contextual
        )

        contextual = self.output_dropout(
            contextual
        )

        return contextual, attention


class ResidualAttentionBlock(nn.Module):
    """Pre-normalized residual-attention block."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        expansion: int = 2,
    ) -> None:
        super().__init__()

        self.norm_attention = nn.LayerNorm(
            d_model
        )

        self.attention = (
            ExplicitMultiheadSelfAttention(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
            )
        )

        self.norm_ffn = nn.LayerNorm(
            d_model
        )

        hidden_dim = (
            expansion * d_model
        )

        self.ffn = nn.Sequential(
            nn.Linear(
                d_model,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_dim,
                d_model,
            ),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        normalized = self.norm_attention(
            x
        )

        attention_output, attention = (
            self.attention(
                normalized
            )
        )

        x = x + attention_output

        x = x + self.ffn(
            self.norm_ffn(x)
        )

        return x, attention


class ResidualAttentionProposer(nn.Module):
    """
    Attention-based proposal network for residual structure.

    IMPORTANT:
    This network is NOT the final AG-NAM predictor.

    Its sole role is to learn structure remaining after the
    cross-fitted main-effect NAM and to propose candidate feature pairs.
    """

    def __init__(
        self,
        feature_specs: Sequence[FeatureSpec],
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()

        if n_layers < 1:
            raise ValueError(
                "n_layers must be at least 1."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must satisfy 0 <= dropout < 1."
            )

        if d_model % n_heads != 0:
            raise ValueError(
                "d_model must be divisible by n_heads."
            )

        self.feature_specs = tuple(
            feature_specs
        )

        self.n_features = len(
            self.feature_specs
        )

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

        self.tokenizer = FeatureTokenizer(
            feature_specs=self.feature_specs,
            d_model=d_model,
        )

        self.blocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(
            d_model
        )

        residual_hidden = max(
            d_model // 2,
            16,
        )

        self.residual_head = nn.Sequential(
            nn.Linear(
                d_model,
                residual_hidden,
            ),
            nn.SiLU(),
            nn.Linear(
                residual_hidden,
                1,
            ),
        )

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.tokenizer.feature_names

    def forward(
        self,
        numeric: torch.Tensor,
        numeric_missing: torch.Tensor,
        categorical: torch.Tensor,
    ) -> ResidualAttentionOutput:
        x = self.tokenizer(
            numeric=numeric,
            numeric_missing=numeric_missing,
            categorical=categorical,
        )

        attention_maps: list[
            torch.Tensor
        ] = []

        for block in self.blocks:
            x, attention = block(x)

            attention_maps.append(
                attention
            )

        x = self.final_norm(x)

        # No CLS token is used because it would introduce a
        # non-feature node into the feature-pair attention graph.
        pooled = x.mean(
            dim=1
        )

        residual_prediction = (
            self.residual_head(
                pooled
            ).squeeze(-1)
        )

        return ResidualAttentionOutput(
            residual_prediction=residual_prediction,
            attention_maps=tuple(
                attention_maps
            ),
            contextual_tokens=x,
        )