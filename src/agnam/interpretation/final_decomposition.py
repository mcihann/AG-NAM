from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from agnam.data.preprocessing import (
    TransformedTabular,
)
from agnam.interpretation.surface_reproducibility import (
    evaluate_pairwise_cross_grid,
    evaluate_pairwise_raw_grid,
)
from agnam.models.agnam import AGNAM
from agnam.utils.reproducibility import (
    get_device,
)


Pair = tuple[str, str]


@dataclass
class PurifiedAGNAMOutput:
    """
    Functionally purified decomposition of a trained AG-NAM.

    The decomposition is empirically identified relative to a fixed
    reference distribution.

    logits:
        Reconstructed logits after purification.

    main_contributions:
        Original main effects plus marginal components reallocated
        from pairwise networks.

    interaction_contributions:
        Functional-ANOVA-purified pairwise effects.

    baseline:
        Baseline after interaction grand means have been transferred.

    max_logit_invariance_error:
        Maximum absolute difference between the original AG-NAM logits
        and the purified decomposition.
    """

    logits: torch.Tensor
    main_contributions: torch.Tensor
    interaction_contributions: torch.Tensor

    baseline: torch.Tensor

    interaction_pairs: tuple[
        Pair,
        ...
    ]

    reference_size: int

    max_logit_invariance_error: float


def _validate_compatible_transforms(
    model: AGNAM,
    transformed: TransformedTabular,
    reference_data: TransformedTabular,
) -> None:
    if len(transformed) == 0:
        raise ValueError(
            "transformed cannot be empty."
        )

    if len(reference_data) == 0:
        raise ValueError(
            "reference_data cannot be empty."
        )

    model_specs = tuple(
        model.feature_specs
    )

    if tuple(
        transformed.feature_specs
    ) != model_specs:
        raise ValueError(
            "transformed feature specifications do not "
            "match the AG-NAM model."
        )

    if tuple(
        reference_data.feature_specs
    ) != model_specs:
        raise ValueError(
            "reference_data feature specifications do not "
            "match the AG-NAM model."
        )


def _slice_transformed(
    transformed: TransformedTabular,
    start: int,
    stop: int,
) -> TransformedTabular:
    return TransformedTabular(
        numeric=(
            transformed
            .numeric[
                start:stop
            ]
        ),
        numeric_missing=(
            transformed
            .numeric_missing[
                start:stop
            ]
        ),
        categorical=(
            transformed
            .categorical[
                start:stop
            ]
        ),
        numeric_feature_names=(
            transformed
            .numeric_feature_names
        ),
        categorical_feature_names=(
            transformed
            .categorical_feature_names
        ),
        feature_specs=(
            transformed
            .feature_specs
        ),
    )


def _to_model_tensors(
    transformed: TransformedTabular,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    numeric = torch.from_numpy(
        transformed.numeric
    ).to(
        device
    )

    missing = torch.from_numpy(
        transformed.numeric_missing
    ).to(
        device
    )

    categorical = torch.from_numpy(
        transformed.categorical
    ).to(
        device
    )

    return (
        numeric,
        missing,
        categorical,
    )


@torch.no_grad()
def compute_purified_agnam_decomposition(
    model: AGNAM,
    transformed: TransformedTabular,
    reference_data: TransformedTabular,
    *,
    query_batch_size: int = 256,
    surface_batch_size: int = 8192,
    device: Optional[
        torch.device
    ] = None,
) -> PurifiedAGNAMOutput:
    """
    Compute an algebraically equivalent functional-ANOVA
    decomposition of a trained AG-NAM.

    For every interaction q = (j, k):

        h_q(x_j, x_k)

    is decomposed as:

        h_q =
            h_q^pure
            + a_j(x_j)
            + a_k(x_k)
            + mu_q

    where:

        a_j(x_j)
            = E_ref,k[h_q(x_j, X_k)] - mu_q

        a_k(x_k)
            = E_ref,j[h_q(X_j, x_k)] - mu_q

        mu_q
            = E_ref,j,k[h_q(X_j, X_k)]

    and:

        h_q^pure =
            h_q
            - E_ref,k[h_q(x_j, X_k)]
            - E_ref,j[h_q(X_j, x_k)]
            + mu_q

    The marginal terms are transferred into their corresponding
    main-effect contributions and mu_q is transferred into the
    baseline.

    Therefore predictions are preserved:

        eta_raw(x) == eta_purified(x)

    up to floating-point tolerance.

    The reference distribution is empirical and corresponds to the
    product of the marginal feature distributions represented by
    reference_data.
    """
    if query_batch_size <= 0:
        raise ValueError(
            "query_batch_size must be positive."
        )

    if surface_batch_size <= 0:
        raise ValueError(
            "surface_batch_size must be positive."
        )

    _validate_compatible_transforms(
        model=model,
        transformed=transformed,
        reference_data=reference_data,
    )

    if device is None:
        device = get_device()

    model = model.to(
        device
    )

    model.eval()

    # ---------------------------------------------------------
    # Interaction grand means on the fixed reference support
    # ---------------------------------------------------------

    grand_means: list[
        float
    ] = []

    for interaction_network in (
        model.interaction_networks
    ):
        reference_grid = (
            evaluate_pairwise_raw_grid(
                model=interaction_network,
                reference_data=(
                    reference_data
                ),
                batch_size=(
                    surface_batch_size
                ),
                device=device,
            )
        )

        grand_means.append(
            float(
                reference_grid.mean()
            )
        )

    grand_mean_sum = float(
        np.sum(
            grand_means
        )
    )

    # ---------------------------------------------------------
    # Query decomposition
    # ---------------------------------------------------------

    all_logits: list[
        torch.Tensor
    ] = []

    all_main: list[
        torch.Tensor
    ] = []

    all_interactions: list[
        torch.Tensor
    ] = []

    max_invariance_error = 0.0

    purified_baseline: torch.Tensor | None = None

    for start in range(
        0,
        len(transformed),
        query_batch_size,
    ):
        stop = min(
            start + query_batch_size,
            len(transformed),
        )

        query_data = _slice_transformed(
            transformed=transformed,
            start=start,
            stop=stop,
        )

        (
            numeric,
            missing,
            categorical,
        ) = _to_model_tensors(
            transformed=query_data,
            device=device,
        )

        # Original model output, used only for invariance auditing.
        raw_output = model(
            numeric=numeric,
            numeric_missing=missing,
            categorical=categorical,
        )

        # Main-effect decomposition before redistribution.
        main_output = (
            model
            .main_effect_model(
                numeric=numeric,
                numeric_missing=missing,
                categorical=categorical,
            )
        )

        adjusted_main = (
            main_output
            .main_contributions
            .clone()
        )

        raw_interactions = (
            model
            .raw_interaction_contributions(
                numeric=numeric,
                numeric_missing=missing,
                categorical=categorical,
            )
        )

        purified_interactions: list[
            torch.Tensor
        ] = []

        for interaction_index, (
            interaction_network,
            grand_mean,
        ) in enumerate(
            zip(
                model.interaction_networks,
                grand_means,
            )
        ):
            # -------------------------------------------------
            # E_k[h(x_j, X_k)]
            # -------------------------------------------------

            row_grid = (
                evaluate_pairwise_cross_grid(
                    model=interaction_network,
                    row_data=query_data,
                    column_data=reference_data,
                    batch_size=surface_batch_size,
                    device=device,
                )
            )

            row_mean = torch.from_numpy(
                row_grid.mean(
                    axis=1
                ).astype(
                    np.float32
                )
            ).to(
                device
            )

            # -------------------------------------------------
            # E_j[h(X_j, x_k)]
            # -------------------------------------------------

            column_grid = (
                evaluate_pairwise_cross_grid(
                    model=interaction_network,
                    row_data=reference_data,
                    column_data=query_data,
                    batch_size=surface_batch_size,
                    device=device,
                )
            )

            column_mean = torch.from_numpy(
                column_grid.mean(
                    axis=0
                ).astype(
                    np.float32
                )
            ).to(
                device
            )

            grand = torch.tensor(
                grand_mean,
                dtype=(
                    raw_interactions.dtype
                ),
                device=device,
            )

            raw_pair = (
                raw_interactions[
                    :,
                    interaction_index,
                ]
            )

            pure_pair = (
                raw_pair
                - row_mean
                - column_mean
                + grand
            )

            purified_interactions.append(
                pure_pair
            )

            first_spec = (
                interaction_network
                .feature_specs[0]
            )

            second_spec = (
                interaction_network
                .feature_specs[1]
            )

            # -------------------------------------------------
            # Transfer marginal pieces to main effects
            # -------------------------------------------------

            adjusted_main[
                :,
                first_spec.original_index,
            ] += (
                row_mean
                - grand
            )

            adjusted_main[
                :,
                second_spec.original_index,
            ] += (
                column_mean
                - grand
            )

        if model.n_interactions == 0:
            interaction_tensor = torch.zeros(
                (
                    len(query_data),
                    0,
                ),
                dtype=(
                    adjusted_main.dtype
                ),
                device=device,
            )

        else:
            interaction_tensor = (
                torch.stack(
                    purified_interactions,
                    dim=1,
                )
            )

        baseline = (
            main_output.baseline
            + torch.tensor(
                grand_mean_sum,
                dtype=(
                    main_output
                    .baseline
                    .dtype
                ),
                device=device,
            )
        )

        reconstructed_logits = (
            baseline
            + adjusted_main.sum(
                dim=1
            )
            + interaction_tensor.sum(
                dim=1
            )
        )

        batch_error = float(
            torch.max(
                torch.abs(
                    reconstructed_logits
                    - raw_output.logits
                )
            )
            .detach()
            .cpu()
            .item()
        )

        max_invariance_error = max(
            max_invariance_error,
            batch_error,
        )

        all_logits.append(
            reconstructed_logits
            .detach()
            .cpu()
        )

        all_main.append(
            adjusted_main
            .detach()
            .cpu()
        )

        all_interactions.append(
            interaction_tensor
            .detach()
            .cpu()
        )

        if purified_baseline is None:
            purified_baseline = (
                baseline
                .detach()
                .cpu()
            )

    if purified_baseline is None:
        raise RuntimeError(
            "Purified decomposition produced no batches."
        )

    return PurifiedAGNAMOutput(
        logits=torch.cat(
            all_logits,
            dim=0,
        ),
        main_contributions=torch.cat(
            all_main,
            dim=0,
        ),
        interaction_contributions=torch.cat(
            all_interactions,
            dim=0,
        ),
        baseline=purified_baseline,
        interaction_pairs=(
            model.interaction_pairs
        ),
        reference_size=len(
            reference_data
        ),
        max_logit_invariance_error=float(
            max_invariance_error
        ),
    )