from __future__ import annotations

import numpy as np
import pandas as pd

from agnam.benchmarking.realx_generator import (
    generate_realx_run,
)
from agnam.benchmarking.realx_protocol import (
    build_realx_schedule,
)


def build_toy_real_x(
    n: int = 600,
) -> pd.DataFrame:
    rng = np.random.default_rng(
        20260908
    )

    data = {}

    for index in range(
        8
    ):
        data[
            f"num_{index + 1}"
        ] = rng.normal(
            loc=(
                index
                * 0.15
            ),
            scale=(
                1.0
                + index
                * 0.05
            ),
            size=n,
        )

    levels = np.asarray(
        [
            "A",
            "B",
            "C",
            "D",
        ],
        dtype=object,
    )

    for index in range(
        6
    ):
        data[
            f"cat_{index + 1}"
        ] = rng.choice(
            levels,
            size=n,
            replace=True,
        )

    frame = pd.DataFrame(
        data
    )

    frame.loc[
        frame.index[
            ::37
        ],
        "num_2",
    ] = np.nan

    frame.loc[
        frame.index[
            ::41
        ],
        "cat_3",
    ] = None

    return frame


def main():
    X = build_toy_real_x()

    schedule = build_realx_schedule()

    null_run = next(
        run
        for run in schedule
        if (
            run.task_id == 49
            and run.realization == 1
            and run.strength_name == "null"
        )
    )

    moderate_run = next(
        run
        for run in schedule
        if (
            run.task_id == 49
            and run.realization == 1
            and run.strength_name == "moderate"
        )
    )

    null_data = generate_realx_run(
        X,
        null_run,
    )

    moderate_data = generate_realx_run(
        X,
        moderate_run,
    )

    print(
        "========================================"
    )

    print(
        "REAL-X GENERATOR SANITY AUDIT"
    )

    print(
        "========================================"
    )

    print(
        f"Rows: {len(moderate_data.X)}"
    )

    print(
        f"Eligible features: "
        f"{len(moderate_data.eligible_features)}"
    )

    print(
        "\nMain-effect features:"
    )

    for feature in (
        moderate_data.main_features
    ):
        print(
            " ",
            feature,
        )

    print(
        "\nTemplate true interactions:"
    )

    for pair in (
        moderate_data.template_true_pairs
    ):
        print(
            " ",
            pair,
        )

    print(
        "\nPurification marginal errors:"
    )

    for (
        pair,
        error,
    ) in (
        moderate_data
        .interaction_marginal_errors
        .items()
    ):
        print(
            f"  {pair}: "
            f"{error:.3e}"
        )

    print(
        "\nComposite standard deviations:"
    )

    print(
        "  main:",
        f"{np.std(moderate_data.main_composite):.6f}",
    )

    print(
        "  interaction:",
        f"{np.std(moderate_data.interaction_composite):.6f}",
    )

    print(
        "\nSplit sizes:"
    )

    print(
        "  train:",
        len(
            moderate_data.train_indices
        ),
    )

    print(
        "  validation:",
        len(
            moderate_data.validation_indices
        ),
    )

    print(
        "  test:",
        len(
            moderate_data.test_indices
        ),
    )

    print(
        "\nNULL CONDITION"
    )

    print(
        "  active true interactions:",
        len(
            null_data.active_true_pairs
        ),
    )

    print(
        "  expected prevalence:",
        f"{null_data.expected_prevalence:.6f}",
    )

    print(
        "  realized prevalence:",
        f"{null_data.realized_prevalence:.6f}",
    )

    print(
        "  interaction signal std:",
        f"{np.std(null_data.interaction_signal):.6f}",
    )

    print(
        "\nMODERATE CONDITION"
    )

    print(
        "  active true interactions:",
        len(
            moderate_data.active_true_pairs
        ),
    )

    print(
        "  expected prevalence:",
        f"{moderate_data.expected_prevalence:.6f}",
    )

    print(
        "  realized prevalence:",
        f"{moderate_data.realized_prevalence:.6f}",
    )

    print(
        "  interaction signal std:",
        f"{np.std(moderate_data.interaction_signal):.6f}",
    )

    same_rows = np.array_equal(
        null_data.source_row_positions,
        moderate_data.source_row_positions,
    )

    same_states = (
        null_data.state_codes.equals(
            moderate_data.state_codes
        )
    )

    same_truth_templates = (
        null_data.template_true_pairs
        == moderate_data.template_true_pairs
    )

    same_uniforms = np.array_equal(
        null_data.uniform_draws,
        moderate_data.uniform_draws,
    )

    same_split = (
        np.array_equal(
            null_data.train_indices,
            moderate_data.train_indices,
        )
        and np.array_equal(
            null_data.validation_indices,
            moderate_data.validation_indices,
        )
        and np.array_equal(
            null_data.test_indices,
            moderate_data.test_indices,
        )
    )

    print(
        "\nCOMMON RANDOM NUMBERS"
    )

    print(
        "  same sampled rows:",
        same_rows,
    )

    print(
        "  same state encoding:",
        same_states,
    )

    print(
        "  same truth templates:",
        same_truth_templates,
    )

    print(
        "  same Bernoulli uniforms:",
        same_uniforms,
    )

    print(
        "  same split:",
        same_split,
    )

    print(
        "\nSTATUS: GENERATOR SANITY AUDIT COMPLETE"
    )


if __name__ == "__main__":
    main()