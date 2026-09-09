from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCENARIOS = (
    "S1",
    "S2",
    "S3",
    "S4",
)

PUBLICATION_ROOT = Path(
    "results/synthetic/publication"
)

OUTPUT_ROOT = (
    PUBLICATION_ROOT
    / "figures"
)

PREDICTIVE_PATH = (
    PUBLICATION_ROOT
    / "table_synthetic_predictive.csv"
)

STRUCTURE_PATH = (
    PUBLICATION_ROOT
    / "table_synthetic_structure.csv"
)

DIAGNOSTICS_PATH = (
    PUBLICATION_ROOT
    / "table_synthetic_diagnostics.csv"
)

INFERENCE_PATH = (
    PUBLICATION_ROOT
    / "table_synthetic_inference.csv"
)


def _load_table(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Required publication table not found: {path}"
        )

    return pd.read_csv(
        path
    )


def _validate_scenarios(
    frame: pd.DataFrame,
    *,
    name: str,
) -> None:
    if "scenario" not in frame.columns:
        raise ValueError(
            f"{name} has no scenario column."
        )

    observed = set(
        frame[
            "scenario"
        ]
        .astype(str)
        .tolist()
    )

    expected = set(
        SCENARIOS
    )

    missing = (
        expected
        - observed
    )

    if missing:
        raise RuntimeError(
            f"{name} is missing scenarios: "
            f"{sorted(missing)}"
        )


def _scenario_rows(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    indexed = (
        frame
        .copy()
        .set_index(
            "scenario"
        )
    )

    return indexed.loc[
        list(
            SCENARIOS
        )
    ].copy()


def _load_inputs():
    predictive = _load_table(
        PREDICTIVE_PATH
    )

    structure = _load_table(
        STRUCTURE_PATH
    )

    diagnostics = _load_table(
        DIAGNOSTICS_PATH
    )

    inference = _load_table(
        INFERENCE_PATH
    )

    _validate_scenarios(
        predictive,
        name="Predictive table",
    )

    _validate_scenarios(
        structure,
        name="Structure table",
    )

    _validate_scenarios(
        diagnostics,
        name="Diagnostics table",
    )

    _validate_scenarios(
        inference,
        name="Inference table",
    )

    predictive = _scenario_rows(
        predictive
    )

    structure = _scenario_rows(
        structure
    )

    # Confirm exactly one primary AG-NAM vs Main NAM AUROC row
    # for each synthetic scenario.
    primary = inference[
        (
            inference[
                "family"
            ]
            .astype(str)
            == "primary"
        )
        & (
            inference[
                "comparison_id"
            ]
            .astype(str)
            == "agnam_vs_main_auroc"
        )
    ].copy()

    if len(
        primary
    ) != 4:
        raise RuntimeError(
            "Expected exactly four primary "
            "synthetic inference rows."
        )

    primary = _scenario_rows(
        primary
    )

    if "family_complete" in primary.columns:
        complete = (
            primary[
                "family_complete"
            ]
            .astype(str)
            .str.casefold()
            .isin(
                [
                    "true",
                    "1",
                ]
            )
        )

        if not complete.all():
            raise RuntimeError(
                "Synthetic primary inference family "
                "is not complete."
            )

    # Cross-check predictive and inference primary Holm p-values.
    for scenario in SCENARIOS:
        table_p = float(
            predictive.loc[
                scenario,
                "primary_p_holm",
            ]
        )

        inference_p = float(
            primary.loc[
                scenario,
                "p_holm",
            ]
        )

        if not np.isclose(
            table_p,
            inference_p,
            atol=1e-15,
            rtol=0.0,
        ):
            raise RuntimeError(
                "Primary p-value mismatch for "
                f"{scenario}."
            )

    sparsification = diagnostics[
        diagnostics[
            "metric"
        ].astype(str)
        == (
            "isr_sparsification_"
            "vs_selection"
        )
    ].copy()

    if len(
        sparsification
    ) != 4:
        raise RuntimeError(
            "Expected four ISR sparsification "
            "diagnostic rows."
        )

    sparsification = _scenario_rows(
        sparsification
    )

    return (
        predictive,
        structure,
        sparsification,
        primary,
    )


def create_synthetic_figure(
    predictive: pd.DataFrame,
    structure: pd.DataFrame,
    sparsification: pd.DataFrame,
):
    x = np.arange(
        len(
            SCENARIOS
        )
    )

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
        }
    ):
        figure, axes = plt.subplots(
            2,
            2,
            figsize=(
                12.4,
                8.6,
            ),
            constrained_layout=True,
        )

        axis_a = axes[
            0,
            0
        ]

        axis_b = axes[
            0,
            1
        ]

        axis_c = axes[
            1,
            0
        ]

        axis_d = axes[
            1,
            1
        ]

        # =====================================================================
        # PANEL A — Interaction recovery
        # =====================================================================

        recovery_mean = (
            structure[
                "interaction_auprc_mean"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        recovery_std = (
            structure[
                "interaction_auprc_std"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        axis_a.errorbar(
            x,
            recovery_mean,
            yerr=recovery_std,
            fmt="o",
            capsize=4,
            linewidth=1.2,
            markersize=6,
        )

        axis_a.set_title(
            "A  Interaction recovery",
            loc="left",
            fontweight="bold",
            pad=8,
        )

        axis_a.set_ylabel(
            "Interaction-ranking AUPRC"
        )

        axis_a.set_xticks(
            x,
            labels=SCENARIOS,
        )

        axis_a.grid(
            axis="y",
            alpha=0.18,
        )

        # =====================================================================
        # PANEL B — Precision refinement
        # =====================================================================

        selection_precision = (
            structure[
                "selection_precision_mean"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        isr_precision = (
            structure[
                "isr_precision_mean"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        width = 0.34

        axis_b.bar(
            x - width / 2,
            selection_precision,
            width,
            label="Selection-stable",
        )

        axis_b.bar(
            x + width / 2,
            isr_precision,
            width,
            label="After ISR",
        )

        axis_b.set_title(
            "B  ISR precision refinement",
            loc="left",
            fontweight="bold",
            pad=8,
        )

        axis_b.set_ylabel(
            "Interaction precision"
        )

        axis_b.set_xticks(
            x,
            labels=SCENARIOS,
        )

        axis_b.set_ylim(
            0.0,
            0.75,
        )

        axis_b.legend(
            frameon=False,
        )

        axis_b.grid(
            axis="y",
            alpha=0.18,
        )

        # =====================================================================
        # PANEL C — ISR sparsification
        # =====================================================================

        sparsification_mean = (
            sparsification[
                "mean"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        sparsification_std = (
            sparsification[
                "std"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        sparsification_n = (
            sparsification[
                "n"
            ]
            .to_numpy(
                dtype=np.int64
            )
        )

        axis_c.errorbar(
            x,
            sparsification_mean,
            yerr=sparsification_std,
            fmt="o",
            capsize=4,
            linewidth=1.2,
            markersize=6,
        )

        axis_c.set_title(
            "C  ISR sparsification",
            loc="left",
            fontweight="bold",
            pad=8,
        )

        axis_c.set_ylabel(
            "Fraction removed from selection-stable set"
        )

        axis_c.set_xticks(
            x,
            labels=SCENARIOS,
        )

        axis_c.set_ylim(
            0.0,
            1.0,
        )

        axis_c.grid(
            axis="y",
            alpha=0.18,
        )

        # S4 has undefined sparsification in four realizations.
        if sparsification_n[-1] != 20:
            axis_c.text(
                0.98,
                0.05,
                (
                    f"S4: n="
                    f"{int(sparsification_n[-1])} "
                    "defined realizations"
                ),
                transform=axis_c.transAxes,
                ha="right",
                va="bottom",
                fontsize=8.5,
            )

        # =====================================================================
        # PANEL D — Predictive ΔAUROC
        # =====================================================================

        delta_mean = (
            predictive[
                "delta_auroc_mean"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        ci_low = (
            predictive[
                "delta_auroc_ci_low"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        ci_high = (
            predictive[
                "delta_auroc_ci_high"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        lower_error = (
            delta_mean
            - ci_low
        )

        upper_error = (
            ci_high
            - delta_mean
        )

        axis_d.axhline(
            0.0,
            linewidth=1.0,
            linestyle="--",
            color="0.45",
        )

        axis_d.errorbar(
            x,
            delta_mean,
            yerr=np.vstack(
                [
                    lower_error,
                    upper_error,
                ]
            ),
            fmt="o",
            capsize=4,
            linewidth=1.2,
            markersize=6,
        )

        axis_d.set_title(
            "D  Predictive ΔAUROC",
            loc="left",
            fontweight="bold",
            pad=8,
        )

        axis_d.set_ylabel(
            "ΔAUROC (AG-NAM − Main NAM)"
        )

        axis_d.set_xticks(
            x,
            labels=SCENARIOS,
        )

        axis_d.grid(
            axis="y",
            alpha=0.18,
        )

        return figure


def save_figure(
    figure,
):
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    png_path = (
        OUTPUT_ROOT
        / (
            "figure_synthetic_"
            "benchmark_evidence.png"
        )
    )

    pdf_path = (
        OUTPUT_ROOT
        / (
            "figure_synthetic_"
            "benchmark_evidence.pdf"
        )
    )

    svg_path = (
        OUTPUT_ROOT
        / (
            "figure_synthetic_"
            "benchmark_evidence.svg"
        )
    )

    figure.savefig(
        str(
            png_path.resolve()
        ),
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )

    figure.savefig(
        str(
            pdf_path.resolve()
        ),
        bbox_inches="tight",
        facecolor="white",
    )

    figure.savefig(
        str(
            svg_path.resolve()
        ),
        bbox_inches="tight",
        facecolor="white",
    )

    return (
        png_path,
        pdf_path,
        svg_path,
    )


def main():
    (
        predictive,
        structure,
        sparsification,
        primary,
    ) = _load_inputs()

    figure = create_synthetic_figure(
        predictive,
        structure,
        sparsification,
    )

    (
        png_path,
        pdf_path,
        svg_path,
    ) = save_figure(
        figure
    )

    plt.close(
        figure
    )

    print(
        "=============================================="
    )

    print(
        "SYNTHETIC PUBLICATION FIGURE GENERATED"
    )

    print(
        "=============================================="
    )

    print(
        "PNG:",
        png_path,
    )

    print(
        "PDF:",
        pdf_path,
    )

    print(
        "SVG:",
        svg_path,
    )

    print(
        "PNG DPI: 600"
    )

    print(
        "Primary scenarios:",
        len(primary),
    )

    print(
        "STATUS: SYNTHETIC PUBLICATION FIGURE GENERATED"
    )


if __name__ == "__main__":
    main()