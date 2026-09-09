def generate_main_realx_figure(
    *,
    primary_dataset: pd.DataFrame,
    dataset_strength: pd.DataFrame,
    primary_result: dict[str, Any],
    png_path: str | Path = (
        DEFAULT_MAIN_FIGURE_PNG
    ),
    pdf_path: str | Path = (
        DEFAULT_MAIN_FIGURE_PDF
    ),
    svg_path: str | Path = (
        DEFAULT_MAIN_FIGURE_SVG
    ),
) -> tuple[
    Path,
    Path,
    Path,
]:
    """
    Generate the final publication-ready Real-X main figure.

    Panel A:
        Prespecified confirmatory primary endpoint.

    Panel B:
        Descriptive interaction-recovery behavior.

    Panel C:
        Descriptive ISR sparsification behavior.

    Panel D:
        Descriptive predictive AUROC change.

    No new statistical analysis is performed here.
    """
    png_path = Path(
        png_path
    )

    pdf_path = Path(
        pdf_path
    )

    svg_path = Path(
        svg_path
    )

    png_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    primary = (
        primary_dataset
        .copy()
        .sort_values(
            "primary_difference"
        )
        .reset_index(
            drop=True
        )
    )

    if len(
        primary
    ) != EXPECTED_DATASETS:
        raise RuntimeError(
            "Primary figure requires exactly nine datasets."
        )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(
            12.4,
            8.6,
        ),
        constrained_layout=True,
    )

    # =========================================================================
    # PANEL A
    # Prespecified confirmatory primary endpoint
    # =========================================================================

    ax = axes[
        0,
        0
    ]

    y = np.arange(
        len(
            primary
        )
    )

    differences = pd.to_numeric(
        primary[
            "primary_difference"
        ],
        errors="raise",
    ).to_numpy(
        dtype=np.float64
    )

    ax.scatter(
        differences,
        y,
        s=48,
        zorder=3,
    )

    ax.axvline(
        0.0,
        linestyle="--",
        linewidth=1.0,
        alpha=0.70,
    )

    ax.set_yticks(
        y
    )

    ax.set_yticklabels(
        primary[
            "dataset_name"
        ]
        .astype(
            str
        )
        .tolist()
    )

    ax.set_xlabel(
        "AUPRC advantage over random ranking"
    )

    ax.set_title(
        "A  Prespecified primary endpoint",
        loc="left",
        fontweight="bold",
        pad=8,
    )

    ax.grid(
        axis="x",
        alpha=0.18,
    )

    primary_annotation = (
        f"W = "
        f"{float(primary_result['wilcoxon_statistic']):.1f}\n"
        f"one-sided p = "
        f"{float(primary_result['p_value']):.4g}\n"
        f"{int(primary_result['positive_dataset_count'])}/"
        f"{int(primary_result['n_datasets'])} "
        f"datasets positive"
    )

    ax.text(
        0.98,
        0.035,
        primary_annotation,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.5,
        linespacing=1.25,
    )

    # =========================================================================
    # PANEL B
    # Interaction recovery
    # =========================================================================

    auprc_series = (
        extract_figure_series(
            dataset_strength,
            metric="mean_interaction_auprc",
            strengths=(
                "weak",
                "moderate",
                "strong",
            ),
        )
    )

    _scatter_box(
        axes[
            0,
            1
        ],
        series=(
            auprc_series
        ),
        strengths=(
            "weak",
            "moderate",
            "strong",
        ),
        ylabel=(
            "Interaction-ranking AUPRC"
        ),
        title=(
            "B  Interaction recovery"
        ),
    )

    # =========================================================================
    # PANEL C
    # ISR filtering / sparsification
    # =========================================================================

    sparsification_series = (
        extract_figure_series(
            dataset_strength,
            metric=(
                "isr_sparsification_fraction"
            ),
            strengths=(
                "null",
                "weak",
                "moderate",
                "strong",
            ),
        )
    )

    _scatter_box(
        axes[
            1,
            0
        ],
        series=(
            sparsification_series
        ),
        strengths=(
            "null",
            "weak",
            "moderate",
            "strong",
        ),
        ylabel=(
            "ISR sparsification fraction"
        ),
        title=(
            "C  ISR sparsification"
        ),
    )

    axes[
        1,
        0
    ].set_ylim(
        -0.04,
        1.04,
    )

    # =========================================================================
    # PANEL D
    # Predictive consequence
    # =========================================================================

    delta_series = (
        extract_figure_series(
            dataset_strength,
            metric="delta_auroc",
            strengths=(
                "null",
                "weak",
                "moderate",
                "strong",
            ),
        )
    )

    _scatter_box(
        axes[
            1,
            1
        ],
        series=(
            delta_series
        ),
        strengths=(
            "null",
            "weak",
            "moderate",
            "strong",
        ),
        ylabel=(
            "ΔAUROC (AG-NAM − Main NAM)"
        ),
        title=(
            "D  Predictive ΔAUROC"
        ),
        zero_line=True,
    )

    # =========================================================================
    # Publication export
    #
    # Deliberately no global figure title.
    # The complete description belongs in the manuscript caption.
    # =========================================================================

    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    fig.savefig(
        svg_path,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return (
        png_path,
        pdf_path,
        svg_path,
    )