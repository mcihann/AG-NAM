from __future__ import annotations

from pathlib import Path


OUTPUT = Path("synthetic_scenario_source.txt")

SEARCH_ROOTS = (
    Path("src/agnam"),
    Path("configs"),
    Path("docs"),
)

TEXT_SUFFIXES = {
    ".py",
    ".yaml",
    ".yml",
    ".json",
    ".md",
}

SCENARIO_TERMS = (
    "S1",
    "S2",
    "S3",
    "S4",
)

CONTEXT_TERMS = (
    "synthetic",
    "scenario",
    "interaction",
    "true_interaction",
    "ground_truth",
    "generator",
    "data-generating",
    "data generating",
)


def relevant_file(
    text: str,
) -> bool:
    lower = text.lower()

    scenario_hits = sum(
        term.lower() in lower
        for term in SCENARIO_TERMS
    )

    context_hits = sum(
        term.lower() in lower
        for term in CONTEXT_TERMS
    )

    # Require substantial scenario evidence rather
    # than dumping unrelated files mentioning one ID.
    return (
        scenario_hits >= 2
        and context_hits >= 1
    )


def main() -> None:
    sections: list[str] = []

    matched_paths: list[Path] = []

    for root in SEARCH_ROOTS:
        if not root.exists():
            continue

        for path in sorted(
            root.rglob("*")
        ):
            if (
                not path.is_file()
                or path.suffix.lower()
                not in TEXT_SUFFIXES
            ):
                continue

            try:
                text = path.read_text(
                    encoding="utf-8",
                    errors="ignore",
                )
            except Exception:
                continue

            if relevant_file(
                text
            ):
                matched_paths.append(
                    path
                )

                sections.append(
                    "\n"
                    + "=" * 100
                    + "\n"
                    + f"FILE: {path}\n"
                    + "=" * 100
                    + "\n\n"
                    + text
                    + "\n"
                )

    header = (
        "AG-NAM SYNTHETIC SCENARIO SOURCE AUDIT\n"
        "======================================\n\n"
        f"Matched files: {len(matched_paths)}\n\n"
        "FILES\n"
        "-----\n"
        + "\n".join(
            str(path)
            for path in matched_paths
        )
        + "\n"
    )

    OUTPUT.write_text(
        header
        + "".join(
            sections
        ),
        encoding="utf-8",
    )

    print(
        "=============================================="
    )
    print(
        "SYNTHETIC SCENARIO AUDIT COMPLETE"
    )
    print(
        "=============================================="
    )
    print(
        "Matched files:",
        len(matched_paths),
    )
    print(
        "Output:",
        OUTPUT.resolve(),
    )
    print(
        "Size:",
        OUTPUT.stat().st_size,
        "bytes",
    )
    print()
    print(
        "No benchmark evidence was modified."
    )


if __name__ == "__main__":
    main()