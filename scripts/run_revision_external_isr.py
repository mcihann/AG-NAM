"""CLI for reviewer-only external proposal x AG-NAM ISR hybrid."""

from __future__ import annotations

import argparse

from agnam.revision.external_isr_runner import (
    METHODS,
    run_external_isr_hybrid_realization,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True, choices=("S1", "S2", "S3", "S4"))
    parser.add_argument("--realization", required=True, type=int)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--stage", choices=("proposals", "isr", "both"), default="proposals")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_external_isr_hybrid_realization(
        scenario=args.scenario,
        realization_number=args.realization,
        methods=tuple(args.methods),
        stage=args.stage,
        force=args.force,
    )


if __name__ == "__main__":
    main()
