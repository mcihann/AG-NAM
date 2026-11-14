from __future__ import annotations

import argparse

from agnam.revision.external_recovery_runner import (
    METHODS,
    run_external_recovery_realization,
)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run reviewer-requested external interaction-recovery baselines."
        )
    )
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--realization", required=True, type=int)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(METHODS),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run_external_recovery_realization(
        scenario=args.scenario,
        realization_number=args.realization,
        methods=tuple(args.methods),
        force=bool(args.force),
    )


if __name__ == "__main__":
    main()
