from __future__ import annotations

import argparse
from pathlib import Path

from agnam.revision.isr_novelty_runner import run_synthetic_isr_novelty


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run reviewer-requested ISR novelty ablation for one locked synthetic realization."
    )
    parser.add_argument("--scenario", required=True, choices=("S1", "S2", "S3", "S4"))
    parser.add_argument("--realization", required=True, type=int)
    parser.add_argument(
        "--results-root",
        default="results/revision/isr_novelty/synthetic",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if not 1 <= args.realization <= 20:
        parser.error("--realization must be in 1..20")

    manifest = (
        Path(args.results_root)
        / args.scenario
        / f"realization_{args.realization:02d}_manifest.json"
    )
    if manifest.exists() and not args.force:
        print(f"SKIP {args.scenario} realization {args.realization}: manifest already exists")
        return

    outputs = run_synthetic_isr_novelty(
        scenario=args.scenario,
        realization_number=args.realization,
        results_root=args.results_root,
        verbose=not args.quiet,
    )
    print("\nSaved revision outputs:")
    for path in outputs.values():
        print(path)


if __name__ == "__main__":
    main()
