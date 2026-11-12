from __future__ import annotations

import argparse
from pathlib import Path

from agnam.revision.isr_novelty_analysis import (
    RevisionISRAnalysisConfig,
    run_analysis,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate reviewer-requested ISR novelty ablation results."
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        required=True,
        help="Scenario names, e.g. S1 or S1 S2 S3 S4.",
    )
    parser.add_argument(
        "--input-root",
        default="results/revision/isr_novelty/synthetic",
    )
    parser.add_argument(
        "--output-root",
        default="results/revision/isr_novelty/analysis",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
    )
    args = parser.parse_args()

    config = RevisionISRAnalysisConfig(
        input_root=Path(args.input_root),
        output_root=Path(args.output_root),
        scenarios=tuple(s.upper() for s in args.scenarios),
        allow_incomplete=bool(args.allow_incomplete),
    )
    outputs = run_analysis(config)

    print("Saved aggregate ISR novelty outputs:")
    for path in outputs.values():
        print(path)


if __name__ == "__main__":
    main()
