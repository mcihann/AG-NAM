from __future__ import annotations

import argparse
from pathlib import Path

from agnam.revision.sensitivity_analysis import (
    SensitivityAnalysisConfig,
    run_analysis,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate reviewer-requested parameter sensitivity results."
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--input-root",
        default="results/revision/sensitivity/synthetic",
    )
    parser.add_argument(
        "--output-root",
        default="results/revision/sensitivity/analysis",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
    )
    args = parser.parse_args()

    config = SensitivityAnalysisConfig(
        input_root=Path(args.input_root),
        output_root=Path(args.output_root),
        scenarios=tuple(s.upper() for s in args.scenarios),
        allow_incomplete=bool(args.allow_incomplete),
    )

    outputs = run_analysis(config)

    print("Saved aggregate sensitivity outputs:")
    for path in outputs.values():
        print(path)


if __name__ == "__main__":
    main()
