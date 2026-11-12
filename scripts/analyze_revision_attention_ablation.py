from __future__ import annotations

import argparse
import json
from pathlib import Path

from agnam.revision.attention_ablation_analysis import (
    analyze_synthetic_attention_ablation,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate reviewer-requested AG-NAM attention-score ablations."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("results") / "revision" / "attention_ablation" / "synthetic",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results") / "revision" / "attention_ablation" / "analysis",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=("S1", "S2", "S3", "S4"),
        default=None,
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--bootstrap-seed", type=int, default=26090806)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    analysis = analyze_synthetic_attention_ablation(
        args.input_root,
        scenarios=args.scenarios,
        allow_incomplete=args.allow_incomplete,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_resamples=args.bootstrap_resamples,
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    analysis.realization_summary.to_csv(
        args.output_root / "realization_summary_long.csv", index=False
    )
    analysis.mode_summary.to_csv(
        args.output_root / "mode_summary.csv", index=False
    )
    analysis.paired_comparisons.to_csv(
        args.output_root / "paired_comparisons.csv", index=False
    )

    manifest = {
        "analysis_status": "reviewer_requested_post_submission",
        "analysis_name": "attention_score_ablation_aggregate",
        "scenarios": args.scenarios,
        "allow_incomplete": bool(args.allow_incomplete),
        "bootstrap_seed": int(args.bootstrap_seed),
        "bootstrap_resamples": int(args.bootstrap_resamples),
        "formal_hypothesis_tests": False,
        "locked_primary_results_modified": False,
        "outputs": {
            "realization_summary": str(args.output_root / "realization_summary_long.csv"),
            "mode_summary": str(args.output_root / "mode_summary.csv"),
            "paired_comparisons": str(args.output_root / "paired_comparisons.csv"),
        },
    }
    (args.output_root / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("Saved aggregate revision outputs:")
    for name in (
        "realization_summary_long.csv",
        "mode_summary.csv",
        "paired_comparisons.csv",
        "analysis_manifest.json",
    ):
        print(args.output_root / name)


if __name__ == "__main__":
    main()
