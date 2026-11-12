from __future__ import annotations

import argparse
import json
from pathlib import Path

from agnam.benchmarking.synthetic_protocol import (
    SyntheticBenchmarkProtocol,
    build_synthetic_schedule,
)
from agnam.benchmarking.synthetic_runner import SyntheticModelConfig
from agnam.revision.attention_ablation_runner import (
    run_synthetic_attention_ablation_realization,
)
from agnam.utils.reproducibility import get_device


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reviewer-requested AG-NAM attention-score ablation. "
            "Runs raw attention, gradient-only, and Grad×Attention "
            "on matched fitted proposers."
        )
    )

    parser.add_argument(
        "--scenario",
        choices=("S1", "S2", "S3", "S4"),
        help="Synthetic scenario for a single realization run.",
    )
    parser.add_argument(
        "--realization",
        type=int,
        help="Human-facing realization number (1-20).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all 80 synthetic realizations.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite already completed revision outputs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results") / "revision" / "attention_ablation" / "synthetic",
    )

    args = parser.parse_args()

    if args.all:
        if args.scenario is not None or args.realization is not None:
            parser.error("--all cannot be combined with --scenario/--realization.")
    else:
        if args.scenario is None or args.realization is None:
            parser.error("Specify --scenario and --realization, or use --all.")
        if not 1 <= args.realization <= 20:
            parser.error("--realization must be between 1 and 20.")

    return args


def _paths(output_root: Path, scenario: str, realization_number: int):
    directory = output_root / scenario
    stem = f"realization_{realization_number:02d}"
    return {
        "directory": directory,
        "runs": directory / f"{stem}_run_metrics.csv",
        "summary": directory / f"{stem}_summary.csv",
        "rankings": directory / f"{stem}_rankings.csv",
        "manifest": directory / f"{stem}_manifest.json",
    }


def _run_one(spec, *, protocol, model_config, device, args) -> None:
    paths = _paths(
        args.output_root,
        spec.scenario,
        spec.realization_number,
    )

    if paths["manifest"].exists() and not args.force:
        print(
            f"SKIP {spec.scenario} realization {spec.realization_number}: "
            "manifest already exists (use --force to rerun)."
        )
        return

    result = run_synthetic_attention_ablation_realization(
        spec=spec,
        protocol=protocol,
        model_config=model_config,
        device=device,
        verbose=True,
    )

    paths["directory"].mkdir(parents=True, exist_ok=True)

    result.run_metrics.to_csv(paths["runs"], index=False)
    result.summary.to_csv(paths["summary"], index=False)
    result.rankings.to_csv(paths["rankings"], index=False)

    manifest = {
        "analysis_status": "reviewer_requested_post_submission",
        "analysis_name": "attention_score_ablation",
        "scenario": spec.scenario,
        "realization_number": spec.realization_number,
        "dataset_seed": spec.dataset_seed,
        "outer_split_seed": spec.outer_split_seed,
        "discovery_base_seed": spec.discovery_base_seed,
        "discovery_seeds": list(spec.discovery_seeds),
        "n_discovery_runs": protocol.n_discovery_runs,
        "selection_threshold": protocol.selection_threshold,
        "score_modes": {
            "raw_attention": "abs(A_jk)",
            "gradient_only": "abs(d r_hat / d A_jk)",
            "grad_x_attention": "abs(A_jk * d r_hat / d A_jk)",
        },
        "locked_primary_results_modified": False,
        "outputs": {
            "run_metrics": str(paths["runs"]),
            "summary": str(paths["summary"]),
            "rankings": str(paths["rankings"]),
        },
    }

    paths["manifest"].write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("\nSaved revision outputs:")
    print(paths["runs"])
    print(paths["summary"])
    print(paths["rankings"])
    print(paths["manifest"])


def main() -> None:
    args = _parse_args()
    protocol = SyntheticBenchmarkProtocol()
    model_config = SyntheticModelConfig()
    schedule = build_synthetic_schedule(protocol)
    device = get_device()

    if args.all:
        selected = schedule
    else:
        selected = tuple(
            spec
            for spec in schedule
            if spec.scenario == args.scenario
            and spec.realization_number == args.realization
        )

    if len(selected) == 0:
        raise RuntimeError("No synthetic realization matched the requested arguments.")

    for spec in selected:
        _run_one(
            spec,
            protocol=protocol,
            model_config=model_config,
            device=device,
            args=args,
        )


if __name__ == "__main__":
    main()
