from __future__ import annotations

import argparse

from agnam.benchmarking.synthetic_protocol import build_synthetic_schedule
from agnam.revision.predictive_threshold_runner import run_predictive_threshold_realization


def main() -> None:
    parser = argparse.ArgumentParser(description="R9-B exploratory final AUROC vs 3x3 R4 thresholds")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--pilot", action="store_true", help="S1 realization 1 only")
    selection.add_argument("--all-selected", action="store_true", help="First 5 realizations of each S1-S4")
    parser.add_argument("--force", action="store_true", help="Explicitly recompute final models; use only for a documented reason")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    args = parser.parse_args()
    specs = tuple(s for s in build_synthetic_schedule() if s.realization_number <= 5)
    if args.pilot:
        specs = (next(s for s in specs if s.scenario == "S1" and s.realization_number == 1),)
    device = None
    if args.device:
        import torch
        device = torch.device(args.device)
    for spec in specs:
        run_predictive_threshold_realization(spec, device=device, force=args.force)


if __name__ == "__main__":
    main()
