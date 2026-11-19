from __future__ import annotations

import argparse

from agnam.revision.strength_fnr_analysis import run_strength_fnr_analysis


def main() -> None:
    p = argparse.ArgumentParser(description="R9-A task-cluster-bootstrap Real-X strength FNR analysis")
    p.add_argument("--source", default="results/realx/benchmark/master_results.csv")
    p.add_argument("--output-root", default="results/revision/strength_fnr_r9/v1")
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20261118)
    args = p.parse_args()
    run_strength_fnr_analysis(args.source, args.output_root, n_boot=args.bootstrap, seed=args.seed)


if __name__ == "__main__":
    main()
