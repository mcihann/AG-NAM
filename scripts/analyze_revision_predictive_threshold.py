from __future__ import annotations

import argparse

from agnam.revision.predictive_threshold_analysis import analyze_predictive_threshold


def main() -> None:
    parser = argparse.ArgumentParser(description="R9-B strict audit and descriptive threshold AUROC aggregation (no training)")
    parser.add_argument("--input-root", default="results/revision/predictive_threshold_r9/v1")
    parser.add_argument("--r4-root", default="results/revision/sensitivity/synthetic")
    parser.add_argument("--original", default="results/synthetic/master_results.csv")
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--offline-audit", action="store_true", help="For detached archives only: do not rehash original benchmark/R4 source files")
    args = parser.parse_args()
    paths = analyze_predictive_threshold(
        input_root=args.input_root,
        r4_root=args.r4_root,
        original_path=args.original,
        bootstrap_repeats=args.bootstrap_repeats,
        verify_sources=not args.offline_audit,
    )
    import pandas as pd
    config = pd.read_csv(paths["configuration_summary.csv"])
    paired = pd.read_csv(paths["paired_vs_reference.csv"])
    factor = pd.read_csv(paths["threshold_factor_summary.csv"])
    audit = pd.read_csv(paths["realization_audit.csv"])
    print("R9-B AUDIT PASS:", len(audit), "/ 20 realizations;", 9 * len(audit), "/ 180 threshold evaluations")
    print("R9-B FINAL FITS:", int(audit.n_distinct_pair_sets.sum()), "(unique within each realization)")
    print("R9-B ZERO PAIR CONFIGURATIONS:", int(audit.n_zero_pair_configs.sum()))
    print("\nSELECTION/ISR FACTORS (descriptive):")
    print(factor.round(6).to_string(index=False))
    print("\nREFERENCE-CONTRASTS (exploratory, 95% bootstrap CIs):")
    print(paired[["config_id", "mean_paired_delta", "delta_ci95_low", "delta_ci95_high", "wins", "ties", "losses"]].round(6).to_string(index=False))
    print("\nSAVED:", paths["manifest"].parent)


if __name__ == "__main__":
    main()
