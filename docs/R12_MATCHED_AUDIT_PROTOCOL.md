# AG-NAM R12 — Matched AG-NAM / pairwise-native SIAN check

This script is **read-only** with respect to R1–R12 benchmark source files. It does not fit or retune any models. It reads the canonical `results/synthetic/<scenario>/realization_XX.csv` and matching `realization_XX_pairs.csv` together with the native-SIAN R12 saved per-realization files.

The check enforces the same scenario, realization, dataset generation seed, outer test-split seed, nested train/validation-split seed, reference/discovery seed, final-model seed and ground-truth feature-pair labels. Prediction metrics for SIAN are re-evaluated from its test labels/probabilities. AG-NAM's archived test metrics are checked for finiteness; its retained interaction set and F1 are recomputed from the frozen pair-membership audits. All read source file SHA-256 values are stored in the output ZIP. No raw AG-NAM prediction vectors were included in the original Phase 4 summary and hence the exact test-label sequences cannot be independently equated from these summaries alone; shared deterministic protocol/seed identities support the comparison.

R12 outcome comparisons are **exploratory after review**. The native SIAN implementation is restricted to pairwise terms (MAX_K=2) and uses R12 hyperparameter settings. Do not compare the SIAN selected-set F1 with stand-alone ranker AUPRC. No final AG-NAM runtime inference is allowed from the original canonical multi-model benchmark timer.

Use: `python scripts/collect_r12_paired_agnam_sian.py --scenario S1` after the 20 S1 native-SIAN runs are complete. Other scenarios work after their 20 R12 native-SIAN runs are complete. The output ZIP includes paired tables and source evidence.
