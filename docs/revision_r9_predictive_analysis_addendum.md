# R9-B predictive threshold sensitivity: 20-realization analysis

This is an **addendum** to `docs/revision_r9_strength_fnr_predictive_protocol.md`,
not a change to the previously fixed study design. The first five realizations of
S1–S4 are an *exploratory convenience subset*; outcomes must not be used to
retune the submitted reference thresholds (selection 0.60, ISR 0.60).

Run after R9-B completes:

```bat
python scripts\analyze_revision_predictive_threshold.py
```

The analysis performs **no model training**. It requires all 20 realization
outputs and verifies 9/9 R4 recovery parity, original reference-model AUROC
parity, per-file SHA-256 values, pair-list/cache consistency, model evaluation
consistency, completeness of the 3×3 grid and preserved source-file SHA-256
values (original benchmark and R4). It fails closed on disagreement.

Outputs under `results/revision/predictive_threshold_r9/v1/analysis/`:

- `realization_summary_long.csv` — 180 config-realization evaluations, with
  paired AUROC deltas, pairs retained and threshold identifiers.
- `configuration_summary.csv` — nine configurations summarized over 20 realizations.
- `scenario_configuration_summary.csv` — four scenarios × nine configurations.
- `paired_vs_reference.csv` — within-realization AUROC changes relative to
  submitted 0.60/0.60, including wins/ties/losses.
- `threshold_factor_summary.csv` — descriptive mean AUROC and mean pairs by
  selection/ISR threshold, averaged across the other factor.
- `realization_audit.csv` — 20 per-realization checks, unique final model counts.
- `analysis_manifest.json` — reproducibility metadata and output hashes.

**Uncertainty and interpretation.** Paired bootstrap percentile 95% intervals
are produced by resampling the *five realizations independently within each
of the four scenarios*, and preserving pairing across all nine configurations.
The scenario weights are equal. Ten thousand resamples with fixed seed 2026
are used. Intervals are **exploratory, not multiplicity-adjusted**; they do
not support confirmatory inference. Report both potentially favorable and
unfavorable threshold effects. These 20 runs must not be treated as a random
sample of all 80 locked realizations. No thresholds are chosen on the test
AUROC. Caching of identical interaction sets within a realization reduces
computation without increasing experimental replication.

The `--offline-audit` option skips rehashing external source files only when
analyzing a detached archive without its R4 and original benchmark inputs.
It still verifies the output-file hashes, model cache, pair identities and
original AUROC parity reported within each R9-B result manifest. On the actual
AG-NAM working tree **do not use** `--offline-audit`.
