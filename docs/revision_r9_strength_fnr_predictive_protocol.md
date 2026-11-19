# R9 — Interaction-strength FNR and predictive threshold sensitivity

**Status:** Post-submission, reviewer-requested exploratory analyses. Nothing in
this revision replaces or retunes the locked primary synthetic/Real-X benchmark.

## R9-A: Real-X interaction-strength false-negative rates

Source: `results/realx/benchmark/master_results.csv` (180 rows: nine tasks,
five realizations, each of weak/moderate/strong/null). The analysis is
**read-only** with respect to the original data. For every positive-interaction
run, compute `selection_FNR = 1 - selection_recall` and
`ISR_FNR = 1 - ISR_recall`. The source supplies three known injected true
interaction pairs per run in the current benchmark. TP and FN are reconstructed
from the number of true pairs and recall, with integer-accounting checks.

Null runs (zero injected interactions) have **undefined** FNR, not zero. All
45 null runs are excluded from FNR estimation. Report TP retained before and
after ISR, the increment in missed pairs, and the per-task/per-strength FNR.

**Uncertainty:** Prespecified percentile 95% confidence intervals from 10,000
paired bootstrap resamples of the **nine OpenML tasks**, sampling all five
realizations within each resampled task. The same task draws are used across
all three strength levels. The resampling unit is *task*, not an individual
run or an injected pair. Intervals are exploratory and do not constitute a new
confirmatory hypothesis test. These pairs are injected semisynthetic truth,
not independently established real clinical interaction mechanisms.

R9-A produces `strength_summary.csv`, `task_strength_summary.csv`,
`positive_run_audit.csv`, and `manifest.json` under
`results/revision/strength_fnr_r9/v1/`.

## R9-B: Final predictive AUROC under interaction threshold variation

The original R4 analysis is a **structural recovery** sensitivity experiment;
its summaries do not contain pair identities or final AUROC by configuration.
This separate follow-up evaluates whether changing the two thresholds changes
predictive AUROC. It does **not** assert that the 80-realization R4 experiment
already supplied this outcome.

The exploratory subset is fixed **before inspecting R9-B predictive outcomes**:
first five locked realization indices (1–5) in each of S1–S4 (20 realizations).
This is an index-based convenience subset, not a random or representative
sample of all 80. It must not be presented as confirmatory. Retain the original
four scenarios, generator seeds, locked outer train/test splits, prediction
preprocessing, final train/validation split and final model training seed.

- Selection threshold: 0.40, 0.60, 0.80.
- ISR acceptance threshold: 0.40, 0.60, 0.80.
- Discovery repeats: locked R = 5 (same first five seeds used by R4).
- Candidate budget: locked K from the synthetic protocol.
- ISR reference support: 512.
- Nine parameter configurations per realization; all other parameters fixed.

Each realization trains **one** five-run discovery stream; three selection
views are constructed from it. ISR is evaluated at a permissive 0.40 for each
selection view, then the saved ISR scores are thresholded at 0.40/0.60/0.80.
**Before any final predictive training**, all nine resulting selection and ISR
recovery summaries must match the archived R4 summary (TP/FP/FN, precision,
recall, F1, FP reduction, candidate budget and thresholds). A mismatch fails
closed; there is no automatic bypass.

Final models call `_prepare_final_prediction_data` and
`_train_interaction_model` from the primary synthetic benchmark. The original
locked Main NAM AUROC is read from its archived master table; it is not
retrained. The first final model evaluated is the submitted reference
(0.60/0.60), whose AUROC is checked against the original benchmark with an
absolute tolerance of 1e-6. Nonreference models are never trained if this
audit fails. Identical normalized interaction sets use the same final-fit
cache key; model results are written incrementally for crash recovery. All
nine configurations are reported; subsets with zero retained interactions
remain in the analysis rather than being excluded.

This is a *predictive* threshold follow-up, not a threshold optimization
exercise. No threshold is selected on outer-test performance. Inference,
if subsequently included, must respect realization/scenario clustering and
multiplicity. Do not generalize exploratory predictive effects beyond this
fixed subset.

R9-B outputs live under
`results/revision/predictive_threshold_r9/v1/<scenario>/realization_<nn>/`.
They include config-to-pair sets, a distinct-pair-set final-model fit cache,
AUROC and other final metrics, hash manifests and parity status. Outputs in
`results/` are gitignored; preserve them separately from the source-code Git
commit. The source cache is per run and does not persist discovery model
weights, so interrupted discovery needs to be repeated; already completed
final fits are reused when hashes match.

## Reproducibility sequence

1. `python -m pytest -q`
2. `python scripts/analyze_revision_strength_fnr.py` (CPU-only, no training)
3. `python scripts/run_revision_predictive_threshold.py --pilot` (S1 R1 pilot; trains five discovery models and the unique final interaction models)
4. Audit pilot against the 9 archived R4 results and original final AUROC.
5. Only after a successful pilot: `python scripts/run_revision_predictive_threshold.py --all-selected`

Do not use `--force` without documenting the reason. Preserve the baseline
benchmark and R4 archives unchanged. If parity fails, investigate rather than
loosening the tolerance or relabeling differences as equivalent.
