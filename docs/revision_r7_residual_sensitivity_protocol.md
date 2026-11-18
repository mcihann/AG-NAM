# R7 — Probability versus signed Bernoulli deviance residuals (post-submission)

## Research question

Does replacing the MainEffectNAM Bernoulli pseudo-residual `y - p` with the
signed Bernoulli deviance residual alter interaction *candidate discovery*?
This **does not** assess the effect on ISR, final prediction or causal
interaction identification. Original synthetic benchmark files are never
modified.

## Data, splits, and paired controls

- Four original synthetic scenarios S1–S4, 20 realizations each.
- The original locked outer 80% development partition is used. Outer-test
  data are not passed to the discovery models.
- Five original discovery seeds per realization, with 60% training, 20%
  validation, and 20% untouched scoring support *within outer development*.
- All 400 MainEffectNAM cached residual runs are first verified using
  `scripts/audit_revision_r7_residual_cache.py`. They derive from 5-fold
  crossfit NAM on discovery training and a separate train-only NAM fit for
  discovery validation. The same cached NAM outputs are used by both arms.
- The proposer architecture, optimizer configuration, features, preprocessing,
  split, K, selection cutoff (0.60) and seed are identical for both arms.
- Both arms are fitted anew to residual targets; scores are evaluated on the
  same held-out interaction-scoring support. No ground-truth interaction is
  used in training, proposal scoring, or stable selection.
- Reference = **probability** target `r = y - p`. Alternative = **raw, not
  standardized** signed Bernoulli deviance:

      r_dev = sqrt(-2 log(p)) if y = 1 else -sqrt(-2 log(1-p))

  with p reconstructed from `y - r` in float64, validated within tolerance
  `1e-6`, and clipped only for numerical safety to `[1e-6,1-1e-6]`.
  Training targets are cast to float32 as in the original trainer.
  The observed full 400-cache audit required **zero** clipping operations.
- These targets have different natural scale; the study isolates the residual
  *definition* as used in the unchanged proposer loss, not every possible
  downstream effect of residual rescaling. Record target standard deviations
  for transparent interpretation.

## Outcomes and interpretation

Per run: interaction-ranking AUPRC, precision/recall/F1@K, fitted epoch,
proposer fit time. Per realization: mean run AUPRC, mean pairwise stability
Jaccard, stable-selection count, true/false positive, precision/recall/F1,
and cross-arm stable-set Jaccard. Selection uses the original 3/5 rule.

The primary paired contrasts are deviance minus probability for (1) mean
interaction AUPRC and (2) stable-selection F1. Analyze 20 realization-level
paired differences per scenario with two-sided Wilcoxon tests and Holm
correction over 8 tests; report paired bootstrap 95% CIs with 10,000 draws.
These are reviewer-requested exploratory analyses, not prespecified primary
confirmatory tests. Do not infer predictive AUROC improvement from discovery
recovery results.

## Running on Windows CMD from repository root

```bat
python scripts\audit_revision_r7_residual_cache.py
python -m pytest -q
python scripts\run_revision_residual_sensitivity.py --scenario S1 --realization 1
python scripts\run_revision_residual_sensitivity.py --all
python scripts\analyze_revision_residual_sensitivity.py
```

Output namespace:
`results/revision/residual_sensitivity/core_synthetic_v1/`.
The original cache and `results/synthetic/` outputs are read-only.
Each run's full ranking plus JSON provenance is saved and validated upon
resume; if any cache or ranking checksum mismatches, halt rather than
silently refitting. Do not use `--force` without documenting the reason.
Only the completed analysis output can justify a claim about efficacy.
