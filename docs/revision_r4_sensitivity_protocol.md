# Reviewer-Requested Sensitivity Analysis Protocol

Status: post-submission, reviewer-requested analysis.

This protocol is frozen before inspecting any sensitivity-analysis outcomes.
It does not modify or replace the locked primary AG-NAM benchmark.

## Purpose

Evaluate whether the principal interaction-discovery conclusions are materially
sensitive to four prespecified choices criticized by the reviewers:

1. selection reproducibility threshold,
2. ISR agreement threshold,
3. number of discovery repeats,
4. candidate budget K,
5. ISR reference-support size.

The submitted configuration remains the reference configuration:

- selection threshold = 0.60,
- ISR threshold = 0.60,
- discovery repeats = 5,
- candidate K = locked AG-NAM rule,
- reference support = 512.

## Synthetic benchmark

All four synthetic scenarios S1-S4 and all 20 locked realizations per scenario
are used. Dataset seeds, outer splits, residual cross-fitting, model
architectures, optimizers, training budgets, and reference seed remain
unchanged.

## Sensitivity design

The design is deliberately not a full Cartesian product. Thresholds are studied
jointly because they form a sequential two-stage filter. The remaining factors
are varied one at a time around the submitted reference configuration.

### A. Threshold grid

Selection threshold:

- 0.40
- 0.60
- 0.80

ISR threshold:

- 0.40
- 0.60
- 0.80

This yields a 3 x 3 grid (9 configurations), with:

- R = 5,
- K = locked,
- reference support = 512.

### B. Discovery-repeat sensitivity

- R = 3
- R = 5 (submitted reference)
- R = 10

All use:

- selection threshold = 0.60,
- ISR threshold = 0.60,
- K = locked,
- reference support = 512.

The same deterministic seed stream is used. R=3 and R=5 use the first 3 and
first 5 runs, respectively, from the 10-run reserved seed block. The original
synthetic schedule already reserves 10 discovery seeds per realization.

### C. Candidate-budget sensitivity

Let K0 be the submitted locked candidate count and P=p(p-1)/2.

- half: K = max(1, ceil(K0 / 2))
- locked: K = K0
- double: K = min(P, 2 K0)

All use:

- R = 5,
- selection threshold = 0.60,
- ISR threshold = 0.60,
- reference support = 512.

The full interaction rankings are unchanged; only the Top-K cut is varied.

### D. Reference-support sensitivity

- 128
- 256
- 512 (submitted reference)

All use:

- R = 5,
- K = locked,
- selection threshold = 0.60,
- ISR threshold = 0.60,
- reference seed = 2026.

## Computation reuse

To isolate sensitivity parameters from unnecessary retraining:

- up to 10 discovery runs are trained once per realization;
- R=3/5/10 are deterministic prefixes of that run sequence;
- K variants reuse the same full interaction rankings;
- selection thresholds are recomputed from the same rankings;
- ISR scores are thresholded after surface agreement is computed;
- pairwise ISR evaluations are cached within a realization whenever the
  discovery view and reference support are identical.

No ground-truth information is used to choose thresholds, K, R, or support.

## Outcomes

For every configuration and realization:

- retained pair count,
- selection-stage TP / FP / FN,
- selection precision / recall / F1,
- ISR-stage TP / FP / FN,
- ISR precision / recall / F1,
- false-positive reduction relative to selection,
- candidate K,
- number of discovery runs,
- selection threshold,
- ISR threshold,
- reference-support size.

Primary interpretation is descriptive robustness, not a new confirmatory
inferential family. Scenario-level means, standard deviations, medians,
paired differences from the submitted reference configuration, win/tie/loss
counts, and paired bootstrap 95% confidence intervals will be reported.

## Inferential boundary

These analyses are reviewer-requested and post-submission. They do not alter
the original locked primary hypothesis tests, family-wise error corrections,
or frozen benchmark results.

Formal hypothesis tests are not introduced for this sensitivity family.

## OpenML follow-up

After the synthetic sensitivity implementation is validated and frozen, the
same principle will be applied to a prespecified OpenML subset. The subset will
be fixed before inspecting OpenML sensitivity outcomes.
