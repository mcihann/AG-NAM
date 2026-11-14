# R5a sanity-run engineering note

During the S1 realization-1 sanity run, recovery outputs for Friedman H,
NID, and SIAN-FID passed the pair-universe and score-integrity checks.

One runtime-accounting implementation issue was identified before any
multi-realization external-baseline run:

- the Friedman-H implementation measured estimator fitting and partial-
  dependence scoring under one elapsed timer;
- the resulting combined elapsed time was placed in `fit_seconds`, leaving
  `scoring_seconds` equal to zero.

This affected runtime labeling only. It did not alter:

- fitted model parameters,
- pair scores,
- pair ranks,
- AUPRC,
- AUROC,
- NDCG,
- precision / recall / F1.

The implementation was corrected immediately after the sanity run so that
GradientBoosting fitting time and partial-dependence interaction-scoring time
are recorded separately. No detector hyperparameter, seed, split, score
definition, or recovery result was changed.
