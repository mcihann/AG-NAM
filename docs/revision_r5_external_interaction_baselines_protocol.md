# Reviewer-Requested External Interaction-Recovery Baselines

Status: post-submission, reviewer-requested analysis.

This protocol is frozen before inspecting any external-baseline recovery result.
It does not modify the locked AG-NAM primary benchmark or its original
confirmatory inference.

## Reviewer target

The interaction-recovery comparison will address the requested detector family:

1. Friedman H-statistic,
2. Neural Interaction Transparency (NIT),
3. Additive Groves interaction detection,
4. iterative Random Forests (iRF),
5. at least one sparse neural interaction model: SIAN.

A Neural Interaction Detection (NID) baseline will also be included as a
secondary neural detector because it provides a direct pair-ranking comparator
and has public reference code.

No requested method may be silently replaced by another method. If a requested
implementation cannot be executed faithfully in the main environment, it will
be run through an isolated environment or external executable where possible.
Any remaining infeasibility must be documented explicitly.

## Benchmark layers

### Synthetic

All four locked scenarios S1-S4 and all 20 realizations per scenario are used.

Each external detector receives the same outer-development partition used for
AG-NAM discovery. The untouched outer test partition is not used to fit or tune
an interaction detector.

Ground-truth interaction labels are used only after model fitting for evaluation.

### Real-X

After synthetic validation, the same detector interface will be applied to the
locked Real-X benchmark under its existing splits. The moderate-strength
condition is the core comparison and null remains a negative-control condition.

## Common output representation

The target comparison unit is an unordered pair of original input features.

Every method must produce one of:

- a full pair ranking with a numeric score, or
- a scored subset of pairs; unreported pairs receive the method-defined floor
  score (normally zero) and are deterministically tied by feature name.

Higher-order interaction sets, where produced, are converted to pairwise scores
only by a prespecified projection rule:

- for a detected set S with |S| >= 2 and method score q,
- every unordered pair contained in S receives candidate score q,
- when multiple detected sets contain the same pair, the pair score is the
  maximum method score across those sets.

The original higher-order set is retained in an audit file so that projection
does not erase provenance.

## Evaluation metrics

### Threshold-free ranking metrics

Primary descriptive detector comparison:

- interaction AUPRC over the complete pair universe.

Secondary:

- NDCG over the complete pair universe,
- AUROC where mathematically defined.

### Common-budget recovery

To avoid rewarding a detector merely for returning more or fewer pairs, report:

- precision / recall / F1 at the locked AG-NAM candidate K,
- precision / recall / F1 at the number of true interactions (oracle-count
  diagnostic; explicitly labeled non-deployable),
- precision / recall / F1 at the count of AG-NAM final ISR-retained pairs
  (count-matched comparison).

### Native sparse output

When a method has a documented native selection/stability criterion, also report
its native retained-set precision / recall / F1. Native thresholds are not tuned
against synthetic ground truth.

## Reproducibility

All baseline randomness is deterministic and derived from the locked realization
seed. No baseline hyperparameter is selected after observing recovery metrics.

Runtime and memory are recorded separately from recovery quality.

## Method-specific plan

### Friedman H-statistic

- Fit a tree-based probabilistic classifier on the outer-development data.
- Compute pairwise Friedman H scores using partial-dependence decomposition.
- Rank all unordered feature pairs by H score.
- The estimator and numerical grid are fixed before recovery inspection.

### NID

- Use a deterministic MLP and the published first-layer weight-decoding
  interaction score.
- Pairwise scores are extracted for every original feature pair.
- Included as a secondary neural detector, not as a substitute for NIT.

### NIT

- NIT is treated as a distinct requested method.
- A faithful paper-based reproduction will be implemented in revision code if
  no compatible official implementation is available.
- Its maximum interaction order will be fixed to pairwise interactions for this
  benchmark because the synthetic and Real-X ground truths are pairwise.
- It will not be relabeled as NID.

### iRF

- Prefer the authors' implementation.
- Because the public Python implementation targets an older Python/scikit-learn
  stack, it must not be installed into the main `agnam` environment if doing so
  would downgrade core dependencies.
- Use an isolated environment or the maintained R package wrapper.
- Pair score is the reported stability score; higher-order sets are projected by
  the common rule above.

### Additive Groves

- Use TreeExtra interaction detection through the official executable/tooling.
- Run in an isolated working directory.
- Parse the pairwise interaction test output into the common ranking schema.
- The executable version and command line are recorded in the manifest.

### SIAN

- Prefer the current `sian` package / author implementation.
- Restrict reported recovery to pairwise detected interactions.
- Keep SIAN-native higher-order output in an audit artifact if produced.

## Statistical boundary

The external detector benchmark is reviewer-requested and post-submission.

- No new confirmatory family is introduced.
- Scenario-level means and paired bootstrap 95% intervals are descriptive.
- Original primary p-values and multiplicity corrections are unchanged.

## Failure policy

A method is not marked "completed" merely because an import succeeds.

For every baseline the manifest must record:

- implementation source/version,
- runtime environment,
- exact command or adapter,
- seed,
- hyperparameters,
- number of scored pairs,
- whether a full ranking was available,
- any projection from higher-order interactions,
- warnings or compatibility deviations.

A baseline is excluded only after a reproducible technical attempt and a
documented reason. Exclusion is never based on poor recovery performance.
