# External proposal + common AG-NAM ISR (completed reviewer-only study)

**Status (2026-11-18):** complete; 80 synthetic realizations, 240 hybrid
method-realization evaluations, 480 audited source CSVs and 391 passing tests.
This post-submission study is separate from the locked original AG-NAM
synthetic benchmark and the standalone single-run Friedman-H, NID and
SIAN-FID interaction-recovery baselines. It is an exploratory reviewer
analysis, not an independently preregistered confirmatory experiment.

## 1. Question and scope

Assess whether the common AG-NAM ISR filter, based on interaction-surface
stability, can refine repeated external proposal sets, and whether its utility
persists after controlling the number of retained pairs. Three complementary
outcomes are distinguished: raw proposal ranking AP, selection-stable
interaction recovery and ISR-filtered recovery. No claim of superior original
AG-NAM proposal ranking follows from the hybrid experiment.

- S1-S4, realizations 01-20 per scenario, on the deterministic original
  synthetic schedule. Thus 80 datasets and 240 method-realization records.
- Each method has five discovery runs (`discovery_base_seed + 0..4`):
  1,200 method-run proposal rankings in total; not 1,200 independent datasets
  or necessarily 1,200 separate model fits.
- Locked outer-development split (80% dataset). Outer test is never passed
  to the proposer, residual generator or ISR.
- Within each run, training/validation/scoring = 60%/20%/20% of outer
  development, as in `run_single_interaction_discovery`.
- Friedman-H: model fit on run training; PD-based scoring on run scoring.
  Validation is unused by this baseline.
- NID and SIAN-FID: common SIAN MLP trained on run training and early-stopped
  on run validation. Scoring uses the fitted model; SIAN-FID uses run scoring
  support. When requested together with both outputs missing, one MLP fit
  can be shared.
- Real MainEffectNAM cross-fitted training residuals and train-fitted
  validation residuals are generated once per run and reused across external
  proposal methods. No dummy proposer model objects are made.
- Full pair rankings are validated. The original pairwise hyperparameters,
  reference size 512, selection threshold 0.60 and ISR threshold 0.60 are
  retained. Ground-truth pairs are for evaluation only, not fitting,
  proposal selection, ISR decisions or threshold tuning.
- Pair selection is restricted to the stable proposal set; ISR only removes
  pairs and cannot retrieve omitted true interactions.
- `fp_reduction` is undefined (NaN) for selection sets with zero FP; do not
  fill those values with zero. Report the valid count for per-realization
  means separately from pooled reduction.

| Scenario | Realizations | Top-K per run | True pairs per realization |
| --- | ---: | ---: | ---: |
| S1 | 20 | 19 | 3 |
| S2 | 20 | 20 | 4 |
| S3 | 20 | 20 | 2 |
| S4 | 20 | 20 | 4 |

## 2. Reproduction (Windows CMD)

Run from the AG-NAM repository root in the `agnam` environment.
The following commands document the complete schedule; **do not rerun
completed experiments merely to reproduce this document**. The runner
validates cached artifacts on restart. Inspect any partial outputs or errors
before continuing. Use `--force` only with a documented reason.

```bat
python -m pytest -q tests\test_external_isr_runner.py tests\test_external_isr_adapter.py
python -m pytest -q
```

For each scenario, complete proposals before ISR. For example, the following
commands reproduce the complete S1 schedule; replace `S1` by `S2`, `S3`, or
`S4` for the other scenarios. These are interactive CMD commands, not batch
file syntax.

```bat
for /L %R in (1,1,20) do python scripts\run_revision_external_isr.py --scenario S1 --realization %R --methods friedman_h_gbdt nid_shared_mlp sian_fid_shared_mlp --stage proposals
for /L %R in (1,1,20) do python scripts\run_revision_external_isr.py --scenario S1 --realization %R --methods friedman_h_gbdt nid_shared_mlp sian_fid_shared_mlp --stage isr
```

Each method/run saves a full ranking and manifest. Each of the five runs
also saves a residual cache with split hashes. All outputs belong under:

```text
results/revision/external_isr_hybrid/core_synthetic/<scenario>/realization_<NN>/
```

Do not write to the locked `results/synthetic/` or
`results/revision/external_interaction_baselines/` directories.

The final read-only analysis of the source results is:

```bat
python scripts\finalize_external_isr_analysis.py
```

It writes `results/revision/external_isr_hybrid/final_analysis_v1/` and
refuses to overwrite an existing analysis directory. To regenerate, use a
separate working copy or explicitly archive the existing analysis first.
The analysis needs no model training or threshold changes.

## 3. Final interaction-recovery results

Means are over 20 realizations per scenario-method group. AP denotes
mean proposal interaction-ranking average precision, not predictive AUPRC.
Selection and ISR F1 are structural recovery metrics against the true pair
set. Delta F1 is computed on unrounded paired scores.

| Scenario | Proposal | AP | Selection F1 | ISR F1 | Delta F1 | Win/loss/tie |
| --- | --- | --- | --- | --- | --- | --- |
| S1 | Friedman-H | 0.8884 | 0.6431 | 0.8742 | +0.2310 | 19/0/1 |
| S1 | NID | 0.6268 | 0.2376 | 0.6031 | +0.3655 | 20/0/0 |
| S1 | SIAN-FID | 0.9743 | 0.3643 | 0.8221 | +0.4579 | 20/0/0 |
| S2 | Friedman-H | 0.7855 | 0.8340 | 0.8238 | -0.0102 | 5/10/5 |
| S2 | NID | 0.6007 | 0.3238 | 0.6838 | +0.3601 | 20/0/0 |
| S2 | SIAN-FID | 0.7870 | 0.4938 | 0.7152 | +0.2214 | 20/0/0 |
| S3 | Friedman-H | 0.7054 | 0.4700 | 0.5236 | +0.0536 | 12/2/6 |
| S3 | NID | 0.5237 | 0.1385 | 0.3244 | +0.1860 | 20/0/0 |
| S3 | SIAN-FID | 0.6547 | 0.2498 | 0.3886 | +0.1389 | 19/1/0 |
| S4 | Friedman-H | 0.3628 | 0.6614 | 0.6395 | -0.0219 | 3/4/13 |
| S4 | NID | 0.2791 | 0.2898 | 0.3767 | +0.0869 | 16/0/4 |
| S4 | SIAN-FID | 0.5050 | 0.6506 | 0.7305 | +0.0800 | 14/3/3 |

ISR improves mean F1 in 10/12 groups (188/240 individual paired
comparisons improve, 20 decline, 32 tie). These are descriptive counts,
not independent statistical sample sizes.

### Pooled pair observations

| Scenario | TP before -> after | FP before -> after | FP reduction | Selected TP retention |
| --- | --- | --- | --- | --- |
| S1 | 160 -> 158 | 529 -> 84 | 84.12% | 98.75% |
| S2 | 195 -> 181 | 368 -> 75 | 79.62% | 92.82% |
| S3 | 101 -> 97 | 584 -> 258 | 55.82% | 96.04% |
| S4 | 124 -> 115 | 101 -> 21 | 79.21% | 92.74% |
| Total | 580 -> 551 | 1582 -> 438 | 72.31% | 95.00% |

Overall, ISR removes 1,144 FP and 29 TP observations, retaining 989 of
2,162 selection-stable observations. Selected TP retention (95.00%) is
conditional on selection and must not be called overall recall. All pooled
counts may include the same pair across methods or realizations.

## 4. Matched-cardinality and tie sensitivity

For each method-realization, set k to the number retained by ISR. Compare
ISR to the k highest selection-probability pairs from the same stable
candidate set, resolving ties lexicographically by the stored `feature_a`
and `feature_b` strings. This is a retrospective cardinality control:
k is supplied by ISR, not an independently chosen deployment budget.

At the cutoff let h be the number of strictly higher-scoring pairs, H the
number of true pairs among them, e the number of tied pairs, E the number
of true tied pairs and b = k - h. The minimum and maximum possible TP counts
are H + max(0, b - (e - E)) and H + min(b, E). Uniform random tie-breaking
has expected TP H + b*E/e when e > 0; when k = 0, TP is zero. At fixed k
and true-pair count T, F1 = 2*TP/(k+T), so the expected F1 follows directly.
The min/max are label-informed oracle sensitivity bounds, not executable
baselines. Robust win/loss means ISR TP lies strictly above/below those
bounds.

| Scenario | Proposal | k | ISR TP | Top-K TP | Oracle TP range | Robust wins/losses |
| --- | --- | --- | --- | --- | --- | --- |
| S1 | Friedman-H | 78 | 59 | 57 | 57-59 | 0/0 |
| S1 | NID | 76 | 40 | 22 | 10-40 | 0/0 |
| S1 | SIAN-FID | 88 | 59 | 35 | 29-60 | 0/0 |
| S2 | Friedman-H | 68 | 61 | 64 | 64-65 | 0/3 |
| S2 | NID | 98 | 60 | 41 | 36-54 | 5/0 |
| S2 | SIAN-FID | 90 | 60 | 60 | 60-62 | 0/1 |
| S3 | Friedman-H | 106 | 38 | 39 | 39-40 | 0/2 |
| S3 | NID | 92 | 21 | 20 | 5-20 | 1/0 |
| S3 | SIAN-FID | 157 | 38 | 30 | 25-39 | 0/1 |
| S4 | Friedman-H | 47 | 42 | 40 | 39-41 | 2/1 |
| S4 | NID | 27 | 20 | 20 | 20-20 | 0/0 |
| S4 | SIAN-FID | 62 | 53 | 44 | 42-49 | 4/0 |

Across 240 evaluations, k sums to 989: ISR TP = 551 and deterministic
Top-K TP = 472 (FP = 438 vs 517). The probability-only oracle TP range
is 426-549, with 141 boundary ties, 12 robust ISR wins and 8 robust losses.
The total oracle advantage of ISR is only two TP observations; it does not
imply general superiority or an inferential result for mean F1.

| Scenario | Proposal | ISR F1 | Top-K F1 | Uniform expected F1 | Oracle best F1 |
| --- | --- | --- | --- | --- | --- |
| S1 | Friedman-H | 0.8742 | 0.8408 | 0.8525 | 0.8742 |
| S1 | NID | 0.6031 | 0.3252 | 0.3688 | 0.6031 |
| S1 | SIAN-FID | 0.8221 | 0.4683 | 0.5932 | 0.8388 |
| S2 | Friedman-H | 0.8238 | 0.8617 | 0.8659 | 0.8742 |
| S2 | NID | 0.6838 | 0.4557 | 0.5006 | 0.6155 |
| S2 | SIAN-FID | 0.7152 | 0.7135 | 0.7191 | 0.7402 |
| S3 | Friedman-H | 0.5236 | 0.5454 | 0.5516 | 0.5579 |
| S3 | NID | 0.3244 | 0.3119 | 0.2389 | 0.3119 |
| S3 | SIAN-FID | 0.3886 | 0.3019 | 0.3476 | 0.3977 |
| S4 | Friedman-H | 0.6395 | 0.6104 | 0.6131 | 0.6270 |
| S4 | NID | 0.3767 | 0.3767 | 0.3767 | 0.3767 |
| S4 | SIAN-FID | 0.7305 | 0.6105 | 0.6282 | 0.6752 |

ISR exceeds the uniform expectation descriptively in 8/12 groups and
exceeds the oracle best mean F1 in four (S2-NID, S3-NID, S4-Friedman-H and
S4-SIAN-FID). The oracle comparison is descriptive, not a hypothesis test.
S1 gains against lexical Top-K are sensitive to tie handling. S2-NID
provides a particularly informative example: ISR F1 = 0.6838 vs 0.4557
(lexical), 0.5006 (uniform expectation) and 0.6155 (oracle best).

## 5. Exploratory paired inference

Use 20 paired realization differences per scenario-method group. Report
unadjusted 95% paired t intervals for the mean difference, two-sided
Wilcoxon signed-rank tests (zero differences omitted; all-zero vectors
assigned p = 1), with Holm adjustment across the entire 24-test family:
12 ISR-vs-selection and 12 ISR-vs-deterministic-probability comparisons.
Rounded differences (12 decimals) avoid spurious floating-point ties.
Wilcoxon p-values are those obtained by the recorded SciPy implementation;
with zeros/ties, do not describe them as exact tests. The t intervals are
not multiplicity-adjusted and need not agree with the corrected tests.
Methods share realizations, so do not treat the 240 records as independent
replicates in a pooled inferential test. No inferential test against the
uniform expectation or the label-informed oracle is included here.

### ISR vs selection

| Scenario | Proposal | Mean delta F1 | 95% CI | Wilcoxon p | Holm p | Significant |
| --- | --- | --- | --- | --- | --- | --- |
| S1 | Friedman-H | +0.2310 | [0.1641, 0.2980] | 0.000129 | 0.002325 | Yes |
| S1 | NID | +0.3655 | [0.3222, 0.4088] | 0.000088 | 0.002084 | Yes |
| S1 | SIAN-FID | +0.4579 | [0.3967, 0.5190] | 0.000088 | 0.002084 | Yes |
| S2 | Friedman-H | -0.0102 | [-0.0576, 0.0373] | 0.690114 | 1.000000 | No |
| S2 | NID | +0.3601 | [0.3235, 0.3966] | 0.000087 | 0.002084 | Yes |
| S2 | SIAN-FID | +0.2214 | [0.1853, 0.2575] | 0.000088 | 0.002084 | Yes |
| S3 | Friedman-H | +0.0536 | [0.0052, 0.1019] | 0.040628 | 0.446907 | No |
| S3 | NID | +0.1860 | [0.1615, 0.2105] | 0.000087 | 0.002084 | Yes |
| S3 | SIAN-FID | +0.1389 | [0.1138, 0.1639] | 0.000102 | 0.002084 | Yes |
| S4 | Friedman-H | -0.0219 | [-0.0772, 0.0335] | 0.734861 | 1.000000 | No |
| S4 | NID | +0.0869 | [0.0597, 0.1140] | 0.000416 | 0.006654 | Yes |
| S4 | SIAN-FID | +0.0800 | [-0.0050, 0.1649] | 0.058077 | 0.580768 | No |

### ISR vs deterministic Probability Top-K

| Scenario | Proposal | Mean delta F1 | 95% CI | Wilcoxon p | Holm p | Significant |
| --- | --- | --- | --- | --- | --- | --- |
| S1 | Friedman-H | +0.0333 | [-0.0147, 0.0814] | 0.157299 | 1.000000 | No |
| S1 | NID | +0.2779 | [0.2295, 0.3264] | 0.000148 | 0.002516 | Yes |
| S1 | SIAN-FID | +0.3538 | [0.2401, 0.4676] | 0.000420 | 0.006654 | Yes |
| S2 | Friedman-H | -0.0379 | [-0.0815, 0.0057] | 0.108809 | 0.979285 | No |
| S2 | NID | +0.2281 | [0.1277, 0.3286] | 0.001439 | 0.020153 | Yes |
| S2 | SIAN-FID | +0.0018 | [-0.0390, 0.0425] | 0.654721 | 1.000000 | No |
| S3 | Friedman-H | -0.0218 | [-0.0801, 0.0365] | 0.285049 | 1.000000 | No |
| S3 | NID | +0.0125 | [-0.0137, 0.0387] | 0.317311 | 1.000000 | No |
| S3 | SIAN-FID | +0.0867 | [0.0286, 0.1449] | 0.006065 | 0.078847 | No |
| S4 | Friedman-H | +0.0292 | [-0.0369, 0.0953] | 0.449692 | 1.000000 | No |
| S4 | NID | +0.0000 | [0.0000, 0.0000] | 1.000000 | 1.000000 | No |
| S4 | SIAN-FID | +0.1200 | [0.0559, 0.1842] | 0.006459 | 0.078847 | No |

After global Holm adjustment at alpha = 0.05, 8/12 selection contrasts
and 3/12 matched-cardinality contrasts are significant improvements. The
latter are S1-NID, S1-SIAN-FID and S2-NID. No significant deterioration is
detected, but non-significance is not evidence of equivalence or safety.
S3-Friedman-H and S4-SIAN-FID must not be called significant based on
prior smaller correction families or raw p-values. All tests are exploratory
post-submission analyses.

## 6. Failure modes and limitations

S3 proxy-substitution category is defined by six unordered pairs:
`(x2,x11)`, `(x1,x12)`, `(x11,x12)`, `(x6,x13)`, `(x5,x14)`,
`(x13,x14)`. Other FP may also involve correlations; they are not defined as
uncorrelated FP.

| Proposal | Proxy FP before/after | Other FP before/after | TP before/after |
| --- | --- | --- | --- |
| Friedman-H | 66/65 | 27/3 | 40/38 |
| NID | 53/53 | 194/18 | 21/21 |
| SIAN-FID | 99/99 | 145/20 | 40/38 |
| Total | 218/217 | 366/41 | 101/97 |

S3: only 1/218 predefined proxy FP observations is removed (0.46%),
whereas 325/366 other FP observations are removed (88.80%). Stable proxy
surfaces can have high ISR; stability is not ground-truth correctness.
Four true-pair observations are lost in two realizations (R17 and R19,
Friedman-H and SIAN-FID). These are not the same as the three robust
matched-cardinality losses.

S4: 9/124 selected TP observations are removed. In R1, SIAN-FID selects
three true pairs and no FP, yet ISR retains only one true pair (F1 drops
from 0.8571 to 0.4000). R1/R2 also contain examples where a false pair
scores higher than a true pair; threshold adjustment cannot resolve that
ordering. The original 0.60 threshold is unchanged. Earlier single-
realization threshold sweeps are descriptive only, not independent tuning.

NID retains all its selected TP across the four scenarios but cannot
recover pairs missed by its proposer. Friedman-H loses mean F1 in S2 and
S4. These observations preclude claims of universal benefit, guaranteed
true-pair retention, or proxy robustness.

## 7. Runtime accounting

All three timing fields are finite and nonnegative for 240 records.
`residual_seconds_sum_shared` agrees across methods for each of the 80
realizations. The columns below are realization-level means in seconds;
each component is the recorded aggregate for the five-run protocol.

| Scenario | Proposal method | Proposal s | Residual s | ISR s | Standalone component total s |
| --- | --- | --- | --- | --- | --- |
| S1 | Friedman-H | 12.34 | 135.65 | 58.08 | 206.07 |
| S1 | NID | 55.42 | 135.65 | 87.95 | 279.02 |
| S1 | SIAN-FID | 100.75 | 135.65 | 101.44 | 337.84 |
| S2 | Friedman-H | 29.32 | 228.48 | 79.91 | 337.71 |
| S2 | NID | 78.35 | 228.48 | 153.64 | 460.47 |
| S2 | SIAN-FID | 205.92 | 228.48 | 112.15 | 546.55 |
| S3 | Friedman-H | 28.97 | 211.32 | 133.67 | 373.97 |
| S3 | NID | 51.21 | 211.32 | 165.79 | 428.33 |
| S3 | SIAN-FID | 179.42 | 211.32 | 243.78 | 634.52 |
| S4 | Friedman-H | 61.58 | 362.39 | 49.56 | 473.53 |
| S4 | NID | 78.90 | 362.39 | 39.14 | 480.43 |
| S4 | SIAN-FID | 579.44 | 362.39 | 73.02 | 1014.85 |

Standalone component total = proposal standalone sum + residual sum +
ISR wall time. This is not a separately measured end-to-end wall time.
Standard deviations of the component totals are computed directly across
realizations in the runtime CSV (not by summing component SDs).

The joint analysis sums the three standalone proposal components and the
three ISR wall times, counting the shared residual once. **This still
includes the shared MLP fit in both NID and SIAN-FID standalone proposal
components when that fit was reused.** Therefore the joint component sum
is an accounting aggregate, not a measured joint execution time or a
fully deduplicated estimate. Accurate joint wall time would require
separate measurement or fit-level accounting from the run manifests.

| Scenario | Sum of standalone proposal s | Sum of ISR s | Shared residual s | Joint component sum s |
| --- | --- | --- | --- | --- |
| S1 | 168.51 | 247.48 | 135.65 | 551.64 |
| S2 | 313.60 | 345.70 | 228.48 | 887.77 |
| S3 | 259.60 | 543.24 | 211.32 | 1014.17 |
| S4 | 719.92 | 161.72 | 362.39 | 1244.03 |

Do not compare these five-run hybrid component totals directly with
single-run standalone baseline or original AG-NAM runtime as equal-budget
measurements. ISR adds non-negligible validation cost; SIAN-FID is
particularly expensive in S4.

## 8. Final artifacts and audit

`final_analysis_v1/` contains the following immutable analysis snapshot:

| Artifact | Contents |
| --- | --- |
| `experiment_summary_240.csv` | 240 summary records and per-record component totals |
| `recovery_by_scenario_method.csv` | Mean, SD, valid n and pooled recovery counts |
| `matched_cardinality_240.csv` | Per-record deterministic, uniform and oracle controls |
| `matched_cardinality_by_scenario_method.csv` | Grouped matched-cardinality results |
| `paired_statistics_global_holm.csv` | 24 paired tests and global Holm adjustment |
| `runtime_by_scenario_method.csv` | Mean and SD of timing components |
| `runtime_joint_by_realization.csv` | Shared-residual accounting for 80 realizations |
| `runtime_joint_by_scenario.csv` | Grouped joint component sums |
| `s3_proxy_by_realization.csv` | 60 S3 method-realization proxy records |
| `s3_proxy_by_method.csv` | Grouped proxy, other FP and TP counts |
| `analysis_manifest.json` | Source paths/hashes, output hashes, script hash and versions |

The finalizer checks source identity, run counts, true-pair counts,
thresholds, finite nonnegative timings, pair uniqueness, selection
probability, ISR decision consistency, summary/pair TP/FP/FN and F1
agreement, tie bounds and group sizes. It hashes all 480 source CSVs and
all ten derived CSVs. This is not a full independent audit of all original
proposal manifests, dataset seeds or training correctness; retain the run
manifests and residual split fingerprints for provenance.

Recorded environment: Python 3.10.19, NumPy 1.26.4, pandas 2.3.3 and
SciPy 1.15.3. The final analysis archive's ten CSV hashes and the finalizer
script hash were checked against `analysis_manifest.json`. The local
finalizer recorded 480 input hashes; the final analysis archive alone
cannot independently rehash all original inputs.

Local full test suite (2026-11-18): `391 passed in 16.47s`.
The tests mock expensive training where appropriate; passing tests show
no detected regression in tested behavior, not empirical correctness of
all external implementations. Preserve the analysis snapshot and source
results before changing the code. Check Git ignore policy before staging
results; never force-add bulk caches or overwrite locked benchmarks.

## 9. Reporting decision

The defensible contribution is a transferable functional-stability filter
that often improves structural precision/F1 and can contribute beyond
selection frequency in specific settings, at additional computation cost.
It is not a universally superior interaction discovery or validation rule.
Report both the selection-relative and cardinality-matched comparisons,
including negative results, multiple-testing adjustment, ties and the S3/S4
failure modes. External hybrids are not unmodified independent methods:
they use common AG-NAM MainEffectNAM residuals and a repeated discovery
protocol. Keep the original AG-NAM benchmark and standalone external
baselines separate; their proposal designs, data use and fit budgets are
not matched by this study. Do not claim superior original AG-NAM raw AP
from hybrid recovery gains.

The planned external-proposal + ISR experiment schedule is complete.
Do not initiate additional training or change thresholds to improve these
reported outcomes. Any subsequent correction must be documented and
versioned separately, with a justified reason and updated manifest.
