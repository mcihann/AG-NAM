# R6 - External interaction recovery on frozen Real-X

Status: reviewer-requested, post-submission. This addendum fixes the scope
before inspecting any R6 recovery outputs. It applies the existing R5a core
scorers to the locked Real-X generator and does not modify any original
synthetic, OpenML, or Real-X benchmark output.

## Scope and estimand

- Nine locked OpenML predictor distributions; five realizations each.
- Moderate (lambda=1.0): 45 runs, 135 detector evaluations. Primary R6 scope.
- Null (lambda=0.0): 45 runs, 135 detector evaluations. Negative control.
- Three existing implementations per run: `friedman_h_gbdt`, `nid_shared_mlp`,
  `sian_fid_shared_mlp`. NID and SIAN-FID share one fitted MLP per run.
- These are rankings, not native sparse selection. SIAN-FID is the detection
  component, not a full final SIAN additive model; NID is not NIT. This does
  not close the protocol's separate NIT/iRF/Additive Groves requests.
- The original AG-NAM mean AP is an average of five discovery runs on internal
  development partitions; R6 external rankings use one fitted detector.
  Direct descriptive comparison is useful but does not imply matched training
  budgets, equivalent architectures, or general superiority.

## Data and preprocessing

Reuse `build_realx_schedule`, `load_realx_openml_X`, `generate_realx_run`,
`derive_realx_model_seeds`, `validate_locked_split`, and `realx_candidate_k`.
The locked generated indices are used directly: 60% train, 20% validation,
20% untouched test. No new stratification or random split. The base model
seed is the original discovery-base seed (label seed + 10). Detector training
and preprocessing see only train; validation supplies early stopping/scoring
support. No predictor from the test partition is transformed for a detector
or used during fitting, scoring, or model selection. The existing Real-X
label generator itself is reused unchanged, including its construction of
synthetic responses from the original predictor distribution.

The R5a scorers require one scalar input per column. Use the existing
`TabularPreprocessor`, fitted on train only, then assemble numerical and
categorical outputs in original column order. Numeric values are median
imputed and standardized; categorical values use deterministic training
vocabulary codes (missing=0, unknown=1, known=2 onward). Numerical missing
indicators are omitted in this scalar-input adapter, and no one-hot feature
expansion is performed. The shared MLP applies its existing training-only
StandardScaler afterwards. Integer coding of nominal predictors can impose
artificial ordering and distances (including H partial dependence and FID
perturbations). It is an explicit implementation limitation, not native
categorical support, and differs from AG-NAM's embeddings/missing indicators.
Do not claim identical representations or tune an alternative encoding using
planted truth. Report dataset-level results and feature regime transparently.

## Frozen evidence and outputs

Before each run, verify the original Phase 6F-A freeze manifest's 182 files
and aggregate hash using the existing `verify_manifest_files`. Validate the
selected original checkpoint; match regenerated feature/sample counts,
partition sizes, seed, candidate K, realized prevalence, template truth and
active truth. If the generator audit CSV exists, also check its source row
hash. Record regenerated source rows, partition indices, label hashes, encoded
matrix hashes, checkpoint SHA-256, and frozen evidence SHA-256. The original
checkpoints do not provide raw label or split index hashes for byte-level
comparison with the regenerated arrays; do not claim such equality solely
from matching metadata. Ground truth is used for regeneration/provenance
checks and scoring evaluation only, never as a detector input or tuning target.

All results go to `results/revision/external_interaction_baselines/core_realx_v1/`.
Each task/realization/strength has summary.csv, pair_scores.csv, manifest.json.
The manifest is written last and stores output hashes, methods, settings,
versions and provenance. A valid triplet is skipped on restart. An incomplete
or corrupted triplet fails closed; use --force only for that selected run
with a documented reason. All three methods are evaluated together to reuse
MLP training. A failed run is not silently treated as complete.

## Metrics

Use all unordered original-feature pairs. Candidate K=min(p-1,20) from the
original Real-X lock. On moderate: AP (average precision), AUROC, NDCG,
precision/recall/F1 at candidate K, oracle truth count, and the original
AG-NAM ISR-retained count. Include TP/FP/FN for K and matched count.
Oracle true count and AG-NAM matched count are evaluation-only budgets;
neither is a standalone learned detector threshold. Lexical tie-breaking
is deterministic and score-independent; note tie effects when relevant.

On null: AP/AUROC/NDCG, recall/F1 and oracle-count metrics are undefined and
stored as empty CSV cells. Preserve the literal `null` label when reading CSV.
Report score summaries by method, not across incompatible native score scales.
A fixed-K ranking selects K false pairs by construction under null. Likewise,
matching AG-NAM's count forces exactly that many false pairs. These are not
native false-positive rates or evidence of null-detection superiority.

Runtime and memory reuse the R5a component measurements (sampled process RSS,
CUDA peak allocated MiB where applicable). Standalone sums include shared MLP
fit for each neural method, so summing across methods double counts training.
They exclude data regeneration/preprocessing and are not full end-to-end
benchmark wall time. NID memory fields primarily reflect shared training.

## Execution from repository root (Windows CMD)

    python -m pytest -q tests/test_external_realx_runner.py
    python scripts/run_revision_external_realx.py --strength both --dry-run

After tests, validate a single moderate run on task 49, realization 1:

    python scripts/run_revision_external_realx.py --task-id 49 --realization 1 --strength moderate

Then validate the matching null pilot before extending:

    python scripts/run_revision_external_realx.py --task-id 49 --realization 1 --strength null

After pilot audit, use the resumable runner:

    python scripts/run_revision_external_realx.py --strength moderate
    python scripts/run_revision_external_realx.py --strength null
    python scripts/analyze_revision_external_realx.py --strength both

The analysis requires every selected run and validates its hashes, protocol
and frozen evidence. It exports per-run, dataset-level mean/SD, equally
weighted dataset macro summaries, and input/output hash manifest. No new
confirmatory tests or changes to the original primary hypothesis are made.

The regression tests use small in-memory data and mock expensive detector
training for the file/resume tests. Real OpenML retrieval, frozen checkpoint
compatibility and the installed SIAN runtime must be verified by the local
pilot; do not mistake mocked tests for completed R6 experiments.

## Execution amendment: degenerate Friedman-H percentile grid

On the full R6 batch, run `realx_7592_r02_null` stopped in the original
Friedman-H sklearn partial-dependence grid constructor because the 5th and
95th percentiles were too close. This is a post-start numerical robustness
amendment, not a change selected from recovery metrics. Keep all prior valid
triplets unchanged; the fallback is opt-in only in R6 (not R5a or ISR hybrid).
Each single-feature grid first attempts the original recursion-based PD call.
Only for the specific percentile-degeneracy exception, construct a sorted
subset of at most 10 actually observed scoring-support values, using evenly
spaced integer indices into the sorted distinct values, including endpoints.
Pairs involving that axis reuse both corresponding single-feature grids via
sklearn `custom_values` (requires sklearn >=1.7); all other calls retain their
original grids. The estimator, splits, centering and H formula are unchanged.
No ground truth, test data, output metrics or randomness enter this fallback.
New manifests record the policy and affected feature names (empty if unused).
Earlier valid manifests lack these fields; under the original code any such
grid failure would have aborted before saving a valid triplet. Keep the old
schema and hashes to permit safe resume. Report fallback cases explicitly in
R6 analysis and flag their support choice as a numerical implementation detail.
