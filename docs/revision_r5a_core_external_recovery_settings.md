# R5a — Core External Interaction-Recovery Settings

Status: reviewer-requested, post-submission analysis.

This addendum is frozen before inspecting any external-baseline recovery
outcome. It instantiates the general external-baseline protocol for the first
three executable detectors:

- Friedman H-statistic,
- Neural Interaction Detection (NID),
- SIAN feature-interaction detection (SIAN-FID / Archipelago).

The original AG-NAM benchmark, thresholds, seed schedule, primary inference,
and frozen primary results remain unchanged.

## Common data partition

For every locked synthetic realization:

1. regenerate the original synthetic dataset with the locked dataset seed;
2. recreate the original 80% outer-development / 20% untouched-test split;
3. do not use the untouched outer test set for external interaction recovery;
4. split outer-development with the locked final-split seed into:
   - 75% detector fitting,
   - 25% detector scoring / validation.

Thus the external detectors use 60% of the complete realization for fitting,
20% for interaction scoring / validation, and leave 20% untouched.

Ground-truth interactions are used only after scores are produced.

## Pair universe

The comparison universe is every unordered pair of original input features.

All three methods produce one numeric score per pair.

Ties are resolved deterministically by feature name only after scores have
been computed. Tie-breaking never changes a score.

## Evaluation

For every method report:

- interaction AUPRC over the complete pair universe,
- interaction AUROC,
- NDCG over the complete pair universe,
- precision / recall / F1 at the locked AG-NAM candidate K,
- precision / recall / F1 at the number of true interactions
  (oracle-count diagnostic; explicitly non-deployable).

The oracle-count diagnostic does not tune a detector. It is used only to
compare ranking quality at a common cardinality.

## Shared neural backbone for NID and SIAN-FID

NID and SIAN-FID intentionally share one fitted MLP. This controls for neural
backbone quality and isolates the effect of the interaction-scoring rule.

Backbone implementation:

- `sian.models.models.MLP`,
- ReLU feed-forward network,
- hidden widths: 128 -> 64 -> 32,
- one binary logit output.

Input preprocessing:

- StandardScaler fit on detector-fitting data only,
- same fitted scaler applied unchanged to detector validation/scoring data.

Optimization:

- optimizer: Adagrad,
- learning rate: 5e-3,
- batch size: 32,
- maximum epochs: 300,
- L1 coefficient: 5e-5,
- gradient clipping norm: 1.0,
- validation AUROC early stopping patience: 20,
- deterministic seed: locked realization discovery-base seed.

These settings deliberately follow the current SIAN package training defaults
where practical (Adagrad, 5e-3, batch 32, 300 epochs, L1 5e-5) while adding a
fixed validation early-stop rule to avoid unnecessary computation.

## NID score

The NID implementation follows the published/reference weight-decoding idea.

For first-hidden-layer unit h:

- compute its downstream absolute influence to the output by multiplying
  absolute downstream weight matrices;
- for pair (j,k), use the pair aggregator
  `min(|W_hj|, |W_hk|)`;
- sum downstream-influence-weighted pair evidence over first-layer units.

The implementation is labeled `nid_shared_mlp`; it is a controlled PyTorch
reproduction of the NID pair decoder, not a claim of bitwise reproduction of
the historical TensorFlow experiments.

## SIAN-FID score

Implementation label: `sian_fid_shared_mlp`.

Use the installed `sian==0.9.5` implementation:

- unmasked `JamArchipelago`,
- masking style: `triangle_marginal`,
- output type: `binary_classification`,
- pair score: `old_arch_inter_score`,
- all unordered original-feature pairs are scored directly,
- deterministic interaction-support sample size: 32 validation observations,
- support observations sampled without replacement using the locked
  realization discovery-base seed,
- `JamArchipelago.batch_size = 512`.

`old_arch_inter_score` is retained because it is explicitly present in the
current unmasked SIAN FID score list and is non-negative.

The support size 32 is a computational-budget choice for exhaustive pair
ranking. It is fixed before any recovery result is inspected and is not tuned
on ground truth.

This baseline evaluates SIAN's feature-interaction detection component, which
is the structure-discovery component relevant to the reviewer's recovery
request. It is therefore reported as `SIAN-FID`, not as predictive performance
of a fully retrained final SIAN additive model.

## Friedman H-statistic

Implementation label: `friedman_h_gbdt`.

Black-box estimator:

- GradientBoostingClassifier,
- 200 trees,
- learning rate 0.05,
- maximum depth 3,
- deterministic locked realization seed.

H is computed from centered one- and two-feature partial dependence on the
detector validation/scoring partition:

- response scale: decision function,
- partial-dependence method: recursion,
- grid resolution: 10,
- percentiles: 0.05 to 0.95.

For pair (j,k):

    H_jk^2 =
        sum [PD_jk - PD_j - PD_k]^2
        --------------------------------
        sum [PD_jk]^2

after centering the relevant partial-dependence functions.

No H threshold is selected; H is used as a complete pair ranking.

## Runtime and memory

Record separately:

- model fitting time,
- interaction scoring time,
- standalone total time,
- process RSS start/end/observed peak,
- CUDA peak allocated memory where applicable.

Because NID and SIAN-FID share the fitted neural backbone, the shared training
cost is reported explicitly. Standalone totals add that common training cost
to each method's own scoring cost.

## Statistical boundary

This is descriptive, reviewer-requested post-submission analysis.

No new confirmatory hypothesis family is introduced.
Original primary p-values and multiplicity adjustments are unchanged.
