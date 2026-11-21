# AG-NAM R12 — Native SIAN pairwise-constrained synthetic pilot

## Purpose

Respond to the reviewer's request for an external **sparse neural model** comparator. This is a *new R12 post-review experiment* and must not be folded into previously frozen R1–R11 primary inferential families.

## Exactly what runs

1. Same synthetic dataset, realization seeds, outer development/test split and nested 60/20/20 final train/validation/test split as the AG-NAM benchmark.
2. StandardScaler fit **only** to the 60% training subset.
3. Reference `sian==0.9.5` **MLP** fitted using existing R5a shared training helper (`train_shared_sian_mlp`, author-package MLP; Adagrad, learning rate 5e-3, L1=5e-5, early stopping on *validation* AUROC).
4. Unmasked `JamArchipelago` scores on a seeded set of 32 *validation* observations; the **native SIAN `layerwise_feature_interaction_selection_algorithm`** selects univariate features and second-order pairs. The fixed, R12-defined selection hyperparameters are MAX_K=2, theta percentile fractions 0.50 for order 1 and 0.10 for order 2, tau=0.50 at order 2. These are not claimed to be the original NeurIPS paper's locked benchmark parameters.
5. The **native `sian.models.models.SIAN` class** is fitted afresh with the selected terms, neural part `dnn_on_or_off=False`, 16→12→8 hidden sizes in its GAM subnetworks, validation-AUROC early stopping, same train-only scaling.
6. Only *after* selection and all early stopping is finished, evaluate final test AUROC/AUPRC once. True interactions (synthetic metadata) are used only to calculate recovery metrics *after* term selection.
7. Results and the native model checkpoint are written to `results/revision/r12_native_sian_pairwise/S1/realization_01/` only; intermediate failures use `.inprogress` to avoid false completed status.

## What the result **can** and **cannot** claim

- This is an **end-to-end SIAN-native workflow constrained to pairwise terms**: reference MLP → library-native FIS → library-native SIAN predictor. It is much stronger than calling the isolated SIAN-FID score a complete SIAN model.
- The upper interaction order is deliberately capped at 2 for the AG-NAM pairwise question. It is **not** a replication of the unrestricted SIAN configuration in its original paper. Performance may differ from the original method with higher-order terms.
- Native-FIS selected-set F1 is **not** the same estimand as full-universe ranking AUPRC (which the R5 SIAN-FID experiments reported). Do not put them in a single league table with common boldface without a comparable estimand.
- A single S1 pilot is **descriptive**. No significance test, real-data generalization or numerical superiority claim may be made from it. Expand to 20 realizations per scenario only after pilot checks and parameter lock.
- AG-NAM vs SIAN model architecture and selection budgets differ. Match data, seed regime and evaluation endpoints, not pretend they are identical architectures.
- R12 cost audit: the *canonical* AG-NAM runner trains six variants, not only the Full AG-NAM path. Its elapsed time is **not a comparable Full-AG-NAM end-to-end runtime**. Use separate isolated profiling for a defensible paired computational-cost analysis.
- Before journal submission, include at least one additional published full detector (NIT/iRF) where feasible or explicitly flag omitted baselines; this pilot alone does **not** satisfy every requested comparator.

## Source/API checked

- Installed local AG-NAM preflight snapshot: `AG_NAM_R12_Preflight.zip`; git HEAD `bd6be1d12bd3d1b7eb493e269cfcf6d6fc8c344a` and branch `revision-reviewers`.
- Reference package: `sian==0.9.5` (Nov 2026), EnouenJ/sian. Native `layerwise_FIS_Hyperparameters`, `layerwise_feature_interaction_selection_algorithm`, `SIAN` class and `JamArchipelago` API checked against source. The author's package is directly imported, **no reimplementation of its FIS or SIAN model**.
- AG-NAM synthetic protocol source: `src/agnam/benchmarking/synthetic_protocol.py`; generator/splits: `src/agnam/benchmarking/synthetic_runner.py`.

## Audit requirement

Record complete `summary.json`, `selected_pairs.csv`, `test_predictions.csv`, `sian_state_dict.pt`, all stage times and git commit. For a full standalone memory audit, wrap the pilot with `scripts/r12_profile_command.py` using a unique label.
