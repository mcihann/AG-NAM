# R10 — Domain-grounded interpretation and final-model interaction diagnostics

**Status:** reviewer-requested, post-submission, exploratory. The frozen primary AG-NAM benchmark and locked case selection are not changed.

## Research question and epistemic boundaries

Three different questions must not be conflated:

1. **Discovery/reproducibility.** Did the existing benchmark retain a feature pair? This is determined solely by the frozen `task_*_pairs.csv` records. It is *not* independent ground truth.
2. **Non-additive fitted-model behavior.** Does the retrained final AG-NAM pair network exhibit a nonzero rectangular mixed difference, on empirically supported held-out feature combinations? This tests a *model function*, not the physical system or a causal relation.
3. **Independent domain evidence.** Does a source independent of the model establish a particular two-feature interaction relevant to the same outcome and scale? Merely discussing both variables, or showing marginal correlations, is insufficient. An external literature annotation requires source, exact table/figure/equation and adjudication; no automatic positive labels are assigned.

No precision/recall/FNR relative to expert labels may be reported without actual, independently validated exhaustive pairwise truth. No *causal interaction*, independent external validation, preregistration, or predictive superiority is implied by R10.

## Frozen case-study selection

Preserve the original publication's one-per-feature-group case selection:

| Group | Task ID | Dataset | Retained pairs |
|---|---:|---|---:|
| Numeric | 146819 | climate-model-simulation-crashes | 2 |
| Mixed | 3021 | sick | 4 |
| Categorical | 3 | kr-vs-kp | 7 |

Total **13 locked pairs**. Additional post-hoc categorical case: **14952 PhishingWebsites**, with **9 retained pairs**. Discuss this separately; it was selected after seeing outputs. Do not replace a locked case with the post-hoc supplement.

Pair names, retained flags, and frequencies come from the frozen OpenML benchmark files, not domain labels. `pair_inventory.csv` is a review worksheet: `external_evidence_status` defaults to `not_yet_adjudicated` for **all** 22 pairs. The script preserves subsequently entered annotation columns instead of silently overwriting them.

## Independent domain sources for manual review

- Lucas et al., *Failure analysis of parameter-induced simulation crashes in climate models*, **Geoscientific Model Development** 6, 1157–1171 (2013), DOI: https://doi.org/10.5194/gmd-6-1157-2013. The paper examines effects of ocean mixing/viscosity parameters on simulation failures. **Do not assert an exact pairwise coefficient or causal interaction until the relevant original equation/table and feature mappings are personally verified.**
- UCI Climate Model Simulation Crashes dataset: https://doi.org/10.24432/c5hg71.
- UCI thyroid/sick dataset documentation: https://archive.ics.uci.edu/dataset/102/thyroid+disease. `TSH`, `FTI`, `T3`, `query_on_thyroxine`, `sex`, `referral_source` are interpretable clinical/data-collection variables; their co-occurrence alone is not statistical interaction proof.
- UCI Chess (King-Rook vs. King-Pawn): https://archive.ics.uci.edu/dataset/22/chess+king+rook+vs+king+pawn. Decode `bkxbq`, `wknck`, etc. from authoritative original documentation before assigning exact chess semantics.
- UCI Phishing Websites: https://archive.ics.uci.edu/dataset/327/phishing+websites. URL, SSL, links, domain and traffic fields offer contextual plausibility, not pairwise truth labels.

Domain annotation suggested categories: `direct_pairwise_evidence`, `related_variables_only`, `contradicted`, `unclear`, `not_yet_adjudicated`; cite precise passages, labels and outcome definitions. Review all pairs, including null or ambiguous interpretations. An annotation is an evidence *classification*, not ground truth by default.

## Model-level diagnostic: final model only

- Reuse task registry, locked predefined outer train/test split, final validation split, final model seed, model architecture, optimizer and training budget from original OpenML benchmark and R8. **Do not rerun discovery or ISR.**
- Reconstruct pair set from *original* audit file. Use the unmodified R8 `_new_model` factory and `train_agnam` training API. Save model `state_dict` and source/output SHA-256 hashes for reproducibility.
- Before reporting, require AG-NAM holdout AUROC to match the frozen benchmark within `1e-6`. Verify predictions from model's standard API match probabilities from decomposition, and verify `logits = baseline + sum(main) + sum(centered_pair)` within `1e-5`.
- On the untouched held-out partition, export per-pair centered contribution values **on logit scale** plus mean absolute magnitude, standard deviation and 95th percentile absolute magnitude. Nonzero variance alone is **not** evidence of a non-additive bivariate function: the pair network itself can encode a sum of univariate functions.
- For each fitted pair `g_jk`, compute the rectangular finite difference on the pair network alone:

  `D = g(a,c) - g(a,d) - g(b,c) + g(b,d)`.

  A *zero* difference is compatible with an additive pair surface for that rectangle; a *nonzero* difference demonstrates non-additive **fitted-model behavior at those four inputs**, not a physical/causal interaction.
- Find two levels of each categorical predictor among observed known categories (`<MISSING>`/`<UNK>` excluded); for numerical features use held-out 25th/75th percentiles. Only calculate this diagnostic when **all four empirical test cells have >=3 rows** (categorical exact levels, numeric low/high midpoint strata). For numeric corners, strata support does **not** guarantee density at the exact quartile intersection: report this limitation explicitly. Unsupported rectangles must be `NaN`, never zero.
- No statistical significance test or multiplicity-corrected discovery claim is made from a single fitted model/rectangle. The 3-observation minimum is an exploratory feasibility filter, not a prespecified clinical support threshold.

## Execution sequence and artifacts

1. `python scripts/run_revision_domain_grounded.py --audit-only` — no network/model training, exports locked/post-hoc 22-pair inventory.
2. Review independent original feature dictionaries and literature annotations.
3. `python scripts/run_revision_domain_grounded.py --pilot` — **one** new final model for frozen numeric task 146819; audit AUROC, decomposition, pair mixed differences.
4. If pilot audit passes, run `--task-id 3021`, `--task-id 3`; optionally `--task-id 14952` as clearly post-hoc.

Outputs: `results/revision/domain_grounded_r10/v1/pair_inventory.csv`; for trained cases `task_<id>/pair_model_evidence.csv`, `test_pair_contributions.csv`, `test_predictions.csv`, `final_agnam_state_dict.pt`, `metadata.json`. These results are *new post-submission outputs*, not original frozen results. Never overwrite unverified task directories; recover/inspect mismatches instead.

## Critical limitations to state in manuscript

- Selection of retained interaction structures is learned on the available data; external evidence should be annotated blind to the model's identity wherever possible, but R10 is explicitly post-hoc.
- Centering is an algebraic translation: it preserves logits and does not purify pair components into statistically identifiable interactions.
- Correlated predictors, sampling, small test cells, and model architecture can produce patterns that look non-additive without corresponding domain interactions. Model decomposition is a functional decomposition, not independent verification.
- Domain evidence in Lucas et al. does not automatically extend to every named retained pair, nor to an externally verified statistical relationship in the OpenML classification label.
