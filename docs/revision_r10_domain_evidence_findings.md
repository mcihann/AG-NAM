# R10 — Domain-grounded evidence and final-model interaction findings

**Status:** Reviewer-requested post-submission exploratory evidence screening; separate from the frozen OpenML benchmark. This document records evidence standards and the results provided by the R10 outputs. The machine-generated `analysis/` files must pass the read-only audit before final manuscript tables are frozen.

## Case roster and reported model diagnostic results

| Case | Task ID | Role | Retained pairs | Empirically supported rectangles | Mean of mean absolute centered logit contributions | Median absolute four-point mixed difference (logit) | Refit AG-NAM AUROC |
|---|---:|---|---:|---:|---:|---:|---:|
| climate-model-simulation-crashes | 146819 | prelocked numeric | 2 | 2 | 0.71976 | 0.04459 | 0.765000 |
| sick | 3021 | prelocked mixed | 4 | 3 | 0.14516 | 0.00104 | 0.944397 |
| kr-vs-kp | 3 | prelocked categorical | 7 | 6 | 1.76562 | 5.50092 | 0.999843 |
| PhishingWebsites | 14952 | **post-hoc supplementary** categorical | 9 | 9 | 0.70223 | 0.79338 | 0.987409 |

**Totals:** 13 pairs in the original three locked cases; 9 additional post-hoc pairs; 22 model interaction candidates; 20 of 22 with a supported four-corner diagnostic. All four R10 AUROCs match the original corresponding benchmark AUROCs within the predeclared tolerance. Exact decomposition errors were zero in the reported runs. The numeric values in this table are descriptive and are not pooled estimates of real-world interaction strength.

## Independent external evidence — what was and was not corroborated

**Source independently checked:** Lucas, D. D. et al. (2013), *Failure analysis of parameter-induced simulation crashes in climate models*, *Geoscientific Model Development* 6, 1157–1171, DOI [10.5194/gmd-6-1157-2013](https://doi.org/10.5194/gmd-6-1157-2013).

- **Pair `vconst_corr × vconst_2`: published surrogate cross-term identified.** Table 1 identifies parameter numbers 1 and 2, respectively; Table 4 reports a polynomial chaos surrogate term `−0.221 P₁(ξ₁) P₁(ξ₂)` for the predicted log failure probability. This is direct **published model-based pairwise evidence for the same parameters and outcome context**; it is neither ground-truth causality nor proof that AG-NAM recovered the same functional form. AG-NAM's observed four-corner contrast for this pair was `−0.089048` logits on a different fitted model and different scale, and **must not** be compared quantitatively with `−0.221` as if coefficients were commensurable.
- **Pair `convect_corr × bckgrnd_vdc1`: contextual, not directly corroborated.** The same paper's Table 1, Figure 2 and Figure 10 discuss these ocean-mixing parameters and failure behavior, but the screened material has not established a directly matching named cross-term for this exact pair. It remains `pairwise_evidence_not_established_in_screened_sources`.
- **Sick (4 pairs), kr-vs-kp (7 pairs), PhishingWebsites (9 pairs): no independently established exact pairwise interaction from the screened data dictionaries and available documents.** Feature descriptions/context are valuable but not sufficient to mark a pair as independently verified. This is **absence of corroboration in reviewed materials, not evidence that the learned relationship is false**. For chess code abbreviations, no undocumented expansion of the codes should be asserted.

Source pointers for feature definitions: [UCI Thyroid Disease](https://archive.ics.uci.edu/dataset/102/thyroid+disease), [UCI Chess (King-Rook vs. King-Pawn)](https://archive.ics.uci.edu/dataset/22/chess+king+rook+vs+king+pawn), and [UCI Phishing Websites](https://archive.ics.uci.edu/dataset/327/phishing+websites). See `domain_evidence_source_screening_draft.csv` for the recorded URLs and per-pair screening status; URLs alone do not imply evidence of interaction.

## Interpretation and limitations

1. A retained AG-NAM pair is a *model-selected structure*, not an observed fact about the physical system. Exact decomposition is a correctness audit of the fitted computation, not a statistical interaction-validation test.
2. The four-point contrast `g(a,c) − g(a,d) − g(b,c) + g(b,d)` assesses **one** selected contrast in the fitted pair network. It is not an estimate of conditional interaction on the data-generating mechanism, and no inferential p-values or confidence intervals were computed. Supported numeric cells represent broad strata, not guaranteed high density at quartile corner values.
3. Two cases (`sick` and `kr-vs-kp`) contain a pair for which at least one rectangle cell has insufficient support; its mixed difference must remain missing, **not zero**.
4. Pair selection was made on the same benchmark family, and PhishingWebsites was added *after* benchmark inspection. Claims of new independent, preregistered, out-of-distribution or clinician/expert validation are **not** warranted.
5. Twenty-one pairs lack direct corroboration **within the screened sources**. The remaining one has a published *surrogate* cross term, not an independently labeled exhaustive set of interaction ground truth. Do not calculate FNR/recall against these 22 source annotations.
6. No new predictive superiority claim follows from high pairwise logit contrasts. In particular, the climate-model case has equal Main NAM and AG-NAM AUROC (`0.765000`).

## Reproduction and reporting

Run `python scripts/analyze_revision_domain_grounded.py` after confirming all four task directories, original benchmark pair files and `domain_evidence_source_screening_draft.csv` exist. The script verifies original source hashes, per-task trained-model and output hashes, saved test-probability AUROC, pair identities and counts, and source-screening classes; it exports an audited case summary, source-screened evidence matrix, and hash manifest without model fitting. To retain the evidence, **archive ignored `results/revision/domain_grounded_r10/v1/` outputs outside Git**; this Git document and code alone do not contain model checkpoints or complete predictions.
