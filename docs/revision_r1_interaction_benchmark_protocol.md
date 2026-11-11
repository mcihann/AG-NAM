\# Reviewer Revision R1 — Interaction-Selection Benchmark and ISR Novelty Analysis



\## Status and inferential boundary



This protocol defines additional analyses requested during peer review of the AG-NAM manuscript. These analyses were specified after the original manuscript submission and therefore do not modify, replace, or retroactively redefine any original prespecified or frozen confirmatory analysis.



All original synthetic, Real-X, and OpenML benchmark results remain unchanged.



Results generated under this protocol will be explicitly described as reviewer-requested post-submission analyses. They will not be presented as originally prespecified confirmatory evidence.



\## Objectives



The revision experiments address three questions.



First, does the AG-NAM residual-guided proposer recover known pairwise interactions more effectively or reproducibly than representative alternative interaction-detection mechanisms?



Second, does repeated discovery provide structural information beyond a single fitted interaction detector?



Third, does purified interaction-surface agreement through Interaction Stability Refinement provide structural refinement beyond selection-frequency filtering alone?



\## Evaluation layers



The interaction-recovery benchmark will use only datasets for which pairwise interaction ground truth is known.



The controlled synthetic benchmark will include all four scenarios S1–S4 and all 20 original realizations per scenario, yielding 80 realization-level experiments.



The core Real-X comparison will use the moderate interaction-strength condition (lambda = 1.0) across all nine datasets and five original realizations per dataset, yielding 45 realization-level experiments. The original train, validation, test, generator, and discovery seeds will be preserved.



The Real-X null condition will additionally be used to examine spurious interaction retention and abstention behavior. Because no ground-truth positive interaction class exists under the null condition, interaction-ranking AUPRC will not be computed there.



Weak and strong Real-X conditions may be reported as additional descriptive strength-response analyses but will not replace the moderate-condition core comparison.



Observational OpenML tasks will not be used to calculate interaction-recovery accuracy because true data-generating interaction identities are unknown.



\## Interaction-selection methods



The target comparison set spans several interaction-selection paradigms.



| Method | Role in revision benchmark |

|---|---|

| AG-NAM Grad×Attention proposer | Proposed residual-guided interaction ranking |

| Neural Interaction Detection (NID) | Neural-network-weight interaction detector |

| H-statistic | Model-based interaction-strength baseline |

| iterative Random Forest (iRF) | Tree-based stable interaction detector |

| Sparse Interaction Additive Networks (SIAN) | Sparse neural interaction-selection competitor |

| GAMI-Net | Same-family neural additive interaction competitor |



Official implementations will be preferred where available.



Before comparative outcomes are inspected, every external implementation will undergo a compatibility audit documenting software version, preprocessing requirements, random-seed behavior, supported feature types, output form, and any unavoidable adaptation required for the common benchmark.



A method will not be silently excluded because of unfavorable performance. Any implementation that cannot be executed faithfully will be documented together with the technical reason for exclusion.



\## Common-data principle



All methods will use the same realization-specific training, validation, and held-out test partitions already defined by the original benchmark.



No method may use the held-out test partition for candidate discovery, interaction ranking, hyperparameter selection, early stopping, or stability filtering.



Preprocessing will be fitted exclusively on the corresponding development/training data.



Ground-truth recovery will always be evaluated in the original feature-pair space.



\## Candidate-budget principle



Ranking-based interaction detectors will be evaluated over their complete available pair ranking whenever possible.



For Top-K comparisons, the same candidate budget used by the original AG-NAM benchmark will be applied to all ranking methods.



For the controlled synthetic benchmark:



K = min\[P, min(20, max(5, ceil(0.10P)))],



where P = p(p−1)/2.



For Real-X:



K = min(p−1, 20).



Methods producing a native selected set rather than a complete pair ranking will additionally be evaluated using their native selection output. Native-set metrics and fixed-K ranking metrics will be reported separately rather than conflated.



\## Interaction-recovery metrics



When a complete interaction ranking is available, the principal structural metrics will be interaction-ranking AUPRC, Precision@K, Recall@K, and F1@K.



For selected interaction sets, structural metrics will include precision, recall, F1 score, false-positive count, false-negative count, and retained-set size.



The false-negative rate will be reported whenever ground-truth interactions are available.



No interaction-recovery metric will be computed from observational OpenML data.



\## Repeated-selection reproducibility



Where technically meaningful, interaction discovery will be repeated using the same deterministic seed schedule used by AG-NAM.



For each method, structural reproducibility will be summarized using pairwise Jaccard similarity across repeated selected sets and pairwise selection frequency.



The primary AG-NAM reproducibility configuration remains five discovery runs with a selection-frequency threshold of 0.60.



These values are retained solely as the original method configuration; reviewer-requested run-count and threshold sensitivity analyses are defined separately.



\## ISR novelty ablation



The incremental structural contribution of ISR will be evaluated through the following controlled variants:



| Variant | Definition |

|---|---|

| Selection-stable only | Pairs satisfying selection probability ≥ 0.60 without ISR |

| Strict selection stability | Pairs satisfying selection probability ≥ 0.80 without ISR |

| Count-matched stability | Highest-selection-frequency pairs retained until the number of pairs equals the ISR-retained count |

| Raw-surface agreement | Cross-run interaction-surface agreement without functional-ANOVA purification |

| Purified ISR | Original AG-NAM purified interaction-surface agreement criterion |



The comparison is designed to distinguish improvement caused merely by retaining fewer pairs from improvement specifically associated with purified functional agreement.



Structural outcomes will include precision, recall, F1, false-positive reduction, false-negative rate, and retained-set size.



Where final predictive models are trained for the compared sets, AUROC and AUPRC will be evaluated using the same final AG-NAM architecture and matched training configuration so that the interaction-selection rule, rather than downstream model capacity, is the principal varying factor.



\## Attention-score ablation



The Residual Attention Proposer will be evaluated using three scoring rules derived from the same trained proposer:



Raw attention:



|A\_jk|



Gradient-only attribution:



|∂r\_hat / ∂A\_jk|



AG-NAM gradient-weighted attention:



|A\_jk × ∂r\_hat / ∂A\_jk|



The proposer architecture, data split, optimization, seeds, and residual targets will remain unchanged across scoring rules.



Interaction-ranking AUPRC, Precision@K, Recall@K, F1@K, and repeated-run Jaccard similarity will be compared.



\## Reviewer-requested sensitivity analyses



Selection-threshold sensitivity will initially evaluate:



tau\_selection ∈ {0.40, 0.60, 0.80}.



ISR-threshold sensitivity will initially evaluate:



tau\_ISR ∈ {0.40, 0.60, 0.80}.



For five discovery runs these selection thresholds correspond to retention in at least 2/5, 3/5, and 4/5 runs, respectively.



Run-count sensitivity will evaluate:



B ∈ {3, 5, 10}.



Candidate-budget sensitivity will evaluate the locked candidate budget K0 together with approximately half and twice the locked budget, subject to the available pair universe:



K ∈ {ceil(0.5K0), K0, min(2K0, P)}.



Reference-support sensitivity will evaluate:



N\_ref ∈ {128, 256, 512}.



Sensitivity analyses will report retained-set size, precision, recall, F1, false-positive and false-negative behavior, interaction-ranking/reproducibility summaries where applicable, and downstream AUROC when a final predictive model is fitted.



\## Residual-target sensitivity



The original probability residual



r\_i = y\_i − p\_hat\_i^OOF



will remain the default AG-NAM residual target.



Its interpretation as the negative gradient of binary cross-entropy with respect to the logit will be stated explicitly in the revised manuscript.



A reviewer-requested alternative residual formulation, including a deviance-residual variant, will be evaluated using otherwise matched discovery settings.



The purpose of this analysis is sensitivity assessment rather than replacement of the original residual definition.



\## Computational reporting



For revision experiments, runtime will be recorded whenever technically possible.



AG-NAM computational reporting will distinguish at least main-effect residualization, repeated proposer discovery, ISR pairwise fitting/refinement, final AG-NAM fitting, and total pipeline time.



Peak memory will be profiled separately under a documented hardware/software environment.



Runtime and memory comparisons will be treated descriptively because implementations span different libraries and computational backends.



\## Statistical reporting



No new result from this protocol will be described as an originally prespecified confirmatory analysis.



Reviewer-requested comparisons will primarily report paired effect estimates, descriptive confidence intervals, win/tie/loss summaries, and structural metrics.



If formal hypothesis tests are used for a clearly defined revision-analysis family, the family and multiplicity correction will be declared before inspecting the corresponding comparative results.



The original frozen inferential analyses remain unchanged.



\## Output separation



Revision outputs will be written under:



results/revision/



and will not overwrite any existing synthetic, Real-X, OpenML, freeze, or publication evidence file.



Original frozen benchmark artifacts remain immutable.



Publication-ready revision summaries will be exported separately after the revision analyses are complete.



\## Reproducibility rule



Every revision experiment must record the dataset/task identity, realization, split seed, discovery seed or seed schedule, method name, method version, candidate budget, thresholds, run count, preprocessing specification, structural metrics, predictive metrics when applicable, runtime, and completion status.



No revision experiment may silently fall back to a different configuration after failure.



Any failed or excluded configuration must remain traceable in the revision audit.

