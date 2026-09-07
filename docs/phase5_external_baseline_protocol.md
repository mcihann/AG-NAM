\# Phase 5C — Locked External Baseline Protocol



\## Objective



External real-world comparators are fixed before AG-NAM OpenML

predictive results are inspected.



The external baseline set contains:



1\. Explainable Boosting Machine

2\. CatBoost



These models serve different methodological roles and are not selected

post hoc according to AG-NAM performance.



\## Explainable Boosting Machine



Implementation:



InterpretML ExplainableBoostingClassifier



Locked package:



interpret-core == 0.7.8



EBM is the principal external interpretable interaction-aware

comparator.



EBM contains nonlinear univariate terms and automatically selected

pairwise interaction terms.



\## EBM Interaction Budget



EBM receives the same interaction candidate budget rule used by

AG-NAM:



P = p(p - 1) / 2



K = clip(

&#x20;   ceil(0.10 x P),

&#x20;   5,

&#x20;   20

)



K is additionally capped by the number of possible feature pairs.



EBM therefore is not artificially constrained to the generally much

smaller post-ISR AG-NAM interaction set.



This creates a strong rather than deliberately weak interpretable

baseline.



\## EBM Training



Locked principal settings:



validation\_size = 0.15



outer\_bags = 14



inner\_bags = 0



learning\_rate = 0.015



max\_rounds = 50000



early\_stopping\_rounds = 100



early\_stopping\_tolerance = 1e-5



max\_bins = 1024



max\_interaction\_bins = 64



min\_samples\_leaf = 4



max\_leaves = 2



Each OpenML task receives a predefined deterministic EBM random seed.



EBM performs its own training-derived internal validation.



The untouched OpenML test fold is never used during EBM fitting or

interaction selection.



\## CatBoost



Locked package:



catboost == 1.2.10



CatBoost is the principal strong black-box tabular predictive

reference.



Its purpose is not explanation but assessment of how much predictive

performance is sacrificed, if any, by the interpretable AG-NAM

framework.



\## CatBoost Processing Unit



CatBoost is locked to:



task\_type = CPU



The GPU implementation is not used for the primary benchmark because

CatBoost GPU training is nondeterministic due to floating-point

summation order.



\## CatBoost Training



Locked principal settings:



iterations = 2000



learning\_rate = 0.03



depth = 6



l2\_leaf\_reg = 3.0



loss\_function = Logloss



eval\_metric = AUC



early\_stopping\_rounds = 100



bootstrap\_type = Bayesian



bagging\_temperature = 1.0



random\_strength = 1.0



CatBoost uses the locked final training subset for fitting and the

locked final validation subset for early stopping.



The untouched OpenML test fold is used only for final evaluation.



Categorical predictors are supplied to CatBoost as categorical

features rather than one-hot encoded predictors.



Numeric missing values are retained for native CatBoost handling.



Categorical missing values are converted deterministically to an

explicit missing category before fitting.



\## Real-World Primary Inference



Experimental unit:



OpenML dataset



Number of locked datasets:



28



Primary metric:



AUROC



Primary comparison:



Full AG-NAM versus Main-Effect NAM



A single two-sided paired Wilcoxon signed-rank test is used for the

primary real-world endpoint.



Alpha:



0.05



Because there is one prespecified primary real-world comparison, no

multiplicity adjustment is required for the primary endpoint.



\## Secondary Inferential Family



The secondary real-world family contains nine prespecified paired

tests:



1\. AG-NAM versus Main NAM — AUPRC

2\. AG-NAM versus EBM — AUROC

3\. AG-NAM versus EBM — AUPRC

4\. AG-NAM versus CatBoost — AUROC

5\. AG-NAM versus CatBoost — AUPRC

6\. AG-NAM versus Random-Pair NAM — AUROC

7\. AG-NAM versus Random-Pair NAM — AUPRC

8\. AG-NAM versus Single-Run AG-NAM — AUROC

9\. AG-NAM versus Single-Run AG-NAM — AUPRC



Holm correction controls family-wise error across these nine secondary

tests.



Balanced Accuracy and F1 are reported descriptively.



\## No-ISR Ablation



Full AG-NAM versus No-ISR AG-NAM is treated primarily as a

sparsification and predictive-preservation analysis.



No superiority, equivalence, or non-inferiority hypothesis is claimed

without a separately justified margin.



\## Bootstrap Reporting



Paired differences are summarized using:



10000 paired bootstrap resamples



bootstrap seed:



20260907



confidence level:



95%



Bootstrap resampling occurs over complete OpenML datasets.



Individual observations within a dataset are not treated as

independent cross-dataset replicates.



\## Ground-Truth Interaction Claims



Neither EBM nor AG-NAM interaction identities on real OpenML datasets

are treated as verified ground truth.



Synthetic interaction-recovery evidence remains the source of

ground-truth structural validation.



\## No Post-Hoc Baseline Tuning



External baseline identities and principal settings are fixed before

the AG-NAM OpenML predictive benchmark is executed.



Baseline hyperparameters are not subsequently altered according to

which configuration produces the most favorable comparison with

AG-NAM.

