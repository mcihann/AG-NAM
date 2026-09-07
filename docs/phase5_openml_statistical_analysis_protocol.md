\# Phase 5F — Locked OpenML Statistical Analysis Protocol



\## Experimental Unit



The statistical experimental unit is one locked OpenML dataset.



The primary real-world benchmark contains 28 independent dataset-level

benchmark outcomes.



Individual observations within one dataset are not treated as

independent replicates for cross-dataset inference.



\## Primary Endpoint



Metric:



AUROC



Comparison:



Full AG-NAM versus Main-Effect NAM



Paired difference:



AUROC\_AG-NAM - AUROC\_Main-NAM



Positive differences favor AG-NAM.



\## Primary Statistical Test



One two-sided paired Wilcoxon signed-rank test is prespecified across

the 28 dataset-level paired AUROC outcomes.



Alpha:



0.05



Because exactly one primary real-world hypothesis test is defined, no

multiplicity correction is required for the primary endpoint.



\## Secondary Inferential Family



The secondary family contains exactly nine prespecified comparisons:



1\. AG-NAM versus Main NAM — AUPRC

2\. AG-NAM versus EBM — AUROC

3\. AG-NAM versus EBM — AUPRC

4\. AG-NAM versus CatBoost — AUROC

5\. AG-NAM versus CatBoost — AUPRC

6\. AG-NAM versus Random-Pair NAM — AUROC

7\. AG-NAM versus Random-Pair NAM — AUPRC

8\. AG-NAM versus Single-Run AG-NAM — AUROC

9\. AG-NAM versus Single-Run AG-NAM — AUPRC



Holm step-down correction controls family-wise error across the nine

secondary tests.



Alpha:



0.05



\## No-ISR Analysis



Full AG-NAM versus No-ISR AG-NAM is not included in the confirmatory

inferential family.



It is treated as a descriptive structural-sparsification and

predictive-preservation analysis.



No formal equivalence or non-inferiority claim is made.



\## Paired Effect Reporting



For every model comparison report:



\- number of datasets

\- mean model-A metric

\- mean model-B metric

\- mean paired difference

\- standard deviation of paired differences

\- median paired difference

\- 25th percentile

\- 75th percentile

\- interquartile range

\- 95% paired bootstrap confidence interval for the mean difference

\- paired rank-biserial correlation

\- win rate

\- loss rate

\- tie rate



\## Bootstrap



Paired bootstrap resampling is performed at the dataset level.



Number of resamples:



10000



Bootstrap seed:



20260907



Confidence level:



95%



Complete paired dataset outcomes are sampled together.



Rows within a dataset are never independently bootstrapped for this

cross-dataset analysis.



\## Wilcoxon Zero Differences



Zero paired differences are excluded from the signed-rank statistic.



The effective non-zero sample size is retained.



\## Interim Benchmark Results



Descriptive paired differences, confidence intervals, effect sizes,

and win rates may be generated during execution for pipeline

verification.



Confirmatory p-values are not exposed until all 28 locked OpenML tasks

have validated benchmark checkpoints.



This prevents intermediate hypothesis-testing results from being used

to alter:



\- AG-NAM architecture

\- interaction thresholds

\- ISR configuration

\- OpenML registry

\- EBM settings

\- CatBoost settings

\- comparison family



\## Real-World Structure Diagnostics



Without verified ground-truth interactions, the benchmark reports

descriptively:



\- mean pairwise Top-K Jaccard

\- number of selection-stable interactions

\- number of ISR-retained interactions

\- ISR sparsification

\- AG-NAM versus No-ISR predictive difference



These diagnostics are additionally summarized descriptively by:



\- all datasets

\- numeric datasets

\- mixed datasets

\- categorical datasets



No subgroup confirmatory hypothesis tests are defined.



\## Ground-Truth Restriction



Real-world interaction identities are not interpreted as verified

causal or ground-truth interactions.



Ground-truth structural recovery claims remain restricted to the

locked synthetic benchmark.

