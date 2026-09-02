\# Phase 4D — Locked Statistical Analysis Protocol



\## Objective



This protocol defines the statistical analysis of the primary

multi-realization synthetic AG-NAM benchmark.



It is fixed after completion of the S1 benchmark but before inspection

of S2-S4 benchmark results.



No AG-NAM architecture, threshold, interaction-scoring rule, or

synthetic scenario definition may be modified based on inferential

results.



\## Experimental Unit



The statistical experimental unit is one independently generated

synthetic realization.



Each scenario contains:



n = 20 independent realizations



Models compared within one realization use the same untouched test

subset.



Therefore, model comparisons are paired across realizations.



Individual observations inside a synthetic realization are not treated

as independent statistical replicates for model-comparison inference.



\## Descriptive Reporting



For scenario-level metrics, report at minimum:



\- mean

\- standard deviation

\- median

\- 25th percentile

\- 75th percentile

\- interquartile range



\## Primary Predictive Endpoint



Primary metric:



AUROC



Primary model comparison:



Full AG-NAM versus Main-Effect NAM



Paired difference:



Delta AUROC =

&#x20;   AUROC\_AG-NAM - AUROC\_Main-NAM



Positive differences favor AG-NAM.



\## Primary Inferential Family



The complete synthetic benchmark contains four scenario-specific

primary tests:



1\. S1: AG-NAM versus Main NAM, AUROC

2\. S2: AG-NAM versus Main NAM, AUROC

3\. S3: AG-NAM versus Main NAM, AUROC

4\. S4: AG-NAM versus Main NAM, AUROC



Each comparison uses a two-sided paired Wilcoxon signed-rank test.



Family-wise error is controlled across these four primary tests using

the Holm step-down procedure.



Alpha:



0.05



Confirmatory Holm-adjusted primary inference is not produced until all

four scenarios have completed all 20 locked realizations.



\## Secondary Inferential Family



Five secondary comparisons are predefined within every scenario:



1\. AG-NAM versus Main NAM, AUPRC

2\. AG-NAM versus Random-Pair NAM, AUROC

3\. AG-NAM versus Random-Pair NAM, AUPRC

4\. AG-NAM versus Single-Run AG-NAM, AUROC

5\. AG-NAM versus Single-Run AG-NAM, AUPRC



Across four scenarios this yields:



20 secondary tests



The complete set of 20 secondary tests forms one secondary

family.



Holm correction is applied across all 20 tests.



Secondary adjusted inference is not produced until all four scenarios

have completed all 20 realizations.



\## Wilcoxon Test



The paired Wilcoxon signed-rank test is two-sided.



Zero paired differences are removed from the signed-rank calculation.



The effective non-zero sample size is reported.



No choice between parametric and non-parametric testing is made after

inspection of normality tests.



\## Effect Size



Every paired inferential comparison reports paired rank-biserial

correlation.



For differences defined as:



model A - model B



positive rank-biserial values favor model A.



The effect size is calculated as:



RBC =

&#x20;   (W\_positive - W\_negative)

&#x20;   /

&#x20;   (W\_positive + W\_negative)



where W\_positive and W\_negative are sums of ranks of the absolute

non-zero paired differences.



\## Paired Differences



For every comparison report:



\- mean paired difference

\- standard deviation of paired differences

\- median paired difference

\- 25th percentile of paired differences

\- 75th percentile of paired differences

\- win rate

\- loss rate

\- tie rate



\## Confidence Intervals



A paired percentile bootstrap is used for confidence intervals of:



\- mean paired difference

\- median paired difference



Number of bootstrap resamples:



10000



Bootstrap seed:



20260902



Confidence level:



95%



Bootstrap resampling occurs at the realization level.



For each resample, complete paired model outcomes from one realization

are sampled together.



Individual observations within a realization are not independently

bootstrapped for the cross-realization model comparison.



\## Oracle Interaction NAM



Oracle Interaction NAM is a synthetic upper-reference model.



No superiority hypothesis test is performed against the oracle model.



The following are reported descriptively:



Oracle AUROC - AG-NAM AUROC



and:



Oracle gain captured =

&#x20;   (AG-NAM AUROC - Main NAM AUROC)

&#x20;   /

&#x20;   (Oracle AUROC - Main NAM AUROC)



The gain-capture ratio is undefined when the oracle-minus-main

denominator is numerically zero.



\## ISR Versus No-ISR



Full AG-NAM versus AG-NAM without ISR is not defined as a superiority

hypothesis test.



ISR is primarily intended to improve structural specificity and

sparsity while preserving useful predictive information.



The following are therefore reported descriptively:



AG-NAM AUROC - No-ISR AUROC



interaction-set reduction from the selection-stable set to the

ISR-retained set



selection-stage precision and recall



ISR-stage precision and recall



false-positive reduction after ISR



No formal equivalence or non-inferiority claim is made without a

separately justified equivalence margin.



\## Single-Run Comparison



Single-Run AG-NAM is a prespecified stability ablation.



Predictive AUROC and AUPRC comparisons against Full AG-NAM belong to

the secondary inferential family.



Interaction-set size is also compared descriptively.



\## Random-Pair Control



Random-Pair NAM controls for additional pairwise neural-network

capacity.



Predictive AUROC and AUPRC comparisons against Full AG-NAM belong to

the secondary inferential family.



Random interaction selection does not use ground-truth interaction

information.



\## Structure Recovery



Structure recovery is summarized separately for every scenario.



Primary descriptive structure outcomes include:



\- interaction-ranking AUPRC

\- mean pairwise Top-K Jaccard similarity

\- selection precision

\- selection recall

\- selection F1

\- ISR precision

\- ISR recall

\- ISR F1

\- false-positive reduction

\- number of selection-stable interactions

\- number of ISR-retained interactions



Synthetic ground truth is used only for structure-recovery evaluation

and for the explicitly labeled oracle control.



\## No Post-Hoc Threshold Selection



The following locked method parameters are not modified using the

synthetic benchmark results:



B = 5



candidate-set rule



selection threshold = 0.60



ISR threshold = 0.60



residual cross-fitting folds = 5



reference size = min(512, N)



attention attribution definition



final AG-NAM architecture



optimization settings



\## Interim Scenario Results



Completed scenarios may be inspected descriptively for engineering

validation.



Raw paired statistics may be calculated.



However, confirmatory Holm-adjusted primary and secondary inference is

deferred until the complete S1-S4 benchmark contains all 80 predefined

realizations.



S1 results are therefore not used to modify the locked method before

S2-S4 execution.

