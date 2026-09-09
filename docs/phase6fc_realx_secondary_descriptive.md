\# Phase 6F-C — Real-X Secondary and Descriptive Analysis



\## Scope



Phase 6F-C begins only after completion of the single prespecified

Real-X confirmatory hypothesis in Phase 6F-B.



All analyses in Phase 6F-C are secondary and descriptive.



No additional confirmatory hypothesis family is introduced.



\## Analysis Gate



Before any secondary result is opened:



1\. the Phase 6F-A evidence freeze must validate;

2\. the Phase 6F-B primary result must exist;

3\. the Phase 6F-B result must reference the identical frozen-evidence

&#x20;  SHA-256;

4\. the primary condition, metric, and statistical test must match the

&#x20;  locked Real-X confirmatory plan.



The frozen evidence is validated again after secondary output

generation.



\## Analysis Unit



Secondary summaries preserve the dataset as the principal reporting

unit.



For each:



dataset × signal strength



the five frozen realizations are first averaged.



Therefore the central secondary table contains:



9 datasets × 4 strengths = 36 dataset-strength observations.



Realization-level observations are not presented as 180 independent

experimental units.



\## Signal Strengths



The four locked conditions are:



\- Null, lambda = 0.0

\- Weak, lambda = 0.5

\- Moderate, lambda = 1.0

\- Strong, lambda = 1.5



The Moderate condition remains the only condition used by the

confirmatory primary hypothesis.



Weak, Strong, and Null summaries remain descriptive.



\## Structure-Recovery Summaries



Descriptive structure summaries include:



\- mean pairwise Jaccard reproducibility;

\- mean interaction-ranking AUPRC;

\- number of selection-stable interactions;

\- number of ISR-retained interactions;

\- ISR sparsification fraction;

\- selection precision;

\- selection recall;

\- selection F1;

\- ISR precision;

\- ISR recall;

\- ISR F1;

\- false-positive reduction.



For runs with no selection-stable interactions, ISR sparsification is

undefined and remains NaN.



For the Null condition, ground-truth interaction-recovery quantities

that require active positives remain undefined.



\## ISR Sparsification



For each realization with at least one selection-stable interaction:



ISR sparsification fraction =



(n\_selection\_stable - n\_isr\_retained)

/

n\_selection\_stable.



This quantity describes the fraction of the reproducible candidate set

removed by ISR.



It is not itself a hypothesis-test statistic.



\## Predictive Summaries



Predictive descriptive summaries include:



\- AUROC;

\- AUPRC;

\- balanced accuracy;

\- F1.



AG-NAM is descriptively compared against:



\- Main-effect NAM;

\- Random-Pair NAM;

\- No-ISR AG-NAM;

\- Single-Run AG-NAM;

\- Oracle Interaction NAM.



Differences are reported using an explicit first-minus-second

orientation.



For example:



AG-NAM minus Main NAM AUROC



and:



Oracle minus AG-NAM AUROC.



\## Dataset-Level Comparator Aggregation



For each dataset and strength, predictive metrics are first averaged

across the five realizations.



Comparator differences are then summarized across the nine datasets.



Reported quantities include:



\- n;

\- mean;

\- standard deviation;

\- median;

\- Q25;

\- Q75;

\- minimum;

\- maximum;

\- positive dataset count;

\- zero dataset count;

\- negative dataset count.



\## Descriptive Bootstrap Confidence Intervals



For predictive comparator mean differences, descriptive 95% percentile

bootstrap confidence intervals are calculated over the nine

dataset-level values.



The locked settings remain:



\- resamples: 10,000

\- seed: 26090806

\- statistic: mean

\- alpha: 0.05



These confidence intervals are descriptive.



They do not create additional confirmatory hypotheses and are not

accompanied by secondary p-values.



\## Feature-Type Summaries



Descriptive subgroup summaries are reported for:



\- numeric;

\- mixed;

\- categorical



predictor regimes.



The feature-type summaries include selected structure and predictive

metrics.



These analyses are descriptive only and must not be interpreted as

formal subgroup-effect tests.



\## Explicit Inferential Boundary



Phase 6F-C produces:



\- no Wilcoxon tests;

\- no t-tests;

\- no Mann-Whitney tests;

\- no Kruskal-Wallis tests;

\- no permutation tests;

\- no secondary p-values;

\- no Holm correction;

\- no FDR correction;

\- no new rejection decisions.



The only confirmatory Real-X p-value remains the prespecified Phase

6F-B primary Wilcoxon result.



\## Outputs



Phase 6F-C produces:



results/realx/benchmark/secondary/dataset\_strength\_values.csv



results/realx/benchmark/secondary/strength\_summary.csv



results/realx/benchmark/secondary/predictive\_comparator\_summary.csv



results/realx/benchmark/secondary/feature\_type\_strength\_summary.csv



results/realx/benchmark/secondary/secondary\_analysis\_manifest.json



\## Evidence Protection



No Phase 6F-C output is written into the frozen evidence directory.



The following files remain unchanged:



\- 180 Phase 6E checkpoints;

\- master\_results.csv;

\- benchmark\_progress.csv;

\- Phase 6F-A evidence manifest.



The frozen aggregate SHA-256 must remain identical before and after the

secondary analysis.



\## Reporting Principle



The secondary analysis is intended to characterize:



\- signal-strength behavior;

\- structure-recovery behavior;

\- ISR sparsification;

\- predictive behavior;

\- comparator gaps;

\- predictor-regime heterogeneity.



Observed patterns may be discussed as descriptive findings.



They must not be retroactively promoted into confirmatory claims.

