\# Phase 6F-B — Prespecified Real-X Confirmatory Statistics



\## Analysis Gate



Phase 6F-B is permitted only after the Phase 6F-A frozen evidence

manifest independently validates.



The analysis program validates the complete frozen evidence set before

opening master\_results.csv.



After analysis outputs are written, the evidence freeze is validated

again.



A changed evidence SHA-256 causes a hard failure.



\## Frozen Evidence Identity



The confirmatory analysis records:



\- aggregate frozen-evidence SHA-256;

\- Phase 6E benchmark Git commit.



Analysis outputs are downstream products and are not themselves part of

the frozen Phase 6E evidentiary set.



\## Experimental Unit



The prespecified experimental unit is the dataset.



There are exactly nine datasets.



Individual realizations are not treated as independent units for

confirmatory inference.



\## Primary Condition



Only the prespecified Moderate condition is used:



lambda = 1.0



This produces:



9 datasets × 5 realizations = 45 Moderate realization rows.



No Null, Weak, or Strong result enters the primary hypothesis test.



\## Interaction-Ranking Universe



The AG-NAM interaction-recovery implementation computes interaction

AUPRC with sklearn average\_precision\_score over the complete ranked

interaction list.



For p predictors, the unordered pairwise interaction universe is:



P = p(p - 1) / 2.



If m interactions are truly active, the positive prevalence of a random

ranking is:



pi = m / P.



For the Moderate Real-X condition:



m = 3.



Therefore the dataset-specific random-ranking reference is:



pi\_d = 3 / C(p\_d, 2).



\## Realization-Level Primary Metric



For dataset d and realization r:



D\_dr =

interaction AUPRC\_dr

\-

random-ranking interaction prevalence\_d.



The AUPRC value used is the prespecified mean interaction AUPRC across

the five repeated discovery runs within that realization.



\## Dataset-Level Aggregation



The five realization-level primary differences are first averaged:



D\_d =

(1 / 5)

sum\_r D\_dr.



Exactly one D\_d value is therefore produced for each of the nine

datasets.



The resulting nine values are the only observations entering

confirmatory inference.



\## Primary Hypothesis



The single prespecified Real-X primary hypothesis is:



H0:

dataset-level interaction-ranking advantage is not greater than zero.



H1:

AG-NAM interaction ranking performs better than random-ranking

prevalence.



\## Primary Test



The locked test is:



one-sided Wilcoxon signed-rank test



with:



alternative = greater

zero\_method = wilcox

correction = false

method = auto



Values numerically equal to zero within absolute tolerance 1e-12 are

removed before testing, consistent with the existing project

Wilcoxon implementation.



Alpha:



0.05



\## Multiplicity



There is exactly one confirmatory Real-X hypothesis.



Therefore:



multiplicity adjustment = none.



No secondary comparison may be promoted into a confirmatory family

after results are inspected.



\## Descriptive Effect Summary



Alongside the confirmatory Wilcoxon test, Phase 6F-B reports:



\- mean dataset-level primary difference;

\- median dataset-level primary difference;

\- number of positive datasets;

\- number of zero datasets;

\- number of negative datasets;

\- minimum dataset-level difference;

\- maximum dataset-level difference.



These quantities are descriptive.



\## Bootstrap Confidence Interval



A descriptive 95% percentile bootstrap confidence interval is computed

for the mean of the nine dataset-level primary differences.



The locked bootstrap settings are:



\- resamples: 10,000

\- seed: 26090806

\- statistic: mean

\- alpha: 0.05



Resampling occurs over the nine complete dataset-level values.



The bootstrap confidence interval does not replace the prespecified

Wilcoxon hypothesis test.



\## Primary Outputs



Phase 6F-B produces:



results/realx/benchmark/statistics/primary\_realization\_values.csv



results/realx/benchmark/statistics/primary\_dataset\_values.csv



results/realx/benchmark/statistics/primary\_confirmatory\_result.json



results/realx/benchmark/statistics/analysis\_manifest.json



\## Secondary Analysis Boundary



All remaining Real-X analyses are secondary and descriptive.



This includes, but is not limited to:



\- Weak results;

\- Strong results;

\- Null results;

\- predictive AUROC comparisons;

\- predictive AUPRC comparisons;

\- balanced accuracy;

\- F1;

\- Oracle comparisons;

\- Random-Pair comparisons;

\- No-ISR comparisons;

\- Single-Run comparisons;

\- ISR sparsification;

\- feature-type subgroup summaries;

\- dataset-level case studies.



No additional confirmatory hypothesis family is introduced.



\## Method Lock



Phase 6F-B does not modify:



\- the 180 frozen checkpoints;

\- master\_results.csv;

\- benchmark\_progress.csv;

\- model architecture;

\- optimizer settings;

\- discovery seeds;

\- candidate K;

\- selection threshold;

\- ISR threshold;

\- interaction strengths;

\- realization count;

\- dataset selection;

\- primary endpoint;

\- primary test;

\- alpha;

\- bootstrap settings.



The phase only evaluates the prespecified hypothesis against the

already frozen evidence.

