\# Phase 6G — Real-X Publication Tables and Figures



\## Purpose



Phase 6G converts the already completed and frozen Real-X analyses into

publication-ready tables and figures.



Phase 6G performs no model fitting, no interaction discovery, and no

new inferential analysis.



\## Evidence Gate



Every publication export first validates the Phase 6F-A frozen

evidence manifest.



The publication layer also requires:



\- the Phase 6F-B primary confirmatory result;

\- the Phase 6F-C secondary descriptive manifest.



Both must reference the identical frozen-evidence SHA-256.



\## Confirmatory Boundary



The only Real-X confirmatory hypothesis remains the prespecified

Moderate-condition primary endpoint from Phase 6F-B.



Its one-sided Wilcoxon result may be displayed in the primary table and

primary figure panel.



No secondary p-values, rejection decisions, significance stars, or

additional confirmatory tests are generated.



\## Main Primary Table



The dataset-level primary table reports:



\- OpenML task ID;

\- dataset name;

\- predictor regime;

\- number of features;

\- complete pairwise interaction universe;

\- random-ranking interaction prevalence;

\- mean interaction-ranking AUPRC;

\- AUPRC minus random-ranking prevalence.



Exactly nine dataset rows are reported.



\## Primary Summary Table



The primary summary table reports the prespecified confirmatory result:



\- experimental unit;

\- Moderate condition;

\- lambda = 1.0;

\- nine datasets;

\- five realizations per dataset;

\- Wilcoxon statistic;

\- one-sided p-value;

\- rejection decision;

\- mean dataset-level difference;

\- median dataset-level difference;

\- descriptive 95% percentile bootstrap CI;

\- positive/zero/negative dataset counts.



\## Secondary Strength Table



The signal-strength table reports dataset-level descriptive summaries

for:



\- interaction-ranking AUPRC;

\- ISR sparsification fraction;

\- AG-NAM minus Main NAM AUROC;

\- AG-NAM minus Main NAM AUPRC.



Null interaction-ranking AUPRC remains undefined because the Null

condition contains no active ground-truth interactions.



\## Predictive Comparator Table



The supplementary comparator table reports descriptive differences for:



\- AG-NAM minus Main NAM;

\- AG-NAM minus Random-Pair NAM;

\- AG-NAM minus No-ISR AG-NAM;

\- AG-NAM minus Single-Run AG-NAM;

\- Oracle minus AG-NAM.



Descriptive 95% percentile bootstrap intervals are retained.



No comparator p-values are produced.



\## Feature-Type Table



Numeric, mixed, and categorical predictor-regime summaries are

reported as descriptive subgroup analyses only.



No formal subgroup-effect test is performed.



\## Main Real-X Figure



The principal Real-X figure contains four panels.



\### Panel A



Prespecified primary endpoint:



interaction AUPRC minus random-ranking interaction prevalence



for each of the nine datasets.



Only this panel displays the prespecified confirmatory Wilcoxon result.



\### Panel B



Interaction-ranking AUPRC across:



\- Weak

\- Moderate

\- Strong



signal strengths.



Null is omitted because interaction-ranking AUPRC is undefined when no

ground-truth interaction is active.



\### Panel C



ISR sparsification fraction across:



\- Null

\- Weak

\- Moderate

\- Strong.



\### Panel D



Predictive AUROC difference:



AG-NAM minus Main NAM



across:



\- Null

\- Weak

\- Moderate

\- Strong.



A zero-reference line is displayed.



\## Figure Output Formats



The principal figure is exported as:



\- PNG at 600 dpi;

\- PDF;

\- SVG.



PDF and SVG preserve vector graphics.



\## Publication Output Directory



Tables:



results/realx/benchmark/publication/tables/



Figures:



results/realx/benchmark/publication/figures/



\## Evidence Protection



Phase 6G never writes to:



\- results/realx/benchmark/runs/

\- master\_results.csv

\- benchmark\_progress.csv

\- the Phase 6F-A freeze directory.



Frozen evidence is validated before and after publication export.



\## Interpretation Boundary



Phase 6G is a reporting layer.



It does not alter:



\- frozen data;

\- models;

\- thresholds;

\- candidate K;

\- ISR;

\- primary endpoint;

\- primary test;

\- secondary-analysis status.



Observed secondary patterns remain descriptive.

