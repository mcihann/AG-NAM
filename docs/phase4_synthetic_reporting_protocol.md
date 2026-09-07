\# Phase 4E — Final Synthetic Benchmark Reporting



\## Status



The locked primary synthetic benchmark contains:



4 scenarios



20 independent realizations per scenario



80 total realizations



All 80 predefined realizations have completed successfully.



\## Source of Truth



The primary source of benchmark evidence is:



results/synthetic/master\_results.csv



This file is derived from the immutable per-realization checkpoints.



Publication tables are regenerated from the master benchmark table

using the locked statistical-analysis protocol.



No individual realization is removed based on model performance.



\## Predictive Publication Table



The predictive table reports, by scenario:



\- Main NAM AUROC

\- AG-NAM AUROC

\- paired AUROC difference

\- 95% paired bootstrap confidence interval

\- paired rank-biserial effect size

\- win rate

\- raw Wilcoxon p-value

\- Holm-adjusted primary p-value

\- Main NAM AUPRC

\- AG-NAM AUPRC

\- paired AUPRC difference

\- Oracle AUROC

\- Random-Pair AUROC

\- No-ISR AUROC

\- Single-Run AUROC



\## Structure Publication Table



The structure table reports, by scenario:



\- interaction-ranking AUPRC

\- Top-K Jaccard similarity

\- selection precision

\- selection recall

\- selection F1

\- ISR precision

\- ISR recall

\- ISR F1

\- false-positive reduction

\- selection-stable interaction count

\- ISR-retained interaction count

\- ISR sparsification relative to selection-stable interactions

\- ISR sparsification relative to the single-run candidate set



Undefined ratio-based metrics remain missing rather than being

artificially replaced by zero.



The number of valid observations is therefore retained for metrics

whose denominator may be zero.



\## Confirmatory Inference



The primary inferential family contains:



4 scenario-specific AG-NAM versus Main NAM AUROC comparisons.



Holm correction controls family-wise error across the four primary

tests.



The secondary inferential family contains:



20 prespecified tests.



Holm correction controls family-wise error across the complete

secondary family.



\## Oracle Interpretation



Oracle Interaction NAM is an upper-reference control.



It is not treated as a deployable competing method.



Oracle gain capture is reported descriptively.



\## ISR Interpretation



ISR is interpreted primarily as a structural specificity and

sparsification mechanism.



No superiority claim against No-ISR is made solely from negligible

predictive differences.



\## Single-Run Interpretation



Full AG-NAM is not claimed to be predictively superior to Single-Run

AG-NAM unless supported by the locked multiplicity-adjusted tests.



The principal value of repeated stability filtering is assessed

through compactness, structural filtering, and predictive

preservation.



\## Random-Pair Interpretation



Random-Pair NAM controls for additional pairwise neural-network

capacity.



Significant Full AG-NAM versus Random-Pair differences support the

interpretation that predictive gains are not explained solely by

adding pairwise subnetworks.



\## Numerical Reproducibility



The publication exporter uses:



10000 paired bootstrap resamples



bootstrap seed:



20260902



alpha:



0.05



The exporter does not refit any predictive model.



The 80 synthetic experiments are not rerun during table generation.

