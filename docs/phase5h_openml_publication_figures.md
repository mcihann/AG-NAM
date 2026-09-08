\# Phase 5H — Publication-Quality OpenML Figures



\## Objective



Publication figures are generated only after the complete 28-task

OpenML benchmark has been cryptographically frozen.



No model is retrained during figure generation.



\## Evidence Integrity



Before any figure is produced, all evidence files listed in:



configs/openml\_benchmark\_freeze.json



are verified using:



\- file existence

\- byte size

\- SHA-256 digest



The figure-generation process stops if any frozen evidence file has

changed.



\## Figure 1 — Real-World Benchmark Evidence



The principal OpenML benchmark figure contains four panels.



\### Panel A — Paired AUROC Differences



For every locked OpenML dataset:



Delta AUROC =

AG-NAM AUROC - Main NAM AUROC



Datasets are ordered by paired AUROC difference.



The panel additionally reports the prespecified primary:



\- mean paired AUROC difference

\- 95% paired bootstrap confidence interval

\- two-sided paired Wilcoxon p-value



This panel does not imply significant superiority when the

prespecified primary null hypothesis is not rejected.



\### Panel B — Final Interaction-Count Distribution



The panel displays the number of datasets retaining:



0, 1, 2, ...



pairwise interactions after ISR.



This is a structural sparsity result.



It does not represent interaction-recovery accuracy.



\### Panel C — ISR Sparsification–Performance Trade-Off



For datasets where sparsification is defined:



x-axis:

ISR sparsification fraction



y-axis:

AG-NAM AUROC - No-ISR AG-NAM AUROC



The analysis is descriptive.



No post-hoc equivalence or non-inferiority claim is made.



\### Panel D — Predictor-Type Descriptive Analysis



Paired:



AG-NAM AUROC - Main NAM AUROC



is displayed separately for:



\- numeric datasets

\- mixed datasets

\- categorical datasets



These are descriptive subgroup summaries only.



No subgroup confirmatory hypothesis test is introduced.



\## Figure 2 — Locked Real-World Interaction Case Studies



Three case studies are used.



Numeric:



OpenML task 146819

climate-model-simulation-crashes



Mixed:



OpenML task 3021

sick



Categorical:



OpenML task 3

kr-vs-kp



The identities were selected using the previously locked rule:



the earliest registry dataset in each feature-type group containing

at least one ISR-retained interaction.



Predictive performance was not used for case-study selection.



\## Network Definition



Each network contains only ISR-retained interactions.



Node:



feature



Edge:



ISR-retained pairwise interaction



Edge label:



selection frequency across the five locked discovery repetitions



Edge width:



monotonic visual encoding of selection frequency



\## ISR Scores



Numeric ISR scores were not persisted in the frozen task-level pair

audit files.



They are therefore not reconstructed after benchmark freeze.



Case-study figures use only information already available in frozen

evidence:



\- interaction identity

\- selection frequency

\- ISR-retained status



\## Interaction Surfaces



Final case-study model weights and interaction-surface grids were not

persisted during the locked OpenML benchmark.



Models are not retrained after benchmark freeze merely to create

publication figures.



Accordingly, the primary frozen case-study visualization is an

interaction network rather than a reconstructed interaction surface.



Any future surface analysis must be explicitly labeled as an

additional post-freeze illustrative analysis.



\## File Formats



Each main figure is exported as:



\- 600 dpi PNG

\- vector PDF

\- vector SVG



No raster screenshot is used as the archival publication master.



\## Interpretation Boundary



The case-study networks demonstrate compact and reproducibility-filtered

interaction structures.



They do not establish:



\- causal interactions

\- biological ground truth

\- physical ground truth

\- interaction-recovery accuracy on real-world data



Ground-truth structural recovery remains restricted to the synthetic

benchmark and future locked semi-synthetic experiments.

