\# Phase 5G — Final OpenML Reporting and Benchmark Freeze



\## Status



The locked real-world benchmark contains:



28 OpenML binary classification tasks



Feature-type composition:



15 numeric



10 mixed



3 categorical



All 28 tasks have completed successfully.



\## Benchmark Freeze



No further modification is permitted to the primary OpenML benchmark

with the purpose of improving AG-NAM performance.



The following remain locked:



\- task registry

\- OpenML predefined split

\- target encoding rule

\- neural architecture

\- candidate budget

\- repeated discovery count

\- residual cross-fitting count

\- selection threshold

\- ISR threshold

\- EBM configuration

\- CatBoost configuration

\- primary endpoint

\- confirmatory comparison family



\## Primary Real-World Result



The prespecified primary comparison is:



Full AG-NAM versus Main-Effect NAM



Metric:



AUROC



Statistical experimental unit:



OpenML dataset



The confirmatory result is retained regardless of whether the null

hypothesis is rejected.



No rerun, dataset removal, or method modification is permitted in

response to the primary p-value.



\## Publication Tables



The final reporting stage exports:



\- task-level predictive results

\- cross-dataset model summaries

\- paired inferential results

\- task-level structure diagnostics

\- feature-type descriptive summaries

\- sparsity-performance diagnostics

\- ISR-retained interaction-count distribution

\- locked illustrative case-study selection

\- benchmark completion table



\## Structure Interpretation



Real-world interaction identities do not have verified ground truth.



Therefore the following are interpreted as structural diagnostics:



\- pairwise Top-K stability

\- number of selection-stable interactions

\- number of ISR-retained interactions

\- ISR sparsification

\- selected interaction identities



No real-world interaction precision or recall is reported.



\## Sparsity-Performance Analysis



For every dataset the final reporting table records:



AG-NAM minus Main NAM AUROC



AG-NAM minus No-ISR AUROC



AG-NAM minus Single-Run AG-NAM AUROC



together with:



\- selection-stable interaction count

\- ISR-retained interaction count

\- ISR sparsification



This analysis is descriptive.



No post-hoc equivalence or non-inferiority claim is introduced.



\## Feature-Type Analysis



Numeric, mixed, and categorical datasets are summarized separately.



These subgroup summaries are descriptive only.



No subgroup confirmatory hypothesis testing is introduced after

inspection of the completed benchmark.



\## Illustrative Case Studies



Three real-world illustrative case studies are selected:



\- one numeric

\- one mixed

\- one categorical



The selection rule is fixed before interaction identities and

case-study surfaces are inspected.



For each feature type:



1\. order datasets according to the locked registry

2\. select the earliest dataset with at least one ISR-retained

&#x20;  interaction

3\. if none retains an interaction, select the earliest registry item

&#x20;  as a transparent fallback



Predictive performance is not used for case-study selection.



Case studies are illustrative and are not additional confirmatory

experiments.



\## Cryptographic Freeze



The completed benchmark is frozen using SHA-256 hashes.



The freeze manifest includes:



\- master\_results.csv

\- all 28 primary task checkpoint files

\- all 28 task-level pair-audit files



The manifest is stored at:



configs/openml\_benchmark\_freeze.json



Any modification to a frozen evidence file changes its SHA-256 digest

and is therefore detectable.



\## Future Method Development



Any future AG-NAM modification must be treated as a new method version.



The completed 28-task OpenML benchmark cannot be reused as an

untouched confirmatory benchmark for a method version developed after

inspection of these results.

