\# Phase 5B — OpenML Pre-Benchmark Data Audit



\## Objective



Before training any predictive model on the locked real-world

benchmark, every OpenML task is independently loaded and audited.



The audit does not inspect AG-NAM performance.



\## Locked Task Identity



The primary identity of a benchmark item is the OpenML task ID.



Dataset names are retained as descriptive metadata.



A cosmetic dataset-name discrepancy is recorded but is not by itself

a reason to replace or remove a locked task.



\## Data Retrieval



For each locked task:



1\. retrieve the OpenML task

2\. retrieve its associated dataset

3\. retrieve predictors and target using the task target attribute

4\. retrieve the predefined task split



No custom replacement train/test split is introduced.



\## Outer Split



The locked split is:



repeat = 0



fold = 0



sample = 0



Development and test indices must:



\- be non-empty

\- contain no duplicates

\- contain no overlap

\- be within dataset bounds

\- together cover the complete task dataset



\## Target Audit



The complete task must contain exactly two observed target classes.



The outer-development subset must contain both classes.



The untouched test subset must contain both classes so that AUROC and

AUPRC are well defined.



Missing target labels are not permitted.



\## Positive-Class Lock



The positive class is determined exclusively from outer-development

labels.



The minority development class becomes the positive class.



If development class counts are tied, the lexicographically smaller

string representation is selected deterministically.



The untouched test distribution is not used in this choice.



The selected positive and negative labels are persisted in the audit

table.



\## Predictor Audit



Observed predictor count is compared against the locked registry.



OpenML's categorical indicator is used to audit:



\- numeric predictor count

\- categorical predictor count

\- overall feature type



The resulting feature type must match one of:



numeric



mixed



categorical



\## Missingness



Missing predictor values are permitted.



For every task the audit records:



\- number of predictors containing missing values

\- overall predictor missing-value fraction



Missing values are handled later only within leakage-safe

training-derived preprocessing.



\## Registry Validation



The following must match the locked registry:



\- sample count

\- predictor count

\- numeric predictor count

\- categorical predictor count

\- feature type



No mismatch is silently corrected.



\## Audit Output



The complete audit is saved to:



results/openml/audit/openml\_task\_audit.csv



Technical loading failures are saved to:



results/openml/audit/openml\_task\_audit\_failures.csv



\## Benchmark Eligibility



The real-world benchmark proceeds only after all 28 locked tasks pass

the pre-benchmark audit.



A failed task is not silently removed or replaced.



Any genuine incompatibility must be examined and documented before

the benchmark protocol proceeds.



\## No Model Training



This phase performs:



\- no residual fitting

\- no attention fitting

\- no interaction discovery

\- no ISR fitting

\- no final NAM fitting

\- no AG-NAM fitting

\- no external-baseline fitting



Therefore Phase 5B remains a pure data/protocol audit.

