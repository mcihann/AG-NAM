\# Phase 6E — Locked Real-X Benchmark Execution



\## Objective



Phase 6E executes the final evidentiary Real-X semi-synthetic

benchmark after completion of all generator and integration audits.



The benchmark contains:



9 datasets × 5 realizations × 4 signal strengths = 180 locked runs.



\## Evidence Boundary



Phase 6D integration-sanity outputs are excluded.



Only checkpoints written beneath:



results/realx/benchmark/runs/



may enter the Phase 6E master result table.



Every accepted checkpoint must contain:



publication\_eligible = true



\## Locked Run Schedule



For every one of the nine prespecified OpenML datasets:



\- five realizations are used;

\- each realization contains Null, Weak, Moderate, and Strong

&#x20; interaction conditions.



The four strength conditions within a dataset-realization share the

same:



\- sampled rows;

\- feature-state encoding;

\- ground-truth interaction template;

\- train/validation/test split;

\- Bernoulli random-number stream;

\- discovery seed family;

\- random-pair seed;

\- final-model seed.



Only the locked interaction coefficient differs across strength

conditions.



\## Candidate Count



Real-X uses:



K = min(p - 1, 20)



The checkpoint is invalid unless both the expected and observed

candidate counts match this rule.



\## Split Ownership



The Real-X generator owns the split.



No additional outer split is permitted.



Final prediction uses:



\- locked training partition for optimization;

\- locked validation partition for early stopping;

\- locked untouched test partition for final metrics.



Interaction discovery uses only the locked training + validation

development pool.



\## Per-Run Pipeline



Each benchmark run performs:



1\. locked real-X semi-synthetic generation;

2\. five-run leakage-safe interaction discovery;

3\. selection reproducibility;

4\. ISR filtering;

5\. Main NAM;

6\. Full AG-NAM;

7\. Oracle Interaction NAM;

8\. Random-Pair NAM;

9\. AG-NAM without ISR;

10\. Single-Run AG-NAM;

11\. untouched-test evaluation;

12\. exact additive-decomposition audit;

13\. test-integrity audit.



\## Checkpointing



Every completed run is written independently.



A checkpoint is accepted only if:



\- schema version matches;

\- benchmark ID matches;

\- run identity matches the locked schedule;

\- status is COMPLETE;

\- publication\_eligible is true;

\- untouched-test integrity is true;

\- candidate K is correct;

\- decomposition error is <= 1e-6;

\- the active true-interaction count matches the signal condition.



Checkpoint creation is atomic.



An interrupted process therefore cannot silently create a valid

partial checkpoint.



\## Resume Behavior



On restart:



\- valid checkpoints are skipped;

\- missing checkpoints are executed;

\- invalid checkpoints are re-executed;

\- failures are logged independently.



Thus machine shutdown, CUDA errors, or process interruption do not

require restarting previously validated runs.



\## Aggregate Tables



The benchmark root contains:



\- master\_results.csv

\- benchmark\_progress.csv

\- failures.csv

\- runs/\*.csv



master\_results.csv contains only individually validated Phase 6E

checkpoints.



\## Null Condition



The Null condition contains no active ground-truth interaction.



Consequently interaction-ranking AUPRC and recall-style recovery

metrics are undefined and are stored as NaN.



False-positive counts and sparsification behavior remain meaningful.



\## Confirmatory Analysis Gate



No final confirmatory Real-X statistical analysis may be performed

until:



180 / 180



locked checkpoints pass validation.



No intermediate result may be used to modify:



\- architecture;

\- learning rate;

\- thresholds;

\- candidate count;

\- ISR configuration;

\- dataset selection;

\- realization count;

\- signal strength;

\- comparator definition;

\- primary endpoint;

\- statistical test.



\## Dry Run



Before any Phase 6E training is started, the execution plan must

report:



Selected: 180

Skipped existing: 0

Invalid existing: 0

Planned: 180



assuming no prior Phase 6E checkpoint exists.



\## Initial Checkpoint Validation



Before launching all 180 runs, exactly one new Phase 6E run is used

to validate checkpoint creation and resume behavior.



The Phase 6D moderate sanity condition is not reused as the initial

checkpoint validation run.



After the checkpoint mechanism is verified, the full locked schedule

is executed without further protocol modification.

