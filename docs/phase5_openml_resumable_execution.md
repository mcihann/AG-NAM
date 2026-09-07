\# Phase 5E — Resumable OpenML Benchmark Execution



\## Objective



The primary real-world benchmark contains 28 computationally

expensive OpenML tasks.



Execution is resumable and checkpointed.



Successful task results must survive:



\- Python interruption

\- machine restart

\- individual task failure

\- later benchmark resumption



\## Immutable Task Checkpoint



Each successfully completed task is stored as:



results/openml/benchmark/task\_<task\_id>.csv



Its interaction audit is stored as:



results/openml/benchmark/task\_<task\_id>\_pairs.csv



The primary task CSV acts as the completion checkpoint.



\## Canonical Task Adoption



Phase 5D already completed OpenML task 49 before the resumable runner

was introduced.



That result is not recomputed.



The runner may adopt:



results/openml/single/task\_49.csv



and:



results/openml/single/task\_49\_pairs.csv



into the primary benchmark.



During adoption the result is enriched with locked:



\- task index

\- AG-NAM seeds

\- final split seed

\- EBM seed

\- CatBoost seed

\- OpenML split identity

\- selection threshold

\- ISR threshold

\- package versions

\- protocol identifier



No predictive metric is changed.



\## Checkpoint Validation



Before an existing checkpoint is skipped, the runner verifies:



\- task ID

\- task index

\- discovery base seed

\- discovery seed stream

\- random-pair seed

\- final split seed

\- final model seed

\- EBM seed

\- CatBoost seed

\- OpenML repeat

\- OpenML fold

\- OpenML sample

\- discovery repetition count

\- residual cross-fitting count

\- selection threshold

\- ISR threshold

\- EBM version

\- CatBoost version

\- benchmark protocol identifier



A mismatched checkpoint is treated as an error.



It is not silently reused.



\## No Overwrite



A valid completed task checkpoint is never overwritten during normal

benchmark execution.



\## Registry-Range Execution



The benchmark can be executed by registry position.



Example:



start = 1



end = 5



runs the first five locked registry tasks.



A specific OpenML task ID may also be selected.



\## Failure Handling



A task-level exception is written to:



results/openml/benchmark/failures.csv



By default, the batch proceeds to the next selected task.



A successfully rerun task removes its stale failure entry.



\## Master Results



After every successful or adopted task the complete master table is

rebuilt from immutable primary task checkpoints:



results/openml/benchmark/master\_results.csv



Pair-audit files are excluded.



\## Benchmark Progress



Completion status for every locked task is stored in:



results/openml/benchmark/benchmark\_progress.csv



The benchmark is complete only when:



28 / 28



locked tasks contain validated primary checkpoints.



\## Dry Run



Dry-run mode performs no model fitting and does not copy canonical

results.



Possible states are:



SKIP



validated primary checkpoint already exists



ADOPT



a canonical Phase 5D result is available for adoption



PLAN



the task remains to be executed



\## Statistical Integrity



No task is removed because AG-NAM performs poorly.



No failed task is silently replaced with another dataset.



No intermediate real-world predictive result is used to modify:



\- task registry

\- AG-NAM architecture

\- candidate rule

\- selection threshold

\- ISR threshold

\- EBM configuration

\- CatBoost configuration

\- predefined inferential family

