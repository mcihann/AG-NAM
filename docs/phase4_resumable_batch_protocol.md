\# Phase 4C — Resumable Synthetic Benchmark Execution



\## Objective



The primary synthetic benchmark contains 80 computationally expensive

realizations.



Execution must therefore be resumable and checkpointed.



A machine restart, individual realization failure, or interrupted

process must not require successful realizations to be recomputed.



\## Primary Benchmark Size



Scenarios:



S1

S2

S3

S4



Independent realizations per scenario:



20



Total:



80 realizations



\## Realization Checkpoint



Immediately after one complete realization finishes successfully, its

SyntheticBenchmarkRecord is written to:



results/synthetic/<scenario>/realization\_XX.csv



where XX is the human-facing realization number.



A corresponding pair-membership audit is written to:



results/synthetic/<scenario>/realization\_XX\_pairs.csv



The primary realization CSV acts as the completion checkpoint.



\## No-Overwrite Rule



A valid completed realization checkpoint is never overwritten during

normal batch execution.



Before skipping an existing realization, identifying fields are

validated against the locked benchmark schedule:



\- scenario

\- realization index

\- dataset seed

\- outer split seed

\- final split seed

\- discovery base seed

\- random-pair seed

\- final-model seed



A mismatched checkpoint is treated as an error rather than silently

reused or overwritten.



\## Resumption



When the batch runner is restarted, every selected realization is

checked.



Valid completed checkpoints are skipped.



Uncompleted realizations are executed.



Thus, completed computational work is preserved across process or

machine restarts.



\## Scenario-Specific Execution



The runner supports execution of:



S1 only



S2 only



S3 only



S4 only



or all scenarios.



\## Realization Range



Human-facing realization ranges may be selected.



For example:



start = 5



end = 10



executes realizations 5 through 10 inclusive for the selected

scenario or scenarios.



\## Failure Handling



An exception inside one realization is written to:



results/synthetic/failures.csv



The log records:



\- scenario

\- realization number

\- all primary seeds

\- UTC timestamp

\- exception type

\- exception message

\- traceback



By default, the batch continues with the next realization.



An optional stop-on-error mode is available for debugging.



If a previously failed realization later completes successfully, its

stale failure entry is removed.



\## Master Results



After each successfully completed realization, all primary realization

checkpoints are re-read and combined into:



results/synthetic/master\_results.csv



The master file is derived data.



It may be rebuilt at any time from the individual immutable

realization checkpoints.



Pair-audit files are excluded from the master result table.



\## Scenario Summary



After each successful realization, the scenario-level descriptive

summary is rebuilt as:



results/synthetic/scenario\_summary.csv



For each available metric and scenario, the summary stores:



\- n

\- mean

\- standard deviation

\- median

\- 25th percentile

\- 75th percentile

\- interquartile range



Scenario-level inference is not performed until the primary benchmark

execution and statistical-analysis protocol are locked.



\## Dry Run



Dry-run mode performs no model training.



It reports whether each selected realization is:



SKIP



because a validated checkpoint already exists,



or:



PLAN



because the realization remains to be executed.



Dry-run mode is used before long benchmark jobs to verify the selected

schedule.



\## Execution Order



The recommended primary execution order is:



1\. S1 realizations 1-20

2\. S2 realizations 1-20

3\. S3 realizations 1-20

4\. S4 realizations 1-20



This ordering is operational only.



It does not alter the predefined realization seeds or statistical

interpretation.



\## Locked Method



The resumable runner changes only benchmark execution and persistence.



It does not change:



\- synthetic scenario definitions

\- AG-NAM architecture

\- interaction scoring

\- candidate-set size

\- selection threshold

\- ISR threshold

\- purification definition

\- optimization settings

\- evaluation metrics



The benchmark runner must not modify method parameters in response to

intermediate benchmark results.

