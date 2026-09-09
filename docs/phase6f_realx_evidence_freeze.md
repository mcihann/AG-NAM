\# Phase 6F-A — Real-X Evidence Freeze



\## Objective



Phase 6F-A cryptographically freezes the completed Real-X benchmark

before confirmatory statistical analysis is opened.



The evidentiary set consists of:



\- 180 individually validated Phase 6E run checkpoints;

\- master\_results.csv;

\- benchmark\_progress.csv.



Therefore the frozen evidence set contains exactly 182 files.



The Phase 6D integration-sanity result is excluded.



\## Preconditions



Evidence freezing is permitted only after:



\- all 180 locked Real-X runs are complete;

\- all run IDs are unique;

\- all checkpoints are publication eligible;

\- all untouched-test integrity audits pass;

\- all candidate-K audits pass;

\- all exact decomposition audits pass;

\- master\_results.csv contains exactly 180 rows;

\- benchmark\_progress.csv contains exactly 180 COMPLETE rows;

\- the Phase 6E execution code has been committed;

\- no tracked repository file has changed after that Phase 6E commit.



Untracked Phase 6F-A implementation files are permitted while the

freeze is created because they do not modify the locked benchmark

implementation.



\## Phase 6E Git Provenance



The freeze manifest records:



\- Git commit SHA;

\- Git branch;

\- Git commit subject.



This identifies the exact committed benchmark implementation from

which the frozen result set was produced.



The freeze validator requires that the recorded commit remains a valid

Git commit in the repository.



\## Per-File SHA-256



Every evidentiary file receives:



\- relative path;

\- role;

\- byte size;

\- SHA-256 digest.



For checkpoint files, the manifest additionally records:



\- run ID;

\- task ID;

\- dataset name;

\- predictor type;

\- realization;

\- strength condition.



\## Aggregate Evidence Hash



A deterministic aggregate SHA-256 is constructed from the sorted

collection of:



role

relative path

byte size

file SHA-256



for all 182 evidentiary files.



Changing any byte of any checkpoint, master\_results.csv, or

benchmark\_progress.csv changes the aggregate evidence hash.



\## Master-Checkpoint Consistency



The freeze process independently reconstructs the 180-run result table

from the individual checkpoint files.



The complete set of checkpoint columns is compared against

master\_results.csv.



The freeze fails if the master table does not represent the validated

per-run checkpoints.



\## Progress Audit



benchmark\_progress.csv must contain exactly the 180 locked run IDs and

every run must have:



state = COMPLETE



\## Failure Log



failures.csv is not part of the evidentiary model-output hash.



If present, its existence and row count are recorded as audit metadata.



A historical transient failure that was subsequently rerun and replaced

by a fully validated checkpoint does not modify the frozen scientific

result.



\## Freeze Outputs



The freeze directory contains:



results/realx/benchmark/freeze/realx\_evidence\_manifest.json



and:



results/realx/benchmark/freeze/realx\_evidence\_files.csv



The JSON manifest is the authoritative freeze record.



\## Immutability



After the evidence freeze is created, the following files must never be

edited or regenerated for the purpose of improving statistical results:



\- the 180 Phase 6E checkpoints;

\- master\_results.csv;

\- benchmark\_progress.csv.



Any change causes freeze validation to fail.



\## Validation



Freeze validation performs both:



1\. cryptographic validation;

2\. semantic benchmark validation.



Cryptographic validation re-computes file SHA-256 values, byte sizes,

and the aggregate evidence hash.



Semantic validation independently repeats the 180-run checkpoint,

master-table, progress-table, candidate-K, decomposition, publication

eligibility, and untouched-test integrity audits.



Both layers must pass.



\## Analysis Gate



Confirmatory statistical analysis remains closed until:



STATUS: FROZEN EVIDENCE VALIDATED



is obtained.



Only after this status may Phase 6F-B compute the prespecified primary

and secondary statistical analyses.



\## No Post-Freeze Model Adaptation



After freezing, no observed Real-X result may be used to change:



\- datasets;

\- realizations;

\- signal strengths;

\- interaction templates;

\- candidate K;

\- discovery repetitions;

\- selection threshold;

\- ISR threshold;

\- model architecture;

\- optimizer settings;

\- random seeds;

\- comparator definitions;

\- primary endpoint;

\- confirmatory statistical procedure.



All subsequent work is analysis and reporting of the frozen evidence.

