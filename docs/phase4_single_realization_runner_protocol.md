\# Phase 4B — Single-Realization Synthetic Benchmark Runner



\## Objective



Before launching the full 80-realization synthetic benchmark, one

complete primary realization must execute through the canonical

benchmark runner.



The runner is subsequently reused unchanged by the multi-realization

batch infrastructure.



\## Locked First Realization



Scenario:



S1



Human-facing realization:



1



Realization index:



0



Dataset seed:



10000



Outer split seed:



20000



Final split seed:



30000



Discovery seeds:



40000

40001

40002

40003

40004



Random-pair seed:



80000



Final-model seed:



90000



\## Shared Evaluation Design



All predictive methods within this realization use:



\- the same untouched test subset

\- the same final training subset

\- the same final validation subset



The same final-model seed is used wherever architecture permits.



No method receives access to the untouched test labels during fitting,

discovery, early stopping, or interaction selection.



\## Models



The single-realization benchmark evaluates:



1\. Main-Effect NAM

2\. Full AG-NAM

3\. Oracle Interaction NAM

4\. Random-Pair NAM

5\. AG-NAM without ISR

6\. Single-Run AG-NAM



\## Full AG-NAM



The final interaction set contains only pairs satisfying:



pi\_jk >= 0.60



and:



ISR\_jk >= 0.60



\## Oracle Interaction NAM



The model uses the known synthetic ground-truth interaction set.



Ground truth is used only for this explicitly labeled oracle control

and for post-hoc structure-recovery evaluation.



It is not exposed to AG-NAM interaction discovery.



\## Random-Pair NAM



The model receives exactly the same number of pairwise subnetworks as

the Full AG-NAM.



Pairs are sampled uniformly without replacement from all valid

non-self feature pairs.



Ground-truth interaction information is not used when sampling random

pairs.



\## AG-NAM Without ISR



This ablation uses all interactions satisfying:



pi\_jk >= 0.60



without applying the ISR filter.



\## Single-Run AG-NAM



This ablation uses the Top-K candidate interaction set from the first

locked discovery run.



It does not use repeated selection reproducibility or ISR filtering.



\## Predictive Training



All interaction-based final predictive models are freshly initialized.



No parameters are transferred from:



\- discovery NAMs

\- attention proposer

\- residual models

\- ISR surface models



\## Primary Output



One SyntheticBenchmarkRecord is produced containing:



\- structure-recovery results

\- Main NAM predictive metrics

\- Full AG-NAM predictive metrics

\- Oracle predictive metrics

\- Random-pair predictive metrics

\- no-ISR ablation metrics

\- single-run ablation metrics

\- exact additive decomposition error

\- total runtime



\## Interpretation



The first realization is an execution audit of the locked benchmark

runner.



It is not used to modify the primary AG-NAM thresholds or architecture.



Scenario-level conclusions require all 20 independent realizations.

