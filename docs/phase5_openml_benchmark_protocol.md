\# Phase 5 — Locked Real-World OpenML Benchmark Protocol



\## Objective



The real-world benchmark evaluates whether AG-NAM provides useful,

stable, sparse, and interpretable interaction modeling across diverse

binary tabular classification tasks.



The task registry is fixed before predictive benchmark results are

inspected.



\## Benchmark Registry



The benchmark contains 28 previously audited binary OpenML tasks.



Feature-type distribution:



\- 15 numeric datasets

\- 10 mixed datasets

\- 3 categorical datasets



No dataset is removed based on subsequent AG-NAM performance.



A dataset may be excluded only for a documented technical reason that

prevents valid execution of the locked benchmark protocol.



Any such exclusion must be reported explicitly.



\## OpenML Outer Split



For every task, the predefined OpenML task split is used:



repeat = 0

fold = 0

sample = 0



The OpenML test fold is the untouched external evaluation subset.



The test fold is not used for:



\- preprocessing fitting

\- positive-class determination

\- residual generation

\- interaction proposal

\- interaction attribution

\- repeated selection

\- ISR

\- final-model early stopping

\- final-model fitting

\- purification-reference construction



\## Positive Class



Binary target labels are encoded only after the OpenML outer split is

retrieved.



The positive class is defined as the minority class in the outer

development subset.



The untouched test-label distribution is not used to choose the

positive class.



If development class counts are exactly tied, the deterministic

lexicographically smaller string representation is used as the

positive label.



The chosen label is stored in the benchmark result.



\## Final Train/Validation Split



The outer-development subset is divided into:



80% final-model training

20% final-model validation



The split is stratified.



Each task has a predefined deterministic final-split seed.



The final validation subset is used only for final-model early

stopping.



\## Locked AG-NAM Discovery



The synthetic benchmark method remains unchanged.



Discovery repetitions:



B = 5



Residual cross-fitting:



K\_r = 5



Candidate rule:



K = clip(

&#x20;   ceil(0.10 x P),

&#x20;   5,

&#x20;   20

)



Selection threshold:



pi\_jk >= 0.60



ISR threshold:



ISR\_jk >= 0.60



ISR reference size:



M = min(512, N)



ISR reference seed:



2026



No threshold or architecture is changed in response to OpenML results.



\## AG-NAM Predictive Models



The core real-world comparison includes:



Main-Effect NAM



Full AG-NAM



Random-Pair NAM



AG-NAM without ISR



Single-Run AG-NAM



All final predictive neural models are freshly initialized.



No parameters are transferred from:



\- residual models

\- attention proposer

\- interaction-discovery models

\- ISR models



\## Ground Truth



Real-world OpenML datasets do not provide verified ground-truth feature

interactions.



Therefore:



\- interaction precision is not reported

\- interaction recall is not reported

\- Oracle Interaction NAM is not used



Synthetic ground-truth interaction claims remain confined to the

synthetic benchmark.



\## Real-World Interaction Diagnostics



The following are retained:



\- mean pairwise Top-K Jaccard

\- number of selection-stable interactions

\- number of ISR-retained interactions

\- ISR sparsification ratio

\- final interaction identities

\- interaction selection frequencies

\- ISR scores

\- runtime



\## Primary Endpoint



Primary metric:



AUROC



Primary paired comparison across datasets:



Full AG-NAM versus Main-Effect NAM



The dataset is the statistical experimental unit.



\## Secondary Predictive Metrics



Secondary metrics include:



\- AUPRC

\- Balanced Accuracy

\- F1



Classification threshold:



0.50



No test-set threshold optimization is permitted.



\## External Baselines



Strong external tabular baselines are added in a locked subsequent

phase before AG-NAM OpenML results are inspected.



The planned baseline families include:



\- a strong tree-ensemble predictive reference

\- an interpretable additive/tabular reference



External baselines must use the same outer OpenML test fold.



\## Statistical Unit



Each of the 28 OpenML tasks contributes one paired benchmark outcome.



Rows or observations within one dataset are not treated as independent

replicates for cross-dataset model-comparison inference.



\## Interpretation



Real-world experiments evaluate predictive utility, interaction-set

stability, sparsity, and interpretability.



They do not validate whether discovered real-world interactions are

causal or ground-truth biological/physical interactions.

