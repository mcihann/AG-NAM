\# Phase 4 — Locked Synthetic Benchmark Protocol



\## Objective



The synthetic benchmark evaluates whether AG-NAM can reproducibly

recover known interaction structure and whether the discovered

structure improves predictive performance.



The benchmark begins only after completion and locking of:



\- main-effect NAM

\- cross-fitted residual generation

\- residual-attention proposal

\- residual-sensitive interaction attribution

\- selection reproducibility

\- interaction-surface reproducibility

\- attention-free final AG-NAM

\- functional-ANOVA final decomposition



No primary AG-NAM threshold or architectural component may be changed

based on the synthetic benchmark results.



\## Scenarios



The primary synthetic benchmark contains four previously defined

scenarios:



S1

S2

S3

S4



The existing scenario definitions in the synthetic data module remain

unchanged.



\## Independent Realizations



Each scenario is evaluated using:



20 independent realizations



Therefore:



4 scenarios x 20 realizations = 80 primary experiments



Dataset generation is repeated independently for every realization.



\## Seed Schedule



The complete seed schedule is generated before inspection of the

multi-realization benchmark results.



Scenario order:



S1, S2, S3, S4



Realization indices:



0 through 19



Human-facing realization numbers:



1 through 20



\### Dataset Seeds



Base:



10000



Scenario offset:



1000 x scenario index



Realization offset:



realization index



\### Outer Split Seeds



Base:



20000



\### Final Train/Validation Split Seeds



Base:



30000



\### Discovery Seeds



Base:



40000



Each scenario receives a separate block of 10000 seeds.



Each realization receives a block of 10 discovery seeds.



With B = 5, the first five seeds in each realization block are used.



\### Random-Pair Control Seeds



Base:



80000



\### Final Predictive Model Seeds



Base:



90000



Within one realization, final predictive models use the same

final-model seed when architecture permits, so that comparisons are

not intentionally advantaged through seed selection.



\## Outer Evaluation



Every realization is first divided into:



80% outer-development



20% untouched test



The test set must not influence:



\- preprocessing

\- residual generation

\- interaction discovery

\- interaction scoring

\- selection reproducibility

\- ISR

\- final-model validation

\- final-model fitting

\- purification reference construction



\## Final Prediction Split



Outer-development data are divided into:



75% final training



25% final validation



This corresponds to:



60% of the complete realization for final training



20% for final validation



20% untouched test



\## Locked AG-NAM Discovery Parameters



Residual cross-fitting:



K\_r = 5



Discovery repetitions:



B = 5



Residual target:



r\_i = y\_i - p\_i



Candidate-set rule:



P = p(p - 1) / 2



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



Reference seed:



2026



\## Structure-Recovery Outcomes



The benchmark records interaction recovery both before and after ISR.



Primary structure-recovery outcomes include:



interaction-ranking AUPRC



true-interaction ranks



mean pairwise Top-K Jaccard similarity



number of selection-stable interactions



selection-stage precision



selection-stage recall



selection-stage F1



number of ISR-retained interactions



ISR-stage precision



ISR-stage recall



ISR-stage F1



false-positive reduction after ISR



final retained interaction-set size



\## Predictive Outcomes



Primary predictive metric:



AUROC



Secondary predictive metrics:



AUPRC



Balanced Accuracy



F1



The classification threshold for Balanced Accuracy and F1 remains:



0.50



No threshold optimization is permitted.



\## Primary Predictive Comparison



Main-Effect NAM



versus



Full AG-NAM



Both models are evaluated on the same untouched test subset within

each realization.



\## Planned Method Controls



The synthetic benchmark infrastructure will additionally support:



Oracle Interaction NAM



Random-Pair NAM



AG-NAM without ISR



Single-Run AG-NAM



These controls are used to distinguish interaction-discovery quality

from simple increases in model capacity.



\## Oracle Interaction NAM



The oracle model receives the true synthetic interaction set.



It does not represent a deployable method.



Its purpose is to estimate the predictive upper reference achievable

when interaction structure is known.



\## Random-Pair NAM



The random-pair control receives the same number of interaction

networks as the full AG-NAM but uses randomly selected non-self

feature pairs.



Random-pair selection is reproducible using the predefined

random-pair seed.



This control tests whether predictive improvement can be explained

merely by adding pairwise neural-network capacity.



\## AG-NAM Without ISR



This ablation retains all interactions satisfying:



pi\_jk >= 0.60



without applying the ISR filter.



It quantifies the incremental effect of functional reproducibility

filtering.



\## Single-Run AG-NAM



This ablation uses the locked interaction ranking from one discovery

run without repeated selection reproducibility.



It quantifies the contribution of repeated discovery stability.



\## Aggregation Across Realizations



Results are summarized separately for each synthetic scenario.



For continuous metrics, at minimum report:



mean



standard deviation



median



interquartile range



For paired predictive comparisons, AG-NAM and its comparators are

evaluated on identical test observations within each realization.



Inferential comparisons will be defined only after the benchmark

runner and output schema have been locked.



\## Interpretation



No single synthetic realization is treated as definitive evidence.



Claims concerning:



interaction recovery



stability



false-positive filtering



predictive improvement



must be based on the multi-realization distributions.



The earlier S1 sanity experiments remain engineering and

methodological audits rather than primary benchmark evidence.

