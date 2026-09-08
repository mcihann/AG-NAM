\# Phase 6D Integration Refinement:

\# Real-X Candidate-K Adapter and CUDA Determinism Guard



\## Trigger



The first Phase 6D Real-X / AG-NAM integration sanity execution

terminated before final predictive evaluation because the locked

Real-X candidate count and the canonical synthetic discovery

candidate count differed.



For the prespecified tic-tac-toe integration dataset:



p = 9



The locked Real-X rule requires:



K = min(p - 1, 20) = 8



The canonical synthetic interaction-discovery engine returned:



K = 5



The integration runner correctly treated this difference as a hard

protocol mismatch.



No accepted Real-X benchmark result was produced.



The Phase 6D run remains explicitly excluded from benchmark evidence.



\## Source-Level Cause



Inspection of the canonical single-run discovery implementation

showed that the complete interaction ranking is generated first:



compute\_interaction\_scores(...)



Only after scoring is complete does the synthetic implementation

calculate:



k = candidate\_count(proposer.n\_features)



Therefore K does not affect:



\- outer discovery resampling

\- residual target generation

\- cross-fitting

\- NAM fitting

\- proposer preprocessing

\- proposer architecture

\- proposer training

\- interaction-score computation

\- the complete ranked interaction list



K controls only how many already-ranked interactions enter the

Top-K reproducibility stage.



\## Implementation Decision



The canonical synthetic discovery implementation is left unchanged.



This preserves all previously frozen synthetic benchmark behavior.



A Real-X-specific adapter now:



1\. executes the canonical repeated discovery pipeline unchanged;

2\. retains the complete interaction ranking from every repetition;

3\. replaces each run's Top-K value with the locked Real-X K;

4\. recomputes selection reproducibility using that locked K.



The interaction scores themselves are not recomputed or altered.



\## Real-X Candidate Rule



The Real-X candidate-count rule remains:



K = min(p - 1, 20)



For the locked Phase 6D sanity run:



p = 9

K = 8



Every repeated discovery run must therefore expose K = 8.



Any mismatch remains a hard failure.



\## Synthetic Benchmark Isolation



The synthetic benchmark continues to call the canonical discovery

engine directly.



It therefore continues to use its original candidate\_count(...)

function and original K values.



The Real-X adapter is invoked only by the Real-X runner.



\## Single-Run Comparator



The Real-X single-run comparator is defined using the first K entries

of the complete first-run interaction ranking.



It therefore uses the same locked Real-X K as the reproducibility

analysis.



\## CUDA Determinism



The first integration execution also emitted a PyTorch warning that

deterministic algorithms were enabled while the cuBLAS workspace

configuration was not explicitly defined.



The standalone Real-X integration entry point now sets:



CUBLAS\_WORKSPACE\_CONFIG=:4096:8



before importing the PyTorch-dependent AG-NAM runner.



The Real-X runner also defines the same environment default before

its own torch import.



This setting changes no model hyperparameter or random seed.



It enables deterministic cuBLAS behavior required by the already

locked deterministic-training policy.



\## Unchanged Scientific Protocol



This integration refinement does not change:



\- any OpenML dataset

\- any Real-X realization

\- any signal strength

\- injected true interactions

\- Real-X outcome generation

\- Real-X train/validation/test split

\- discovery repetitions

\- residual cross-fitting

\- selection threshold

\- ISR threshold

\- model architecture

\- optimizer settings

\- training epochs

\- early stopping

\- random seeds

\- comparator definitions

\- primary endpoint

\- statistical analysis



The refinement only ensures that the already locked Real-X

candidate-count rule is actually propagated into the reused AG-NAM

discovery infrastructure.



\## Evidence Boundary



Phase 6D remains an integration sanity phase.



All numerical outputs from this phase remain:



publication\_eligible = false



They must not be used in final benchmark tables or inferential

analyses.

