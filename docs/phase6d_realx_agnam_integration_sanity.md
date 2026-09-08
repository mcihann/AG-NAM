\# Phase 6D — Real-X / AG-NAM End-to-End Integration Sanity



\## Purpose



Phase 6D validates the connection between the locked Real-X

semi-synthetic generator and the existing canonical AG-NAM training,

interaction-discovery, reproducibility, ISR, and final prediction

pipeline.



This phase is an implementation sanity test.



Its numerical results are not benchmark evidence.



\## Prespecified Sanity Run



The integration run is fixed before execution:



\- OpenML task: 49

\- Dataset: tic-tac-toe

\- Real-X realization: 1

\- Interaction strength: Moderate

\- Interaction coefficient: 1.0



This condition is chosen because it is the first locked Real-X

dataset and first realization, not because of any observed model

performance.



\## Evidence Restriction



All Phase 6D outputs are marked:



publication\_eligible = false



The Phase 6D AUROC, AUPRC, interaction-recovery, selection, ISR, and

control-model values must not be transferred into the final Real-X

benchmark tables or inferential analyses.



The final Real-X benchmark begins only after the integration pipeline

has been validated and committed.



\## Reuse of Existing AG-NAM Engine



Phase 6D does not create a second model-training implementation.



It delegates to the canonical components already used by the locked

synthetic benchmark:



\- MainEffectNAM

\- AGNAM

\- reproducible interaction discovery

\- leakage-safe residual cross-fitting

\- interaction ranking

\- selection reproducibility

\- ISR

\- Main NAM training

\- AG-NAM training

\- Oracle interaction control

\- random-pair control

\- no-ISR control

\- single-run control

\- exact additive decomposition audit



The default SyntheticModelConfig is reused unchanged.



\## Model-Side Seeds



The Real-X generator already defines a dataset/realization seed family

that is independent of interaction strength.



The model-side seed family extends this structure.



If the locked label seed is:



group + 51



then:



\- discovery seed = group + 61

\- random-pair seed = group + 71

\- final-model seed = group + 81



Therefore Null, Weak, Moderate, and Strong conditions within the same

dataset and realization receive identical model-side seeds.



\## Candidate Count



The locked Real-X interaction-candidate rule is:



K = min(p - 1, 20)



where p is the number of predictors.



The integration runner verifies that every discovery repetition uses

exactly this K.



A mismatch causes a hard failure before the pipeline can be accepted.



\## Locked Split Ownership



The Real-X generator owns the outer data split.



No new train\_test\_split operation is permitted.



The locked partitions are:



\- training: approximately 60%

\- validation: approximately 20%

\- untouched test: remainder, approximately 20%



Integer rounding follows the already locked Real-X generator.



\## Discovery Development Pool



Interaction discovery is permitted to use:



training + validation



which forms the 80% development pool.



The untouched Real-X test partition is excluded from:



\- Main-NAM residual discovery

\- cross-fitting

\- proposer training

\- repeated interaction ranking

\- selection stability

\- ISR

\- early stopping

\- pair selection



\## Final Prediction Data



Final predictive model training uses exactly:



\- locked training partition for parameter optimization

\- locked validation partition for early stopping

\- locked test partition for final evaluation



The tabular preprocessor is fitted only on the locked training

partition.



Validation and test predictors are transformed using that fitted

preprocessor.



\## ISR



The Real-X integration protocol inherits the previously validated ISR

reference design:



\- reference size: 512

\- reference seed: 4026

\- ISR threshold: 0.60



No Real-X outcome is used to alter these settings.



\## Control Interaction Sets



The integration pipeline constructs:



\- Oracle interaction set:

&#x20; known injected true interactions



\- Random-pair set:

&#x20; same number of pairs as the ISR-retained set, sampled without using

&#x20; ground-truth information



\- No-ISR set:

&#x20; selection-stable interactions before ISR



\- Single-run set:

&#x20; top-K interactions from the first discovery repetition



\## Exact Decomposition Audit



The final AG-NAM test logits must satisfy:



logit =

baseline

\+ sum(main effects)

\+ sum(pairwise interactions)



The maximum absolute reconstruction error on the untouched test set

must not exceed:



1e-6



\## Test Integrity Audit



The raw Real-X test predictor rows and semi-synthetic labels are

snapshotted before discovery and training.



After all models are fitted, the integration runner verifies that:



\- the raw test predictor values are unchanged

\- the raw test labels are unchanged

\- FinalPredictionData contains exactly the locked test labels



Failure constitutes a hard integration error.



\## Acceptance Criteria



Phase 6D passes only when:



\- the unit-test suite passes

\- the locked sanity condition resolves uniquely

\- the candidate-K audit passes

\- discovery completes

\- selection reproducibility completes

\- ISR completes

\- all six predictive branches complete

\- exact decomposition passes

\- untouched test integrity passes



Only after these conditions are met may the Real-X batch benchmark

runner be implemented.



\## Interpretation Boundary



Even when the Phase 6D predictive or interaction-recovery results

appear favorable, they are implementation diagnostics only.



They must not influence:



\- protocol parameters

\- dataset selection

\- interaction strength

\- thresholds

\- model architecture

\- optimizer settings

\- primary endpoint

\- inferential analysis



The complete locked Real-X benchmark remains the sole evidentiary

analysis.

