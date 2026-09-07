\# Phase 5D — Canonical Single-Task OpenML Runner



\## Objective



Before launching all 28 locked OpenML tasks, the complete real-world

benchmark pipeline is executed on one predefined task.



The purpose is execution validation, not method tuning.



\## Canonical First Task



OpenML task:



49



Dataset:



tic-tac-toe



Feature type:



categorical



The task was selected because it is the first item in the locked

OpenML registry.



It was not selected according to AG-NAM predictive performance.



\## Outer Evaluation



The predefined OpenML task split is used:



repeat = 0



fold = 0



sample = 0



The OpenML test fold remains untouched during:



\- preprocessing fitting

\- positive-class selection

\- interaction discovery

\- repeated selection

\- ISR

\- final-model fitting

\- neural early stopping

\- EBM fitting

\- CatBoost fitting and early stopping



\## Positive Class



The positive label is the minority class of the outer-development

subset.



Test labels are not used to select the positive class.



\## Interaction Discovery



The already locked AG-NAM method is used without modification.



Discovery repetitions:



B = 5



Residual cross-fitting:



K\_r = 5



Selection threshold:



pi >= 0.60



ISR threshold:



ISR >= 0.60



Candidate rule:



K = clip(

&#x20;   ceil(0.10 x P),

&#x20;   5,

&#x20;   20

)



\## Final Neural Split



Outer-development data are divided into:



80% final training



20% final validation



The split is stratified.



The OpenML test fold remains unchanged.



\## Final Neural Models



The canonical task evaluates:



1\. Main-Effect NAM

2\. Full AG-NAM

3\. Random-Pair NAM

4\. AG-NAM without ISR

5\. Single-Run AG-NAM



All final models are freshly initialized.



\## Explainable Boosting Machine



EBM uses the locked Phase 5C external-baseline protocol.



It is fitted only on outer-development data.



Its internal validation is training-derived.



The EBM interaction budget uses the same locked candidate-budget rule

as AG-NAM.



\## CatBoost



CatBoost uses the locked Phase 5C CPU protocol.



It is fitted on the final training subset.



The final validation subset is used for early stopping.



The untouched OpenML test fold is used only for final prediction.



\## Real-World Structural Interpretation



There is no verified ground-truth interaction set.



Therefore the canonical real-world output records:



\- interaction identities

\- selection frequencies

\- selection-stable membership

\- ISR-retained membership

\- ISR score when available

\- random-control membership

\- single-run membership



No real-world interaction precision or recall is calculated.



\## Predictive Outputs



For each model report:



\- AUROC

\- AUPRC



Additionally report Balanced Accuracy and F1 for:



\- Main NAM

\- Full AG-NAM

\- EBM

\- CatBoost



The classification threshold remains 0.50.



\## Additive Audit



The final AG-NAM decomposition is reconstructed on the untouched test

fold.



Maximum absolute logit reconstruction error is recorded.



\## No Post-Hoc Modification



The canonical task result must not be used to modify:



\- AG-NAM architecture

\- candidate rule

\- selection threshold

\- ISR threshold

\- EBM settings

\- CatBoost settings

\- OpenML task registry

\- final statistical-analysis family



After successful execution, the same runner becomes the basis of the

28-task resumable OpenML benchmark.

