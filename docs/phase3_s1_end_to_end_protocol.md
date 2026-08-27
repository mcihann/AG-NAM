\# Phase 3 — S1 End-to-End Sanity Protocol



\## Objective



This experiment is a methodological end-to-end sanity audit of the

complete AG-NAM pipeline.



It is not part of the final multi-realization benchmark and must not

be used for post-hoc hyperparameter selection.



\## Data



Synthetic scenario:



S1 — Sparse strong pairwise interactions



Dataset seed:



42



Sample size:



n = 5000



Number of predictors:



p = 20



Ground-truth interactions are used only for post-experiment diagnostic

reporting and are not provided to the AG-NAM discovery or prediction

pipeline.



\## Outer holdout



The dataset is first divided into:



\- 80% outer-development data

\- 20% untouched test data



The split is stratified.



Random seed:



42



The test subset is not used for:



\- preprocessing

\- residual generation

\- interaction proposal

\- interaction scoring

\- selection reproducibility

\- ISR

\- early stopping

\- model fitting

\- model selection



\## Interaction discovery



Interaction discovery is performed exclusively within the

outer-development data.



Primary locked parameters:



B = 5



Discovery seeds:



42, 43, 44, 45, 46



Candidate-set rule:



K = clip(ceil(0.10 \* P), 5, 20)



Selection threshold:



pi\_jk >= 0.60



Residual cross-fitting:



K\_r = 5



Residual target:



r\_i = y\_i - p\_i



Residual-attention proposer:



\- d\_model = 64

\- n\_heads = 4

\- n\_layers = 2

\- dropout = 0.10



Interaction attribution:



|A\_jk \* d(r\_hat) / d(A\_jk)|



\## Interaction Surface Reproducibility



Only selection-stable interactions proceed to ISR.



Common reference support:



M = min(512, N)



Reference seed:



2026



Pairwise surface model:



\- feature embedding dimension = 16

\- hidden width = 64

\- depth = 2

\- activation = SiLU

\- dropout = 0.10



Pairwise optimization:



\- MSE loss

\- AdamW

\- learning rate = 1e-3

\- weight decay = 1e-5

\- batch size = 256

\- maximum epochs = 200

\- patience = 20



ISR threshold:



ISR\_jk >= 0.60



The final interaction set S\* contains only interactions satisfying:



pi\_jk >= 0.60



and



ISR\_jk >= 0.60



\## Final prediction split



After interaction discovery is complete, the same outer-development

set is divided into:



\- 75% final-model training

\- 25% final-model validation



Because outer-development represents 80% of the full dataset, this

corresponds to:



\- 60% full-data final training

\- 20% full-data final validation

\- 20% untouched test



The final validation subset is used only for early stopping of the

final predictive models.



\## Predictive models



Two models are compared.



\### Main-Effect NAM



eta(x) = beta\_0 + sum\_j f\_j(x\_j)



\### Final AG-NAM



eta(x) =

&#x20;   beta\_0

&#x20;   + sum\_j f\_j(x\_j)

&#x20;   + sum\_(j,k in S\*) f\_jk(x\_j, x\_k)



The final AG-NAM is freshly initialized.



No parameters from:



\- the attention proposer

\- residual models

\- ISR pairwise models



are transferred to the final classifier.



Attention is absent from the final predictive model.



\## Final model configuration



Main-effect networks:



\- hidden width = 64

\- depth = 2

\- dropout = 0.10



Interaction networks:



\- feature embedding dimension = 16

\- hidden width = 64

\- depth = 2

\- dropout = 0.10



Optimization:



\- BCEWithLogitsLoss

\- AdamW

\- learning rate = 1e-3

\- weight decay = 1e-5

\- batch size = 256

\- maximum epochs = 300

\- early-stopping patience = 30



Early stopping metric:



Validation AUROC



\## Predictive evaluation



Primary metric:



AUROC



Secondary metrics:



\- AUPRC

\- Balanced Accuracy

\- F1



The classification threshold for Balanced Accuracy and F1 is fixed at:



0.50



No threshold optimization is performed.



\## Interpretability audit



The final AG-NAM must satisfy:



logit =

&#x20;   baseline

&#x20;   + sum(main contributions)

&#x20;   + sum(interaction contributions)



within floating-point tolerance.



The final predictor must contain no attention module.



\## Interpretation of this experiment



This experiment evaluates whether the complete locked pipeline can:



1\. discover a stable interaction set without test-data access,

2\. construct a fresh attention-free additive classifier,

3\. preserve exact additive decomposition,

4\. improve predictive performance over the main-effect-only NAM.



No method component or threshold will be modified based solely on this

single S1 realization.

