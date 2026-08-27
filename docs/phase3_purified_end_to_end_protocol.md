\# Phase 3E — Purified End-to-End AG-NAM Audit



\## Objective



This experiment verifies the final explanation interface of AG-NAM

after functional-ANOVA purification.



No model-selection rule, interaction-discovery rule, threshold,

architecture, or optimization hyperparameter is modified in this

phase.



The experiment must demonstrate that the empirically purified

decomposition is algebraically equivalent to the already defined

attention-free final AG-NAM classifier.



\## Synthetic Scenario



Scenario:



S1 — Sparse strong pairwise interactions



Dataset seed:



42



Sample size:



n = 5000



Number of predictors:



p = 20



Ground-truth interaction labels are not used for model fitting,

interaction discovery, ISR filtering, final-model training, or

purification.



\## Outer Split



The complete dataset is divided into:



\- 80% outer-development data

\- 20% untouched test data



The split is stratified.



Random seed:



42



The untouched test subset must not influence:



\- preprocessing

\- residual generation

\- interaction proposal

\- interaction attribution

\- selection reproducibility

\- ISR

\- final training

\- early stopping

\- purification reference construction



\## Interaction Discovery



Interaction discovery is performed only inside the outer-development

subset using the previously locked AG-NAM protocol.



Repeated discovery:



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



Residual-sensitive attention attribution:



|A\_jk \* d(r\_hat) / d(A\_jk)|



\## Interaction Surface Reproducibility



Only selection-stable interaction candidates proceed to ISR.



Reference size:



M = min(512, N)



Reference seed:



2026



ISR threshold:



ISR\_jk >= 0.60



An interaction enters the final stable set S\* only when:



pi\_jk >= 0.60



and



ISR\_jk >= 0.60



\## Final Prediction Split



After discovery is completed, outer-development data are divided into:



\- 75% final-model training

\- 25% final-model validation



This corresponds to:



\- 60% of the complete dataset for final training

\- 20% for final validation

\- 20% untouched testing



Final split seed:



142



\## Final AG-NAM



The final AG-NAM is freshly initialized after S\* has been fixed.



No parameter is transferred from:



\- the residual-attention proposer

\- residual NAM models

\- interaction-discovery networks

\- ISR pairwise networks



The final classifier is:



eta(x) =

&#x20;   beta\_0

&#x20;   + sum\_j f\_j(x\_j)

&#x20;   + sum\_(j,k in S\*) h\_jk(x\_j, x\_k)



Attention is absent from the final predictor.



\## Final Classification Training



Main-effect configuration:



\- hidden width = 64

\- depth = 2

\- dropout = 0.10

\- categorical embedding dimension = 16



Interaction configuration:



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

\- patience = 30



Early stopping:



validation AUROC



Final-model seed:



50042



\## Purification Reference



Functional-ANOVA purification uses observations sampled exclusively

from the final-model training subset.



Reference size:



M = min(512, N\_train)



Reference seed:



2026



The untouched test set must not contribute to the purification

reference distribution.



\## Functional-ANOVA Purification



For each fitted pairwise function h\_jk:



mu\_jk =

&#x20;   E\_ref,j,k\[

&#x20;       h\_jk(X\_j, X\_k)

&#x20;   ]



a\_j(x\_j) =

&#x20;   E\_ref,k\[

&#x20;       h\_jk(x\_j, X\_k)

&#x20;   ] - mu\_jk



a\_k(x\_k) =

&#x20;   E\_ref,j\[

&#x20;       h\_jk(X\_j, x\_k)

&#x20;   ] - mu\_jk



and:



h\_jk^pure(x\_j, x\_k) =

&#x20;   h\_jk(x\_j, x\_k)

&#x20;   - E\_ref,k\[h\_jk(x\_j, X\_k)]

&#x20;   - E\_ref,j\[h\_jk(X\_j, x\_k)]

&#x20;   + mu\_jk



Marginal components are reallocated to the corresponding main

effects.



Interaction grand means are transferred to the baseline.



\## Required Invariance



For every untouched test observation:



eta\_raw(x) =

&#x20;   eta\_purified(x)



up to floating-point tolerance.



Therefore:



sigmoid(eta\_raw(x)) =

&#x20;   sigmoid(eta\_purified(x))



up to floating-point tolerance.



The following predictive metrics must remain unchanged apart from

numerical precision:



\- AUROC

\- AUPRC

\- Balanced Accuracy

\- F1



\## Required Additive Identity



The purified explanation must satisfy:



eta\_purified(x) =

&#x20;   beta\_0^\*

&#x20;   + sum\_j f\_j^\*(x\_j)

&#x20;   + sum\_(j,k in S\*) h\_jk^pure(x\_j, x\_k)



within floating-point tolerance.



\## Required Interaction Identifiability Audit



For every retained interaction, its purified reference grid must have

approximately zero empirical marginal means:



E\_ref,k\[

&#x20;   h\_jk^pure(x\_j, X\_k)

] approximately 0



and:



E\_ref,j\[

&#x20;   h\_jk^pure(X\_j, x\_k)

] approximately 0



The global mean of each purified interaction surface must also be

approximately zero.



\## Interpretation



Purification is an explanation-level algebraic reparameterization.



It is not:



\- model retraining

\- post-hoc prediction correction

\- threshold optimization

\- causal interaction estimation



The reported purified interactions represent joint predictive

structure relative to the fixed empirical training-reference

distribution.



No method component may be changed based on the results of this audit.

