\# Phase 6C — Locked OpenML Real-X Generator Audit



\## Purpose



This phase validates the locked Real-X semi-synthetic generator on

the nine prespecified real OpenML predictor distributions before any

AG-NAM model is trained on semi-synthetic outcomes.



No predictive metric and no interaction-recovery metric is produced

in this phase.



\## Data Loading



Each of the nine locked OpenML tasks is loaded through the OpenML

task API.



The original target is requested only because it is part of the

OpenML dataset-loading interface.



The returned target object is immediately discarded.



It is not returned by the Real-X loader and is never supplied to the

semi-synthetic generator.



\## Predictor Metadata



OpenML categorical-indicator metadata is preserved.



Predictors marked categorical by OpenML are explicitly cast to

categorical pandas dtype before Real-X generation.



This prevents nominal predictors that happen to use integer codes

from being incorrectly interpreted as continuous numeric variables.



Predictors not marked categorical are converted to numeric values.



\## Locked Feature-Type Audit



The previously locked predictor regimes are checked against OpenML

metadata:



Numeric:

all predictors must be numeric.



Categorical:

all predictors must be categorical.



Mixed:

at least one numeric and at least one categorical predictor must be

present.



This audit occurs before any semi-synthetic performance result is

generated.



\## Dataset-Realization Audit



For each of:



9 datasets × 5 realizations = 45 dataset-realization units



all four locked strength conditions are regenerated:



\- Null

\- Weak

\- Moderate

\- Strong



No model is trained.



\## Common-Random-Number Audit



Within each dataset-realization unit, all four strength conditions

must have identical:



\- sampled source rows

\- raw sampled X

\- state encoding

\- selected main-effect features

\- true interaction templates

\- main-effect composite

\- interaction composite

\- Bernoulli random numbers

\- train indices

\- validation indices

\- test indices



Only the interaction coefficient may differ.



\## Signal-Strength Audit



Because the interaction composite is standardized to unit standard

deviation, the interaction-signal standard deviations must be:



\- Null: 0.0

\- Weak: 0.5

\- Moderate: 1.0

\- Strong: 1.5



\## Purification Audit



The locked interaction purification algorithm converges at

1e-10 before global component standardization.



Because multiplication by a floating-point scaling constant may

produce residual numerical differences around machine precision, the

post-standardization reporting audit permits a maximum marginal error

of 1e-8.



This numerical audit margin does not change the generator,

purification algorithm, or locked convergence threshold.



\## Prevalence Audit



For every strength condition the expected Bernoulli prevalence must

equal the locked target prevalence of 0.50 to numerical tolerance.



Realized prevalence is recorded but is not used to alter the

protocol.



\## Class-Viability Audit



Every generated:



\- training split

\- validation split

\- test split



must contain both outcome classes for every locked strength

condition.



This is checked before model training.



A structurally invalid realization therefore fails before any

predictive or recovery result is observed.



\## Audit Outputs



The audit records:



\- OpenML task identity

\- real-X dimensions

\- OpenML predictor-type counts

\- missing-value count

\- sampled row count

\- eligible feature count

\- selected main-effect features

\- injected true interaction templates

\- hashes of sampled rows and truth structure

\- state cardinalities

\- purification marginal error

\- component standard deviations

\- split sizes

\- expected prevalence

\- realized prevalence

\- interaction-signal strength

\- class viability

\- common-random-number invariance



The audit table is saved to:



results/realx/audit/realx\_generator\_audit.csv



\## Interpretation Boundary



This phase does not provide evidence that AG-NAM recovers the injected

interactions.



It validates only that the prespecified semi-synthetic data-generating

mechanism is valid and reproducible on all nine locked real predictor

distributions.



Interaction-recovery evaluation begins only after this audit has

passed.

