\# Phase 6 — Locked Real-X Semi-Synthetic Interaction Recovery Benchmark



\## Purpose



The synthetic benchmark provides known interaction ground truth but

uses artificial predictor distributions.



The locked OpenML benchmark provides real predictor distributions but

does not provide verified interaction ground truth.



The Real-X semi-synthetic benchmark bridges these two settings:



real predictor distributions + known injected interaction structure.



The original OpenML outcome labels are not used.



\## Dataset Selection



The source registry is the previously locked 28-task OpenML registry.



No AG-NAM predictive or structural result is used for dataset

selection.



Within each predictor-type stratum, three positions are selected:



1\. first

2\. lower median

3\. last



This produces nine locked Real-X datasets.



\### Categorical



\- tic-tac-toe

\- kr-vs-kp

\- PhishingWebsites



\### Mixed



\- dresses-sales

\- credit-g

\- adult



\### Numeric



\- kc2

\- pc4

\- jm1



Large datasets are deterministically subsampled to at most 5000 rows

per realization.



The sampled X values remain real observations.



\## Original Outcomes



Original OpenML labels are discarded before semi-synthetic outcome

generation.



They are never used for:



\- feature selection

\- interaction selection

\- surface generation

\- parameter tuning

\- synthetic label generation



\## Realizations



Each dataset contains five locked realizations.



Each realization contains four paired interaction-strength

conditions:



\- Null: 0.0

\- Weak: 0.5

\- Moderate: 1.0

\- Strong: 1.5



Total benchmark size:



9 datasets × 5 realizations × 4 strengths = 180 runs.



\## Common Random Numbers



Within one dataset and realization, all four strength conditions

share exactly the same:



\- sampled X rows

\- train/validation/test split

\- selected main-effect features

\- selected interaction features

\- latent main-effect surfaces

\- latent interaction surfaces

\- Bernoulli uniform random numbers



Only the interaction coefficient changes.



Therefore strength comparisons are paired by construction.



\## Data Split



For every realization:



\- 60% training

\- 20% validation

\- 20% untouched test



The split seed is predetermined by the locked schedule.



\## Ground-Truth Feature Structure



Each non-null realization contains:



\- 3 disjoint true pairwise interactions

\- 3 additional disjoint main-effect features



Therefore nine distinct eligible predictors are required.



The interaction and main-effect feature identities are chosen using a

predetermined feature-selection seed.



No model output is used to select true features.



\## State Representation



Semi-synthetic label generation uses a generic state representation

that supports numeric, categorical, mixed, and missing-valued data.



\### Numeric Predictors



Numeric features are represented using empirical quantile states.



At most eight states are used.



\### Categorical Predictors



Categorical features use observed categories.



High-cardinality predictors retain the most frequent levels and

collapse remaining levels into an OTHER state.



\### Missing Values



Missingness is represented as an explicit state during the

semi-synthetic generation process.



This generation representation is separate from the predictive

model's preprocessing pipeline.



\## Main-Effect Functions



Each selected main-effect feature receives a random state-effect

function.



The function is empirically centered and standardized.



The main-effect composite has unit standard deviation.



The locked main-effect coefficient is:



1.0



\## Pairwise Interaction Functions



For every true pair, a random Gaussian cell surface is created over

the empirical state grid.



The surface is purified using empirical weighted two-way

functional-ANOVA centering.



The objective is to remove additive row and column components from

the injected interaction surface.



Each purified pairwise contribution is standardized before the

interaction composite is formed.



\## Outcome Model



The binary semi-synthetic outcome follows a logistic model:



eta =

intercept

\+ 1.0 × standardized main-effect composite

\+ lambda × standardized interaction composite



where lambda is:



\- 0.0 for Null

\- 0.5 for Weak

\- 1.0 for Moderate

\- 1.5 for Strong



The intercept is calibrated so that the expected positive prevalence

is approximately 0.50.



Labels are then sampled from the resulting Bernoulli probabilities.



\## Null Condition



The Null condition uses the same latent pair templates as the

corresponding non-null conditions, but their interaction coefficient

is exactly zero.



For structural evaluation the active ground-truth interaction set in

the Null condition is empty.



The Null condition is used as a negative control for false

interaction discovery.



\## AG-NAM Discovery Protocol



The Real-X benchmark reuses the locked AG-NAM interaction protocol:



\- 5 repeated discovery runs

\- 5-fold residual cross-fitting

\- selection-frequency threshold = 0.60

\- ISR threshold = 0.60



Candidate budget:



min(p - 1, 20)



where p is the number of eligible predictors.



\## Predictive Comparators



The locked Real-X benchmark contains:



\- Main NAM

\- Full AG-NAM

\- Oracle AG-NAM

\- Random-Pair NAM

\- No-ISR AG-NAM

\- Single-Run AG-NAM



EBM and CatBoost are not included in this structural benchmark because

their role as external predictive comparators has already been

evaluated in the frozen real-world OpenML benchmark.



\## Structural Metrics — Non-Null Conditions



The benchmark reports:



\- interaction AUPRC

\- true-interaction ranks

\- NDCG

\- selection precision

\- selection recall

\- selection F1

\- ISR precision

\- ISR recall

\- ISR F1

\- exact recovery

\- false-discovery rate

\- false-positive reduction after ISR

\- number of selection-stable pairs

\- number of ISR-retained pairs



Ground-truth interaction identity is known by construction.



\## Structural Metrics — Null Condition



Because the active ground-truth interaction set is empty, precision

and recall are not interpreted in the ordinary way.



The Null condition reports:



\- number of selection-stable false interactions

\- number of ISR-retained false interactions

\- probability of retaining any false interaction

\- false-positive reduction after ISR



\## Predictive Metrics



On the untouched test set report:



\- AUROC

\- AUPRC

\- balanced accuracy

\- F1



for the prespecified predictive comparators.



\## Primary Real-X Endpoint



The experimental unit for confirmatory inference is the dataset.



The primary condition is:



Moderate interaction strength, lambda = 1.0.



For each dataset, the five realizations are first averaged.



The primary metric is:



interaction AUPRC - random-ranking interaction prevalence.



The prespecified primary hypothesis test is a one-sided Wilcoxon

signed-rank test across the nine dataset-level differences.



Alternative hypothesis:



AG-NAM interaction ranking performs better than random ranking.



Alpha:



0.05



Because there is exactly one primary Real-X hypothesis test, no

multiplicity correction is applied.



\## Secondary Analyses



All remaining Real-X analyses are secondary and descriptive.



No additional confirmatory hypothesis family is introduced after

results are inspected.



\## Bootstrap



Dataset-level descriptive confidence intervals use:



10000 bootstrap resamples



Seed:



26090806



\## Method Lock



The following may not be changed after the first Real-X result is

generated:



\- nine dataset identities

\- row cap

\- number of realizations

\- signal-strength levels

\- number of true interactions

\- number of main-effect features

\- state encoding rules

\- purification rule

\- main-effect coefficient

\- expected class prevalence

\- candidate budget

\- discovery repetitions

\- residual cross-fitting

\- selection threshold

\- ISR threshold

\- comparator set

\- primary endpoint

\- primary statistical test



Any later modification constitutes a new method or benchmark version.

