\# AG-NAM Phase 0 Protocol Lock



\## Methodological objective



AG-NAM investigates whether attention-guided interaction proposals can be filtered according to cross-resample reproducibility to construct an intrinsically decomposable neural additive predictor.



Attention is not treated as an explanation and is not part of the final predictive model.



\## Final model form



For binary classification:



eta(x) = beta\_0 + sum\_j f\_j(x\_j) + sum\_(j,k in S\*) f\_jk(x\_j, x\_k)



p(y=1|x) = sigmoid(eta(x))



The prediction is therefore exactly decomposable into:



\- baseline contribution,

\- univariate main effects,

\- explicit pairwise interaction effects.



\## Interaction discovery



Candidate interactions are proposed using a residual-targeted attention mechanism.



Raw attention weights are not interpreted as feature importance.



Candidate rankings are derived from the contribution of attention connections to residual prediction.



\## Selection reproducibility



For interaction (j,k):



pi\_jk = (1/B) \* sum\_b I\[(j,k) belongs to TopK\_b]



Primary discovery setting:



B = 5



An interaction must satisfy:



pi\_jk >= 0.60



\## Interaction Surface Reproducibility (ISR)



Selection frequency alone is insufficient.



For every repeatedly selected interaction, the learned interaction functions are evaluated on a common reference support sampled from the outer-training data.



Interaction surfaces learned across resamples are centered and compared for functional agreement.



An interaction must satisfy:



ISR\_jk >= 0.60



\## Candidate-set size



Let:



P = p(p-1)/2



The number of candidate pairs is:



K = clip(ceil(0.10 \* P), 5, 20)



where clip restricts K to the interval \[5, 20].



\## Final interaction acceptance



An interaction is retained only if both:



pi\_jk >= 0.60



and



ISR\_jk >= 0.60



are satisfied.



\## General benchmark datasets



The following OpenML-CC18 binary-classification tasks were selected before model evaluation:



1\. Task 49 — tic-tac-toe

2\. Task 14952 — PhishingWebsites

3\. Task 29 — credit-approval

4\. Task 167141 — churn

5\. Task 14965 — bank-marketing

6\. Task 7592 — adult

7\. Task 9957 — qsar-biodeg

8\. Task 9978 — ozone-level-8hr

9\. Task 43 — spambase

10\. Task 3904 — jm1



Dataset selection was based only on metadata, including sample size, dimensionality, predictor type, missingness, class imbalance, and domain diversity.



No model-performance results were inspected before dataset selection.



\## Biomedical case studies



\### Case Study 1

WDBC / Breast Cancer Wisconsin Diagnostic



Purpose:

\- global main-effect visualization,

\- pairwise interaction visualization,

\- local additive decomposition.



\### Case Study 2

Diabetes 130-US Hospitals



Target:

30-day readmission versus all other outcomes.



Repeated encounters from the same patient must not cross evaluation partitions.



patient\_nbr is used only as a grouping variable and not as a predictor.



\## Synthetic scenarios



Four predefined synthetic scenarios will be used:



\- S1: sparse strong pairwise interactions

\- S2: multiple nonlinear interactions

\- S3: correlated nuisance features

\- S4: weak interactions under high-dimensional noise and class imbalance



Each scenario will use 20 independent dataset realizations.



\## Primary predictive metric



AUROC



\## Secondary predictive metrics



\- AUPRC

\- Balanced Accuracy

\- F1



Accuracy may be reported descriptively but is not a primary metric.



\## Interaction-recovery metrics



Primary:



\- Interaction AUPRC



Secondary:



\- Precision@K

\- Recall@K

\- NDCG@K

\- Exact Recovery Rate

\- False Discovery Rate

\- Selection Reproducibility

\- Interaction Surface Reproducibility



\## Evaluation protocol



General OpenML benchmarks:



\- official OpenML 10-fold outer evaluation

\- 3-fold stratified inner validation

\- no outer-test information is used for preprocessing, interaction discovery, hyperparameter selection, or early stopping



Biomedical repeated-patient data:



\- group-aware stratified evaluation

\- patient identity is used only for grouping



\## Statistical analysis



Across benchmark datasets:



\- Friedman test

\- Wilcoxon signed-rank pairwise comparisons

\- Holm correction for multiple comparisons

\- mean ranks

\- effect sizes where appropriate



\## Prohibited leakage



The following operations must never use outer-test data:



\- imputation

\- scaling

\- categorical vocabulary construction

\- interaction proposal

\- interaction selection

\- ISR estimation

\- hyperparameter optimization

\- early stopping

\- classification-threshold optimization



\## Positive-class definition



Positive-class mappings were fixed before any predictive model was evaluated.



The predefined positive labels are:



| OpenML task | Dataset | Positive raw label |

| --- | --- | --- |

| 49 | tic-tac-toe | positive |

| 14952 | PhishingWebsites | -1 |

| 29 | credit-approval | + |

| 167141 | churn | 1 |

| 14965 | bank-marketing | 2 |

| 7592 | adult | >50K |

| 9957 | qsar-biodeg | 2 |

| 9978 | ozone-level-8hr | 2 |

| 43 | spambase | 1 |

| 3904 | jm1 | True |



All models will internally encode the predefined positive class as 1 and the alternative class as 0.



This mapping must not be changed based on predictive performance.



\## Residual target for interaction proposal



For binary classification, the interaction-proposal stage uses

Bernoulli negative-gradient pseudo-residuals:



r\_i = y\_i - p\_i



where:



p\_i = sigmoid(eta\_i)



This residual corresponds exactly to the negative derivative of

binary cross-entropy with respect to the model logit.



Pearson and deviance residuals are not used as primary residual

targets.



Residuals used for interaction proposal must be cross-fitted.



Within each outer-training partition, 5-fold cross-fitting is used

to obtain out-of-fold main-effect NAM predictions:



r\_i^(OOF) = y\_i - sigmoid(eta\_hat^(-k(i))(x\_i))



Thus, no observation receives a residual obtained from a

main-effect NAM that was trained on that same observation.



Fixed residual cross-fitting parameter:



K\_r = 5



\## Proposer validation leakage rule



When a residual-attention proposer uses a separate validation subset,

residual targets must be generated only after the proposer

train/validation partition is fixed.



Training residuals are cross-fitted exclusively within the proposer

training subset.



Validation residuals are obtained from a main-effect NAM trained

exclusively on proposer-training observations.



Thus, proposer-validation observations cannot influence the residual

targets used for proposer optimization.



\## Synthetic first-look interaction audit



Before inspecting interaction-recovery results, the S1 first-look

protocol is fixed as follows:



\- 60% proposer-training subset

\- 20% proposer early-stopping validation subset

\- 20% untouched interaction-scoring holdout subset

\- stratified partitioning

\- fixed seed = 42



Residual targets for proposer training and early stopping are

constructed without access to the interaction-scoring holdout.



The trained proposer is applied to the untouched scoring holdout

without using target labels.



Interaction scores are computed using the predefined residual-sensitive

attention attribution:



|A\_jk \* d(r\_hat) / d(A\_jk)|



Ground-truth synthetic interactions are used only after ranking has

been produced, solely for evaluation.



The predefined S1 candidate-set rule gives K = 19 for p = 20.



This first-look experiment is a methodological sanity audit and is not

a substitute for the final 20-realization synthetic benchmark.



\## Selection reproducibility implementation



Primary interaction-selection reproducibility uses five repeated

discovery runs:



B = 5



The predefined run seeds are:



42, 43, 44, 45, 46



Each run independently applies the locked:



\- 60% proposer-training partition

\- 20% proposer-validation partition

\- 20% untouched interaction-scoring partition



The interaction-scoring method and candidate-set size remain unchanged

across runs.



For each interaction (j,k):



pi\_jk = number of Top-K selections / B



The primary acceptance threshold is:



pi\_jk >= 0.60



Thus, with B = 5, an interaction must appear in the Top-K candidate

set in at least three runs.



Full interaction ranks are retained in every run, including when an

interaction falls outside Top-K. No artificial rank such as K+1 is

assigned.



Mean pairwise Jaccard similarity between Top-K sets is reported as a

global diagnostic of candidate-set reproducibility.



Selection probability is the primary criterion; mean and median ranks

are descriptive diagnostics only.



\## Interaction Surface Reproducibility (ISR)



Selection reproducibility alone is not considered sufficient evidence

for retaining an interaction.



Only interactions satisfying:



pi\_jk >= 0.60



proceed to functional reproducibility assessment.



For each accepted candidate pair, an explicit pairwise residual model

is fitted independently in every discovery run in which that pair was

selected in the Top-K candidate set.



A fixed common raw-data reference support is used:



M = min(512, N)



Reference observations are sampled with fixed seed 2026 from the

available outer-training/development data.



Before cross-run comparison, each learned pairwise function is

purified using an empirical functional-ANOVA projection.



For interaction function h\_jk and reference observations

{(x\_j\_i, x\_k\_i)}:



v\_i =

&#x20;   h(x\_j\_i, x\_k\_i)

&#x20;   - mean\_m h(x\_j\_i, x\_k\_m)

&#x20;   - mean\_m h(x\_j\_m, x\_k\_i)

&#x20;   + mean\_mn h(x\_j\_m, x\_k\_n)



Thus, univariate marginal structure and the global intercept are

removed before functional comparison.



Interaction Surface Reproducibility is defined as the mean pairwise

Pearson correlation between purified interaction vectors obtained

from the discovery runs in which the interaction was selected.



If a purified interaction vector has effectively zero variance, its

functional agreement is treated as zero.



The primary ISR threshold is:



ISR\_jk >= 0.60



Final interaction retention requires both:



pi\_jk >= 0.60



and



ISR\_jk >= 0.60



Neither threshold may be changed based on S1 first-look results.



\## Pairwise surface model configuration



The explicit pairwise residual networks used for ISR are fixed before

inspection of ISR results.



Architecture:



\- feature embedding dimension = 16

\- hidden width = 64

\- hidden depth = 2

\- activation = SiLU

\- dropout = 0.10



Optimization:



\- loss = mean squared error

\- optimizer = AdamW

\- learning rate = 1e-3

\- weight decay = 1e-5

\- batch size = 256

\- maximum epochs = 200

\- early-stopping patience = 20



Each pairwise model is trained on the leakage-safe residual targets

associated with the corresponding discovery run.



The pairwise model is not itself treated as an explanation until

functional-ANOVA purification has been applied.

