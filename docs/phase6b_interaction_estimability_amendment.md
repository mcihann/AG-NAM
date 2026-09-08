\# Phase 6 Pre-Outcome Protocol Amendment:

\# Empirical Interaction Estimability



\## Trigger



During the pre-model Real-X generator audit, OpenML task 3

(kr-vs-kp) produced a degenerate purified interaction surface.



No AG-NAM model had been trained on Real-X outcomes.



No predictive metric, interaction-recovery metric, or comparative

benchmark result had been observed.



\## Cause



The originally seeded feature-selection procedure required each

feature to contain at least two observed states but did not verify

that a selected pair possessed a non-zero pure pairwise interaction

subspace on the empirical joint support.



For strongly dependent categorical predictors, a pair can contain

multiple states while its empirical support still admits no

non-additive two-way degree of freedom.



In this case functional-ANOVA purification correctly removes the

entire proposed surface, producing zero observed interaction

variance.



\## Locked Amendment



A candidate ground-truth interaction pair is now eligible only when

its empirical pure-interaction degrees of freedom satisfy:



interaction\_df >= 1



For the bipartite empirical support graph:



interaction\_df = E - V + C



where:



\- E is the number of observed joint-state cells,

\- V is the number of active row and column states,

\- C is the number of connected support components.



This quantity is the cycle rank of the empirical bipartite support

graph.



A zero value denotes absence of an estimable pure two-way

interaction component on the observed support.



\## Pair Selection



All empirically estimable candidate pairs are identified before

surface generation.



The locked feature seed determines a reproducible ordering of

candidate pairs.



A deterministic backtracking search selects three disjoint

estimable pairs.



Three main-effect features are subsequently selected from predictors

not appearing in the selected interaction pairs.



No outcome, model result, predictive performance, interaction

recovery result, or AG-NAM score is used.



\## Surface Safety



For an estimable pair, Gaussian cell surfaces are purified using the

unchanged empirical weighted two-way functional-ANOVA algorithm.



Up to 32 deterministic Gaussian draws are permitted if a projected

surface is numerically degenerate.



The random-number stream is fixed by the previously locked surface

seed.



\## Unchanged Protocol Elements



This amendment does not change:



\- the nine locked OpenML datasets,

\- the five realizations,

\- the four interaction-strength conditions,

\- the 5000-row cap,

\- the train/validation/test split,

\- the number of true interactions,

\- the number of main effects,

\- the state encoding,

\- the interaction purification algorithm,

\- the signal coefficients,

\- the class prevalence target,

\- the AG-NAM discovery protocol,

\- the comparator set,

\- the primary endpoint,

\- the primary statistical test.



\## Timing



This amendment was made during generator feasibility auditing and

before the first Real-X predictive or interaction-recovery result was

generated.



It therefore constitutes a pre-outcome feasibility refinement rather

than performance-driven post-hoc tuning.

