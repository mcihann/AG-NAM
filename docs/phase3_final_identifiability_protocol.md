\# Phase 3 — Final AG-NAM Interaction Identifiability



\## Objective



The final AG-NAM classifier is exactly additive, but unconstrained

pairwise neural networks can contain both genuine joint structure and

relearned marginal effects.



Therefore, reported interaction effects are subjected to an empirical

functional-ANOVA purification after final model training.



This operation is an algebraically equivalent reparameterization of

the fitted classifier and must not alter predictive outputs.



\## Raw Final Model



The fitted classifier has the form:



eta(x) =

&#x20;   beta\_0

&#x20;   + sum\_j f\_j(x\_j)

&#x20;   + sum\_(j,k in S\*) h\_jk(x\_j, x\_k)



The attention proposer is not part of the final classifier.



\## Empirical Reference Distribution



Purification uses a fixed reference support sampled exclusively from

the final-model training data.



Primary reference size:



M = min(512, N\_train)



Reference seed:



2026



The empirical functional-ANOVA projection is defined relative to the

product of the empirical marginal distributions represented by this

reference support.



The untouched test set must never be used to construct the reference

distribution.



\## Pairwise Purification



For pairwise function h\_jk:



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



The purified interaction is:



h\_jk^pure(x\_j, x\_k) =

&#x20;   h\_jk(x\_j, x\_k)

&#x20;   - E\_ref,k\[h\_jk(x\_j, X\_k)]

&#x20;   - E\_ref,j\[h\_jk(X\_j, x\_k)]

&#x20;   + mu\_jk



\## Reallocation



Marginal components removed from each pairwise network are not

discarded.



They are reallocated to the corresponding main effects:



f\_j^\*(x\_j) =

&#x20;   f\_j(x\_j)

&#x20;   + sum of interaction-derived a\_j(x\_j) terms



The interaction grand means are transferred to the baseline.



Therefore:



eta\_raw(x) =

&#x20;   eta\_purified(x)



up to floating-point precision.



\## Reported Explanation



The explanation interface reports:



\- purified baseline

\- adjusted main-effect contributions

\- purified interaction contributions



The raw unconstrained pairwise network output is not reported as the

final interaction explanation.



\## Interpretation



The resulting interaction surfaces satisfy empirical zero-marginal

constraints relative to the fixed reference distribution.



Thus, an interaction contribution is interpreted as joint structure

remaining after marginal components associated with either feature

have been removed.



This is a reference-distribution-based functional decomposition, not

a claim of causal interaction.



\## Predictive Invariance



Purification is not a refitting step.



It must not alter:



\- logits

\- probabilities

\- AUROC

\- AUPRC

\- Balanced Accuracy

\- F1



A unit test must verify numerical equivalence between raw and purified

logits.



\## Locked Parameters



Reference size:



M = min(512, N\_train)



Reference seed:



2026



These choices are fixed before inspection of purified S1 final-model

effects and must not be modified based on the resulting interaction

surfaces.

