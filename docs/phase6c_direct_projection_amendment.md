\# Phase 6 Pre-Outcome Numerical Amendment:

\# Direct Weighted Functional-ANOVA Projection



\## Trigger



During the locked Real-X generator feasibility audit, OpenML task

7592 (adult) produced a failure in the numerical interaction

purification solver.



The selected ground-truth interaction pair had already passed the

prespecified positive empirical interaction-degree-of-freedom

criterion.



The failure therefore did not indicate an undefined interaction

space.



Instead, the previously implemented alternating weighted row/column

centering procedure did not reach the locked numerical tolerance of

1e-10 within 1000 iterations.



The final reported marginal error was approximately:



3.849e-05



No AG-NAM model had been trained on Real-X outcomes.



No Real-X predictive metric, interaction-recovery metric, or

comparative benchmark result had been observed.



\## Mathematical Target



The definition of the injected pure pairwise interaction is unchanged.



For a proposed joint-state surface g(i,j), the required

functional-ANOVA decomposition is:



g(i,j) = additive(i,j) + h(i,j)



where h(i,j) must be empirically orthogonal to the additive

row/column function space under the observed joint-state

distribution.



Equivalently, the purified interaction must satisfy weighted

zero-marginal conditions along both axes.



\## Numerical Refinement



The numerical solver is changed from alternating weighted marginal

centering to direct weighted least-squares projection.



For observed joint-state cells, an additive design matrix containing:



\- an intercept

\- row-state indicators

\- column-state indicators



is constructed.



Let A denote this additive design matrix, W the diagonal empirical

cell-count weight matrix, and g the proposed surface values on

observed cells.



The additive component is estimated by:



beta\_hat = argmin\_beta || W^(1/2) (g - A beta) ||\_2^2



The purified interaction is:



h = g - A beta\_hat



Thus h is the weighted least-squares residual from the full

estimable additive subspace.



\## Sparse and Disconnected Support



The additive design may be rank deficient when the empirical

bipartite support contains disconnected components or redundant

indicator columns.



The implementation therefore uses an SVD-based least-squares

solution.



A unique coefficient vector is not required.



Only the fitted additive projection and its residual are required,

and these remain well defined in the estimable observation space.



\## Relationship to the Previous Method



This change does not redefine the interaction.



Alternating marginal centering and direct weighted least-squares

projection target the same weighted additive orthogonal complement.



The direct solver is used because it reaches the target projection

without dependence on slow iterative convergence under highly

imbalanced sparse joint supports.



\## Locked Tolerance



The functional-ANOVA purification tolerance remains:



1e-10



It is not relaxed.



A second direct projection pass is allowed only as a floating-point

cleanup operation if the first pass exceeds the locked tolerance.



Failure after that cleanup remains a hard error.



\## Surface Standardization



After purification, the observed pairwise contribution is

standardized to unit standard deviation.



A post-standardization numerical marginal-error guard of 1e-8 is

retained.



If a Gaussian draw is numerically unsuitable after purification and

standardization, another draw is taken from the previously locked

deterministic random-number stream.



At most 32 deterministic attempts are permitted.



\## Unchanged Protocol Elements



This amendment does not change:



\- the nine OpenML datasets

\- dataset-selection rules

\- the 5000-row cap

\- five realizations

\- four signal strengths

\- true-interaction count

\- main-effect count

\- selected feature seeds

\- row-sampling seeds

\- split seeds

\- surface seeds

\- Bernoulli seeds

\- empirical state encoding

\- positive interaction-degree-of-freedom requirement

\- disjoint pair-matching rule

\- main-effect coefficient

\- interaction-strength coefficients

\- target prevalence

\- AG-NAM training protocol

\- candidate budget

\- discovery repetitions

\- cross-fitting

\- selection threshold

\- ISR threshold

\- comparator set

\- primary endpoint

\- primary statistical test



\## Timing and Interpretation



The numerical refinement was made during generator feasibility

auditing and before generation of the first Real-X model-performance

or interaction-recovery result.



It is therefore a pre-outcome numerical implementation refinement

rather than result-driven tuning.

