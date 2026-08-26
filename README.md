\# AG-NAM



\*\*Attention-Guided Neural Additive Models for Reproducible Feature Interaction Discovery\*\*



AG-NAM is a research framework for intrinsically interpretable tabular learning. The method uses attention only as an interaction-proposal mechanism and retains feature interactions according to their reproducibility across resampled training subsets.



The final predictive model is attention-free and decomposes predictions into explicit main-effect and pairwise-interaction contributions.



\## Research status



This repository is under active research development.



The methodological design, benchmark datasets, evaluation protocol, and primary hypotheses are defined before final benchmark evaluation to reduce researcher degrees of freedom.



\## Core principles



\- Intrinsically decomposable prediction

\- Attention-guided interaction proposal

\- Selection reproducibility

\- Interaction-surface reproducibility

\- Sparse pairwise interaction modeling

\- Leakage-free nested evaluation

\- Reproducible experiments



\## Repository structure



```text

AG-NAM/

├── configs/

├── data/

├── docs/

├── experiments/

├── figures/

├── results/

├── scripts/

├── src/

│   └── agnam/

└── tests/

