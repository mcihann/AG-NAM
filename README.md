# AG-NAM

**Attention-Guided Neural Additive Models for Reproducible and Interpretable Feature Interaction Learning**

AG-NAM is a reproducibility-aware neural additive modeling framework for selective and interpretable pairwise feature interaction learning in tabular data.

The framework is designed to determine when a candidate pairwise interaction is sufficiently supported to be included in an interpretable additive predictor. Rather than adding all possible feature interactions, AG-NAM combines residual-guided candidate discovery, repeated-selection reproducibility, and functional interaction-surface reproducibility.

## Method overview

AG-NAM follows five main stages:

1. **Main-effect residualization**  
   A Neural Additive Model (NAM) is trained using first-order feature effects. Leakage-safe out-of-fold probability residuals are then generated.

2. **Residual-guided interaction discovery**  
   A dedicated Residual Attention Proposer models residual structure. Candidate pairs are ranked using gradient-weighted attention attribution rather than raw attention magnitude.

3. **Repeated-selection reproducibility**  
   The complete discovery process is repeated across five deterministic runs. A pair is considered selection-stable when it appears in the Top-K candidate set in at least three of five runs.

4. **Interaction Stability Refinement (ISR)**  
   Independent pairwise residual models are trained for selection-stable candidates. Their interaction surfaces are purified to remove lower-order effects, and cross-run functional agreement is quantified using Pearson correlation.

5. **Fresh final AG-NAM training**  
   Only interactions satisfying both the selection-reproducibility and ISR criteria are included in a freshly trained final model.

The final predictor remains explicitly decomposable into:

```text
baseline + main effects + sparse pairwise interactions
```

Attention is used only for interaction proposal and is not treated as the explanation of the final predictor.

## Evaluation design

The study uses three complementary evaluation layers.

### Controlled synthetic benchmark

Four synthetic scenarios were designed to evaluate interaction recovery under complementary conditions, including sparse strong interactions, nonlinear interactions, correlated nuisance predictors, and weak interactions under high-dimensional irrelevant predictors and class imbalance.

Each scenario contains 20 independent realizations.

Publication-ready synthetic outputs are stored in:

```text
results/synthetic/publication/
```

### Real-X semi-synthetic benchmark

Real-X preserves empirical predictor distributions from nine OpenML datasets while introducing known pairwise interactions at controlled strengths.

The benchmark contains:

```text
9 datasets × 5 realizations × 4 interaction strengths = 180 runs
```

The four interaction-strength conditions are:

```text
null
weak
moderate
strong
```

The moderate condition is the prespecified confirmatory Real-X condition.

Publication outputs are stored in:

```text
results/realx/benchmark/publication/
```

The complete 180-run frozen benchmark evidence is stored in:

```text
results/realx/benchmark/runs/
```

The corresponding master and progress files are:

```text
results/realx/benchmark/master_results.csv
results/realx/benchmark/benchmark_progress.csv
```

Frozen evidence metadata and integrity information are stored in:

```text
results/realx/benchmark/freeze/
```

The evidence manifest defines the frozen Real-X evidence set used for the manuscript analyses.

### OpenML observational benchmark

The observational benchmark contains 28 heterogeneous binary-classification OpenML tasks spanning numeric, mixed, and categorical predictor regimes.

Because true interaction identities are unknown in this setting, retained OpenML pairs are interpreted as reproducible model structure rather than verified true or causal interactions.

Publication-ready OpenML outputs are stored in:

```text
results/openml/publication/
```

## Repository structure

```text
AG-NAM/
├── configs/        # Locked benchmark and baseline configurations
├── data/           # Dataset placeholders; raw/processed data are not versioned
├── docs/           # Prespecified methodological and statistical protocols
├── experiments/    # Experiment-related resources
├── figures/        # Figure resources
├── results/        # Frozen/publication benchmark outputs
├── scripts/        # Benchmark, audit, analysis, export, and figure scripts
├── src/
│   └── agnam/      # Core AG-NAM implementation
├── tests/          # Automated test suite
├── environment.yml
├── pyproject.toml
├── CITATION.cff
├── LICENSE
└── README.md
```

## Installation

AG-NAM requires Python 3.10.

The recommended setup uses Conda:

```bash
conda env create -f environment.yml
conda activate agnam
```

Alternatively, using an existing Python 3.10 environment:

```bash
pip install -e ".[dev,baselines]"
```

The project dependencies are defined in `pyproject.toml`.

The reference development environment used:

```text
Python: 3.10.19
PyTorch: 2.5.1+cu121
InterpretML / interpret: 0.7.8
CatBoost: 1.2.10
```

GPU acceleration is supported by PyTorch but is not required for inspecting the repository or publication outputs.

## Testing

Run the complete automated test suite from the repository root:

```bash
python -m pytest
```

The manuscript-release codebase passes the complete test suite.

## Benchmark execution

Synthetic benchmark scripts include:

```text
scripts/run_single_synthetic_benchmark.py
scripts/run_synthetic_batch.py
scripts/analyze_synthetic_benchmark.py
scripts/audit_synthetic_scenarios.py
scripts/export_synthetic_publication_tables.py
scripts/generate_synthetic_figures.py
```

Real-X benchmark scripts include:

```text
scripts/run_realx_integration_sanity.py
scripts/run_realx_batch.py
scripts/analyze_realx_benchmark.py
scripts/analyze_realx_secondary.py
scripts/freeze_realx_benchmark.py
scripts/export_realx_publication_tables.py
scripts/generate_realx_publication_figures.py
```

OpenML benchmark scripts include:

```text
scripts/audit_openml_tasks.py
scripts/run_single_openml_benchmark.py
scripts/run_openml_batch.py
scripts/analyze_openml_benchmark.py
scripts/export_openml_publication_tables.py
scripts/generate_openml_figures.py
```

Detailed methodological, execution, statistical-analysis, freezing, and reporting protocols are available under:

```text
docs/
```

These protocol documents should be consulted before rerunning full benchmarks because they record the locked experimental design and analysis decisions used for the manuscript.

## Configuration files

Important locked configuration files include:

```text
configs/benchmark_targets.yaml
configs/openml_benchmark_freeze.json
configs/realx_semisynthetic_v1.yaml
configs/external_baselines_requirements.txt
```

## External baselines

The observational benchmark includes Explainable Boosting Machine and CatBoost as external secondary comparators.

Locked baseline dependencies include:

```text
interpret-core==0.7.8
catboost==1.2.10
```

## Reproducibility and frozen evidence

The repository distinguishes generated experiment outputs from the publication evidence used in the manuscript.

For Real-X, the frozen evidence package contains the complete 180 run-level files together with the benchmark master/progress files and freeze metadata. The evidence manifest is intended to allow verification that the publication analyses are based on the locked benchmark outputs rather than subsequently regenerated results.

Publication tables and figures are generated from the corresponding frozen or finalized benchmark outputs.

## Interpretation scope

AG-NAM is intended as a selective interaction-augmentation framework, not as a universally more accurate replacement for a main-effect NAM.

Repeated discovery is used as a structural reproducibility criterion rather than as a guaranteed predictive-performance enhancement.

ISR is primarily a structural refinement mechanism. It evaluates whether independently learned purified interaction surfaces are reproducible across discovery runs.

In observational datasets, retained feature pairs should not be interpreted as causal interactions solely because they are reproducibly selected.

Exact additive reconstruction establishes computational faithfulness of the reported components to the fitted predictor; it does not establish causal validity.

## Data availability

The real-world benchmark datasets are obtained through OpenML. Dataset redistribution is therefore subject to the original data sources and their applicable terms and licenses.

Raw and processed datasets are intentionally excluded from version control.

## Manuscript

This repository accompanies the manuscript:

**AG-NAM: Attention-Guided Neural Additive Models for Reproducible and Interpretable Feature Interaction Learning**

The repository version associated with the manuscript will be preserved as a tagged release.

## Citation

If you use AG-NAM, its benchmarking framework, or the accompanying publication artifacts, please cite the software using the metadata provided in `CITATION.cff`.

The manuscript citation and DOI will be added after publication.

## License

The AG-NAM source code is released under the MIT License. See `LICENSE` for details.

Third-party datasets and dependencies remain subject to their respective licenses.