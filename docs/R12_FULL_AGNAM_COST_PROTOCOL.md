# R12 isolated cost measurement of the Full AG-NAM pipeline

This new runner uses the installed AG-NAM's own canonical stage functions. It measures the exact sequence:

1. Fresh synthetic data generation and identical raw train/validation/test split as SIAN.
2. Five-run cross-fitted residual attention discovery.
3. Interaction Stability Refinement (ISR).
4. Fresh Full AG-NAM training and test evaluation (and final split-specific preprocessing).

It intentionally **does not train** Main NAM, Oracle, Random-Pair, Single-Run or No-ISR predictive comparator variants. These are unrelated to the deployed *Full AG-NAM* computational path, although the main-effect fits **within discovery** remain included.

It uses the locked `SyntheticBenchmarkProtocol()` and `SyntheticModelConfig()` without retuning. If native SIAN `summary.json` is available for the same scenario/realization, it **requires exact SHA-256 identity of train, validation, and test raw sample indices**. The original AG-NAM synthetic result is only read for a descriptive reproducibility comparison; it is never overwritten.

**Run from AG-NAM root, existing `agnam` environment:**

```bat
python -m unittest discover -s tests -p "test_r12_agnam_isolated_cost.py" -v
python scripts/run_r12_agnam_isolated_cost.py --scenario S1 --realization 1
python scripts/r12_profile_command.py --label r12_s1_01_agnam_fullpath --execute -- python scripts/run_r12_agnam_isolated_cost.py --scenario S1 --realization 1 --execute
```

The last command launches **one complete fresh AG-NAM training** with external wall-time and sampled CPU/GPU memory measurement. Expected output:

- `results/revision/r12_agnam_fullpath_cost/S1/realization_01/summary.json`
- `results/revision/r12_cost_audit/r12_s1_01_agnam_fullpath/measurement.json`

The files are write-once: the same realization or profiler label cannot overwrite existing data. For repeated trials, use a different realization and label. Never use the original six-model `run_r12_agnam_canonical_cost.py` when reporting a Full AG-NAM **only** runtime.

**Comparison caveats:** AG-NAM and SIAN use different computational graphs, optimizers, feature preprocessing and term budgets; this is a real-world procedural cost comparison rather than a controlled model architecture study. SIAN itself is run with the R12 `MAX_K=2` restriction. Elapsed and sampled process memory are environment-specific. Do not label a one-run time as a scenario mean or an accurate GPU peak.

No full SIAM/NIT/iRF/Real-X generality follows from this experiment. These gaps must be addressed through transparent scope limits or further specifically targeted experiments.

**After successful training, collect the finished pilot evidence without training again:**

```bat
python scripts/collect_r12_agnam_fullpath_cost.py --scenario S1 --realization 1
```

Upload `C:\Users\mucah\AG-NAM\AG_NAM_R12_S1_R01_Fullpath_Cost_Evidence.zip`. It will include the AG-NAM full-path timing and metrics, the external resource profile, the relevant SIAN split hashes (where available) and the original AG-NAM benchmark row, with SHA-256 manifest entries. All outputs are write-once.
