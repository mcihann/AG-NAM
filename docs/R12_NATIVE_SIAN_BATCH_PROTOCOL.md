# R12 Native SIAN pairwise-constrained — checkpointed synthetic benchmark

## Scope

This package adds only two new scripts and their tests. It reuses the **unchanged** R12 pilot script for actual model fitting and the existing command-level profiler. The SHA-256 of the pilot runner is frozen to `41166131baac2c6b11995f1639479781f16521892ad52e3cb5345849c70a1d47`; a mismatch aborts the batch before training. The native SIAN package version is locked to 0.9.5 by the existing pilot script.

The pilot script runs the trained SIAN reference MLP, native FIS constrained to `MAX_K=2`, then trains the native SIAN sparse additive model with a fixed R12 adaptation of the author's package. It must **not** be described as unconstrained full-order SIAN replication. Data seeds and held-out test partitions are the same as the canonical synthetic protocol. Ground-truth true pairs enter recovery evaluation **after** selection, not FIS tuning. No parameters may be changed in response to test results.

## Executing safely

Use `python scripts/run_r12_sian_batch.py --scenario S1 --start 2 --end 5` to check all future work **without training**. Add `--execute` to train; never remove a `.inprogress` folder or overwrite a completed folder without a separate documented forensic assessment. Running this command again skips valid completed realizations; it stops on partial evidence or a failed run. No R1-R11 result is overwritten. A new process and new memory audit is created for each realization.

After 5 S1 runs: `python scripts/collect_r12_sian_batch.py --scenario S1 --start 1 --end 5`. The exporter recomputes AUROC, AUPRC, TP, FP, FN, precision, recall and F1 from archived per-run CSVs. It checks split-size invariants, native pairwise settings and profiler completion. It includes per-run summaries, predictions, interaction pairs, profiled wall-clock measurements, per-run CSV and descriptive summary in a new ZIP. The SIAN checkpoint `.pt` is not included in aggregate ZIP (it remains on the local computer). It refuses an existing output ZIP rather than overwriting.

If the first five runs pass the scientific review, execute S1 realizations 6–20. Only after S1 validation should you extend the same locked protocol to S2, S3 and S4 (each 1–20), subject to GPU capacity and the observed cost. Do not combine incomplete and complete cohorts in an inferential result. Inference and paired AG-NAM comparisons should occur only after reproducible benchmark completion.

## Reporting and caveats

- Selected-set recovery F1 and full-universe detector AUPRC are different metrics. Do not place them in a single ranking table.
- Native SIAN has a different parameterization and selection budget from AG-NAM. Matched datasets and test partitions permit predictive pairing, not an architecture-equal claim.
- Retained-pair Jaccard computed across different generated data realizations measures variation in selected pairs across *different datasets*. It is a descriptive cross-realization quantity, **not** within-dataset model retraining stability; do not claim cross-run reproducibility superiority from it.
- An AG-NAM canonical six-variant benchmark runtime is not the isolated Full AG-NAM cost. A separate isolated end-to-end AG-NAM timing protocol is required.
- The profiler reports sampled RSS and wall-clock time. PyTorch GPU memory-allocated peak is a different statistic. A missing nvidia-smi value must remain missing.
- R12 is an exploratory, post-review comparison; it is separate from previously frozen R1–R11 inferential families. Full iRF/NIT comparisons remain unrun.
