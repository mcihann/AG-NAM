# R8 — Exploratory calibration and final-model resource-use audit

## Purpose and fixed scope

Post-submission response to reviewer requests for calibration and computational
cost. Tasks **49, 3918, 3021, 167141, 9952, 14952, 14965, 7592** were
chosen after inspecting original OpenML AUROC and ISR retained-pair counts.
This is a **post hoc, deliberately heterogeneous exploratory sample**, *not*
a representative selection of 28 tasks, confirmatory inference, or a fresh
claim of full AG-NAM superiority.

- Reuse locked OpenML outer task split via `load_openml_task` and the original
  `OpenMLBenchmarkProtocol` schedule (repeat=fold=sample=0).
- Use original `prepare_final_prediction_data` (train-only preprocessing,
  identical final validation seed/fraction) and `SyntheticModelConfig()`.
- Read original ISR `isr_retained` flags from task-specific pair audits;
  verify count and exact feature-name membership against the loaded task.
  No rediscovery, selection, ISR or other baseline training is performed.
- Fit **only** MainEffectNAM and final AGNAM, from fresh initialization,
  with exactly the original `final_model_seed`, training configuration,
  early stopping and prediction routines. Empty ISR selections are kept.
- Record prediction probabilities on the **untouched outer test split**.
  Never calibrate, tune thresholds, select models or adjust hyperparameters
  using the outer test split.
- Save per-task probability vectors, equal-width positive-class ten-bin
  reliability data, Brier, ECE, AUROC, AUPRC, elapsed final fit + prediction
  time, peak PyTorch CUDA memory allocated/reserved and optional sampled
  process RSS. CPU peak RSS is sampled, not an exact instantaneous peak.
- Compare rerun AUROC to frozen original AUROC to 1e-6, reporting discrepancies
  rather than forcing a match or silently changing the protocol.
- Store hashes of source benchmark/pair files, frozen outer indices and
  original test labels, test predictions/summary/reliability bins.
- The original `runtime_seconds` is **entire benchmark execution** and is
  not directly comparable with R8 **final-fit plus test prediction** times.
  CUDA peak allocated is process-level during the audited interval; it is
  not a hardware-global memory peak or total training pipeline cost.
- This does not estimate calibration over all 28 OpenML tasks or external
  validity. No post-hoc significance claims should be made on the eight tasks.

## Invocation (Windows CMD from repository root)

```bat
python -m pytest -q
python scripts\run_revision_openml_calibration.py --task-id 49
python scripts\run_revision_openml_calibration.py --all
python scripts\analyze_revision_openml_calibration.py
```

Files saved to `results/revision/openml_calibration_r8/v1/task_<id>` and
`results/revision/openml_calibration_r8/v1/analysis` (including PNG reliability diagrams). These output directories
are outside the locked original benchmark. Resume verifies input and output
hashes; a stale result is not automatically replaced. `--force` is available
only for an explicitly documented rerun after auditing a mismatch.

## Calibration definition

For observed labels y in {0,1}, predicted positive-class probabilities p,
Brier is mean((p-y)^2). Equal-width ten-bin ECE is
sum_bin(n_bin/N * abs(mean(p_bin) - mean(y_bin))); empty bins do not
contribute. The rightmost bin contains p=1. Reliability bins are descriptive;
model training and evaluation occur on strictly separated splits.

## Audit interpretation

A successful 1/1 task parity check is not proof the other seven tasks will
match. Review task-level AUROC deltas and parity flags before running all.
The task-49 no-pair case measures the additive fallback and should be
identified explicitly. Save the complete source and output hashes.
