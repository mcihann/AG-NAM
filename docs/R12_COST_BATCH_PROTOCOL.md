# R12 matched AG-NAM/SIAN cost replication — locked protocol

This package adds two new scripts and one test file. Existing scripts and frozen experiment outputs are not changed.

## Scope
- S1–S4, realizations 01–05, to obtain 5 matched cost pairs per scenario (20 matched realizations).
- AG-NAM: exact five-run discovery, ISR, fresh final model from the existing isolated full-path runner.
- SIAN: native SIAN (version 0.9.5) with pairwise limit `MAX_K=2`, reuses already-completed cost measurement; no SIAN retraining.
- Data: exact raw split identity is enforced via SHA-256 index checks for each pair.
- Source pinning: compares hashes of all relevant AG-NAM modules with the validated S1/01 anchor before running. Stops on changes.
- Runtime: separate command-level wall-clock measurements including process launch and Python setup. No cross-language normalizations.
- Memory: sampled process-tree RSS and library-specific PyTorch peak CUDA allocation; not interchangeable with total device memory.
- Results are descriptive and exploratory; comparison is not a controlled benchmark of optimization quality or equal parameter counts.

## Commands (run from AG-NAM root in `agnam` environment)

1. Tests: `python -m unittest discover -s tests -p "test_r12_agnam_cost_batch.py" -v`
2. No-training preflight: `python scripts/run_r12_agnam_cost_batch.py --scenarios S1 S2 S3 S4 --start 1 --end 5`
3. **Explicitly starts fresh AG-NAM training** for incomplete run records only:
   `python scripts/run_r12_agnam_cost_batch.py --scenarios S1 S2 S3 S4 --start 1 --end 5 --execute`
4. Read-only aggregate collection:
   `python scripts/collect_r12_agnam_cost_batch.py --scenarios S1 S2 S3 S4 --start 1 --end 5`
5. Upload `AG_NAM_R12_AGNAM_S1_S4_R01_R05_Matched_Cost_Evidence.zip`.

The validated S1/01 pilot will be reused; the batch is resumable. A partial or failed run will STOP and requires diagnosis; existing folders are NEVER overwritten or silently deleted. On rerun, already completed and archived runs are skipped.
