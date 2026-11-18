"""Read-only audit of all locked synthetic MainEffectNAM residual caches for R7.

Run from AG-NAM repository root:
    python scripts/audit_revision_r7_residual_cache.py

No model fitting and no mutation of the original cache/benchmark files.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from agnam.benchmarking.synthetic_protocol import SyntheticBenchmarkProtocol
from agnam.revision.external_isr_runner import (
    _discovery_split,
    _load_residual_cache,
    _prepare_development,
    _spec_for,
)

ROOT = Path("results/revision/external_isr_hybrid/core_synthetic")
OUT = Path("results/revision/residual_sensitivity/r7_cache_audit.csv")
EPSILON = 1e-6
TOL = 1e-6


def validate_probabilities(y: np.ndarray, residual: np.ndarray, context: str):
    """Recover predictions from y-p; reject grossly invalid residual arrays."""
    y = np.asarray(y, dtype=np.float64)
    residual = np.asarray(residual, dtype=np.float64).reshape(-1)
    if y.ndim != 1 or residual.shape != y.shape:
        raise AssertionError(f"Shape mismatch: {context}")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(residual).all():
        raise AssertionError(f"Invalid labels or residual values: {context}")
    raw_p = y - residual
    if not np.isfinite(raw_p).all():
        raise AssertionError(f"Nonfinite reconstructed probabilities: {context}")
    if np.any((raw_p < -TOL) | (raw_p > 1.0 + TOL)):
        raise AssertionError(f"Reconstructed probabilities out of bounds: {context}")
    p = np.clip(raw_p, 0.0, 1.0)
    p_safe = np.clip(p, EPSILON, 1.0 - EPSILON)
    # Signed Bernoulli deviance residual, for y in {0, 1}.
    dev = np.where(y == 1, np.sqrt(-2.0 * np.log(p_safe)),
                   -np.sqrt(-2.0 * np.log1p(-p_safe)))
    if not np.isfinite(dev).all() or np.any((dev > 0) & (y == 0)) or np.any((dev < 0) & (y == 1)):
        raise AssertionError(f"Invalid deviance residuals: {context}")
    return {
        "n": len(y),
        "p_min": float(raw_p.min()),
        "p_max": float(raw_p.max()),
        "p_clipped": int(np.count_nonzero(p != p_safe)),
        "prob_residual_abs_mean": float(np.abs(residual).mean()),
        "deviance_residual_abs_mean": float(np.abs(dev).mean()),
    }


def main() -> None:
    protocol = SyntheticBenchmarkProtocol()
    if protocol.n_realizations != 20 or protocol.n_discovery_runs != 5:
        raise AssertionError("Protocol differs from the locked 20x5 schedule")
    rows = []
    for scenario in protocol.scenarios:
        for realization in range(1, 21):
            spec = _spec_for(scenario, realization, protocol)
            X, y, _, _ = _prepare_development(spec, protocol)
            features = tuple(map(str, X.columns))
            for run_index in range(protocol.n_discovery_runs):
                path = ROOT / scenario / f"realization_{realization:02d}" / "residual_runs" / f"run_{run_index + 1:02d}.npz"
                split = _discovery_split(y, spec.discovery_base_seed + run_index)
                arrays, metadata = _load_residual_cache(
                    path, spec=spec, run_index=run_index, split=split,
                    n_development_rows=len(X), expected_feature_names=features,
                )
                train = validate_probabilities(y[split[0]], arrays["train_residuals"], str(path) + ":train")
                valid = validate_probabilities(y[split[1]], arrays["validation_residuals"], str(path) + ":validation")
                rows.append({
                    "scenario": scenario,
                    "realization": realization,
                    "run": run_index + 1,
                    "seed": spec.discovery_base_seed + run_index,
                    "cache_path": path.as_posix(),
                    "metadata_split_sha256": metadata["split_sha256"],
                    **{f"train_{k}": v for k, v in train.items()},
                    **{f"validation_{k}": v for k, v in valid.items()},
                })
        print(f"VALIDATED {scenario}: 100 / 100 residual caches", flush=True)
    result = pd.DataFrame(rows)
    assert len(result) == 400
    assert len(result.drop_duplicates(["scenario", "realization", "run"])) == 400
    assert all(result.groupby("scenario").size().eq(100))
    assert result["train_n"].min() > 0 and result["validation_n"].min() > 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT, index=False, float_format="%.12g")
    print("TOTAL VERIFIED: 400 / 400")
    print("TOTAL TRAIN/VALIDATION PROBABILITY CLIPS:",
          int(result["train_p_clipped"].sum()), int(result["validation_p_clipped"].sum()))
    print("RECONSTRUCTED PROBABILITY RANGE:",
          f"{min(result['train_p_min'].min(), result['validation_p_min'].min()):.9f}",
          f"{max(result['train_p_max'].max(), result['validation_p_max'].max()):.9f}")
    print("OUTPUT:", OUT)
    print("R7 CACHE AUDIT COMPLETE: no model training performed")


if __name__ == "__main__":
    main()
