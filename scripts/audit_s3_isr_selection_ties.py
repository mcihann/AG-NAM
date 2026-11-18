"""Read-only, tie-aware ISR vs selection-probability audit for AG-NAM S3.

Run from the repository root:
    python scripts/audit_s3_isr_selection_ties.py --last 5

The selection-probability baseline retains the same number of pairs as ISR.
Reported oracle tie bounds are sensitivity bounds, NOT deployable baselines.
No experiment results or source files are modified.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

METHODS = ("friedman_h_gbdt", "nid_shared_mlp", "sian_fid_shared_mlp")
PROXY_PAIRS = {
    frozenset(pair)
    for pair in (
        ("x2", "x11"), ("x1", "x12"), ("x11", "x12"),
        ("x6", "x13"), ("x5", "x14"), ("x13", "x14"),
    )
}
REQUIRED = {"feature_a", "feature_b", "selection_probability", "isr_retained", "is_true"}


def checked_bool(series: pd.Series, name: str) -> pd.Series:
    values = series.astype(str).str.strip().str.lower()
    if not values.isin(("true", "false")).all():
        raise ValueError(f"Invalid or missing Boolean value in {name}")
    return values.eq("true")


def class_bounds(strict_mask: np.ndarray, tied_mask: np.ndarray,
                 class_mask: np.ndarray, needed: int) -> tuple[int, int]:
    """Best/worst class counts among all permissible probability-tie breaks."""
    strict_class = int((strict_mask & class_mask).sum())
    tied_class = int((tied_mask & class_mask).sum())
    tied_nonclass = int(tied_mask.sum()) - tied_class
    return (
        strict_class + max(0, needed - tied_nonclass),
        strict_class + min(needed, tied_class),
    )


def analyze_pair_file(path: Path) -> dict:
    q = pd.read_csv(path)
    missing = REQUIRED.difference(q.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if q[["feature_a", "feature_b", "selection_probability"]].isna().any().any():
        raise ValueError(f"{path}: missing features or selection probabilities")
    q["is_true"] = checked_bool(q["is_true"], "is_true")
    q["isr_retained"] = checked_bool(q["isr_retained"], "isr_retained")
    probs = pd.to_numeric(q["selection_probability"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(probs).all() or ((probs < 0) | (probs > 1)).any():
        raise ValueError(f"{path}: invalid selection probability")
    pairs = [tuple(sorted((str(a), str(b)))) for a, b in zip(q.feature_a, q.feature_b)]
    if any(a == b for a, b in pairs) or len(set(pairs)) != len(pairs):
        raise ValueError(f"{path}: duplicate/self interaction pair")
    proxy = np.asarray([frozenset(pair) in PROXY_PAIRS for pair in pairs], dtype=bool)
    true = q.is_true.to_numpy(dtype=bool)
    retained = q.isr_retained.to_numpy(dtype=bool)
    if (proxy & true).any():
        raise ValueError(f"{path}: proxy reference unexpectedly overlaps true pair")
    k = int(retained.sum())

    # Use the exact deterministic tie-break previously tested by the user.
    alphabetic_top_k = q.sort_values(
        ["selection_probability", "feature_a", "feature_b"],
        ascending=[False, True, True], kind="mergesort",
    ).head(k)
    det_tp = int(alphabetic_top_k.is_true.sum())
    det_proxy = sum(
        (not bool(t)) and frozenset((str(a), str(b))) in PROXY_PAIRS
        for a, b, t in zip(alphabetic_top_k.feature_a,
                           alphabetic_top_k.feature_b, alphabetic_top_k.is_true)
    )

    if k == 0:
        tp_low = tp_high = proxy_low = proxy_high = 0
        has_boundary_tie = False
    else:
        cutoff = float(np.sort(probs)[-k])
        tolerance = 1e-10
        strict = probs > cutoff + tolerance
        tied = np.abs(probs - cutoff) <= tolerance
        need = k - int(strict.sum())
        if need < 0 or need > int(tied.sum()):
            raise AssertionError(f"{path}: inconsistent cutoff partition")
        tp_low, tp_high = class_bounds(strict, tied, true, need)
        proxy_low, proxy_high = class_bounds(strict, tied, proxy, need)
        has_boundary_tie = 0 < need < int(tied.sum())
        if not (tp_low <= det_tp <= tp_high and proxy_low <= det_proxy <= proxy_high):
            raise AssertionError(f"{path}: deterministic result is outside tie bounds")

    isr_tp = int((retained & true).sum())
    isr_proxy = int((retained & proxy).sum())
    return {
        "k": k, "isr_tp": isr_tp, "prob_tp": det_tp,
        "prob_tp_min": tp_low, "prob_tp_max": tp_high,
        "isr_proxy_fp": isr_proxy, "prob_proxy_fp": det_proxy,
        "prob_proxy_min": proxy_low, "prob_proxy_max": proxy_high,
        "boundary_tie": has_boundary_tie,
        "robust_isr_tp_win": isr_tp > tp_high,
        "robust_isr_tp_loss": isr_tp < tp_low,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=int, default=1)
    parser.add_argument("--last", type=int, default=5)
    parser.add_argument("--root", type=Path, default=Path(
        "results/revision/external_isr_hybrid/core_synthetic/S3"))
    args = parser.parse_args()
    if not (1 <= args.first <= args.last <= 20):
        parser.error("Expected 1 <= first <= last <= 20")

    rows = []
    for realization in range(args.first, args.last + 1):
        for method in METHODS:
            prefix = args.root / f"realization_{realization:02d}" / method
            pairs_path = prefix.with_name(method + "_isr_pairs.csv")
            summary_path = prefix.with_name(method + "_summary.csv")
            if not pairs_path.is_file() or not summary_path.is_file():
                raise FileNotFoundError(f"Expected both {pairs_path} and {summary_path}")
            result = analyze_pair_file(pairs_path)
            summary = pd.read_csv(summary_path)
            if len(summary) != 1:
                raise ValueError(f"{summary_path}: expected one summary row")
            s = summary.iloc[0]
            if (int(s.n_isr) != result["k"] or int(s.isr_tp) != result["isr_tp"]
                    or str(s.method) != method or int(s.realization_number) != realization):
                raise ValueError(f"{summary_path}: summary/pair mismatch")
            rows.append({"realization": realization, "method": method, **result})

    d = pd.DataFrame(rows)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    print("PER REALIZATION (probability bounds cover ALL cutoff-tie choices):")
    print(d.to_string(index=False))
    columns = ("k", "isr_tp", "prob_tp", "prob_tp_min", "prob_tp_max",
               "isr_proxy_fp", "prob_proxy_fp", "prob_proxy_min", "prob_proxy_max",
               "boundary_tie", "robust_isr_tp_win", "robust_isr_tp_loss")
    print("\nMETHOD TOTALS:")
    print(d.groupby("method")[list(columns)].sum().to_string())
    print("\nOVERALL TOTALS:")
    print(d[list(columns)].sum().to_string())
    print("\nINTERPRETATION: prob_tp_min/max and prob_proxy_min/max are ")
    print("oracle WORST/BEST tie-break sensitivity bounds, not executable baselines.")
    print("A robust ISR TP win requires isr_tp > prob_tp_max; a robust loss")
    print("requires isr_tp < prob_tp_min. Do not treat method-realization")
    print("observations as independent experimental units.")


if __name__ == "__main__":
    main()
