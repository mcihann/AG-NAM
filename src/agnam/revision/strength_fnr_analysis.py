"""R9-A: Real-X interaction-strength false-negative-rate audit.

Inference uses a task-cluster bootstrap. Each cluster is an OpenML task,
including its five matched realizations at each positive strength.
No model is trained and null-interaction runs never receive an FNR of zero.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

STRENGTHS = ("weak", "moderate", "strong")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_realx_frame(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "task_id", "strength_name", "n_true_interactions",
        "selection_recall", "isr_recall", "delta_auroc",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Missing Real-X columns: {missing}")

    frame = raw.copy()
    frame["n_true_interactions"] = pd.to_numeric(frame["n_true_interactions"], errors="raise")
    frame["strength_name"] = frame["strength_name"].astype("string").str.lower()
    pos = frame[frame.n_true_interactions > 0].copy()
    null = frame[frame.n_true_interactions == 0].copy()
    if len(pos) + len(null) != len(frame):
        raise ValueError("Negative/invalid true-interaction count")
    if not pos.strength_name.isin(STRENGTHS).all():
        raise ValueError("Unexpected or missing strength label for a positive run")
    if not null.strength_name.isna().all():
        raise ValueError("Null runs have a non-null interaction-strength label")

    for tag in ("selection", "isr"):
        recall = pd.to_numeric(pos[f"{tag}_recall"], errors="raise").to_numpy(float)
        truth = pos.n_true_interactions.to_numpy(float)
        count = recall * truth
        if not (np.isfinite(recall).all() and ((0 <= recall) & (recall <= 1)).all()):
            raise ValueError(f"Invalid {tag} recall")
        if not np.allclose(count, np.rint(count), atol=1e-6):
            raise ValueError(f"Non-integer implied TP counts for {tag}")
        pos[f"{tag}_tp"] = np.rint(count).astype(int)
        pos[f"{tag}_fn"] = pos.n_true_interactions.astype(int) - pos[f"{tag}_tp"]
        pos[f"{tag}_fnr"] = pos[f"{tag}_fn"] / pos.n_true_interactions
    pos["tp_lost_by_isr"] = pos.selection_tp - pos.isr_tp
    if (pos.tp_lost_by_isr < 0).any():
        raise ValueError("ISR TP exceeds selection TP")
    if "realization_index" in pos.columns:
        if pos.duplicated(["task_id", "strength_name", "realization_index"]).any():
            raise ValueError("Duplicate Real-X task/strength/realization")
    if pos.groupby(["strength_name", "task_id"]).size().nunique() != 1:
        raise ValueError("Unbalanced strength/task realization counts")
    if pos.groupby("strength_name").task_id.nunique().nunique() != 1:
        raise ValueError("Strengths have unequal task counts")
    task_sets = [set(pos.loc[pos.strength_name == s, "task_id"]) for s in STRENGTHS]
    if not (task_sets[0] == task_sets[1] == task_sets[2]):
        raise ValueError("Strengths must contain the same task clusters")
    return pos, null


def summarize_strength_fnr(
    pos: pd.DataFrame,
    *,
    n_boot: int = 10000,
    seed: int = 20261118,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if n_boot < 100:
        raise ValueError("At least 100 bootstrap samples are required")
    keys = ["strength_name", "task_id"]
    per_task = pos.groupby(keys, as_index=False).agg(
        runs=("n_true_interactions", "size"),
        n_true=("n_true_interactions", "sum"),
        selection_tp=("selection_tp", "sum"),
        selection_fn=("selection_fn", "sum"),
        isr_tp=("isr_tp", "sum"),
        isr_fn=("isr_fn", "sum"),
        tp_lost_by_isr=("tp_lost_by_isr", "sum"),
        runs_with_tp_loss=("tp_lost_by_isr", lambda z: int((z > 0).sum())),
        mean_delta_auroc=("delta_auroc", "mean"),
    )
    per_task["selection_fnr"] = per_task.selection_fn / per_task.n_true
    per_task["isr_fnr"] = per_task.isr_fn / per_task.n_true
    per_task["fnr_increase_by_isr"] = per_task.isr_fnr - per_task.selection_fnr

    # Sample the same nine task indices for each strength and each bootstrap
    # replicate. This preserves the paired design across strength levels.
    task_ids = sorted(per_task.task_id.unique().tolist())
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(task_ids), size=(n_boot, len(task_ids)))
    records = []
    for strength in STRENGTHS:
        subset = per_task[per_task.strength_name == strength].set_index("task_id").loc[task_ids]
        values = subset[["n_true", "selection_fn", "isr_fn"]].to_numpy(float)
        boot = values[draws].sum(axis=1)
        selection_boot = boot[:, 1] / boot[:, 0]
        isr_boot = boot[:, 2] / boot[:, 0]
        delta_boot = isr_boot - selection_boot
        counts = subset[["runs", "n_true", "selection_tp", "selection_fn", "isr_tp", "isr_fn", "tp_lost_by_isr", "runs_with_tp_loss"]].sum()
        truth = float(counts["n_true"])
        records.append({
            "strength_name": strength,
            "n_tasks": len(task_ids),
            "n_runs": int(counts["runs"]),
            "n_true": int(truth),
            "selection_tp": int(counts["selection_tp"]),
            "selection_fn": int(counts["selection_fn"]),
            "isr_tp": int(counts["isr_tp"]),
            "isr_fn": int(counts["isr_fn"]),
            "tp_lost_by_isr": int(counts["tp_lost_by_isr"]),
            "runs_with_tp_loss": int(counts["runs_with_tp_loss"]),
            "selection_fnr": float(counts["selection_fn"] / truth),
            "selection_fnr_ci_low": float(np.quantile(selection_boot, .025)),
            "selection_fnr_ci_high": float(np.quantile(selection_boot, .975)),
            "isr_fnr": float(counts["isr_fn"] / truth),
            "isr_fnr_ci_low": float(np.quantile(isr_boot, .025)),
            "isr_fnr_ci_high": float(np.quantile(isr_boot, .975)),
            "fnr_increase_by_isr": float(counts["tp_lost_by_isr"] / truth),
            "fnr_increase_ci_low": float(np.quantile(delta_boot, .025)),
            "fnr_increase_ci_high": float(np.quantile(delta_boot, .975)),
            "mean_delta_auroc": float(pos.loc[pos.strength_name == strength, "delta_auroc"].mean()),
        })
    result = pd.DataFrame(records)
    return result, per_task


def run_strength_fnr_analysis(
    source: str | Path = "results/realx/benchmark/master_results.csv",
    output_root: str | Path = "results/revision/strength_fnr_r9/v1",
    *,
    n_boot: int = 10000,
    seed: int = 20261118,
) -> dict[str, Path]:
    source = Path(source)
    out = Path(output_root)
    pos, null = audit_realx_frame(pd.read_csv(source))
    summary, per_task = summarize_strength_fnr(pos, n_boot=n_boot, seed=seed)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": out / "strength_summary.csv",
        "per_task": out / "task_strength_summary.csv",
        "audit": out / "positive_run_audit.csv",
        "manifest": out / "manifest.json",
    }
    summary.to_csv(paths["summary"], index=False, float_format="%.12g")
    per_task.to_csv(paths["per_task"], index=False, float_format="%.12g")
    pos.to_csv(paths["audit"], index=False, float_format="%.12g")
    meta = {
        "analysis": "R9-A Real-X interaction-strength FNR",
        "status": "reviewer_requested_post_submission",
        "not_a_confirmatory_inferential_test": True,
        "source": str(source),
        "source_sha256": _sha256(source),
        "n_positive_runs": len(pos),
        "n_null_runs": len(null),
        "n_task_clusters": int(pos.task_id.nunique()),
        "bootstrap": "paired task-cluster resampling (all realizations within task)",
        "bootstrap_replicates": n_boot,
        "bootstrap_seed": seed,
        "null_fnr": "undefined and excluded",
        "outputs_sha256": {name: _sha256(path) for name, path in paths.items() if name != "manifest"},
    }
    paths["manifest"].write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("R9-A VERIFIED POSITIVE RUNS:", len(pos), "NULL RUNS:", len(null))
    print(summary[["strength_name", "n_runs", "n_true", "selection_tp", "isr_tp", "tp_lost_by_isr", "selection_fnr", "isr_fnr", "isr_fnr_ci_low", "isr_fnr_ci_high"]].round(5).to_string(index=False))
    print("R9-A SAVED:", out)
    return paths
