"""Build provenance-audited AG-NAM and standalone-baseline reporting tables.

This is analysis-only: no model training, no source-result mutations.
Scientific guardrail: AG-NAM repeated-discovery AUPRC is kept separate from
standalone full-pair AUPRC. The 5-realization S1 TreeExtra pilot is not
presented as an 80-realization four-scenario comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

SCENARIOS = ("S1", "S2", "S3", "S4")
METHODS = ("friedman_h_gbdt", "nid_shared_mlp", "sian_fid_shared_mlp")
PILOT_METHODS = METHODS + ("additive_groves",)
METRICS = ("interaction_auprc", "interaction_auroc", "ndcg", "f1_at_k", "f1_at_true_count")
LABELS = {
    "friedman_h_gbdt": "Friedman-H (GBDT)",
    "nid_shared_mlp": "NID (shared MLP)",
    "sian_fid_shared_mlp": "SIAN-FID (shared MLP)",
    "additive_groves": "Additive Groves (TreeExtra)",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _csv(path: Path, required: tuple[str, ...] = ()) -> pd.DataFrame:
    _require(path.is_file(), f"Missing source file: {path}")
    data = pd.read_csv(path)
    missing = set(required) - set(data.columns)
    _require(not missing, f"Missing columns in {path}: {sorted(missing)}")
    return data


def _summary(rows: pd.DataFrame, by: tuple[str, ...]) -> pd.DataFrame:
    result = []
    for group_keys, group in rows.groupby(list(by), sort=True):
        keys = group_keys if isinstance(group_keys, tuple) else (group_keys,)
        record = dict(zip(by, keys))
        record["n"] = len(group)
        for name in METRICS:
            vals = pd.to_numeric(group[name], errors="coerce")
            _require(vals.notna().all(), f"Missing or invalid {name} for {record}")
            _require(vals.between(0, 1).all(), f"Out of range {name} for {record}")
            record[f"{name}_mean"] = float(vals.mean())
            record[f"{name}_sd"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        times = pd.to_numeric(group["standalone_seconds"], errors="coerce")
        record["runtime_measured_n"] = int(times.notna().sum())
        record["runtime_mean_seconds_measured_only"] = (
            float(times.mean()) if times.notna().any() else None
        )
        result.append(record)
    return pd.DataFrame(result)


def _validate_pairs(path: Path, expected_count: int = 190) -> None:
    data = _csv(path, ("feature_a", "feature_b", "score"))
    _require(len(data) == expected_count, f"Expected {expected_count} pair scores: {path}")
    pairs = [tuple(sorted((str(a), str(b)))) for a, b in zip(data.feature_a, data.feature_b)]
    _require(len(set(pairs)) == expected_count, f"Duplicate feature pairs in {path}")
    _require(all(a != b for a, b in pairs), f"Self-pair in {path}")
    _require(pd.to_numeric(data.score, errors="coerce").notna().all(), f"Non-numeric score in {path}")


def collect(root: Path) -> tuple[dict[str, pd.DataFrame], dict]:
    ext = root / "results/revision/external_interaction_baselines"
    synthetic = root / "results/synthetic"
    source_paths: list[Path] = []
    core_rows: list[pd.DataFrame] = []
    agnam_rows = []
    pilot_rows = []
    groves_native_rows = []

    for scenario in SCENARIOS:
        for realization in range(1, 21):
            path = ext / "core_synthetic" / scenario / f"realization_{realization:02d}_summary.csv"
            frame = _csv(path, ("scenario", "realization_number", "dataset_seed", "method", "standalone_seconds") + METRICS)
            _require(len(frame) == 3, f"Expected 3 independent baselines: {path}")
            _require(set(frame.method.astype(str)) == set(METHODS), f"Baseline method coverage: {path}")
            _require(set(frame.scenario.astype(str)) == {scenario}, f"Scenario mismatch: {path}")
            _require(set(frame.realization_number.astype(int)) == {realization}, f"Realization mismatch: {path}")
            seeds = set(frame.dataset_seed.astype(int))
            _require(len(seeds) == 1, f"Baseline seed disagreement: {path}")
            source_paths.append(path)
            core_rows.append(frame)

            ag_path = synthetic / scenario / f"realization_{realization:02d}.csv"
            ag = _csv(ag_path, ("scenario", "realization_index", "dataset_seed", "mean_interaction_auprc", "selection_f1", "isr_f1"))
            _require(len(ag) == 1, f"AG-NAM source is not one row: {ag_path}")
            ag_row = ag.iloc[0]
            _require(str(ag_row.scenario) == scenario, f"AG-NAM scenario mismatch: {ag_path}")
            _require(int(ag_row.realization_index) == realization - 1, f"AG-NAM zero-based realization mismatch: {ag_path}")
            _require(int(ag_row.dataset_seed) == next(iter(seeds)), f"Dataset seed misalignment: {ag_path}")
            agnam_rows.append(ag)
            source_paths.append(ag_path)
            if scenario == "S1" and realization <= 5:
                pilot_rows.append(frame)

    core = pd.concat(core_rows, ignore_index=True)
    _require(len(core) == 240, f"Expected 240 full-scope baseline records, got {len(core)}")
    full = _summary(core, ("scenario", "method"))
    _require(len(full) == 12 and (full.n == 20).all(), "Core baseline coverage not 4 x 3 x 20")

    for realization in range(1, 6):
        folder = ext / "additive_groves" / "S1" / f"realization_{realization:02d}"
        path = folder / "additive_groves_summary.csv"
        frame = _csv(path, ("scenario", "realization_number", "dataset_seed", "method", "standalone_seconds") + METRICS)
        _require(len(frame) == 1 and str(frame.iloc[0].method) == "additive_groves", f"Unexpected TreeExtra summary: {path}")
        _require(str(frame.iloc[0].scenario) == "S1" and int(frame.iloc[0].realization_number) == realization,
                 f"TreeExtra realization mismatch: {path}")
        ag_seed = int(_csv(synthetic / "S1" / f"realization_{realization:02d}.csv").iloc[0].dataset_seed)
        _require(int(frame.iloc[0].dataset_seed) == ag_seed, f"TreeExtra seed mismatch: {path}")
        manifest_path = folder / "additive_groves_manifest.json"
        _require(manifest_path.is_file(), f"Missing TreeExtra manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _require(int(manifest["n_all_pairs"]) == 190 and int(manifest["n_true_pairs"]) == 3,
                 f"TreeExtra pair-universe / truth mismatch: {manifest_path}")
        if realization >= 2:
            _require(int(manifest["treeextra_training_parameters"]["bagging_iterations"]) ==
                     {2: 140, 3: 140, 4: 210, 5: 210}[realization],
                     f"Uncorrected TreeExtra bagging setting: {manifest_path}")
            _require(math.isfinite(float(manifest["total_cpp_stage_seconds"])),
                     f"Missing measured TreeExtra runtime: {manifest_path}")
        pairs = folder / "additive_groves_pair_scores.csv"
        _validate_pairs(pairs)
        source_paths.extend((path, manifest_path, pairs))
        pilot_rows.append(frame)
        groves_native_rows.append({
            "scenario": "S1", "realization_number": realization,
            "native_tp": int(manifest["native_tp"]), "native_fp": int(manifest["native_fp"]),
            "native_fn": int(manifest["native_fn"]), "native_f1": float(manifest["native_f1"]),
            "tested_pairs": int(manifest["n_tested_pairs"]),
            "untested_pairs": int(manifest["n_untested_pairs"]),
            "bagging_iterations": (
                int(manifest["treeextra_training_parameters"]["bagging_iterations"])
                if "treeextra_training_parameters" in manifest else None
            ),
        })

    pilot = pd.concat(pilot_rows, ignore_index=True)
    _require(len(pilot) == 20, f"Expected 5 realizations x 4 methods, got {len(pilot)}")
    for i in range(1, 6):
        block = pilot[pilot.realization_number == i]
        _require(set(block.method) == set(PILOT_METHODS), f"Pilot method coverage at R{i:02d}")
        _require(len(set(block.dataset_seed.astype(int))) == 1, f"Pilot seed mismatch at R{i:02d}")
    pilot_summary = _summary(pilot, ("method",))
    _require((pilot_summary.n == 5).all(), "Pilot counts must be exactly five per method")

    ag_table_path = synthetic / "publication/table_synthetic_structure.csv"
    ag_table = _csv(ag_table_path, ("scenario", "n_realizations", "interaction_auprc_mean", "interaction_auprc_std", "selection_f1_mean", "isr_f1_mean", "false_positive_reduction_mean"))
    _require(set(ag_table.scenario.astype(str)) == set(SCENARIOS), "AG-NAM aggregate scenario coverage")
    _require((ag_table.n_realizations.astype(int) == 20).all(), "AG-NAM aggregate must have 20 runs per scenario")
    source_paths.append(ag_table_path)
    ag_all = pd.concat(agnam_rows, ignore_index=True)
    _require(len(ag_all) == 80, "Expected 80 AG-NAM synthetic realization records")
    for _, row in ag_table.iterrows():
        obs = ag_all[ag_all.scenario == row.scenario].mean_interaction_auprc.astype(float)
        _require(abs(float(obs.mean()) - float(row.interaction_auprc_mean)) < 1e-5,
                 f"AG-NAM publication mean does not match realization records: {row.scenario}")

    paired_diffs = []
    groves_ap = pilot[pilot.method == "additive_groves"].set_index("realization_number").interaction_auprc
    for method in METHODS:
        comparator = pilot[pilot.method == method].set_index("realization_number").interaction_auprc
        differences = groves_ap.sort_index() - comparator.sort_index()
        paired_diffs.append({"scenario": "S1", "method_a": "additive_groves",
                             "method_b": method, "n_paired": 5,
                             "mean_ap_difference_a_minus_b": float(differences.mean()),
                             "median_ap_difference_a_minus_b": float(differences.median()),
                             "individual_differences": ";".join(f"{x:.8f}" for x in differences)})

    report_tables = {
        "table_standalone_full_scope.csv": full,
        "table_standalone_s1_paired_pilot.csv": pilot_summary,
        "table_agnam_synthetic_structure_separate.csv": ag_table,
        "table_additive_groves_native_decisions.csv": pd.DataFrame(groves_native_rows),
        "table_additive_groves_paired_ap_differences.csv": pd.DataFrame(paired_diffs),
    }
    audit = {
        "scope": {"standalone_full": "4 scenarios x 20 paired realizations x 3 independent detectors",
                  "extended_s1_pilot": "S1 only x 5 paired realizations x 4 independent detectors",
                  "agnam_structure": "4 scenarios x 20 realizations, repeated discovery summaries"},
        "counts": {"full_standalone_rows": len(core), "pilot_rows": len(pilot),
                   "agnam_realization_rows": len(ag_all), "treeextra_rows": 5},
        "evaluation_caveats": [
            "AG-NAM mean_interaction_auprc is mean over repeated discovery runs and is not pooled into independent-detector ranking table.",
            "AG-NAM pairwise-selection F1 / ISR F1 are not equivalent to independent-detector F1@K.",
            "TreeExtra/Additive Groves is restricted to S1 n=5; this is a descriptive paired pilot, not a four-scenario comparison.",
            "SIAN-FID Shared MLP is an interaction-detection component, not the complete SIAN system.",
            "Additive Groves realization 01 runtime was not measured; never impute it.",
            "The external ranking universe is 190 pairs for S1. True pairs filtered during TreeExtra feature selection stay in the full universe at the shared untested floor score.",
        ],
        "source_sha256": {str(p.relative_to(root)).replace('\\', '/'): _digest(p)
                           for p in sorted(set(source_paths))},
    }
    return report_tables, audit


def _v(mean: float, sd: float) -> str:
    return f"{mean:.4f} ± {sd:.4f}"


def render_manuscript_tables(tables: dict[str, pd.DataFrame]) -> str:
    full = tables["table_standalone_full_scope.csv"]
    pilot = tables["table_standalone_s1_paired_pilot.csv"]
    ag = tables["table_agnam_synthetic_structure_separate.csv"]
    native = tables["table_additive_groves_native_decisions.csv"]
    lines = [
        "# Experimental results — interaction recovery and refinement",
        "",
        "## Independent interaction-detection baselines (S1–S4; n=20 per scenario)",
        "",
        "| Scenario | Detector | AUPRC | AUROC | NDCG | F1@K |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, r in full.iterrows():
        lines.append(f"| {r.scenario} | {LABELS[r.method]} | "
                     f"{_v(r.interaction_auprc_mean,r.interaction_auprc_sd)} | "
                     f"{_v(r.interaction_auroc_mean,r.interaction_auroc_sd)} | "
                     f"{_v(r.ndcg_mean,r.ndcg_sd)} | "
                     f"{_v(r.f1_at_k_mean,r.f1_at_k_sd)} |")
    lines += [
        "",
        "*Values are means ± sample standard deviations across 20 independent synthetic realizations within each scenario. "
        "Friedman-H is implemented with GBDT; NID and SIAN-FID are shared-MLP detector implementations. "
        "The latter does not constitute an evaluation of the complete SIAN framework.*",
        "",
        "## Extended S1 detector comparison (paired pilot; n=5)",
        "",
        "| Independent detector | AUPRC | AUROC | NDCG | F1@K |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, r in pilot.sort_values("interaction_auprc_mean", ascending=False).iterrows():
        lines.append(f"| {LABELS[r.method]} | "
                     f"{_v(r.interaction_auprc_mean,r.interaction_auprc_sd)} | "
                     f"{_v(r.interaction_auroc_mean,r.interaction_auroc_sd)} | "
                     f"{_v(r.ndcg_mean,r.ndcg_sd)} | "
                     f"{_v(r.f1_at_k_mean,r.f1_at_k_sd)} |")
    lines += [
        "",
        "*Five matched S1 synthetic realizations (same dataset seeds and train/validation protocol) were used to assess "
        "Additive Groves via TreeExtra against three independent detectors. Performance differences are descriptive "
        "and should not be treated as evidence of significance or generalization across S2–S4.*",
        "",
        "## AG-NAM repeated-discovery and structural-refinement analysis (n=20 per scenario)",
        "",
        "| Scenario | Mean discovery-run interaction AUPRC | Selection F1 | ISR F1 | Mean false-positive reduction |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, r in ag.sort_values("scenario").iterrows():
        lines.append(f"| {r.scenario} | {_v(r.interaction_auprc_mean,r.interaction_auprc_std)} | "
                     f"{r.selection_f1_mean:.4f} | {r.isr_f1_mean:.4f} | {r.false_positive_reduction_mean:.4f} |")
    lines += [
        "",
        "*AG-NAM interaction AUPRC is averaged across repeated discovery runs and is reported separately from "
        "single-ranking standalone-detector AUPRC. Selection F1 and ISR F1 evaluate accepted pair sets, whereas "
        "F1@K evaluates ranked candidates at a fixed K; these are different targets.*",
        "",
        "## Additive Groves native interaction decisions (S1 paired pilot)",
        "",
        "| Realization | TP | FP | FN | Native F1 | Evaluated pairs |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in native.iterrows():
        lines.append(f"| {int(r.realization_number):02d} | {int(r.native_tp)} | {int(r.native_fp)} | "
                     f"{int(r.native_fn)} | {r.native_f1:.4f} | {int(r.tested_pairs)}/190 |")
    lines += [
        "",
        "*TreeExtra first performs feature selection; interaction tests are restricted to retained features. "
        "In ranking evaluation, untested pairs are assigned a common score below all tested pairs, "
        "preserving the 190-pair S1 universe. Native decisions and ranking metrics are reported separately. "
        "TreeExtra runtime was prospectively recorded for four of five realizations; the first runtime is unavailable.*",
        "",
        "## Suggested Results narrative (English; self-contained)",
        "",
        "The interaction-detection comparison covered four synthetic scenarios, with 20 independent "
        "realizations per scenario. Across these experiments, the strongest mean ranking method depended "
        "on the interaction structure: the SIAN-FID detector achieved the highest mean AUPRC in S1, "
        "S2, and S4, whereas Friedman-H narrowly led in S3. This variation indicates that interaction "
        "ranking is sensitive to the scenario and that no single detector dominates all settings.",
        "",
        "A complementary paired comparison in S1 evaluated the TreeExtra implementation of Additive "
        "Groves on five realizations. This experiment assessed the method's feature-selection and "
        "restricted-interaction-testing pipeline within a common 190-pair ranking universe. As the "
        "extended comparison comprises five S1 realizations, its conclusions are limited to that "
        "setting and are interpreted descriptively.",
        "",
        "AG-NAM was additionally evaluated for reproducible interaction discovery and structural "
        "refinement over 20 realizations in each scenario. The ISR-refined selection F1 exceeded "
        "the initial selection F1 across S1–S4. These set-recovery results characterize the "
        "reliability-oriented selection stage and should not be equated with standalone detectors' "
        "F1@K or their single-ranking AUPRC values.",
        "",
        "## Reporting boundaries",
        "",
        "- No claim of AG-NAM ranking superiority follows from these tables.",
        "- This comparison does not represent complete NIT, SIAN, or iRF pipelines.",
        "- No inferential significance is claimed for the five S1 Additive Groves comparisons.",
        "- No unmeasured runtime is imputed; hardware/back-end differences require explicit context in any runtime discussion.",
        "- The final text must be reconciled against the manuscript's existing experimental sections and table numbering before insertion.",
        "",
    ]
    return "\n".join(lines)


def write_outputs(root: Path, destination: Path, overwrite: bool = False) -> None:
    tables, audit = collect(root)
    targets = [destination / name for name in tables] + [destination / "r11_comparison_audit.json", destination / "r11_manuscript_tables_draft.md"]
    existing = [p for p in targets if p.exists()]
    _require(overwrite or not existing, "Refusing to overwrite generated output without --overwrite: " + ", ".join(map(str, existing)))
    destination.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(destination / name, index=False, float_format="%.10g")
    (destination / "r11_comparison_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (destination / "r11_manuscript_tables_draft.md").write_text(render_manuscript_tables(tables), encoding="utf-8")
    print("=== R11 REPORTING TABLES VERIFIED ===", flush=True)
    print("Standalone full scope: 4 scenarios x 20 realizations x 3 methods = 240 records", flush=True)
    print("S1 paired pilot: 5 realizations x 4 methods = 20 records", flush=True)
    print("AG-NAM repeated-discovery results: 4 scenarios x 20 realizations = 80 records", flush=True)
    print("Additive Groves: five verified 190-pair rankings; first runtime left unmeasured", flush=True)
    print("OUTPUTS:", flush=True)
    for target in targets:
        print(" ", target, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path, default=Path.cwd())
    ap.add_argument("--output-dir", type=Path, default=Path("results/revision/r11_publication_tables"))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    root = args.repo_root.resolve()
    destination = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    write_outputs(root, destination, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
