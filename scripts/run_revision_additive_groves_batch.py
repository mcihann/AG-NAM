"""Run independent TreeExtra Additive Groves pair recovery with prospective telemetry.

Standalone Windows CLI; does not alter AG-NAM source or previous baseline outputs.
Uses the exact synthetic train/validation split from the external-recovery runner.

Pilot: python scripts/run_revision_additive_groves_batch.py --scenario S1 --realization 2
Batch: python scripts/run_revision_additive_groves_batch.py --all
Resume an interrupted incomplete run: add --resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil


MODEL_RE = re.compile(
    r"Best model\s*:\s*Alpha\s*=\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)"
    r"\s*N\s*=\s*(\d+)",
    re.IGNORECASE,
)
BAG_RE = re.compile(r"\b(\d+)\s+bagging iterations\b", re.IGNORECASE)
PERF_RE = re.compile(
    r"Average performance:\s*([0-9.]+),\s*std:\s*([0-9.]+)",
    re.IGNORECASE,
)
ACTION_RE = re.compile(r"Suggested action:\s*(ag_[a-z_]+)([^\r\n]*)", re.IGNORECASE)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def suggested_action(output: str) -> tuple[str, list[str]]:
    """Extract exactly the last TreeExtra next-step recommendation."""
    matches = list(ACTION_RE.finditer(output))
    if not matches:
        raise ValueError("No TreeExtra 'Suggested action' found; stop instead of guessing")
    last = matches[-1]
    return last.group(1).lower(), last.group(2).strip().split()


def parse_model(output: str) -> tuple[float, int, int]:
    models = list(MODEL_RE.finditer(output))
    bags = list(BAG_RE.finditer(output))
    if not models or not bags:
        raise ValueError("Missing best alpha/N or number of bagging iterations")
    alpha = float(models[-1].group(1))
    n = int(models[-1].group(2))
    # TreeExtra ag_expand logs the CURRENT training budget in its header,
    # followed by the PREVIOUS model's bagging budget in a nested section.
    # The final regex hit is therefore stale (e.g. current=140, old=90).
    # Only accept the current run's pre-iteration header value.
    header = output.split("Model already trained:", 1)[0]
    header = header.split("Iteration 1 out of", 1)[0]
    current_bags = list(BAG_RE.finditer(header))
    if len(current_bags) != 1:
        raise ValueError("Expected exactly one current bagging count in TreeExtra header")
    b = int(current_bags[0].group(1))
    if not (math.isfinite(alpha) and alpha > 0 and n > 0 and b > 0):
        raise ValueError("Invalid best model parameters")
    return alpha, n, b


def parse_feature_selection_performance(output: str) -> tuple[float, float]:
    matches = list(PERF_RE.finditer(output))
    if not matches:
        raise ValueError("Missing final mean/std in ag_fs log")
    mean, std = map(float, matches[-1].groups())
    if not (math.isfinite(mean) and math.isfinite(std) and std > 0):
        raise ValueError("Invalid feature-selection performance statistics")
    return mean, std


def validate_expansion_action(program: str, args: list[str]) -> list[str]:
    """Accept the official TreeExtra expansion axes -a, -n, and -b.

    -a expands the alpha search down to a positive finite decimal value.
    -n and -b expand the number of terms and bagging iterations.  Preserve
    TreeExtra's exact argument order and numeric spelling for traceability.
    Reject malformed, duplicated, or unknown flags instead of guessing.
    """
    if program != "ag_expand":
        raise ValueError(f"Not an expansion action: {program}")
    if not args or len(args) % 2:
        raise ValueError(f"Unexpected ag_expand recommendation: {args}")
    seen: set[str] = set()
    for name, value in zip(args[::2], args[1::2]):
        if name in seen or name not in ("-a", "-n", "-b"):
            raise ValueError(f"Unexpected ag_expand recommendation: {args}")
        seen.add(name)
        if name == "-a":
            # Unlike -n/-b, alpha is a strictly positive real quantity.
            # Accept standard decimal/scientific notation, but not NaN/Inf.
            if not re.fullmatch(r"\+?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[Ee][+-]?[0-9]+)?", value):
                raise ValueError(f"Unexpected ag_expand recommendation: {args}")
            alpha = float(value)
            if not math.isfinite(alpha) or alpha <= 0:
                raise ValueError(f"Unexpected ag_expand recommendation: {args}")
        elif not re.fullmatch(r"[0-9]+", value) or int(value) < 1:
            raise ValueError(f"Unexpected ag_expand recommendation: {args}")
    return args


def atomic_json(path: Path, data: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def measure_process(cmd: list[str], workdir: Path, log_path: Path, heartbeat: int = 60) -> dict[str, Any]:
    """Measure spawned C++ process; not retroactively inferred from output."""
    print("RUN", " ".join(cmd), flush=True)
    print("LOG", log_path.relative_to(workdir), flush=True)
    begin = time.perf_counter()
    samples: list[int] = []
    observed_peak: int | None = None
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(cmd, cwd=workdir, stdout=log, stderr=subprocess.STDOUT)
        worker = psutil.Process(process.pid)
        last_report = begin
        while process.poll() is None:
            try:
                info = worker.memory_info()
                samples.append(int(info.rss))
                # Windows psutil may expose a kernel-reported peak working set.
                observed_peak = max(observed_peak or 0, int(getattr(info, "peak_wset", info.rss)))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            now = time.perf_counter()
            if heartbeat > 0 and now - last_report >= heartbeat:
                print(f"  running: {now - begin:.0f} seconds", flush=True)
                last_report = now
            time.sleep(0.15)
        rc = process.wait()
    elapsed = time.perf_counter() - begin
    result: dict[str, Any] = {
        "command": cmd,
        "log": log_path.name,
        "exit_code": rc,
        "seconds": round(elapsed, 6),
        "rss_start_mib": samples[0] / 2**20 if samples else None,
        "rss_end_mib": samples[-1] / 2**20 if samples else None,
        "rss_peak_mib": max(max(samples, default=0), observed_peak or 0) / 2**20
        if samples or observed_peak is not None else None,
        "memory_samples": len(samples),
        "memory_scope": "spawned_treeextra_process_only",
    }
    print(f"EXIT {rc} | seconds={elapsed:.3f} | peak_RSS_MiB={result['rss_peak_mib']}", flush=True)
    if rc != 0:
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-25:])
        raise RuntimeError(f"C++ command failed (exit {rc}): {cmd}\n{tail}")
    return result


def _last_model_search_log(workdir: Path, checkpoint: dict[str, Any]) -> Path:
    """Return the final successful ag_train/ag_expand log, not the FS log."""
    stages = checkpoint.get("stages", {})
    names = sorted((k for k in stages if re.fullmatch(r"ag_expand_\d+", k)),
                   key=lambda k: int(k.rsplit("_", 1)[1]))
    last = names[-1] if names else "ag_train"
    entry = stages.get(last)
    if not entry or entry.get("exit_code") != 0:
        raise ValueError(f"Missing successful model-search stage: {last}")
    log = workdir / f"{last}.log"
    if not log.is_file():
        raise FileNotFoundError(f"Model-search log missing: {log}")
    text = log.read_text(encoding="utf-8", errors="replace")
    if suggested_action(text)[0] != "ag_save":
        raise ValueError(f"Model search has not converged at {last}")
    return log


def _completed_bagging_mismatch(workdir: Path) -> tuple[int, int] | None:
    """Compare recorded FS bagging to the true, current model-search budget."""
    checkpoint_file = workdir / "ag_batch_checkpoint.json"
    if not checkpoint_file.is_file():
        return None  # Manually run S1-R01 has no batch checkpoint.
    checkpoint = json.loads(checkpoint_file.read_text(encoding="utf-8"))
    log = _last_model_search_log(workdir, checkpoint)
    _, _, expected = parse_model(log.read_text(encoding="utf-8", errors="replace"))
    stage = checkpoint["stages"].get("ag_fs")
    if not stage or stage.get("exit_code") != 0:
        raise ValueError("Completed run lacks successful FS telemetry")
    command = stage.get("command", [])
    if command.count("-b") != 1:
        raise ValueError("Completed FS command missing unique -b argument")
    actual = int(command[command.index("-b") + 1])
    return (actual, expected) if actual != expected else None


def prepare_bagging_repair(workdir: Path, ref_manifest: Path) -> dict[str, int]:
    """Archive a completed inconsistent run; reuse training/grid search only.

    Never silently overwrite the old run. On any invalid or incomplete input,
    fail closed without changing its checkpoint or results.
    """
    checkpoint_path = workdir / "ag_batch_checkpoint.json"
    manifest_path = workdir / "additive_groves_manifest.json"
    summary_path = workdir / "additive_groves_summary.csv"
    pairs_path = workdir / "additive_groves_pair_scores.csv"
    required = (checkpoint_path, manifest_path, summary_path, pairs_path,
                workdir / "ag_fs.log", workdir / "ag_interactions.log",
                workdir / "ag_fs_model.bin", workdir / "data.fs.attr")
    absent = [str(p) for p in required if not p.is_file()]
    if absent:
        raise FileNotFoundError(f"Refusing incomplete repair: missing {absent}")
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if checkpoint.get("reference_manifest_sha256") != sha256(ref_manifest):
        raise ValueError("Reference manifest changed; refusing bagging repair")
    if not checkpoint.get("finished_analysis"):
        raise ValueError("Only fully completed experiments may use --repair-bagging")
    if checkpoint.get("scenario") != json.loads(manifest_path.read_text(encoding="utf-8")).get("scenario"):
        raise ValueError("Experiment identities differ")
    mismatch = _completed_bagging_mismatch(workdir)
    if mismatch is None:
        raise ValueError("No bagging mismatch; repair is not necessary")
    actual, expected = mismatch
    old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    old_params = old_manifest.get("treeextra_training_parameters", {})
    if old_params.get("bagging_iterations") != actual:
        raise ValueError("Manifest and FS command bagging do not agree")
    stages = checkpoint["stages"]
    for name in ("ag_fs", "ag_interactions"):
        if stages.get(name, {}).get("exit_code") != 0:
            raise ValueError(f"Invalid recorded stage: {name}")
    # Verify the original output checksums before changing anything.
    for p in (pairs_path, summary_path, manifest_path):
        if checkpoint.get("outputs", {}).get(p.name) != sha256(p):
            raise ValueError(f"Original output hash mismatch: {p}")
    archive = workdir / "repair_archive" / f"bagging_{actual}_to_{expected}_pre_repair"
    if archive.exists():
        raise FileExistsError(f"Repair archive already exists: {archive}")
    files = list(required)
    optional = (workdir / "core_features.txt",)
    files.extend(p for p in optional if p.is_file())
    files.extend(p for p in workdir.glob("*.effect.txt") if p.is_file())
    files.extend(p for p in workdir.glob("*.interaction.txt") if p.is_file())
    files = list(dict.fromkeys(files))
    archive.mkdir(parents=True, exist_ok=False)
    for p in files:
        shutil.copy2(p, archive / p.name)
    archive_meta = {
        "reason": "ag_expand preceding-model bagging count was incorrectly selected",
        "previous_fs_bagging": actual,
        "correct_current_search_bagging": expected,
        "archived_sha256": {p.name: sha256(p) for p in files},
        "preserved_model_search_logs": sorted(
            name for name in stages if name == "ag_train" or name.startswith("ag_expand_")),
    }
    atomic_json(archive / "repair_audit.json", archive_meta)
    # Archive verified: now remove ONLY downstream artifacts to force their
    # prospective re-execution with the corrected current bagging budget.
    for p in files:
        if p != checkpoint_path:
            p.unlink()
    for name in ("ag_fs", "ag_interactions"):
        del stages[name]
    checkpoint.pop("finished_analysis", None)
    checkpoint.pop("outputs", None)
    checkpoint["bagging_repair"] = {
        "previous_fs_bagging": actual,
        "correct_current_search_bagging": expected,
        "archive": str(archive.relative_to(workdir)),
    }
    atomic_json(checkpoint_path, checkpoint)
    print(f"REPAIR ARCHIVED: old bagging={actual}, correct bagging={expected}", flush=True)
    print(f"REPAIR ARCHIVE: {archive}", flush=True)
    print("REUSE: ag_train and ag_expand stages; RE-RUN: ag_fs, ag_interactions, metrics", flush=True)
    return {"actual": actual, "expected": expected}


def run_one(repo: Path, scenario: str, realization: int, resume: bool, max_expansions: int,
            repair_bagging: bool = False) -> None:
    from types import SimpleNamespace

    import pandas as pd

    from agnam.benchmarking.synthetic_protocol import SyntheticBenchmarkProtocol
    from agnam.revision.external_recovery_runner import _prepare_data
    from run_revision_additive_groves_analysis import analyze

    root = repo / "results/revision/external_interaction_baselines"
    ref_manifest = root / "core_synthetic" / scenario / f"realization_{realization:02d}_manifest.json"
    if not ref_manifest.is_file():
        raise FileNotFoundError(f"Locked external baseline manifest missing: {ref_manifest}")
    ref = json.loads(ref_manifest.read_text(encoding="utf-8"))
    if ref.get("scenario") != scenario or int(ref.get("realization_number", -1)) != realization:
        raise ValueError("Reference manifest scenario/realization mismatch")

    workdir = root / "additive_groves" / scenario / f"realization_{realization:02d}"
    summary_path = workdir / "additive_groves_summary.csv"
    pairs_path = workdir / "additive_groves_pair_scores.csv"
    manifest_path = workdir / "additive_groves_manifest.json"
    if repair_bagging:
        prepare_bagging_repair(workdir, ref_manifest)
        resume = True
    elif all(p.is_file() for p in (summary_path, pairs_path, manifest_path)):
        mismatch = _completed_bagging_mismatch(workdir)
        if mismatch is not None:
            actual, expected = mismatch
            raise RuntimeError(
                f"INVALID COMPLETED RUN {scenario} R{realization:02d}: "
                f"FS bagging={actual}, current search bagging={expected}. "
                "Run this realization with --repair-bagging instead of skipping it."
            )
        print(f"SKIP {scenario} R{realization:02d}: three verified analysis outputs already exist", flush=True)
        return
    if workdir.exists() and any(workdir.iterdir()) and not resume:
        raise FileExistsError(f"Partial or preexisting run at {workdir}; inspect it and use --resume to continue")
    workdir.mkdir(parents=True, exist_ok=True)

    executable_dir = repo / "external/TreeExtra/win/Release"
    exes = {name: (executable_dir / (name + ".exe")).resolve()
            for name in ("ag_train", "ag_expand", "ag_fs", "ag_interactions")}
    for name, path in exes.items():
        if not path.is_file():
            raise FileNotFoundError(f"Required TreeExtra executable not present: {name}: {path}")

    checkpoint_path = workdir / "ag_batch_checkpoint.json"
    if checkpoint_path.is_file():
        if not resume:
            raise FileExistsError(f"Found prior checkpoint; pass --resume: {checkpoint_path}")
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("reference_manifest_sha256") != sha256(ref_manifest):
            raise ValueError("Locked reference manifest changed; refusing resume")
    else:
        checkpoint = {
            "scenario": scenario,
            "realization_number": realization,
            "reference_manifest_sha256": sha256(ref_manifest),
            "treeextra_version": "2.7.0 (Windows v143 Release/Win32 build)",
            "stages": {},
        }
        atomic_json(checkpoint_path, checkpoint)

    def stage(name: str, program: str, arguments: list[str]) -> str:
        path = workdir / f"{name}.log"
        full_cmd = [str(exes[program]), *arguments]
        existing = checkpoint["stages"].get(name)
        if existing is not None:
            if existing.get("exit_code") != 0 or existing.get("command") != full_cmd or not path.is_file():
                raise RuntimeError(f"Invalid checkpoint for {name}; manual review required")
            print(f"RESUME {name}: completed stage reused", flush=True)
            return path.read_text(encoding="utf-8", errors="replace")
        if path.is_file():
            raise FileExistsError(f"Uncheckpointed existing log {path}; manual review required")
        tel = measure_process(full_cmd, workdir, path)
        checkpoint["stages"][name] = tel
        atomic_json(checkpoint_path, checkpoint)
        return path.read_text(encoding="utf-8", errors="replace")

    spec = SimpleNamespace(
        scenario=scenario,
        dataset_seed=int(ref["dataset_seed"]),
        outer_split_seed=int(ref["outer_split_seed"]),
        final_split_seed=int(ref["final_split_seed"]),
    )
    print(f"=== INDEPENDENT ADDITIVE GROVES {scenario} R{realization:02d} ===", flush=True)
    xf, yf, xv, yv, names, _ = _prepare_data(spec, SyntheticBenchmarkProtocol())
    x_train = xf.to_numpy(dtype=np.float64)
    x_valid = xv.to_numpy(dtype=np.float64)
    if x_train.ndim != 2 or x_train.shape[1] != int(ref["n_features"]):
        raise ValueError("Generated feature count does not match locked baseline")
    if x_valid.shape[1] != x_train.shape[1] or not np.isfinite(x_train).all() or not np.isfinite(x_valid).all():
        raise ValueError("TreeExtra only accepts finite, numeric, equally shaped features")
    if not (set(np.unique(yf)).issubset({0, 1}) and set(np.unique(yv)).issubset({0, 1})):
        raise ValueError("Expected binary class labels 0 and 1")
    if not all(re.fullmatch(r"[A-Za-z][A-Za-z_0-9]*", name) for name in names):
        raise ValueError("Unsupported TreeExtra feature name")

    train_path = workdir / "data.train"
    valid_path = workdir / "data.valid"
    attr_path = workdir / "data.attr"
    if not train_path.is_file() and not valid_path.is_file() and not attr_path.is_file():
        np.savetxt(train_path, np.column_stack((yf, x_train)), delimiter="\t",
                   fmt=["%d"] + ["%.17g"] * len(names))
        np.savetxt(valid_path, np.column_stack((yv, x_valid)), delimiter="\t",
                   fmt=["%d"] + ["%.17g"] * len(names))
        attr_path.write_text("class: 0,1 (class).\n" + "".join(f"{n}: cont.\n" for n in names)
                             + "contexts:\n", encoding="utf-8")
        print(f"EXPORTED train={len(yf)}, validation={len(yv)}, features={len(names)}", flush=True)
    elif not all(p.is_file() for p in (train_path, valid_path, attr_path)):
        raise ValueError("Partial TreeExtra data export; refusing to guess")

    # Regenerate exact bytes and compare to on-disk inputs, so --resume cannot
    # silently substitute a different split or features.
    from io import StringIO
    for path, y, data in ((train_path, yf, x_train), (valid_path, yv, x_valid)):
        buf = StringIO()
        np.savetxt(buf, np.column_stack((y, data)), delimiter="\t",
                   fmt=["%d"] + ["%.17g"] * len(names))
        if path.read_text(encoding="utf-8") != buf.getvalue():
            raise ValueError(f"Input data differs from locked reconstruction: {path}")
    expected_attr = "class: 0,1 (class).\n" + "".join(f"{n}: cont.\n" for n in names) + "contexts:\n"
    if attr_path.read_text(encoding="utf-8") != expected_attr:
        raise ValueError("Attribute file differs from locked reconstruction")

    random_seed = str(ref["baseline_seed"])
    common = ["-t", "data.train", "-v", "data.valid", "-r", "data.attr"]
    output = stage("ag_train", "ag_train", common + ["-s", "layered", "-c", "roc", "-i", random_seed])
    for expansion in range(1, max_expansions + 2):
        action, flags = suggested_action(output)
        if action == "ag_save":
            print("MODEL SEARCH CONVERGED:", action, " ".join(flags), flush=True)
            break
        if action != "ag_expand":
            raise ValueError(f"Unexpected TreeExtra recommendation {action}: {flags}")
        if expansion > max_expansions:
            raise RuntimeError(f"Reached max expansions {max_expansions} without ag_save recommendation")
        output = stage(f"ag_expand_{expansion:02d}", "ag_expand", validate_expansion_action(action, flags))
    alpha, n, bagging = parse_model(output)
    print(f"SELECTED alpha={alpha:g}, N={n}, bagging={bagging}", flush=True)

    fs_args = common + ["-a", format(alpha, ".15g"), "-n", str(n), "-b", str(bagging),
                        "-c", "roc", "-i", random_seed, "-m", "ag_fs_model.bin"]
    fs_text = stage("ag_fs", "ag_fs", fs_args)
    mean, std = parse_feature_selection_performance(fs_text)
    if not (workdir / "data.fs.attr").is_file() or not (workdir / "ag_fs_model.bin").is_file():
        raise FileNotFoundError("ag_fs did not produce selected attribute / model files")

    interaction_args = ["-t", "data.train", "-v", "data.valid", "-r", "data.fs.attr",
                        "-a", format(alpha, ".15g"), "-n", str(n), "-b", str(bagging),
                        "-ave", format(mean, ".15g"), "-std", format(std, ".15g"),
                        "-m", "ag_fs_model.bin", "-c", "roc", "-i", random_seed]
    stage("ag_interactions", "ag_interactions", interaction_args)

    # Reuse the separately regression-tested analysis writer; it enforces exact
    # original pair universe and correct tested-pair coverage.
    analyze(repo, scenario, realization)
    if not all(p.is_file() for p in (summary_path, pairs_path, manifest_path)):
        raise RuntimeError("Analysis writer did not produce the three required outputs")

    # Prospective telemetry: only genuinely measured process values are written.
    stages = checkpoint["stages"]
    fit_names = [name for name in stages if name != "ag_interactions"]
    fit_secs = sum(stages[name]["seconds"] for name in fit_names)
    score_secs = stages["ag_interactions"]["seconds"]
    memory_peaks = [st["rss_peak_mib"] for st in stages.values() if st.get("rss_peak_mib") is not None]
    first_samples = [st["rss_start_mib"] for st in stages.values() if st.get("rss_start_mib") is not None]
    last_samples = [st["rss_end_mib"] for st in stages.values() if st.get("rss_end_mib") is not None]
    frame = pd.read_csv(summary_path)
    if len(frame) != 1 or str(frame.iloc[0]["method"]) != "additive_groves":
        raise ValueError("Unexpected analysis summary schema/content")
    frame.loc[0, "fit_seconds"] = fit_secs
    frame.loc[0, "scoring_seconds"] = score_secs
    frame.loc[0, "standalone_seconds"] = fit_secs + score_secs
    frame.loc[0, "rss_start_mib"] = first_samples[0] if first_samples else np.nan
    frame.loc[0, "rss_end_mib"] = last_samples[-1] if last_samples else np.nan
    frame.loc[0, "rss_peak_mib"] = max(memory_peaks) if memory_peaks else np.nan
    frame.to_csv(summary_path, index=False, float_format="%.12g")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["wall_time_and_peak_memory"] = (
        "prospectively measured; seconds sum C++ stages, "
        "max peak RSS across spawned processes; excludes Python data preparation"
    )
    manifest["treeextra_training_parameters"] = {"alpha": alpha, "n": n, "bagging_iterations": bagging,
                                                 "metric": "roc", "mode": "layered", "random_seed": int(random_seed)}
    manifest["stage_telemetry"] = stages
    manifest["total_cpp_stage_seconds"] = fit_secs + score_secs
    manifest["max_spawned_process_peak_rss_mib"] = max(memory_peaks) if memory_peaks else None
    atomic_json(manifest_path, manifest)
    checkpoint["finished_analysis"] = True
    checkpoint["outputs"] = {p.name: sha256(p) for p in (pairs_path, summary_path, manifest_path)}
    atomic_json(checkpoint_path, checkpoint)
    print(f"COMPLETE {scenario} R{realization:02d} | stage_seconds={fit_secs+score_secs:.3f} | "
          f"peak_RSS_MiB={max(memory_peaks) if memory_peaks else 'NA'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("S1", "S2", "S3", "S4"), default="S1")
    parser.add_argument("--realization", type=int, default=2)
    parser.add_argument("--all", action="store_true", help="Run/skip all 80 locked scenario-realizations")
    parser.add_argument("--resume", action="store_true", help="Continue verified checkpointed incomplete runs")
    parser.add_argument("--repair-bagging", action="store_true",
                        help="Archive and correct one previously completed run with a stale bagging count")
    parser.add_argument("--max-expansions", type=int, default=12)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    if args.realization < 1 or args.realization > 20 or args.max_expansions < 1:
        parser.error("Realization must be 1..20; max expansions must be at least one")
    if args.repair_bagging and args.all:
        parser.error("--repair-bagging is permitted for one explicit realization only")
    repo = args.repo_root.resolve()
    pairs = ((s, r) for s in ("S1", "S2", "S3", "S4") for r in range(1, 21)) if args.all else ((args.scenario, args.realization),)
    for scenario, realization in pairs:
        run_one(repo, scenario, realization, args.resume, args.max_expansions,
                repair_bagging=args.repair_bagging)


if __name__ == "__main__":
    main()
